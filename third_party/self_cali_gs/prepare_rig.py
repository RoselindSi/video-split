"""Build a calibration-free COLMAP initializer for a six-camera rig.

This script is intentionally kept outside the Self-Cali-GS checkout.  It uses
PyCOLMAP's native rig representation: six sensor intrinsics, one fixed sensor
transform per camera, and one pose per synchronized frame.  No values from the
hardware calibration file are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def six_camera_rig_config():
    return [{
        "cameras": [
            {"image_prefix": f"cam{camera}/", "ref_sensor": camera == 0}
            for camera in range(6)
        ]
    }]


def select_reconstruction(reconstructions):
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce a reconstruction")
    return max(
        reconstructions.values(),
        key=lambda reconstruction: (
            reconstruction.num_reg_images(), reconstruction.num_points3D()))


def load_best_reconstruction(pycolmap, root):
    models = []
    for path in Path(root).iterdir():
        if path.is_dir() and (path / "images.bin").is_file():
            models.append(pycolmap.Reconstruction(path))
    if not models:
        raise RuntimeError(f"no completed COLMAP models in {root}")
    return max(
        models,
        key=lambda reconstruction: (
            reconstruction.num_reg_images(), reconstruction.num_points3D()))


def reconstruction_report(reconstruction):
    images = {}
    for image_id, image in reconstruction.images.items():
        if image.has_pose:
            images[image.name] = {
                "image_id": int(image_id),
                "camera_id": int(image.camera_id),
                "projection_center": image.projection_center().tolist(),
            }
    cameras = {
        str(camera_id): {
            "model": camera.model_name,
            "width": int(camera.width),
            "height": int(camera.height),
            "params": camera.params.tolist(),
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    rigs = {
        str(rig_id): rig.todict(recursive=True)
        for rig_id, rig in reconstruction.rigs.items()
    }
    return {
        "registered_images": reconstruction.num_reg_images(),
        "points3D": reconstruction.num_points3D(),
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
        "cameras": cameras,
        "rigs": rigs,
        "images": images,
    }


def prepare(scene_root, device="auto", resume=False):
    import pycolmap

    root = Path(scene_root).resolve()
    image_dir = root / "input"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing input images: {image_dir}")
    expected = [image_dir / f"cam{camera}" for camera in range(6)]
    if not all(path.is_dir() for path in expected):
        raise ValueError("input must use cam0/..cam5/ rig folders")

    work = root / "colmap"
    database_path = work / "database.db"
    unrigged_path = work / "unrigged"
    rigged_path = work / "rigged"
    prepared_path = root / "self_cali"
    if resume:
        if not database_path.is_file() or not unrigged_path.is_dir():
            raise FileNotFoundError("resume requires database.db and unrigged models")
        if prepared_path.exists() or any(rigged_path.iterdir()):
            raise FileExistsError("rigged or Self-Cali output already exists")
        reconstruction = load_best_reconstruction(
            pycolmap, unrigged_path)
    else:
        occupied = [path for path in (database_path, unrigged_path, rigged_path,
                                      prepared_path) if path.exists()]
        if occupied:
            raise FileExistsError(f"initializer output already exists: {occupied}")
        work.mkdir(parents=True)
        unrigged_path.mkdir()
        rigged_path.mkdir()

        device_value = getattr(pycolmap.Device, device)
        pycolmap.extract_features(
            database_path, image_dir,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            camera_model="OPENCV_FISHEYE", device=device_value)
        pycolmap.match_exhaustive(database_path, device=device_value)

        options = pycolmap.IncrementalPipelineOptions()
        options.multiple_models = True
        options.min_model_size = 6
        options.min_num_matches = 12
        options.ba_refine_sensor_from_rig = True
        options.mapper.init_min_num_inliers = 50
        options.mapper.abs_pose_min_num_inliers = 15
        reconstructions = pycolmap.incremental_mapping(
            database_path, image_dir, unrigged_path, options=options)
        reconstruction = select_reconstruction(reconstructions)
        reconstruction.write(unrigged_path / "best")

    config_path = work / "rig_config.json"
    with open(config_path, "w", encoding="utf-8") as stream:
        json.dump(six_camera_rig_config(), stream, indent=2)
        stream.write("\n")
    database = pycolmap.Database.open(database_path)
    try:
        pycolmap.apply_rig_config(
            pycolmap.read_rig_config(config_path), database, reconstruction)
    finally:
        database.close()

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = True
    ba_options.refine_rig_from_world = True
    ba_options.refine_sensor_from_rig = True
    ba_options.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
    ba_options.loss_function_scale = 1.0
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    reconstruction.write(rigged_path)

    pycolmap.undistort_images(
        prepared_path, rigged_path, image_dir, output_type="COLMAP")
    sparse = prepared_path / "sparse"
    sparse_zero = sparse / "0"
    if not sparse_zero.exists():
        sparse_zero.mkdir()
        for path in list(sparse.iterdir()):
            if path != sparse_zero:
                shutil.move(path, sparse_zero / path.name)
    raw_target = prepared_path / "fish" / "images"
    shutil.copytree(image_dir, raw_target)

    report = {
        "schema": "video-split.seam360-colmap-rig.v1",
        "uses_camera_parameters": False,
        "camera_grouping": "one-intrinsic-model-per-physical-camera",
        "pose_grouping": "one-rig-pose-per-synchronized-time",
        "rig_config": six_camera_rig_config(),
        **reconstruction_report(reconstruction),
    }
    with open(root / "colmap_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.scene, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
