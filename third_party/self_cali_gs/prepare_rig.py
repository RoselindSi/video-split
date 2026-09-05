"""Build a calibration-free COLMAP initializer for a six-camera rig.

This script is intentionally kept outside the Self-Cali-GS checkout.  It uses
PyCOLMAP's native rig representation: six sensor intrinsics, one fixed sensor
transform per camera, and one pose per synchronized frame.  No values from the
hardware calibration file are read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics


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


def camera_model_name(camera):
    model_name = getattr(camera, "model_name", None)
    if model_name is not None:
        return str(model_name)
    model = camera.model
    return str(getattr(model, "name", model))


def summarize_errors(errors):
    finite = sorted(error for error in errors if math.isfinite(error))

    def percentile(fraction):
        if not finite:
            return None
        position = fraction * (len(finite) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return finite[lower]
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "observations": len(errors),
        "finite_observations": len(finite),
        "nonfinite_observations": len(errors) - len(finite),
        "mean_px": statistics.fmean(finite) if finite else None,
        "median_px": percentile(0.5),
        "p90_px": percentile(0.9),
        "p99_px": percentile(0.99),
        "max_px": finite[-1] if finite else None,
        "over_10px": sum(error > 10.0 for error in finite),
        "over_100px": sum(error > 100.0 for error in finite),
    }


def observation_error_report(reconstruction):
    by_camera = defaultdict(list)
    all_errors = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        camera_errors = by_camera[int(image.camera_id)]
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            try:
                xyz = reconstruction.points3D[point2D.point3D_id].xyz
                projected = image.project_point(xyz)
                error = math.hypot(
                    float(projected[0]) - float(point2D.x),
                    float(projected[1]) - float(point2D.y),
                )
            except (IndexError, KeyError, OverflowError, TypeError, ValueError):
                error = math.inf
            camera_errors.append(error)
            all_errors.append(error)
    return {
        "all": summarize_errors(all_errors),
        "by_camera_id": {
            str(camera_id): summarize_errors(errors)
            for camera_id, errors in sorted(by_camera.items())
        },
    }


def rig_report(rig):
    sensors = []
    for sensor in sorted(rig.sensor_ids(), key=lambda item: int(item.id)):
        entry = {
            "type": str(sensor.type).split(".")[-1],
            "id": int(sensor.id),
            "reference": bool(rig.is_ref_sensor(sensor)),
        }
        if not entry["reference"]:
            entry["sensor_from_rig"] = rig.sensor_from_rig(
                sensor).matrix().tolist()
        sensors.append(entry)
    return {
        "num_sensors": int(rig.num_sensors()),
        "sensors": sensors,
    }


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
            "model": camera_model_name(camera),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": camera.params.tolist(),
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    rigs = {
        str(rig_id): rig_report(rig)
        for rig_id, rig in reconstruction.rigs.items()
    }
    return {
        "registered_images": reconstruction.num_reg_images(),
        "points3D": reconstruction.num_points3D(),
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
        "cameras": cameras,
        "rigs": rigs,
        "images": images,
        "observation_errors": observation_error_report(reconstruction),
    }


def write_report(root, reconstruction):
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


def prepare(scene_root, device="auto", resume=False, report_only=False):
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
    if report_only:
        if not (rigged_path / "images.bin").is_file():
            raise FileNotFoundError("report-only requires a completed rigged model")
        return write_report(root, pycolmap.Reconstruction(rigged_path))
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

    return write_report(root, reconstruction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.report_only:
        parser.error("--resume and --report-only are mutually exclusive")
    print(json.dumps(prepare(
        args.scene, args.device, args.resume, args.report_only), indent=2))


if __name__ == "__main__":
    main()
