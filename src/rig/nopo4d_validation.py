"""Prepare a six-camera clip for a pose-free NoPo4D validation.

The production panorama is deliberately not involved.  In ``raw`` mode this
module only splits the three packed stereo videos and resizes their six eyes;
it does not read calibration and does not align source pixels.  ``rectified``
is an explicit ablation that uses lens calibration for preprocessing only.

NoPo4D expects camera-major names::

    cam0_t0.png, cam0_t1.png, ..., cam5_t0.png, cam5_t1.png

Hardware camera ``cam1`` maps to NoPo4D ``cam0``.  The zero-based names are an
upstream file-format convention, not a camera reorder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np


MODULES = (
    ("cam12", "cam1", "cam2"),
    ("cam34", "cam3", "cam4"),
    ("cam56", "cam5", "cam6"),
)
DEFAULT_SIZE = (448, 336)
IMAGE_NAME = re.compile(r"cam(?P<camera>\d+)_t(?P<time>\d+)\.png")


def videos_from_databag(databag):
    root = Path(databag).resolve()
    videos = {key: root / f"{key}.mp4" for key, _, _ in MODULES}
    missing = [os.fspath(path) for path in videos.values()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing packed six-camera video: " + ", ".join(missing))
    return {key: os.fspath(path) for key, path in videos.items()}


def split_module_frames(module_frames):
    """Return hardware-camera frames in cam1..cam6 order."""
    from src.rig.render_wide import split_halves

    cameras = {}
    for module, left_name, right_name in MODULES:
        if module not in module_frames:
            raise ValueError(f"missing synchronized module frame: {module}")
        frame = module_frames[module]
        if frame.ndim != 3 or frame.shape[1] % 2:
            raise ValueError(
                f"{module} is not an even-width packed stereo frame: "
                f"{frame.shape}")
        cameras[left_name], cameras[right_name] = split_halves(frame)
    return cameras


def resize_without_camera_parameters(image, size=DEFAULT_SIZE):
    """Centre-crop to the target aspect and resize without camera geometry."""
    import cv2

    width, height = (int(v) for v in size)
    if width <= 0 or height <= 0 or width % 14 or height % 14:
        raise ValueError("NoPo4D width and height must be positive multiples of 14")
    src_h, src_w = image.shape[:2]
    target_aspect = width / height
    source_aspect = src_w / src_h
    if source_aspect > target_aspect:
        crop_w = max(1, int(round(src_h * target_aspect)))
        x0 = (src_w - crop_w) // 2
        image = image[:, x0:x0 + crop_w]
    elif source_aspect < target_aspect:
        crop_h = max(1, int(round(src_w / target_aspect)))
        y0 = (src_h - crop_h) // 2
        image = image[y0:y0 + crop_h]
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def inspect_image_grid(image_dir, expected_cameras=None):
    """Validate and return camera-major paths from a NoPo4D image folder."""
    root = Path(image_dir)
    grid = {}
    for path in root.glob("*.png"):
        match = IMAGE_NAME.fullmatch(path.name)
        if match:
            key = (int(match.group("camera")), int(match.group("time")))
            if key in grid:
                raise ValueError(f"duplicate NoPo4D input {key}: {path}")
            grid[key] = path
    if not grid:
        raise ValueError(f"no camN_tN.png inputs in {root}")

    cameras = sorted({key[0] for key in grid})
    times = sorted({key[1] for key in grid})
    if cameras != list(range(len(cameras))):
        raise ValueError(f"camera indexes must be contiguous from zero: {cameras}")
    if times != list(range(len(times))):
        raise ValueError(f"time indexes must be contiguous from zero: {times}")
    if expected_cameras is not None and len(cameras) != int(expected_cameras):
        raise ValueError(
            f"expected {expected_cameras} cameras, found {len(cameras)}")
    missing = [(camera, time) for camera in cameras for time in times
               if (camera, time) not in grid]
    if missing:
        raise ValueError(f"incomplete camera/time grid: missing {missing}")
    return [grid[(camera, time)] for camera in cameras for time in times]


def _rectification_maps(databag, size, fov_scale):
    calibration = Path(databag) / "calibration.yaml"
    if not calibration.is_file():
        raise FileNotFoundError(
            f"rectified ablation requires {calibration}")
    from src.rig.calibration import RigCalibration
    from src.rig.stabstitch_export import rectify_map

    rig = RigCalibration(os.fspath(calibration))
    return {name: rectify_map(rig, name, size=size, fov_scale=fov_scale)
            for _, left, right in MODULES for name in (left, right)}


def export_clip(databag, output_dir, start=3000, frames=2, stride=1,
                size=DEFAULT_SIZE, mode="raw", fov_scale=0.6):
    """Export one synchronized six-camera clip and write its provenance."""
    import cv2

    if frames < 1 or stride < 1 or start < 0:
        raise ValueError("start must be non-negative; frames and stride positive")
    if mode not in ("raw", "rectified"):
        raise ValueError("mode must be 'raw' or 'rectified'")
    size = tuple(int(v) for v in size)
    # Validate patch alignment even in rectified mode.
    resize_without_camera_parameters(np.zeros((14, 14, 3), np.uint8), size)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"NoPo4D output directory is not empty: {root}")

    videos = videos_from_databag(databag)
    maps = (_rectification_maps(databag, size, fov_scale)
            if mode == "rectified" else None)
    captures = {}
    try:
        for module, path in videos.items():
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise OSError(f"cannot open {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            captures[module] = capture

        written = []
        source_fps = float(captures["cam12"].get(cv2.CAP_PROP_FPS) or 30.0)
        for time_index in range(int(frames)):
            if time_index:
                for capture in captures.values():
                    for _ in range(int(stride) - 1):
                        if not capture.grab():
                            raise EOFError("source ended while advancing clip stride")
            packed = {}
            for module, capture in captures.items():
                ok, image = capture.read()
                if not ok:
                    frame = start + time_index * stride
                    raise EOFError(f"cannot read {module} frame {frame}")
                packed[module] = image
            camera_frames = split_module_frames(packed)
            for hardware_index, camera_name in enumerate(sorted(camera_frames)):
                image = camera_frames[camera_name]
                if maps is None:
                    image = resize_without_camera_parameters(image, size)
                else:
                    map_x, map_y = maps[camera_name]
                    image = cv2.remap(
                        image, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
                name = f"cam{hardware_index}_t{time_index}.png"
                path = root / name
                if not cv2.imwrite(os.fspath(path), image):
                    raise OSError(f"cannot write {path}")
                written.append(name)
    finally:
        for capture in captures.values():
            capture.release()

    # The model sorts filenames, so verify the exact camera-major grid after
    # writing instead of trusting loop order.
    ordered = inspect_image_grid(root, expected_cameras=6)
    manifest = {
        "schema": "video-split.nopo4d-input.v1",
        "databag": os.fspath(Path(databag).resolve()),
        "videos": videos,
        "preprocessing": mode,
        "uses_camera_parameters": mode == "rectified",
        "alignment_uses_camera_parameters": False,
        "camera_mapping": {f"cam{i - 1}": f"cam{i}" for i in range(1, 7)},
        "start_frame": int(start),
        "frames": int(frames),
        "stride": int(stride),
        "source_fps": source_fps,
        "frame_indexes": [int(start + i * stride) for i in range(frames)],
        "image_size": list(size),
        "fov_scale": float(fov_scale) if mode == "rectified" else None,
        "camera_major_files": [path.name for path in ordered],
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def wide_intrinsics(width, height, hfov_deg=150.0, vfov_deg=90.0):
    """Pixel-space intrinsics for the single virtual wide render."""
    if not (1.0 < hfov_deg < 179.0 and 1.0 < vfov_deg < 179.0):
        raise ValueError("virtual field of view must be between 1 and 179 degrees")
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def central_camera(c2w, central_index=2):
    """Use a learned central orientation at the robust learned rig centre."""
    poses = np.asarray(c2w, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected [camera,4,4] c2w poses, got {poses.shape}")
    if not 0 <= central_index < len(poses):
        raise ValueError(f"central camera {central_index} outside {len(poses)} poses")
    target = poses[central_index].copy()
    target[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    return target


def run_command(python, runner, nopo_root, image_dir, output_dir,
                model_dir, da3_model_dir, extra=()):
    command = [
        python, os.fspath(Path(runner).resolve()),
        "--nopo_root", os.fspath(Path(nopo_root).resolve()),
        "--image_dir", os.fspath(Path(image_dir).resolve()),
        "--output_dir", os.fspath(Path(output_dir).resolve()),
        "--model", os.fspath(Path(model_dir).resolve()),
        "--da3_model", os.fspath(Path(da3_model_dir).resolve()),
        "--num_cameras", "6",
    ]
    command.extend(extra)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--databag", required=True)
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--start", type=int, default=3000)
    export_parser.add_argument("--frames", type=int, default=2)
    export_parser.add_argument("--stride", type=int, default=1)
    export_parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    export_parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    export_parser.add_argument("--mode", choices=("raw", "rectified"),
                               default="raw")
    export_parser.add_argument("--fov_scale", type=float, default=0.6)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--nopo_root", required=True)
    run_parser.add_argument("--images", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--da3_model", required=True)
    run_parser.add_argument("--python", default="python3")
    run_parser.add_argument(
        "--runner",
        default=Path(__file__).resolve().parents[2] /
                "third_party/nopo4d/wide_inference.py")
    run_parser.add_argument("--gpu", default="0")
    run_parser.add_argument("--hfov", type=float, default=150.0)
    run_parser.add_argument("--vfov", type=float, default=90.0)
    run_parser.add_argument("--render_inputs", action="store_true")
    args = parser.parse_args()

    if args.command == "export":
        manifest = export_clip(
            args.databag, args.out, start=args.start, frames=args.frames,
            stride=args.stride, size=(args.width, args.height), mode=args.mode,
            fov_scale=args.fov_scale)
        print(json.dumps(manifest, indent=2))
        return

    inspect_image_grid(args.images, expected_cameras=6)
    extra = ["--gpu", args.gpu, "--hfov", str(args.hfov),
             "--vfov", str(args.vfov)]
    if args.render_inputs:
        extra.append("--render_inputs")
    command = run_command(
        args.python, args.runner, args.nopo_root, args.images, args.out,
        args.model, args.da3_model, extra)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
