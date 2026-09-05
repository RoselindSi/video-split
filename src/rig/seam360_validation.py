"""Prepare raw six-camera clips for self-calibrating seam experiments.

The exporter deliberately does not read ``calibration.yaml``.  Camera identity
and capture time are retained in each filename so an external optimiser can
share one lens model per physical camera and one rig pose per timestamp.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np

from src.rig.nopo4d_validation import MODULES, split_module_frames, videos_from_databag


FLAT_IMAGE_NAME = re.compile(
    r"cam(?P<camera>[0-5])_t(?P<time>\d{3,})\.(?:jpg|png)")
RIG_IMAGE_NAME = re.compile(
    r"cam(?P<camera>[0-5])/t(?P<time>\d{3,})\.(?:jpg|png)")
DEFAULT_SIZE = (768, 608)


def parse_rig_image_name(name):
    """Return zero-based physical camera and timestamp indexes."""
    normalized = os.fspath(name).replace("\\", "/")
    match = RIG_IMAGE_NAME.fullmatch(normalized)
    if not match:
        match = FLAT_IMAGE_NAME.fullmatch(Path(normalized).name)
    if not match:
        raise ValueError(f"not a six-camera image name: {name}")
    return int(match.group("camera")), int(match.group("time"))


def resize_full_frame(image, size=DEFAULT_SIZE):
    """Resize without cropping, rectification, or a camera model."""
    import cv2

    width, height = (int(value) for value in size)
    if width < 2 or height < 2:
        raise ValueError("image dimensions must be at least two pixels")
    interpolation = (cv2.INTER_AREA
                     if image.shape[1] > width or image.shape[0] > height
                     else cv2.INTER_LINEAR)
    return cv2.resize(image, (width, height), interpolation=interpolation)


def inspect_rig_grid(image_dir, expected_cameras=6):
    """Validate a complete time-major six-camera image grid."""
    root = Path(image_dir)
    grid = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            key = parse_rig_image_name(path.relative_to(root))
        except ValueError:
            continue
        if key in grid:
            raise ValueError(f"duplicate camera/time image {key}: {path}")
        grid[key] = path
    if not grid:
        raise ValueError(f"no camN_tNNN images in {root}")
    cameras = sorted({camera for camera, _ in grid})
    times = sorted({time for _, time in grid})
    if cameras != list(range(expected_cameras)):
        raise ValueError(f"expected cameras 0..{expected_cameras - 1}, found {cameras}")
    if times != list(range(len(times))):
        raise ValueError(f"time indexes must be contiguous from zero: {times}")
    missing = [(camera, time) for time in times for camera in cameras
               if (camera, time) not in grid]
    if missing:
        raise ValueError(f"incomplete six-camera grid: missing {missing}")
    return [grid[(camera, time)] for time in times for camera in cameras]


def export_clip(databag, output_dir, start=3000, frames=12, stride=2,
                size=DEFAULT_SIZE, extension="png", layout="rig"):
    """Export synchronized raw eyes while preserving camera/time identity."""
    import cv2

    if start < 0 or frames < 1 or stride < 1:
        raise ValueError("start must be non-negative; frames and stride positive")
    if extension not in ("jpg", "png"):
        raise ValueError("extension must be jpg or png")
    if layout not in ("rig", "flat"):
        raise ValueError("layout must be rig or flat")
    size = tuple(int(value) for value in size)
    resize_full_frame(np.zeros((4, 4, 3), np.uint8), size)

    root = Path(output_dir)
    image_dir = root / "input"
    image_dir.mkdir(parents=True, exist_ok=True)
    if any(image_dir.iterdir()):
        raise FileExistsError(f"Self-Cali input directory is not empty: {image_dir}")

    videos = videos_from_databag(databag)
    captures = {}
    try:
        for module, path in videos.items():
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise OSError(f"cannot open {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            captures[module] = capture

        fps = float(captures["cam12"].get(cv2.CAP_PROP_FPS) or 30.0)
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
                    source_index = start + time_index * stride
                    raise EOFError(f"cannot read {module} frame {source_index}")
                packed[module] = image
            cameras = split_module_frames(packed)
            for camera_index, camera_name in enumerate(sorted(cameras)):
                image = resize_full_frame(cameras[camera_name], size)
                if layout == "rig":
                    path = image_dir / f"cam{camera_index}" / f"t{time_index:03d}.{extension}"
                    path.parent.mkdir(exist_ok=True)
                else:
                    path = image_dir / f"cam{camera_index}_t{time_index:03d}.{extension}"
                options = ([int(cv2.IMWRITE_JPEG_QUALITY), 95]
                           if extension == "jpg" else [])
                if not cv2.imwrite(os.fspath(path), image, options):
                    raise OSError(f"cannot write {path}")
    finally:
        for capture in captures.values():
            capture.release()

    ordered = inspect_rig_grid(image_dir)
    manifest = {
        "schema": "video-split.seam360-input.v1",
        "method": "self-cali-gs+seam360gs-rig6",
        "databag": os.fspath(Path(databag).resolve()),
        "videos": videos,
        "uses_camera_parameters": False,
        "camera_model": "learned-per-camera",
        "rig_model": "one-pose-per-time+one-fixed-transform-per-camera",
        "layout": layout,
        "start_frame": int(start),
        "frames": int(frames),
        "stride": int(stride),
        "source_fps": fps,
        "frame_indexes": [int(start + time * stride) for time in range(frames)],
        "image_size": list(size),
        "files": [path.relative_to(image_dir).as_posix() for path in ordered],
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def _project_to_rotation(matrix):
    u, _, vt = np.linalg.svd(np.asarray(matrix, np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def mean_transform(transforms):
    """Average a short set of rigid transforms without averaging scale."""
    transforms = np.asarray(transforms, np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("expected one or more [4,4] transforms")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _project_to_rotation(transforms[:, :3, :3].mean(axis=0))
    result[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    return result


def factor_rigid_rig(camera_c2w, anchor_camera=0):
    """Factor per-image poses into per-time rig poses and fixed camera poses.

    ``camera_c2w`` has shape ``[time,camera,4,4]``.  This is an initializer and
    diagnostic for the constrained optimiser; it never reads hardware
    calibration.
    """
    poses = np.asarray(camera_c2w, np.float64)
    if poses.ndim != 4 or poses.shape[2:] != (4, 4):
        raise ValueError("expected camera poses with shape [time,camera,4,4]")
    _, cameras = poses.shape[:2]
    if not 0 <= anchor_camera < cameras:
        raise ValueError("anchor camera is outside the pose grid")
    rig_c2w = poses[:, anchor_camera].copy()
    camera_to_rig = np.repeat(np.eye(4)[None], cameras, axis=0)
    for camera in range(cameras):
        relative = [np.linalg.inv(rig_c2w[time]) @ poses[time, camera]
                    for time in range(len(rig_c2w))]
        camera_to_rig[camera] = mean_transform(relative)
    reconstructed = rig_c2w[:, None] @ camera_to_rig[None]
    return rig_c2w, camera_to_rig, reconstructed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--databag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--extension", choices=("jpg", "png"), default="png")
    parser.add_argument("--layout", choices=("rig", "flat"), default="rig")
    args = parser.parse_args()
    manifest = export_clip(
        args.databag, args.out, start=args.start, frames=args.frames,
        stride=args.stride, size=(args.width, args.height),
        extension=args.extension, layout=args.layout)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
