"""Prepare and run a real stereo-pair StabStitch++ validation.

This is deliberately separate from :mod:`src.rig.stabstitch_export`.  That
module exports one eye from each of the three rig modules for a three-view
stitch.  This module exports both eyes of exactly one physical stereo module:

    cam12.mp4 -> case/video1/*.jpg (cam1), case/video2/*.jpg (cam2)

The two fisheye images are stereo-rectified onto the same perspective canvas
before they reach StabStitch++.  Calibration is only lens/epipolar
normalisation here; StabStitch++ still estimates the content-dependent warp.

The official inference program has two path quirks which ``run`` contains in
one place: it must be launched from its Codes directory and its output path
must end in a slash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


OUT_SIZE = (960, 720)
FOV_SCALE = 0.8
STABSTITCH_COMMIT = "646dad01f2f1159aeca10bcef2fbb1416d658df7"


def pair_key(module):
    return f"{module.left.name}{module.right.name[-1]}"


def find_module(rig, key):
    for module in rig.modules:
        if pair_key(module) == key:
            return module
    have = ", ".join(pair_key(module) for module in rig.modules)
    raise ValueError(f"unknown stereo pair {key!r}; calibrated pairs: {have}")


def stereo_rectify_maps(rig, module, size=OUT_SIZE, fov_scale=FOV_SCALE,
                        balance=0.0):
    """Return perspective remap grids for the two eyes of ``module``."""
    import cv2

    K1, D1, K2, D2, sensor_size, R, T = rig.stereo_pair(module)
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, sensor_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=tuple(size), balance=float(balance),
        fov_scale=float(fov_scale))
    map1 = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, tuple(size), cv2.CV_32FC1)
    map2 = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, tuple(size), cv2.CV_32FC1)
    return map1, map2, P1, P2


def valid_fraction(remap, sensor_size):
    mx, my = remap
    width, height = sensor_size
    valid = ((mx >= 0) & (mx < width - 1) &
             (my >= 0) & (my < height - 1))
    return float(valid.mean())


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _epipolar_residual(image1, image2):
    """Sparse vertical match residual; diagnostic only, never a warp input."""
    import cv2

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    matches = sorted(matches, key=lambda match: match.distance)[:500]
    if len(matches) < 20:
        return None
    dx = np.array([kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]
                   for m in matches], dtype=np.float32)
    dy = np.array([kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1]
                   for m in matches], dtype=np.float32)
    # Reject obvious cross-check accidents using the property rectification is
    # meant to establish. The retained statistic still exposes a bad map.
    keep = np.abs(dy - np.median(dy)) <= 12.0
    if keep.sum() < 20:
        return None
    return {
        "matches": int(keep.sum()),
        "abs_vertical_median_px": float(np.median(np.abs(dy[keep]))),
        "abs_vertical_p90_px": float(np.percentile(np.abs(dy[keep]), 90)),
        "horizontal_median_px": float(np.median(dx[keep])),
    }


def export_pair(rig, video, out_root, key="cam12", start=3000, n=120,
                stride=1, size=OUT_SIZE, fov_scale=FOV_SCALE, balance=0.0,
                quality=95, case_name=None, preview=True):
    """Export one side-by-side stereo recording in official dataset layout."""
    import cv2
    from src.rig.render_wide import split_halves

    if n < 1 or stride < 1:
        raise ValueError("n and stride must both be positive")
    module = find_module(rig, key)
    case_name = case_name or f"{key}_f{start:06d}_n{n:06d}_s{stride:02d}"
    case_dir = Path(out_root) / case_name
    dirs = (case_dir / "video1", case_dir / "video2")
    if case_dir.exists():
        raise FileExistsError(
            f"{case_dir} already exists; choose another --case_name")
    for directory in dirs:
        directory.mkdir(parents=True)

    maps1, maps2, P1, P2 = stereo_rectify_maps(
        rig, module, size=size, fov_scale=fov_scale, balance=balance)
    sensor_size = (module.left.width, module.left.height)
    validity = [valid_fraction(maps1, sensor_size),
                valid_fraction(maps2, sensor_size)]
    cap = cv2.VideoCapture(os.fspath(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) / stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

    writer = None
    if preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            os.fspath(case_dir / "rectified_pair.mp4"), fourcc, fps,
            (size[0] * 2, size[1]))
        if not writer.isOpened():
            cap.release()
            raise OSError("cannot create rectified_pair.mp4")

    residuals = []
    written = 0
    try:
        for index in range(n):
            if index:
                for _ in range(stride - 1):
                    if not cap.grab():
                        return _finish_manifest(
                            rig, video, case_dir, module, start, n, stride,
                            size, fov_scale, balance, quality, fps, written,
                            validity, residuals, P1, P2)
            ok, frame = cap.read()
            if not ok:
                break
            eye1, eye2 = split_halves(frame)
            expected = (module.left.height, module.left.width)
            if eye1.shape[:2] != expected or eye2.shape[:2] != expected:
                raise ValueError(
                    f"video halves are {eye1.shape[:2]} and {eye2.shape[:2]}, "
                    f"calibration expects {expected}")
            image1 = cv2.remap(eye1, maps1[0], maps1[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            image2 = cv2.remap(eye2, maps2[0], maps2[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            name = f"{index:06d}.jpg"
            if not cv2.imwrite(os.fspath(dirs[0] / name), image1, params):
                raise OSError(f"cannot write {dirs[0] / name}")
            if not cv2.imwrite(os.fspath(dirs[1] / name), image2, params):
                raise OSError(f"cannot write {dirs[1] / name}")
            if writer is not None:
                writer.write(np.concatenate((image1, image2), axis=1))
            if index % max(1, round(fps / 2)) == 0:
                residual = _epipolar_residual(image1, image2)
                if residual:
                    residuals.append(residual)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return _finish_manifest(
        rig, video, case_dir, module, start, n, stride, size, fov_scale,
        balance, quality, fps, written, validity, residuals, P1, P2)


def _finish_manifest(rig, video, case_dir, module, start, requested, stride,
                     size, fov_scale, balance, quality, fps, written,
                     validity, residuals, P1, P2):
    vertical_medians = [r["abs_vertical_median_px"] for r in residuals]
    vertical_p90s = [r["abs_vertical_p90_px"] for r in residuals]
    manifest = {
        "schema": "video-split.stabstitch-pair.v1",
        "source": {
            "video": os.path.abspath(os.fspath(video)),
            "video_sha256": _sha256(video),
            "calibration": os.path.abspath(rig.path),
            "calibration_sha256": _sha256(rig.path),
        },
        "pair": pair_key(module),
        "cameras": [module.left.name, module.right.name],
        "frames": {
            "start": int(start), "stride": int(stride),
            "requested": int(requested), "written": int(written),
            "fps": float(fps),
        },
        "rectification": {
            "output_size": [int(size[0]), int(size[1])],
            "fov_scale": float(fov_scale), "balance": float(balance),
            "jpeg_quality": int(quality),
            "valid_fraction": validity,
            "P1": np.asarray(P1).tolist(), "P2": np.asarray(P2).tolist(),
        },
        "epipolar_diagnostic": {
            "sample_count": len(residuals),
            "abs_vertical_median_px": (
                float(np.median(vertical_medians)) if vertical_medians else None),
            "abs_vertical_p90_px": (
                float(np.median(vertical_p90s)) if vertical_p90s else None),
        },
        "stabstitch": {
            "upstream_commit": STABSTITCH_COMMIT,
            "model_family": "full_model_tra",
            "warp_mode": "NORMAL",
            "fusion_mode": "LINEAR",
        },
    }
    with open(case_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest, case_dir


def run_official(stabstitch_root, input_root, output_dir, python="python3",
                 gpu="0"):
    """Run the unmodified official two-view inference program."""
    root = Path(stabstitch_root).resolve()
    codes = root / "Full_model_inference" / "Codes"
    script = codes / "test_online_tra.py"
    model_dir = root / "Full_model_inference" / "full_model_tra"
    missing = [model_dir / name for name in
               ("spatial_warp.pth", "temporal_warp.pth", "smooth_warp.pth")
               if not (model_dir / name).is_file()]
    if not script.is_file():
        raise FileNotFoundError(f"missing official inference script: {script}")
    if missing:
        raise FileNotFoundError("missing checkpoints: " +
                                ", ".join(os.fspath(path) for path in missing))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    command = [python, script.name, "--gpu", str(gpu),
               "--test_path", os.fspath(Path(input_root).resolve()),
               "--output_path", os.fspath(Path(output_dir).resolve()) + os.sep,
               "--warp_mode", "NORMAL", "--fusion_mode", "LINEAR"]
    subprocess.run(command, cwd=os.fspath(codes), check=True)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--calibration", required=True)
    export_parser.add_argument("--video", required=True)
    export_parser.add_argument("--pair", default="cam12")
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--start", type=int, default=3000)
    export_parser.add_argument("--n", type=int, default=120)
    export_parser.add_argument("--stride", type=int, default=1)
    export_parser.add_argument("--width", type=int, default=OUT_SIZE[0])
    export_parser.add_argument("--height", type=int, default=OUT_SIZE[1])
    export_parser.add_argument("--fov_scale", type=float, default=FOV_SCALE)
    export_parser.add_argument("--balance", type=float, default=0.0)
    export_parser.add_argument("--quality", type=int, default=95)
    export_parser.add_argument("--case_name")
    export_parser.add_argument("--no_preview", action="store_true")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--stabstitch_root", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--python", default="python3")
    run_parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    if args.command == "export":
        from src.rig.calibration import RigCalibration
        rig = RigCalibration(args.calibration)
        manifest, case_dir = export_pair(
            rig, args.video, args.out, key=args.pair, start=args.start,
            n=args.n, stride=args.stride, size=(args.width, args.height),
            fov_scale=args.fov_scale, balance=args.balance,
            quality=args.quality, case_name=args.case_name,
            preview=not args.no_preview)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"\nStabStitch++ input -> {case_dir}")
    else:
        command = run_official(
            args.stabstitch_root, args.input, args.output,
            python=args.python, gpu=args.gpu)
        print("completed:", " ".join(command))


if __name__ == "__main__":
    main()
