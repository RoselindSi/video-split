"""Temporal audit for precomputed Fast-FoundationStereo disparity.

Raw frame-to-frame disparity differences mix depth instability with ordinary
image motion.  This audit estimates backward optical flow in the rectified
left view, transports the previous disparity to the current frame, and only
scores pixels whose forward/backward flow is consistent.  Values are reported
in full-resolution disparity pixels even though flow is estimated smaller.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


MIN_DISPARITY = 0.5
FLOW_SCALE = 0.5
FLOW_FB_MAX_PX = 1.5
JUMP_PX = 3.0


def temporal_disparity_error(previous_image, current_image,
                             previous_disparity, current_disparity,
                             scale=FLOW_SCALE):
    """Flow-aligned disparity change from previous to current frame."""
    import cv2

    if previous_image.shape != current_image.shape:
        raise ValueError("temporal images have different shapes")
    if previous_disparity.shape != current_disparity.shape:
        raise ValueError("temporal disparities have different shapes")
    height, width = current_disparity.shape
    if previous_image.shape[:2] != (height, width):
        raise ValueError("image and disparity shapes differ")
    small_w = max(32, int(round(width * scale)))
    small_h = max(24, int(round(height * scale)))
    sx, sy = small_w / width, small_h / height

    def gray(image):
        value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(value, (small_w, small_h),
                          interpolation=cv2.INTER_AREA)

    previous_gray, current_gray = gray(previous_image), gray(current_image)
    options = (0.5, 4, 31, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, *options)
    forward = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, *options)

    yy, xx = np.mgrid[:small_h, :small_w].astype(np.float32)
    map_x = xx + backward[..., 0]
    map_y = yy + backward[..., 1]
    inside = ((map_x >= 0) & (map_x < small_w - 1) &
              (map_y >= 0) & (map_y < small_h - 1))
    forward_x = cv2.remap(forward[..., 0], map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    forward_y = cv2.remap(forward[..., 1], map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    fb_error = np.hypot(backward[..., 0] + forward_x,
                        backward[..., 1] + forward_y)

    previous_small = cv2.resize(
        np.asarray(previous_disparity, np.float32), (small_w, small_h),
        interpolation=cv2.INTER_LINEAR) * sx
    current_small = cv2.resize(
        np.asarray(current_disparity, np.float32), (small_w, small_h),
        interpolation=cv2.INTER_LINEAR) * sx
    previous_warped = cv2.remap(
        previous_small, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)
    valid = (inside & (fb_error <= FLOW_FB_MAX_PX) &
             np.isfinite(current_small) & np.isfinite(previous_warped) &
             (current_small > MIN_DISPARITY * sx) &
             (previous_warped > MIN_DISPARITY * sx))
    if valid.sum() < 100:
        return None
    error = np.abs(current_small[valid] - previous_warped[valid]) / sx
    return {
        "support_fraction": float(valid.mean()),
        "median_px": float(np.median(error)),
        "p90_px": float(np.percentile(error, 90)),
        "jump_gt3_fraction": float(np.mean(error > JUMP_PX)),
    }


def _summary(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def evaluate(case_dir, disparity_dir):
    """Evaluate a frame directory and matching ``000000.npy`` disparities."""
    import cv2

    case_dir, disparity_dir = Path(case_dir), Path(disparity_dir)
    images = sorted((case_dir / "video1").glob("*.jpg"))
    disparities = sorted(disparity_dir.glob("*.npy"))
    if not images or len(images) != len(disparities):
        raise ValueError(
            f"frame counts differ: images {len(images)}, "
            f"disparities {len(disparities)}")
    previous_image = cv2.imread(os.fspath(images[0]))
    previous_disparity = np.load(disparities[0])
    rows = []
    for index in range(1, len(images)):
        current_image = cv2.imread(os.fspath(images[index]))
        current_disparity = np.load(disparities[index])
        if previous_image is None or current_image is None:
            raise OSError(f"cannot decode frame {index - 1} or {index}")
        metrics = temporal_disparity_error(
            previous_image, current_image,
            previous_disparity, current_disparity)
        if metrics is not None:
            rows.append({"frame": index, **metrics})
        previous_image, previous_disparity = current_image, current_disparity
    if not rows:
        raise ValueError("no frame pair had enough consistent flow support")
    keys = ("support_fraction", "median_px", "p90_px",
            "jump_gt3_fraction")
    summary = {key: _summary(rows, key) for key in keys}
    summary["pairs"] = len(rows)
    summary["worst_frames"] = [
        row["frame"] for row in sorted(
            rows, key=lambda row: row["p90_px"], reverse=True)[:8]]
    return {"summary": summary, "frames": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--disparity_dir", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    result = evaluate(args.case, args.disparity_dir)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nmetrics -> {args.out_json}")


if __name__ == "__main__":
    main()
