"""Measure whether StabStitch++ actually aligns a stereo pair at hard edges.

The six-view panorama seam ratio needs an ownership boundary. StabStitch++
instead blends most of the overlapping canvas, so pretending it has one seam
would measure an invented coordinate. This audit uses the same principle at
the representation StabStitch++ exposes: corresponding edges in its two
warped source layers should occupy the same pixels before they are blended.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def edge_chamfer(image1, image2, mask=None):
    """Symmetric nearest-edge distance in pixels inside the valid overlap."""
    import cv2

    if image1.shape != image2.shape:
        raise ValueError(f"image shapes differ: {image1.shape} vs {image2.shape}")
    height, width = image1.shape[:2]
    if mask is None:
        valid = np.ones((height, width), np.uint8)
    else:
        valid = np.asarray(mask, dtype=np.uint8)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    edge1 = (cv2.Canny(gray1, 60, 120) > 0) & valid
    edge2 = (cv2.Canny(gray2, 60, 120) > 0) & valid
    if edge1.sum() < 50 or edge2.sum() < 50:
        return None
    distance1 = cv2.distanceTransform((~edge1).astype(np.uint8),
                                      cv2.DIST_L2, 3)
    distance2 = cv2.distanceTransform((~edge2).astype(np.uint8),
                                      cv2.DIST_L2, 3)
    distances = np.concatenate((distance2[edge1], distance1[edge2]))
    # Cap gross unmatched/occluded edges. They still land at the cap and move
    # the tail, without one view-only border making every percentile useless.
    distances = np.clip(distances, 0, 32)
    return {
        "edge_count": int(len(distances)),
        "chamfer_median_px": float(np.median(distances)),
        "chamfer_p90_px": float(np.percentile(distances, 90)),
        "chamfer_p99_px": float(np.percentile(distances, 99)),
        # Linear fusion turns these unmatched edges into visible double
        # contours. Three pixels is already obvious on the 1090 px output.
        "ghost_risk_gt3_fraction": float(np.mean(distances > 3.0)),
    }


def _read_directory_pair(case_dir):
    import cv2

    names1 = sorted(glob.glob(os.path.join(case_dir, "video1", "*.jpg")))
    names2 = sorted(glob.glob(os.path.join(case_dir, "video2", "*.jpg")))
    if not names1 or len(names1) != len(names2):
        raise ValueError("input video1/video2 frame counts are empty or unequal")
    for name1, name2 in zip(names1, names2):
        image1, image2 = cv2.imread(name1), cv2.imread(name2)
        if image1 is None or image2 is None:
            raise OSError(f"cannot decode {name1} or {name2}")
        yield image1, image2, None


def _read_warped_pair(video_path, mask_path):
    import cv2

    video = cv2.VideoCapture(os.fspath(video_path))
    masks = cv2.VideoCapture(os.fspath(mask_path))
    if not video.isOpened() or not masks.isOpened():
        raise OSError("cannot open warped-pair or mask video")
    try:
        while True:
            ok1, frame = video.read()
            ok2, mask_frame = masks.read()
            if not ok1 or not ok2:
                break
            width = frame.shape[1] // 2
            if frame.shape[1] % 2 or mask_frame.shape[1] != frame.shape[1]:
                raise ValueError("warped debug videos are not equal-width pairs")
            mask1 = mask_frame[:, :width, 0] > 127
            mask2 = mask_frame[:, width:, 0] > 127
            yield frame[:, :width], frame[:, width:], mask1 & mask2
    finally:
        video.release()
        masks.release()


def _summarize(rows, key):
    values = np.array([row[key] for row in rows], dtype=float)
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _measure_regions(image1, image2, mask):
    full = edge_chamfer(image1, image2, mask)
    split = image1.shape[0] // 2
    near_mask = None if mask is None else mask[split:]
    near = edge_chamfer(image1[split:], image2[split:], near_mask)
    if full is None or near is None:
        return None
    return {**full, **{f"near_{key}": value for key, value in near.items()}}


def evaluate(case_dir, warped_video, mask_video):
    before = [_measure_regions(*triple) for triple in
              _read_directory_pair(case_dir)]
    after = [_measure_regions(*triple) for triple in
             _read_warped_pair(warped_video, mask_video)]
    if len(before) != len(after):
        raise ValueError(f"frame count differs: input {len(before)}, warp {len(after)}")
    rows = []
    for index, (raw, warped) in enumerate(zip(before, after)):
        if raw is None or warped is None:
            continue
        row = {"frame": index,
               **{f"before_{key}": value for key, value in raw.items()},
               **{f"after_{key}": value for key, value in warped.items()}}
        denominator = max(raw["chamfer_p90_px"], 1e-6)
        row["p90_ratio"] = warped["chamfer_p90_px"] / denominator
        rows.append(row)
    keys = ("before_chamfer_median_px", "before_chamfer_p90_px",
            "after_chamfer_median_px", "after_chamfer_p90_px",
            "before_near_chamfer_p90_px", "after_near_chamfer_p90_px",
            "before_ghost_risk_gt3_fraction",
            "after_ghost_risk_gt3_fraction",
            "before_near_ghost_risk_gt3_fraction",
            "after_near_ghost_risk_gt3_fraction",
            "p90_ratio")
    summary = {key: _summarize(rows, key) for key in keys}
    summary["frames"] = len(rows)
    summary["worst_after_frames"] = [
        row["frame"] for row in sorted(
            rows, key=lambda row: row["after_chamfer_p90_px"], reverse=True)[:5]]
    summary["worst_ratio_frames"] = [
        row["frame"] for row in sorted(
            rows, key=lambda row: row["p90_ratio"], reverse=True)[:5]]
    return {"summary": summary, "frames": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--warped", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    result = evaluate(args.case, args.warped, args.masks)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nmetrics -> {args.out_json}")


if __name__ == "__main__":
    main()
