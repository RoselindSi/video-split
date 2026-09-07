"""Probe all six-view pairs with Tiny RoMa and piecewise homographies."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np

from src.rig.multi_homography import (
    PairGeometryConfig,
    accepted_components,
    analyze_pair,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def load_tiny_roma(runtime, xfeat_source, roma_weights, xfeat_weights,
                   device):
    """Load pinned local artifacts without Torch Hub or network access."""
    import sys
    import torch

    for path in (os.fspath(runtime), os.fspath(xfeat_source)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import romatch
    from modules.xfeat import XFeat

    xfeat = XFeat(os.fspath(xfeat_weights), top_k=4096).net
    weights = torch.load(
        os.fspath(roma_weights), map_location=device, weights_only=True)
    model = romatch.tiny_roma_v1_outdoor(
        device=device, weights=weights, xfeat=xfeat)
    model.eval()
    return model


def person_masks(images, weights, confidence=0.15, dilation=9):
    """Return conservative source-view person masks for match exclusion."""
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.ops import scale_masks

    model = YOLO(os.fspath(weights))
    results = model.predict(
        images, classes=[0], conf=float(confidence), verbose=False)
    masks = []
    for image, result in zip(images, results):
        if result.masks is None:
            mask = np.zeros(image.shape[:2], bool)
        else:
            scaled = scale_masks(
                result.masks.data[:, None], image.shape[:2],
                padding=True, mode="bilinear")[:, 0]
            mask = np.any(scaled.detach().cpu().numpy() >= 0.5, axis=0)
        if int(dilation) > 0:
            radius = int(dilation)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        masks.append(mask)
    return masks


def draw_pair(image_a, image_b, report, output, limit=300):
    import cv2

    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    canvas = np.zeros((height, width_a + image_b.shape[1], 3), np.uint8)
    canvas[:image_a.shape[0], :width_a] = image_a
    canvas[:image_b.shape[0], width_a:] = image_b
    labels = report["labels"]
    accepted = np.flatnonzero(labels >= 0)
    if len(accepted) > int(limit):
        step = int(np.ceil(len(accepted) / int(limit)))
        accepted = accepted[::step]
    colours = (
        (230, 80, 80), (80, 210, 80), (80, 130, 240), (220, 180, 60),
        (190, 80, 210), (70, 210, 210), (160, 160, 70), (210, 120, 150),
    )
    for index in accepted:
        first = tuple(np.rint(report["source"][index]).astype(int))
        second_point = np.rint(report["target"][index]).astype(int)
        second = (int(second_point[0] + width_a), int(second_point[1]))
        colour = colours[int(labels[index]) % len(colours)]
        cv2.line(canvas, first, second, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, first, 2, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, second, 2, colour, -1, cv2.LINE_AA)
    verdict = "PASS" if report["accepted"] else "REJECT"
    cv2.putText(
        canvas,
        f"{verdict} static={report['static_matches']} "
        f"F={report['fundamental_inliers']} H={report['explained_matches']} "
        f"p90={report['piecewise_p90_px']:.2f}px",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (60, 220, 80) if report["accepted"] else (60, 80, 230),
        2, cv2.LINE_AA)
    cv2.imwrite(os.fspath(output), canvas)


def probe(args):
    import cv2
    import torch

    image_paths = [Path(args.scene) / "images" / f"cam{camera}"
                   / f"t{args.time:03d}.jpg" for camera in range(6)]
    images = [cv2.imread(os.fspath(path)) for path in image_paths]
    missing = [os.fspath(path) for path, image in zip(image_paths, images)
               if image is None]
    if missing:
        raise OSError(f"cannot decode source images: {missing}")
    exclusions = (
        person_masks(images, args.person_weights, args.person_confidence,
                     args.person_dilation)
        if args.person_weights else [None] * len(images)
    )
    model = load_tiny_roma(
        args.runtime, args.xfeat_source, args.roma_weights,
        args.xfeat_weights, args.device)
    config = PairGeometryConfig(
        min_certainty=args.min_certainty,
        fundamental_threshold_px=args.fundamental_threshold,
        homography_threshold_px=args.homography_threshold,
        min_fundamental_inliers=args.min_fundamental_inliers,
        min_fundamental_ratio=args.min_fundamental_ratio,
        min_homography_inliers=args.min_homography_inliers,
        max_homographies=args.max_homographies,
        min_explained_matches=args.min_explained_matches,
        min_explained_ratio=args.min_explained_ratio,
        min_grid_coverage=args.min_grid_coverage,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.inference_mode():
        for first, second in combinations(range(6), 2):
            warp, certainty = model.match(
                os.fspath(image_paths[first]), os.fspath(image_paths[second]))
            matches, scores = model.sample(warp, certainty, num=args.matches)
            source, target = model.to_pixel_coordinates(
                matches, *images[first].shape[:2], *images[second].shape[:2])
            report = analyze_pair(
                source.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                images[first].shape[:2], images[second].shape[:2],
                scores.detach().cpu().numpy(), exclusions[first],
                exclusions[second], config)
            report["pair"] = [first, second]
            reports.append(report)
            draw_pair(
                images[first], images[second], report,
                output / f"pair_{first}_{second}.jpg")
            print(
                f"cam{first}-cam{second}: "
                f"{'PASS' if report['accepted'] else 'reject'} "
                f"static={report['static_matches']} "
                f"F={report['fundamental_inliers']} "
                f"H={report['explained_matches']} "
                f"p90={report['piecewise_p90_px']:.2f}px "
                f"coverage={report['source_grid_coverage']:.2f}/"
                f"{report['target_grid_coverage']:.2f}", flush=True)
    components = accepted_components(6, reports)
    serializable = []
    for report in reports:
        row = {key: value for key, value in report.items()
               if key not in ("source", "target", "certainty", "labels")}
        serializable.append(row)
    result = {
        "matcher": "tiny-roma-v1-outdoor",
        "time": int(args.time),
        "pairs": serializable,
        "accepted_pairs": [row["pair"] for row in serializable
                           if row["accepted"]],
        "components": components,
        "connected": len(components) == 1,
        "config": vars(args),
    }
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, default=_json_value)
        stream.write("\n")
    print(f"components: {components}")
    print(f"report -> {output / 'report.json'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--runtime", default="/workspace/roma_runtime")
    parser.add_argument(
        "--xfeat-source",
        default="/workspace/roma_runtime/accelerated_features-main")
    parser.add_argument(
        "--roma-weights",
        default="/workspace/models/tiny_roma_v1_outdoor.pth")
    parser.add_argument(
        "--xfeat-weights", default="/workspace/models/xfeat.pt")
    parser.add_argument("--person-weights")
    parser.add_argument("--person-confidence", type=float, default=0.15)
    parser.add_argument("--person-dilation", type=int, default=9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matches", type=int, default=5000)
    parser.add_argument("--min-certainty", type=float, default=0.05)
    parser.add_argument("--fundamental-threshold", type=float, default=2.0)
    parser.add_argument("--homography-threshold", type=float, default=3.0)
    parser.add_argument("--min-fundamental-inliers", type=int, default=80)
    parser.add_argument("--min-fundamental-ratio", type=float, default=0.10)
    parser.add_argument("--min-homography-inliers", type=int, default=32)
    parser.add_argument("--max-homographies", type=int, default=8)
    parser.add_argument("--min-explained-matches", type=int, default=120)
    parser.add_argument("--min-explained-ratio", type=float, default=0.12)
    parser.add_argument("--min-grid-coverage", type=float, default=0.06)
    args = parser.parse_args()
    probe(args)


if __name__ == "__main__":
    main()
