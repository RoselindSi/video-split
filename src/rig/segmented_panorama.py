"""Render a six-view panorama from pixels and matches, without calibration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.rig.dynamic_panorama import (
    compose_gated_boundary_blend,
    compose_single_source,
    geometric_owner,
    regularize_semantic_owner,
)
from src.rig.multi_homography import (
    PairGeometryConfig,
    analyze_pair,
    assign_segments_to_models,
    segmented_warp,
    transformed_support_points,
)
from src.rig.multi_homography_probe import load_tiny_roma, person_masks


def content_segments(image, person_mask=None, count=450, compactness=10.0):
    """SLIC regions plus indivisible semantic foreground components."""
    import cv2

    image = np.asarray(image)
    try:
        from skimage.segmentation import slic
    except ImportError:
        height, width = image.shape[:2]
        cell = max(1, int(round(np.sqrt(height * width / int(count)))))
        yy, xx = np.indices((height, width))
        columns = int(np.ceil(width / cell))
        segments = (yy // cell * columns + xx // cell).astype(np.int32)
    else:
        segments = slic(
            image[..., ::-1], n_segments=int(count),
            compactness=float(compactness), start_label=0,
            convert2lab=True, enforce_connectivity=True, slic_zero=True,
            channel_axis=-1).astype(np.int32)
    if person_mask is None:
        return segments
    person_mask = np.asarray(person_mask, bool)
    if person_mask.shape != segments.shape:
        raise ValueError("person mask and image shapes differ")
    component_count, components = cv2.connectedComponents(
        person_mask.astype(np.uint8), connectivity=8)
    next_label = int(segments.max()) + 1
    for component in range(1, component_count):
        selected = components == component
        if selected.any():
            segments[selected] = next_label
            next_label += 1
    return segments


def canvas_from_footprints(footprints, anchor_shape, margin=24,
                           max_size=(6000, 3000)):
    """Build a translated anchor-coordinate canvas from robust footprints."""
    points = [np.asarray(value, np.float32) for value in footprints
              if len(value)]
    if not points:
        raise ValueError("no transformed image footprints")
    points = np.concatenate(points)
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        raise ValueError("all transformed footprint points are non-finite")
    tail = 0.5 if len(finite) >= 200 else 0.0
    lower = np.percentile(finite, tail, axis=0)
    upper = np.percentile(finite, 100.0 - tail, axis=0)
    anchor_height, anchor_width = anchor_shape[:2]
    lower = np.minimum(lower, [0.0, 0.0]) - float(margin)
    upper = np.maximum(upper, [anchor_width - 1, anchor_height - 1]) \
        + float(margin)
    width, height = np.ceil(upper - lower + 1).astype(int)
    if width > int(max_size[0]) or height > int(max_size[1]):
        raise ValueError(
            f"piecewise warp requested unsafe canvas {width}x{height}")
    transform = np.asarray([
        [1.0, 0.0, -lower[0]],
        [0.0, 1.0, -lower[1]],
        [0.0, 0.0, 1.0],
    ], np.float64)
    return transform, (int(width), int(height)), {
        "anchor_bounds": [float(lower[0]), float(lower[1]),
                          float(upper[0]), float(upper[1])],
        "size": [int(width), int(height)],
    }


def exposure_compensate(images, valid):
    """Block-gain compensation over overlaps, preserving invalid pixels."""
    import cv2

    keys = sorted(images)
    corners = [(0, 0)] * len(keys)
    adjusted = [np.asarray(images[key]).copy() for key in keys]
    masks = [(np.asarray(valid[key], bool) * 255).astype(np.uint8)
             for key in keys]
    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners, adjusted, masks)
    for index, _ in enumerate(keys):
        compensator.apply(index, corners[index], adjusted[index], masks[index])
    return {key: image for key, image in zip(keys, adjusted)}


def graphcut_owner(images, valid, cost):
    """Find low-disagreement hard seams, then fill any graph-cut holes."""
    import cv2

    keys = sorted(images)
    masks = [(np.asarray(valid[key], bool) * 255).astype(np.uint8)
             for key in keys]
    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    finder.find(
        [np.asarray(images[key], np.float32) for key in keys],
        [(0, 0)] * len(keys), masks)
    cut_valid = {key: mask > 0 for key, mask in zip(keys, masks)}
    owner = geometric_owner(cut_valid, cost)
    fallback = geometric_owner(valid, cost)
    owner[owner < 0] = fallback[owner < 0]
    return owner, cut_valid


def crop_valid(images, arrays, valid, margin=0):
    """Crop a common bounding box around all visible source pixels."""
    union = np.logical_or.reduce([np.asarray(value, bool)
                                  for value in valid.values()])
    rows, columns = np.nonzero(union)
    if not len(rows):
        raise ValueError("render has no valid pixels")
    margin = int(margin)
    top = max(0, int(rows.min()) - margin)
    bottom = min(union.shape[0], int(rows.max()) + margin + 1)
    left = max(0, int(columns.min()) - margin)
    right = min(union.shape[1], int(columns.max()) + margin + 1)
    cropped_images = {key: value[top:bottom, left:right]
                      for key, value in images.items()}
    cropped_arrays = [value[top:bottom, left:right] for value in arrays]
    cropped_valid = {key: value[top:bottom, left:right]
                     for key, value in valid.items()}
    return cropped_images, cropped_arrays, cropped_valid, \
        [left, top, right, bottom]


def _serializable_report(report):
    row = {key: value for key, value in report.items()
           if key not in ("source", "target", "certainty", "labels",
                          "fundamental")}
    row["homographies"] = [{
        key: value for key, value in model.items()
        if key not in ("homography", "inlier_indices")
    } for model in report["homographies"]]
    return row


def render(args):
    import cv2
    import torch

    root = Path(args.scene) / "images"
    paths = [root / f"cam{camera}" / f"t{args.time:03d}.jpg"
             for camera in range(6)]
    images = [cv2.imread(os.fspath(path)) for path in paths]
    missing = [os.fspath(path) for path, image in zip(paths, images)
               if image is None]
    if missing:
        raise OSError(f"cannot decode source images: {missing}")
    semantic = person_masks(
        images, args.person_weights, args.person_confidence,
        args.person_dilation)
    matcher = load_tiny_roma(
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

    anchor = int(args.anchor)
    camera_models = {
        anchor: {
            "homographies": [np.eye(3, dtype=np.float64)],
            "model_map": np.zeros(images[anchor].shape[:2], np.int16),
            "report": None,
        }
    }
    with torch.inference_mode():
        for camera in range(6):
            if camera == anchor:
                continue
            warp, certainty = matcher.match(
                os.fspath(paths[camera]), os.fspath(paths[anchor]))
            matches, scores = matcher.sample(
                warp, certainty, num=int(args.matches))
            source, target = matcher.to_pixel_coordinates(
                matches, *images[camera].shape[:2],
                *images[anchor].shape[:2])
            report = analyze_pair(
                source.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                images[camera].shape[:2], images[anchor].shape[:2],
                scores.detach().cpu().numpy(), semantic[camera],
                semantic[anchor], config)
            report["pair"] = [camera, anchor]
            if not report["accepted"]:
                print(f"cam{camera}->cam{anchor}: rejected", flush=True)
                continue
            segments = content_segments(
                images[camera], semantic[camera], args.segments,
                args.compactness)
            model_map = assign_segments_to_models(
                segments, report["source"], report["labels"],
                len(report["homographies"]))
            camera_models[camera] = {
                "homographies": [model["homography"]
                                 for model in report["homographies"]],
                "model_map": model_map,
                "report": report,
            }
            print(
                f"cam{camera}->cam{anchor}: "
                f"H={report['explained_matches']} "
                f"models={len(report['homographies'])} "
                f"p90={report['piecewise_p90_px']:.2f}px", flush=True)

    if len(camera_models) < 3:
        raise RuntimeError("fewer than three cameras connected to the anchor")
    footprints = []
    for camera, values in camera_models.items():
        footprints.append(transformed_support_points(
            values["model_map"], values["homographies"]))
    canvas_transform, canvas_size, canvas_stats = canvas_from_footprints(
        footprints, images[anchor].shape, margin=args.canvas_margin,
        max_size=(args.max_canvas_width, args.max_canvas_height))

    warped, valid, warped_semantic = {}, {}, {}
    for camera, values in camera_models.items():
        warped[camera], valid[camera] = segmented_warp(
            images[camera], values["model_map"], values["homographies"],
            canvas_transform, canvas_size)
        semantic_image, _ = segmented_warp(
            (semantic[camera].astype(np.uint8) * 255),
            values["model_map"], values["homographies"],
            canvas_transform, canvas_size, interpolation=cv2.INTER_NEAREST)
        warped_semantic[camera] = semantic_image > 127

    adjusted = exposure_compensate(warped, valid)
    cost = {}
    for camera in sorted(valid):
        distance = cv2.distanceTransform(
            valid[camera].astype(np.uint8), cv2.DIST_L2, 3)
        cost[camera] = -distance.astype(np.float32)
    owner, graph_masks = graphcut_owner(adjusted, valid, cost)
    owner, protected, semantic_stats = regularize_semantic_owner(
        owner, warped_semantic, valid, cost,
        close_px=args.semantic_close, dilate_px=args.semantic_dilate,
        max_component_fraction=0.35)
    black = np.zeros((*owner.shape, 3), np.float32)
    no_background = np.zeros(owner.shape, bool)
    hard, _, _ = compose_single_source(
        adjusted, valid, owner, black, no_background)
    gated, transition, gated_out = compose_gated_boundary_blend(
        adjusted, valid, cost, owner, black, no_background,
        protected=protected, gate=args.blend_gate / 255.0,
        boundary_px=args.blend_width, temperature=args.blend_temperature)

    adjusted, arrays, valid, crop = crop_valid(
        adjusted, [hard, gated, owner, protected, transition, gated_out],
        valid)
    hard, gated, owner, protected, transition, gated_out = arrays
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.fspath(output / "panorama_hard.png"),
                np.clip(hard * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(os.fspath(output / "panorama_gated.png"),
                np.clip(gated * 255.0, 0, 255).astype(np.uint8))
    palette = np.asarray([
        [230, 80, 80], [80, 210, 80], [80, 130, 240],
        [220, 180, 60], [190, 80, 210], [70, 210, 210],
    ], np.uint8)
    owner_image = np.zeros((*owner.shape, 3), np.uint8)
    for camera in range(6):
        owner_image[owner == camera] = palette[camera]
    cv2.imwrite(os.fspath(output / "owner.png"), owner_image)
    cv2.imwrite(os.fspath(output / "semantic_protected.png"),
                protected.astype(np.uint8) * 255)
    cv2.imwrite(os.fspath(output / "blend_transition.png"),
                transition.astype(np.uint8) * 255)
    cv2.imwrite(os.fspath(output / "blend_gated_out.png"),
                gated_out.astype(np.uint8) * 255)

    result = {
        "method": "tiny-roma-segmented-multi-homography",
        "uses_camera_parameters": False,
        "anchor": anchor,
        "active_cameras": sorted(camera_models),
        "canvas": canvas_stats,
        "crop": crop,
        "output_size": [int(hard.shape[1]), int(hard.shape[0])],
        "protected_fraction": float(protected.mean()),
        "blend_transition_fraction": float(transition.mean()),
        "blend_gated_fraction": float(gated_out.mean()),
        "semantic_ownership": semantic_stats,
        "pairs": {
            str(camera): _serializable_report(values["report"])
            for camera, values in camera_models.items()
            if values["report"] is not None
        },
        "config": vars(args),
    }
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "active_cameras", "canvas", "crop", "output_size",
        "protected_fraction", "blend_transition_fraction",
        "blend_gated_fraction")}, indent=2))
    print(f"render -> {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--anchor", type=int, choices=range(6), default=3)
    parser.add_argument("--runtime", default="/workspace/roma_runtime")
    parser.add_argument(
        "--xfeat-source",
        default="/workspace/roma_runtime/accelerated_features-main")
    parser.add_argument(
        "--roma-weights",
        default="/workspace/models/tiny_roma_v1_outdoor.pth")
    parser.add_argument(
        "--xfeat-weights", default="/workspace/models/xfeat.pt")
    parser.add_argument(
        "--person-weights", default="/workspace/models/yolo11n-seg.pt")
    parser.add_argument("--person-confidence", type=float, default=0.15)
    parser.add_argument("--person-dilation", type=int, default=9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matches", type=int, default=5000)
    parser.add_argument("--segments", type=int, default=450)
    parser.add_argument("--compactness", type=float, default=10.0)
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
    parser.add_argument("--canvas-margin", type=int, default=24)
    parser.add_argument("--max-canvas-width", type=int, default=6000)
    parser.add_argument("--max-canvas-height", type=int, default=3000)
    parser.add_argument("--semantic-close", type=int, default=12)
    parser.add_argument("--semantic-dilate", type=int, default=2)
    parser.add_argument("--blend-width", type=int, default=3)
    parser.add_argument("--blend-gate", type=float, default=20.0)
    parser.add_argument("--blend-temperature", type=float, default=8.0)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
