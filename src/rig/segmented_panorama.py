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
    fit_stable_affine,
    guard_segment_models,
    regularized_match_map,
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


def enforce_anchor_authority(owner, anchor_valid, anchor):
    """Keep the anchor's complete raw view free of internal camera seams."""
    owner = np.asarray(owner, np.int16).copy()
    anchor_valid = np.asarray(anchor_valid, bool)
    if owner.shape != anchor_valid.shape:
        raise ValueError("owner and anchor validity shapes differ")
    owner[anchor_valid] = int(anchor)
    return owner


def minimum_vertical_seam(cost, allowed, nominal, max_step=6,
                          smoothness=0.03):
    """Find a top-to-bottom seam through a pair's valid overlap."""
    cost = np.asarray(cost, np.float32)
    allowed = np.asarray(allowed, bool)
    if cost.shape != allowed.shape or cost.ndim != 2:
        raise ValueError("cost and allowed must share a 2D shape")
    height, width = cost.shape
    nominal = int(np.clip(nominal, 0, width - 1))
    max_step = max(1, int(max_step))
    columns = np.arange(width)
    position = 0.02 * ((columns - nominal) / max(width, 1)) ** 2
    row_cost = cost + position[None]
    row_cost = np.where(allowed, row_cost, 1e3 + position[None])
    # If an overlap disappears for one row, keep the path near the closest
    # valid row instead of making the seam jump to the image boundary.
    empty = ~allowed.any(axis=1)
    row_cost[empty] = position

    previous = row_cost[0].astype(np.float64)
    back = np.zeros((height, width), np.int16)
    offsets = np.arange(-max_step, max_step + 1)
    for row in range(1, height):
        candidates = np.full((len(offsets), width), np.inf, np.float64)
        for index, offset in enumerate(offsets):
            if offset < 0:
                candidates[index, -offset:] = previous[:offset]
            elif offset > 0:
                candidates[index, :-offset] = previous[offset:]
            else:
                candidates[index] = previous
            candidates[index] += float(smoothness) * abs(int(offset))
        choice = np.argmin(candidates, axis=0)
        back[row] = np.clip(columns + offsets[choice], 0, width - 1)
        previous = row_cost[row] + candidates[choice, columns]
    seam = np.empty(height, np.int32)
    seam[-1] = int(np.argmin(previous))
    for row in range(height - 1, 0, -1):
        seam[row - 1] = int(back[row, seam[row]])
    return seam


def pair_seam_cost(left, right, left_valid, right_valid,
                   left_protected=None, right_protected=None,
                   protect_margin=0):
    """Prefer photometrically agreeing, low-gradient, non-person pixels."""
    import cv2

    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    colour = np.abs(left - right).mean(axis=2) / 255.0
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    gradient = np.zeros(colour.shape, np.float32)
    for gray in (gray_left, gray_right):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient += np.sqrt(gx * gx + gy * gy) / (8.0 * 255.0)
    cost = colour + 0.35 * gradient
    protected = np.zeros(colour.shape, bool)
    if left_protected is not None:
        protected |= np.asarray(left_protected, bool)
    if right_protected is not None:
        protected |= np.asarray(right_protected, bool)
    margin = max(int(protect_margin), 0)
    if protected.any() and margin:
        distance = cv2.distanceTransform(
            (~protected).astype(np.uint8), cv2.DIST_L2, 3)
        cost += 20.0 * np.clip(1.0 - distance / margin, 0.0, 1.0)
    else:
        cost[protected] += 20.0
    allowed = np.asarray(left_valid, bool) & np.asarray(right_valid, bool)
    return cost, allowed


def three_band_owner(images, valid, protected, cameras=(1, 3, 5),
                     max_step=6, smoothness=0.03, protect_margin=0):
    """Compose left/centre/right modules with two constrained seams."""
    left, centre, right = (int(value) for value in cameras)
    if any(camera not in images for camera in (left, centre, right)):
        raise ValueError("all three band cameras must be available")
    centre_columns = np.nonzero(np.asarray(valid[centre], bool))[1]
    if not len(centre_columns):
        raise ValueError("centre camera has no valid pixels")
    low, high = int(centre_columns.min()), int(centre_columns.max())
    first_nominal = int(round(low + (high - low) / 3.0))
    second_nominal = int(round(low + 2.0 * (high - low) / 3.0))

    first_cost, first_allowed = pair_seam_cost(
        images[left], images[centre], valid[left], valid[centre],
        protected.get(left), protected.get(centre), protect_margin)
    first_seam = minimum_vertical_seam(
        first_cost, first_allowed, first_nominal, max_step, smoothness)
    second_cost, second_allowed = pair_seam_cost(
        images[centre], images[right], valid[centre], valid[right],
        protected.get(centre), protected.get(right), protect_margin)
    second_seam = minimum_vertical_seam(
        second_cost, second_allowed, second_nominal, max_step, smoothness)

    height, width = first_cost.shape
    columns = np.broadcast_to(np.arange(width), (height, width))
    owner = np.full((height, width), -1, np.int16)
    bands = (
        (left, columns <= first_seam[:, None]),
        (centre, (columns > first_seam[:, None])
         & (columns <= second_seam[:, None])),
        (right, columns > second_seam[:, None]),
    )
    for camera, band in bands:
        selected = band & np.asarray(valid[camera], bool)
        owner[selected] = camera
    # Keep each preferred band's source when possible, but never leave a hole
    # where another one of the three primary cameras has pixels.
    for camera in (centre, left, right):
        selected = (owner < 0) & np.asarray(valid[camera], bool)
        owner[selected] = camera
    return owner, first_seam, second_seam


def hierarchical_six_owner(
        images, valid, semantic, cost, pairs=((0, 1), (2, 3), (4, 5)),
        max_step=6, smoothness=0.03, protect_margin=0,
        semantic_close=12, semantic_dilate=2):
    """Resolve three stereo modules before placing two inter-module seams."""
    pair_owners = {}
    module_images = {}
    module_valid = {}
    module_semantic = {}
    module_cost = {}
    pair_stats = {}
    pair_seams = {}
    shape = next(iter(valid.values())).shape
    columns = np.broadcast_to(np.arange(shape[1]), shape)

    for module, pair in enumerate(pairs):
        left, right = (int(value) for value in pair)
        overlap = np.asarray(valid[left], bool) & np.asarray(valid[right], bool)
        overlap_columns = np.nonzero(overlap)[1]
        if not len(overlap_columns):
            raise ValueError(f"stereo pair {pair} has no overlap")
        low, high = int(overlap_columns.min()), int(overlap_columns.max())
        fraction = (module + 0.5) / len(pairs)
        nominal = int(round(low + fraction * (high - low)))
        seam_cost, allowed = pair_seam_cost(
            images[left], images[right], valid[left], valid[right],
            semantic.get(left), semantic.get(right), protect_margin)
        seam = minimum_vertical_seam(
            seam_cost, allowed, nominal, max_step, smoothness)
        pair_owner = np.full(shape, -1, np.int16)
        for camera, region in (
                (left, columns <= seam[:, None]),
                (right, columns > seam[:, None])):
            selected = region & np.asarray(valid[camera], bool)
            pair_owner[selected] = camera
        pair_valid = {camera: valid[camera] for camera in pair}
        pair_cost = {camera: cost[camera] for camera in pair}
        fallback = geometric_owner(pair_valid, pair_cost)
        pair_owner[pair_owner < 0] = fallback[pair_owner < 0]
        pair_semantic = {camera: semantic[camera] for camera in pair}
        pair_owner, pair_protected, stats = regularize_semantic_owner(
            pair_owner, pair_semantic, pair_valid, pair_cost,
            close_px=semantic_close, dilate_px=semantic_dilate,
            max_component_fraction=0.35)
        pair_owners[module] = pair_owner
        pair_seams[module] = seam
        pair_stats[str(module)] = stats
        black = np.zeros((*shape, 3), np.float32)
        composed, filled, _ = compose_single_source(
            {camera: images[camera] for camera in pair}, pair_valid,
            pair_owner, black, np.zeros(shape, bool))
        module_images[module] = np.clip(
            composed * 255.0, 0, 255).astype(np.uint8)
        module_valid[module] = filled
        module_semantic[module] = pair_protected
        stacked_cost = np.stack([
            np.where(valid[camera], cost[camera], np.inf)
            for camera in pair
        ])
        module_cost[module] = np.min(stacked_cost, axis=0)

    module_owner, first_seam, second_seam = three_band_owner(
        module_images, module_valid, module_semantic, cameras=(0, 1, 2),
        max_step=max_step, smoothness=smoothness,
        protect_margin=protect_margin)
    fallback = geometric_owner(module_valid, module_cost)
    module_owner[module_owner < 0] = fallback[module_owner < 0]
    module_owner, protected, module_stats = regularize_semantic_owner(
        module_owner, module_semantic, module_valid, module_cost,
        close_px=semantic_close, dilate_px=semantic_dilate,
        max_component_fraction=0.35)
    owner = np.full(shape, -1, np.int16)
    for module, pair_owner in pair_owners.items():
        selected = module_owner == module
        owner[selected] = pair_owner[selected]
    return owner, protected, {
        "pair_semantic": pair_stats,
        "module_semantic": module_stats,
        "pair_seams": {
            str(module): {
                "min": int(seam.min()),
                "median": float(np.median(seam)),
                "max": int(seam.max()),
            } for module, seam in pair_seams.items()
        },
        "module_seams": [{
            "min": int(seam.min()),
            "median": float(np.median(seam)),
            "max": int(seam.max()),
        } for seam in (first_seam, second_seam)],
    }


def compose_frequency_selective_blend(
        images, valid, owner, hard, protected=None, sigma=24.0,
        boundary_px=72, protect_dilate_px=12):
    """Blend low-frequency colour across seams while retaining owned detail."""
    import cv2

    owner = np.asarray(owner, np.int16)
    hard = np.asarray(hard, np.float32)
    if hard.shape[:2] != owner.shape:
        raise ValueError("hard image and owner shapes differ")
    sigma = max(float(sigma), 0.1)
    keys = sorted(images)
    low_images = {}
    weights = {}
    for key in keys:
        image = np.asarray(images[key])
        image_float = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):
            image_float /= float(np.iinfo(image.dtype).max)
        source_valid = np.asarray(valid[key], bool)
        valid_float = source_valid.astype(np.float32)
        normalizer = cv2.GaussianBlur(
            valid_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        numerator = cv2.GaussianBlur(
            image_float * valid_float[..., None], (0, 0),
            sigmaX=sigma, sigmaY=sigma)
        low_images[key] = np.divide(
            numerator, normalizer[..., None],
            out=np.zeros_like(numerator),
            where=normalizer[..., None] > 1e-5)
        owned = ((owner == key) & source_valid).astype(np.float32)
        weights[key] = cv2.GaussianBlur(
            owned, (0, 0), sigmaX=sigma, sigmaY=sigma) * valid_float

    total = np.zeros(owner.shape, np.float32)
    blended_low = np.zeros_like(hard)
    selected_low = np.zeros_like(hard)
    for key in keys:
        weight = weights[key]
        total += weight
        blended_low += low_images[key] * weight[..., None]
        selected = owner == key
        selected_low[selected] = low_images[key][selected]
    blended_low = np.divide(
        blended_low, total[..., None], out=selected_low.copy(),
        where=total[..., None] > 1e-5)

    boundary = np.zeros(owner.shape, bool)
    horizontal = (owner[:, 1:] != owner[:, :-1]) \
        & (owner[:, 1:] >= 0) & (owner[:, :-1] >= 0)
    vertical = (owner[1:] != owner[:-1]) \
        & (owner[1:] >= 0) & (owner[:-1] >= 0)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    overlap = np.sum(
        np.stack([np.asarray(valid[key], bool) for key in keys]), axis=0)
    transition = boundary | (distance <= max(int(boundary_px), 0))
    transition &= overlap >= 2
    transition &= owner >= 0
    if protected is not None:
        protected = np.asarray(protected, bool)
        radius = max(int(protect_dilate_px), 0)
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            protected = cv2.dilate(
                protected.astype(np.uint8), kernel) > 0
        transition &= ~protected

    # The hard-selected source contributes every high-frequency residual.
    # Only its slowly varying colour and illumination are mixed.
    candidate = np.clip(blended_low + hard - selected_low, 0.0, 1.0)
    output = hard.copy()
    output[transition] = candidate[transition]
    return output, transition


def overlay_regularized_match_warp(
        global_image, global_valid, source_image, mapping, active,
        canvas_transform, canvas_size, interpolation=None):
    """Replace an affine overlap with one continuous, bounded match warp."""
    import cv2

    global_image = np.asarray(global_image).copy()
    global_valid = np.asarray(global_valid, bool).copy()
    source_image = np.asarray(source_image)
    mapping = np.asarray(mapping, np.float32)
    active = np.asarray(active, bool)
    if mapping.shape[:2] != active.shape or mapping.shape[2:] != (2,):
        raise ValueError("mapping and active mask shapes differ")
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    remapped = cv2.remap(
        source_image, mapping[..., 0], mapping[..., 1], interpolation,
        borderMode=cv2.BORDER_CONSTANT)
    width, height = (int(value) for value in canvas_size)
    transform = np.asarray(canvas_transform, np.float64)
    dense_canvas = cv2.warpPerspective(
        remapped, transform, (width, height), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT)
    active_canvas = cv2.warpPerspective(
        active.astype(np.uint8), transform, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    global_image[active_canvas] = dense_canvas[active_canvas]
    global_valid[active_canvas] = True
    return global_image, global_valid


def overlay_dense_anchor_warp(
        global_image, global_valid, source_image, source_protected,
        anchor_protected, anchor_to_source, certainty, canvas_transform,
        canvas_size, certainty_floor=0.08, certainty_feather=0.08):
    """Refine static overlap in anchor coordinates with dense matches."""
    import cv2

    global_image = np.asarray(global_image).copy()
    global_valid = np.asarray(global_valid, bool).copy()
    source_image = np.asarray(source_image)
    source_protected = np.asarray(source_protected, bool)
    anchor_protected = np.asarray(anchor_protected, bool)
    mapping = np.asarray(anchor_to_source, np.float32)
    certainty = np.asarray(certainty, np.float32)
    if mapping.shape[:2] != anchor_protected.shape or \
            mapping.shape[2:] != (2,) or certainty.shape != anchor_protected.shape:
        raise ValueError("dense mapping must match the anchor image shape")
    source_height, source_width = source_image.shape[:2]
    # Tiny RoMa/grid_sample use pixel-centre normalized coordinates. OpenCV
    # remap uses integer pixel centres, hence the half-pixel conversion.
    map_x = source_width / 2.0 * (mapping[..., 0] + 1.0) - 0.5
    map_y = source_height / 2.0 * (mapping[..., 1] + 1.0) - 0.5
    inside = (
        np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0.0) & (map_x <= source_width - 1)
        & (map_y >= 0.0) & (map_y <= source_height - 1)
    )
    dense_image = cv2.remap(
        source_image, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)
    dense_protected = cv2.remap(
        source_protected.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT) > 0
    feather = max(float(certainty_feather), 1e-6)
    alpha = np.clip(
        (certainty - float(certainty_floor)) / feather, 0.0, 1.0)
    alpha *= inside & ~dense_protected & ~anchor_protected
    width, height = (int(value) for value in canvas_size)
    dense_canvas = cv2.warpPerspective(
        dense_image, np.asarray(canvas_transform, np.float64),
        (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)
    alpha_canvas = cv2.warpPerspective(
        alpha.astype(np.float32), np.asarray(canvas_transform, np.float64),
        (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)
    selected = alpha_canvas > 1e-4
    blend = alpha_canvas[..., None]
    global_image[selected] = np.clip(
        global_image[selected].astype(np.float32)
        * (1.0 - blend[selected])
        + dense_canvas[selected].astype(np.float32) * blend[selected],
        0, 255).astype(global_image.dtype)
    global_valid[selected] = True
    return global_image, global_valid, float(selected.mean())


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

    np.random.seed(int(args.seed))
    cv2.setRNGSeed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

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
            stable, stable_inliers = fit_stable_affine(
                report["source"], report["target"],
                threshold_px=args.affine_threshold)
            if stable is None or int(stable_inliers.sum()) \
                    < args.min_affine_inliers:
                print(f"cam{camera}->cam{anchor}: no stable affine", flush=True)
                continue
            homographies = [stable] + [
                model["homography"] for model in report["homographies"]]
            model_map = assign_segments_to_models(
                segments, report["source"], report["labels"] + 1,
                len(homographies), default_model=0,
                min_votes=args.segment_min_votes,
                min_vote_fraction=args.segment_vote_fraction)
            model_map = guard_segment_models(
                segments, model_map, homographies, fallback_index=0,
                max_delta_px=args.max_local_delta,
                max_relative_span=args.max_local_span)
            mesh_mapping = mesh_active = mesh_report = None
            if args.continuous_refine:
                mesh_matches = report["labels"] >= 0
                try:
                    mesh_mapping, mesh_active, mesh_report = \
                        regularized_match_map(
                            images[camera].shape, images[anchor].shape,
                            report["source"][mesh_matches],
                            report["target"][mesh_matches], stable,
                            cell_px=args.mesh_cell,
                            smooth_sigma=args.mesh_smooth,
                            max_correction_px=args.mesh_max_correction,
                            hull_feather_px=args.mesh_feather)
                except (ValueError, np.linalg.LinAlgError) as error:
                    print(
                        f"cam{camera}->cam{anchor}: "
                        f"continuous refine skipped: {error}", flush=True)
            reverse_warp = reverse_certainty = None
            if args.dense_refine:
                reverse_warp_tensor, reverse_certainty_tensor = matcher.match(
                    os.fspath(paths[anchor]), os.fspath(paths[camera]))
                reverse_warp = reverse_warp_tensor[..., 2:].detach().cpu().numpy()
                reverse_certainty = \
                    reverse_certainty_tensor.detach().cpu().numpy()
            camera_models[camera] = {
                "homographies": homographies,
                "model_map": model_map,
                "report": report,
                "stable_affine_inliers": int(stable_inliers.sum()),
                "mesh_mapping": mesh_mapping,
                "mesh_active": mesh_active,
                "mesh_report": mesh_report,
                "anchor_to_source": reverse_warp,
                "dense_certainty": reverse_certainty,
            }
            print(
                f"cam{camera}->cam{anchor}: "
                f"H={report['explained_matches']} "
                f"models={len(report['homographies'])} "
                f"affine={int(stable_inliers.sum())} "
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
    dense_fractions = {}
    mesh_reports = {}
    for camera, values in camera_models.items():
        warped[camera], valid[camera] = segmented_warp(
            images[camera], values["model_map"], values["homographies"],
            canvas_transform, canvas_size)
        semantic_image, _ = segmented_warp(
            (semantic[camera].astype(np.uint8) * 255),
            values["model_map"], values["homographies"],
            canvas_transform, canvas_size, interpolation=cv2.INTER_NEAREST)
        warped_semantic[camera] = semantic_image > 127
        if values.get("mesh_mapping") is not None:
            warped[camera], valid[camera] = overlay_regularized_match_warp(
                warped[camera], valid[camera], images[camera],
                values["mesh_mapping"], values["mesh_active"],
                canvas_transform, canvas_size)
            semantic_image, _ = overlay_regularized_match_warp(
                semantic_image, semantic_image > 127,
                semantic[camera].astype(np.uint8) * 255,
                values["mesh_mapping"], values["mesh_active"],
                canvas_transform, canvas_size,
                interpolation=cv2.INTER_NEAREST)
            warped_semantic[camera] = semantic_image > 127
            mesh_reports[camera] = values["mesh_report"]
        if values.get("anchor_to_source") is not None:
            warped[camera], valid[camera], dense_fractions[camera] = \
                overlay_dense_anchor_warp(
                    warped[camera], valid[camera], images[camera],
                    semantic[camera], semantic[anchor],
                    values["anchor_to_source"], values["dense_certainty"],
                    canvas_transform, canvas_size,
                    args.dense_certainty, args.dense_feather)

    adjusted = exposure_compensate(warped, valid)
    cost = {}
    for camera in sorted(valid):
        distance = cv2.distanceTransform(
            valid[camera].astype(np.uint8), cv2.DIST_L2, 3)
        cost[camera] = -distance.astype(np.float32)
    if args.composition == "hierarchical-six":
        owner, protected, semantic_stats = hierarchical_six_owner(
            adjusted, valid, warped_semantic, cost,
            max_step=args.seam_max_step,
            smoothness=args.seam_smoothness,
            protect_margin=args.seam_protect_margin,
            semantic_close=args.semantic_close,
            semantic_dilate=args.semantic_dilate)
        graph_masks = None
    elif args.composition == "three-band":
        band_cameras = tuple(int(value) for value in args.band_cameras.split(","))
        if len(band_cameras) != 3:
            raise ValueError("--band-cameras must contain three camera ids")
        owner, first_seam, second_seam = three_band_owner(
            adjusted, valid, warped_semantic, band_cameras,
            args.seam_max_step, args.seam_smoothness,
            args.seam_protect_margin)
        fallback = geometric_owner(valid, cost)
        owner[owner < 0] = fallback[owner < 0]
        graph_masks = None
    else:
        owner, graph_masks = graphcut_owner(adjusted, valid, cost)
    if args.composition != "hierarchical-six":
        owner, protected, semantic_stats = regularize_semantic_owner(
            owner, warped_semantic, valid, cost,
            close_px=args.semantic_close, dilate_px=args.semantic_dilate,
            max_component_fraction=0.35)
    if args.composition == "anchor" or (
            args.composition == "graphcut" and args.anchor_authority):
        owner = enforce_anchor_authority(owner, valid[anchor], anchor)
    black = np.zeros((*owner.shape, 3), np.float32)
    no_background = np.zeros(owner.shape, bool)
    hard, _, _ = compose_single_source(
        adjusted, valid, owner, black, no_background)
    gated, transition, gated_out = compose_gated_boundary_blend(
        adjusted, valid, cost, owner, black, no_background,
        protected=protected, gate=args.blend_gate / 255.0,
        boundary_px=args.blend_width, temperature=args.blend_temperature)
    frequency_protected = (
        protected if args.frequency_preserve_protected else None)
    frequency, frequency_transition = compose_frequency_selective_blend(
        adjusted, valid, owner, hard, protected=frequency_protected,
        sigma=args.frequency_sigma, boundary_px=args.frequency_width,
        protect_dilate_px=args.frequency_protect_dilate)

    adjusted, arrays, valid, crop = crop_valid(
        adjusted, [hard, gated, frequency, owner, protected, transition,
                   gated_out, frequency_transition],
        valid)
    hard, gated, frequency, owner, protected, transition, gated_out, \
        frequency_transition = arrays
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.fspath(output / "panorama_hard.png"),
                np.clip(hard * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(os.fspath(output / "panorama_gated.png"),
                np.clip(gated * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(os.fspath(output / "panorama_frequency.png"),
                np.clip(frequency * 255.0, 0, 255).astype(np.uint8))
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
    cv2.imwrite(os.fspath(output / "frequency_transition.png"),
                frequency_transition.astype(np.uint8) * 255)

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
        "frequency_transition_fraction": float(
            frequency_transition.mean()),
        "dense_refine_fraction": dense_fractions,
        "continuous_refine": mesh_reports,
        "semantic_ownership": semantic_stats,
        "composition": args.composition,
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
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--affine-threshold", type=float, default=6.0)
    parser.add_argument("--min-affine-inliers", type=int, default=100)
    parser.add_argument("--segment-min-votes", type=int, default=4)
    parser.add_argument("--segment-vote-fraction", type=float, default=0.55)
    parser.add_argument("--max-local-delta", type=float, default=180.0)
    parser.add_argument("--max-local-span", type=float, default=2.5)
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
    parser.add_argument("--frequency-sigma", type=float, default=24.0)
    parser.add_argument("--frequency-width", type=int, default=72)
    parser.add_argument("--frequency-protect-dilate", type=int, default=12)
    parser.add_argument(
        "--frequency-preserve-protected",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--anchor-authority", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument(
        "--composition",
        choices=("hierarchical-six", "three-band", "anchor", "graphcut"),
        default="three-band")
    parser.add_argument("--band-cameras", default="1,3,5")
    parser.add_argument("--seam-max-step", type=int, default=6)
    parser.add_argument("--seam-smoothness", type=float, default=0.03)
    parser.add_argument("--seam-protect-margin", type=int, default=64)
    parser.add_argument(
        "--dense-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dense-certainty", type=float, default=0.08)
    parser.add_argument("--dense-feather", type=float, default=0.08)
    parser.add_argument(
        "--continuous-refine", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--mesh-cell", type=int, default=32)
    parser.add_argument("--mesh-smooth", type=float, default=1.0)
    parser.add_argument("--mesh-max-correction", type=float, default=120.0)
    parser.add_argument("--mesh-feather", type=float, default=48.0)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
