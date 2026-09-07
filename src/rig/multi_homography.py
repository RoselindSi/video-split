"""Parameter-free pair geometry for parallax-tolerant panorama stitching.

Dense correspondences are first checked against one epipolar model, then
partitioned into several local homographies.  The epipolar model answers
whether two cameras see the same static scene; the homographies describe the
piecewise image warps needed when one global plane cannot explain parallax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairGeometryConfig:
    min_certainty: float = 0.05
    fundamental_threshold_px: float = 2.0
    homography_threshold_px: float = 3.0
    min_fundamental_inliers: int = 80
    min_fundamental_ratio: float = 0.10
    min_homography_inliers: int = 32
    max_homographies: int = 8
    min_explained_matches: int = 120
    min_explained_ratio: float = 0.12
    min_grid_coverage: float = 0.06
    grid_rows: int = 8
    grid_columns: int = 8


def _points(value, name):
    points = np.asarray(value, np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape [matches, 2]")
    return points


def _sample_mask(mask, points):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("exclusion masks must be two-dimensional")
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    sampled = np.ones(len(points), bool)
    sampled[inside] = mask[y[inside], x[inside]]
    return sampled


def static_correspondence_mask(source, target, certainty=None,
                               source_exclusion=None, target_exclusion=None,
                               min_certainty=0.05):
    """Select finite, in-frame correspondences outside dynamic masks."""
    source = _points(source, "source")
    target = _points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target match counts differ")
    keep = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    if certainty is not None:
        certainty = np.asarray(certainty, np.float32).reshape(-1)
        if len(certainty) != len(source):
            raise ValueError("certainty and match counts differ")
        keep &= np.isfinite(certainty) & (certainty >= float(min_certainty))
    if source_exclusion is not None:
        keep &= ~_sample_mask(source_exclusion, source)
    if target_exclusion is not None:
        keep &= ~_sample_mask(target_exclusion, target)
    return keep


def grid_coverage(points, image_shape, rows=8, columns=8):
    """Fraction of coarse image cells containing at least one point."""
    points = _points(points, "points")
    height, width = (int(value) for value in image_shape[:2])
    rows, columns = int(rows), int(columns)
    if height < 1 or width < 1 or rows < 1 or columns < 1:
        raise ValueError("image and grid dimensions must be positive")
    if not len(points):
        return 0.0
    x = np.floor(points[:, 0] / width * columns).astype(np.int64)
    y = np.floor(points[:, 1] / height * rows).astype(np.int64)
    inside = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    occupied = np.unique(y[inside] * columns + x[inside])
    return float(len(occupied)) / float(rows * columns)


def symmetric_homography_error(homography, source, target):
    """Maximum forward/backward transfer error for every correspondence."""
    import cv2

    source = _points(source, "source")
    target = _points(target, "target")
    homography = np.asarray(homography, np.float64)
    if homography.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full(len(source), np.inf, np.float32)
    forward = cv2.perspectiveTransform(source[:, None], homography)[:, 0]
    backward = cv2.perspectiveTransform(target[:, None], inverse)[:, 0]
    error = np.maximum(
        np.linalg.norm(forward - target, axis=1),
        np.linalg.norm(backward - source, axis=1),
    )
    return error.astype(np.float32)


def fundamental_inlier_mask(source, target, threshold_px=2.0):
    """Robust epipolar support, used to reject unrelated camera pairs."""
    import cv2

    source = _points(source, "source")
    target = _points(target, "target")
    if len(source) < 8:
        return None, np.zeros(len(source), bool)
    try:
        fundamental, inliers = cv2.findFundamentalMat(
            source, target, cv2.USAC_MAGSAC, float(threshold_px), 0.999, 10000)
    except cv2.error:
        fundamental, inliers = cv2.findFundamentalMat(
            source, target, cv2.FM_RANSAC, float(threshold_px), 0.999)
    if fundamental is None or np.asarray(fundamental).shape != (3, 3) or \
            inliers is None:
        return None, np.zeros(len(source), bool)
    return np.asarray(fundamental, np.float64), inliers.reshape(-1).astype(bool)


def fit_piecewise_homographies(source, target,
                               config=PairGeometryConfig()):
    """Greedily explain correspondences with bounded local homographies."""
    import cv2

    source = _points(source, "source")
    target = _points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target match counts differ")
    remaining = np.arange(len(source), dtype=np.int64)
    labels = np.full(len(source), -1, np.int16)
    models = []
    for model_index in range(int(config.max_homographies)):
        if len(remaining) < max(4, int(config.min_homography_inliers)):
            break
        try:
            homography, _ = cv2.findHomography(
                source[remaining], target[remaining], cv2.USAC_MAGSAC,
                float(config.homography_threshold_px), None, 10000, 0.999)
        except cv2.error:
            homography, _ = cv2.findHomography(
                source[remaining], target[remaining], cv2.RANSAC,
                float(config.homography_threshold_px))
        if homography is None:
            break
        error = symmetric_homography_error(
            homography, source[remaining], target[remaining])
        local_inliers = np.isfinite(error) \
            & (error <= float(config.homography_threshold_px))
        count = int(local_inliers.sum())
        if count < int(config.min_homography_inliers):
            break
        selected = remaining[local_inliers]
        labels[selected] = model_index
        selected_error = error[local_inliers]
        models.append({
            "homography": np.asarray(homography, np.float64),
            "inlier_indices": selected,
            "inliers": count,
            "median_error_px": float(np.median(selected_error)),
            "p90_error_px": float(np.percentile(selected_error, 90.0)),
        })
        remaining = remaining[~local_inliers]
    return models, labels


def analyze_pair(source, target, source_shape, target_shape, certainty=None,
                 source_exclusion=None, target_exclusion=None,
                 config=PairGeometryConfig()):
    """Summarize whether a camera pair has useful piecewise-static support."""
    source = _points(source, "source")
    target = _points(target, "target")
    keep = static_correspondence_mask(
        source, target, certainty, source_exclusion, target_exclusion,
        min_certainty=config.min_certainty)
    static_source, static_target = source[keep], target[keep]
    fundamental, epipolar = fundamental_inlier_mask(
        static_source, static_target, config.fundamental_threshold_px)
    models, labels = fit_piecewise_homographies(
        static_source, static_target, config)
    explained = labels >= 0
    explained_count = int(explained.sum())
    static_count = int(len(static_source))
    fundamental_count = int(epipolar.sum())
    explained_errors = np.concatenate([
        symmetric_homography_error(
            model["homography"],
            static_source[model["inlier_indices"]],
            static_target[model["inlier_indices"]],
        )
        for model in models
    ]) if models else np.empty(0, np.float32)
    source_coverage = grid_coverage(
        static_source[explained], source_shape,
        config.grid_rows, config.grid_columns)
    target_coverage = grid_coverage(
        static_target[explained], target_shape,
        config.grid_rows, config.grid_columns)
    fundamental_ratio = fundamental_count / max(static_count, 1)
    explained_ratio = explained_count / max(static_count, 1)
    residual_p90 = (float(np.percentile(explained_errors, 90.0))
                    if len(explained_errors) else float("inf"))
    checks = {
        "enough_epipolar_inliers": (
            fundamental_count >= config.min_fundamental_inliers),
        "enough_epipolar_ratio": (
            fundamental_ratio >= config.min_fundamental_ratio),
        "enough_piecewise_inliers": (
            explained_count >= config.min_explained_matches),
        "enough_piecewise_ratio": (
            explained_ratio >= config.min_explained_ratio),
        "enough_source_coverage": (
            source_coverage >= config.min_grid_coverage),
        "enough_target_coverage": (
            target_coverage >= config.min_grid_coverage),
        "bounded_piecewise_residual": (
            residual_p90 <= config.homography_threshold_px),
    }
    return {
        "matches": int(len(source)),
        "static_matches": static_count,
        "fundamental": fundamental,
        "fundamental_inliers": fundamental_count,
        "fundamental_ratio": float(fundamental_ratio),
        "homographies": models,
        "labels": labels,
        "explained_matches": explained_count,
        "explained_ratio": float(explained_ratio),
        "piecewise_p90_px": residual_p90,
        "source_grid_coverage": source_coverage,
        "target_grid_coverage": target_coverage,
        "checks": checks,
        "accepted": bool(all(checks.values())),
        "source": static_source,
        "target": static_target,
        "certainty": (None if certainty is None
                      else np.asarray(certainty, np.float32).reshape(-1)[keep]),
    }


def accepted_components(camera_count, pair_reports):
    """Connected components induced by accepted pair reports."""
    camera_count = int(camera_count)
    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    adjacency = {camera: set() for camera in range(camera_count)}
    for report in pair_reports:
        if not report.get("accepted", False):
            continue
        first, second = (int(value) for value in report["pair"])
        if first not in adjacency or second not in adjacency:
            raise ValueError("pair contains an out-of-range camera")
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = []
    unseen = set(adjacency)
    while unseen:
        pending = [min(unseen)]
        component = []
        unseen.remove(pending[0])
        while pending:
            camera = pending.pop()
            component.append(camera)
            neighbours = sorted(adjacency[camera] & unseen, reverse=True)
            for neighbour in neighbours:
                unseen.remove(neighbour)
                pending.append(neighbour)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def assign_segments_to_models(segments, points, labels, model_count,
                              default_model=None, min_votes=1,
                              min_vote_fraction=0.0):
    """Assign each content segment one model without crossing its boundary."""
    segments = np.asarray(segments)
    if segments.ndim != 2:
        raise ValueError("segments must be two-dimensional")
    points = _points(points, "points")
    labels = np.asarray(labels, np.int64).reshape(-1)
    model_count = int(model_count)
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")
    if model_count < 1:
        raise ValueError("model_count must be positive")
    segment_values, inverse = np.unique(segments, return_inverse=True)
    segment_map = inverse.reshape(segments.shape)
    votes = np.zeros((len(segment_values), model_count), np.int64)
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    valid = (
        (labels >= 0) & (labels < model_count)
        & (x >= 0) & (x < segments.shape[1])
        & (y >= 0) & (y < segments.shape[0])
    )
    np.add.at(votes, (segment_map[y[valid], x[valid]], labels[valid]), 1)
    assignment = np.argmax(votes, axis=1).astype(np.int16)
    totals = votes.sum(axis=1)
    strongest = votes.max(axis=1)
    seeded = (totals >= int(min_votes)) & (
        strongest / np.maximum(totals, 1) >= float(min_vote_fraction))
    if not seeded.any():
        fill = 0 if default_model is None else int(default_model)
        return np.full(segments.shape, fill, np.int16)

    if default_model is not None:
        assignment[~seeded] = int(default_model)
        return assignment[segment_map]

    # Empty segments inherit the nearest seeded segment in source-image
    # coordinates. This extrapolates only across content regions and never
    # averages two model transforms inside one region.
    yy, xx = np.indices(segments.shape)
    area = np.bincount(segment_map.ravel(), minlength=len(segment_values))
    centroid_x = np.bincount(
        segment_map.ravel(), weights=xx.ravel(),
        minlength=len(segment_values)) / np.maximum(area, 1)
    centroid_y = np.bincount(
        segment_map.ravel(), weights=yy.ravel(),
        minlength=len(segment_values)) / np.maximum(area, 1)
    seeded_indices = np.flatnonzero(seeded)
    for index in np.flatnonzero(~seeded):
        distance = (
            (centroid_x[seeded_indices] - centroid_x[index]) ** 2
            + (centroid_y[seeded_indices] - centroid_y[index]) ** 2
        )
        assignment[index] = assignment[seeded_indices[np.argmin(distance)]]
    return assignment[segment_map]


def fit_stable_affine(source, target, threshold_px=4.0):
    """Fit a non-projective fallback that remains finite outside overlap."""
    import cv2

    source = _points(source, "source")
    target = _points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target match counts differ")
    if len(source) < 3:
        return None, np.zeros(len(source), bool)
    affine, inliers = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold_px), maxIters=10000,
        confidence=0.999, refineIters=10)
    if affine is None or inliers is None:
        return None, np.zeros(len(source), bool)
    homography = np.eye(3, dtype=np.float64)
    homography[:2] = affine
    return homography, inliers.reshape(-1).astype(bool)


def regularized_match_map(
        source_shape, target_shape, source, target, affine,
        cell_px=32, smooth_sigma=1.0, max_correction_px=120.0,
        hull_feather_px=48):
    """Build a continuous target-to-source map from robust match residuals."""
    import cv2

    source = _points(source, "source")
    target = _points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target match counts differ")
    source_height, source_width = (
        int(value) for value in source_shape[:2])
    target_height, target_width = (
        int(value) for value in target_shape[:2])
    affine = np.asarray(affine, np.float64)
    if affine.shape != (3, 3):
        raise ValueError("affine must be 3x3")
    inverse = np.linalg.inv(affine)
    baseline = cv2.perspectiveTransform(target[:, None], inverse)[:, 0]
    residual = source - baseline
    magnitude = np.linalg.norm(residual, axis=1)
    in_frame = (
        np.isfinite(target).all(axis=1)
        & np.isfinite(residual).all(axis=1)
        & (target[:, 0] >= 0) & (target[:, 0] < target_width)
        & (target[:, 1] >= 0) & (target[:, 1] < target_height)
    )
    finite_magnitude = magnitude[in_frame]
    if len(finite_magnitude) < 6:
        raise ValueError("not enough finite matches for a continuous map")
    median = float(np.median(finite_magnitude))
    mad = float(np.median(np.abs(finite_magnitude - median)))
    robust_limit = median + max(4.0 * 1.4826 * mad, 8.0)
    limit = min(float(max_correction_px), robust_limit)
    keep = in_frame & (magnitude <= limit)
    if int(keep.sum()) < 6:
        raise ValueError("not enough bounded matches for a continuous map")

    cell_px = max(int(cell_px), 4)
    cells = {}
    for point, value in zip(target[keep], residual[keep]):
        key = (int(point[1] // cell_px), int(point[0] // cell_px))
        cells.setdefault(key, [[], []])
        cells[key][0].append(point)
        cells[key][1].append(value)
    control_points = np.asarray([
        np.median(values[0], axis=0) for values in cells.values()
    ], np.float32)
    control_residuals = np.asarray([
        np.median(values[1], axis=0) for values in cells.values()
    ], np.float32)
    if len(control_points) < 6:
        raise ValueError("not enough occupied mesh cells")

    grid_x = np.unique(np.r_[
        np.arange(0, target_width, cell_px), target_width - 1])
    grid_y = np.unique(np.r_[
        np.arange(0, target_height, cell_px), target_height - 1])
    coarse = np.zeros((len(grid_y), len(grid_x), 2), np.float32)
    known = np.zeros(coarse.shape[:2], bool)
    for point, value in zip(control_points, control_residuals):
        column = int(np.argmin(np.abs(grid_x - point[0])))
        row = int(np.argmin(np.abs(grid_y - point[1])))
        coarse[row, column] = value
        known[row, column] = True
    missing = (~known).astype(np.uint8)
    for channel in range(2):
        coarse[..., channel] = cv2.inpaint(
            coarse[..., channel], missing, 2.0, cv2.INPAINT_TELEA)
    if float(smooth_sigma) > 0.0:
        coarse = cv2.GaussianBlur(
            coarse, (0, 0), sigmaX=float(smooth_sigma),
            sigmaY=float(smooth_sigma))
    correction = cv2.resize(
        coarse, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    hull = cv2.convexHull(control_points.astype(np.float32))
    support = np.zeros((target_height, target_width), np.uint8)
    cv2.fillConvexPoly(support, np.rint(hull).astype(np.int32), 1)
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 3)
    feather = max(float(hull_feather_px), 1.0)
    weight = np.clip(distance / feather, 0.0, 1.0)
    correction *= weight[..., None]
    correction_norm = np.linalg.norm(correction, axis=2)
    scale = np.minimum(
        1.0, float(max_correction_px) / np.maximum(correction_norm, 1e-6))
    correction *= scale[..., None]

    grid_columns, grid_rows = np.meshgrid(
        np.arange(target_width, dtype=np.float32),
        np.arange(target_height, dtype=np.float32))
    target_grid = np.stack(
        (grid_columns, grid_rows), axis=-1).reshape(-1, 1, 2)
    base_map = cv2.perspectiveTransform(target_grid, inverse).reshape(
        target_height, target_width, 2)
    mapping = base_map + correction
    inside = (
        np.isfinite(mapping).all(axis=2)
        & (mapping[..., 0] >= 0.0)
        & (mapping[..., 0] <= source_width - 1)
        & (mapping[..., 1] >= 0.0)
        & (mapping[..., 1] <= source_height - 1)
    )
    active = (weight > 1e-4) & inside
    return mapping.astype(np.float32), active, {
        "input_matches": int(len(source)),
        "bounded_matches": int(keep.sum()),
        "control_points": int(len(control_points)),
        "correction_limit_px": float(limit),
        "active_fraction": float(active.mean()),
    }


def guard_segment_models(segments, model_map, homographies,
                         fallback_index=0, max_delta_px=180.0,
                         max_relative_span=2.5):
    """Reject local transforms that extrapolate wildly over one segment."""
    import cv2

    segments = np.asarray(segments)
    model_map = np.asarray(model_map, np.int16).copy()
    if segments.shape != model_map.shape or segments.ndim != 2:
        raise ValueError("segments and model_map must share a 2D shape")
    homographies = [np.asarray(value, np.float64) for value in homographies]
    fallback_index = int(fallback_index)
    fallback = homographies[fallback_index]
    for segment in np.unique(segments):
        selected = segments == segment
        models, counts = np.unique(model_map[selected], return_counts=True)
        model_index = int(models[np.argmax(counts)])
        if model_index == fallback_index:
            continue
        rows, columns = np.nonzero(selected)
        x0, x1 = float(columns.min()), float(columns.max())
        y0, y1 = float(rows.min()), float(rows.max())
        samples = np.asarray([
            [x0, y0], [x1, y0], [x1, y1], [x0, y1],
            [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        ], np.float32)
        local_points = cv2.perspectiveTransform(
            samples[:, None], homographies[model_index])[:, 0]
        fallback_points = cv2.perspectiveTransform(
            samples[:, None], fallback)[:, 0]
        finite = np.isfinite(local_points).all() and \
            np.isfinite(fallback_points).all()
        if finite:
            delta = float(np.max(np.linalg.norm(
                local_points - fallback_points, axis=1)))
            local_span = np.ptp(local_points[:4], axis=0)
            fallback_span = np.maximum(
                np.ptp(fallback_points[:4], axis=0), 1.0)
            span_ratio = local_span / fallback_span
            finite = delta <= float(max_delta_px) and bool(np.all(
                (span_ratio >= 1.0 / float(max_relative_span))
                & (span_ratio <= float(max_relative_span))))
        if not finite:
            model_map[selected] = fallback_index
    return model_map


def transformed_support_points(model_map, homographies, step=24):
    """Sample a segmented image's transformed footprint for canvas sizing."""
    import cv2

    model_map = np.asarray(model_map, np.int64)
    if model_map.ndim != 2:
        raise ValueError("model_map must be two-dimensional")
    homographies = [np.asarray(value, np.float64) for value in homographies]
    height, width = model_map.shape
    step = max(1, int(step))
    x = np.unique(np.r_[np.arange(0, width, step), width - 1])
    y = np.unique(np.r_[np.arange(0, height, step), height - 1])
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
    labels = model_map[yy.ravel(), xx.ravel()]
    transformed = []
    for model_index, homography in enumerate(homographies):
        selected = labels == model_index
        if selected.any():
            values = cv2.perspectiveTransform(
                points[selected, None], homography)[:, 0]
            transformed.append(values[np.isfinite(values).all(axis=1)])
    return (np.concatenate(transformed).astype(np.float32)
            if transformed else np.empty((0, 2), np.float32))


def segmented_warp(image, model_map, homographies, canvas_transform,
                   canvas_size, interpolation=None):
    """Warp one image with one homography per complete content segment."""
    import cv2

    image = np.asarray(image)
    model_map = np.asarray(model_map, np.int64)
    if image.shape[:2] != model_map.shape:
        raise ValueError("image and model_map shapes differ")
    width, height = (int(value) for value in canvas_size)
    if width < 1 or height < 1:
        raise ValueError("canvas dimensions must be positive")
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    if image.ndim == 2:
        output = np.zeros((height, width), image.dtype)
    else:
        output = np.zeros((height, width, image.shape[2]), image.dtype)
    valid = np.zeros((height, width), bool)
    canvas_transform = np.asarray(canvas_transform, np.float64)
    for model_index, homography in enumerate(homographies):
        source_mask = (model_map == model_index).astype(np.uint8)
        if not source_mask.any():
            continue
        transform = canvas_transform @ np.asarray(homography, np.float64)
        warped_mask = cv2.warpPerspective(
            source_mask, transform, (width, height), flags=cv2.INTER_NEAREST)
        selected = (warped_mask > 0) & ~valid
        if not selected.any():
            continue
        warped = cv2.warpPerspective(
            image, transform, (width, height), flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT)
        output[selected] = warped[selected]
        valid[selected] = True
    return output, valid
