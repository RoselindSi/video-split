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
