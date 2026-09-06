"""Stereo geometry derived from learned camera poses, not hardware calibration.

Self-Cali-GS estimates one perspective camera for every undistorted rig view.
The synchronized physical pairs can therefore be rectified from those learned
poses, and dense disparity can place moving foreground at its current depth.
This module contains only the geometry and deterministic z-buffer; inference
stays in :mod:`src.rig.fast_foundation_stereo`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearnedStereoRectification:
    map_left: tuple
    map_right: tuple
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    left_camera_to_world: np.ndarray
    baseline: float


@dataclass(frozen=True)
class EquirectangularLayer:
    rgb: np.ndarray
    range_m: np.ndarray
    valid: np.ndarray
    source: np.ndarray


def camera_matrix_from_fov(width, height, fov_x, fov_y):
    """Centered pinhole matrix matching a learned Self-Cali camera."""
    width, height = int(width), int(height)
    if width <= 1 or height <= 1:
        raise ValueError("camera dimensions must be greater than one")
    if not 0.0 < float(fov_x) < np.pi or not 0.0 < float(fov_y) < np.pi:
        raise ValueError("camera FOV must be in (0, pi)")
    return np.asarray([
        [width / (2.0 * np.tan(float(fov_x) / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(float(fov_y) / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0],
    ], np.float64)


def scale_camera_matrix(camera_matrix, source_size, target_size):
    """Scale pinhole intrinsics for an independently resized image."""
    source_width, source_height = (float(value) for value in source_size)
    target_width, target_height = (float(value) for value in target_size)
    if min(source_width, source_height, target_width, target_height) <= 0.0:
        raise ValueError("camera dimensions must be positive")
    scaled = np.asarray(camera_matrix, np.float64).copy()
    if scaled.shape != (3, 3):
        raise ValueError("camera matrix must be 3x3")
    scaled[0] *= target_width / source_width
    scaled[1] *= target_height / source_height
    return scaled


def temporal_change_mask(current, references, threshold=24.0 / 255.0):
    """Detect current content absent from nearby frames of one camera."""
    import cv2

    current = np.asarray(current, np.float32)
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("current image must have shape [height, width, 3]")
    if current.size and float(np.nanmax(current)) > 1.5:
        current = current / 255.0
    resized = []
    for reference in references:
        reference = np.asarray(reference, np.float32)
        if reference.ndim != 3 or reference.shape[2] != 3:
            raise ValueError("reference images must have three channels")
        if reference.shape[:2] != current.shape[:2]:
            reference = cv2.resize(
                reference, (current.shape[1], current.shape[0]),
                interpolation=cv2.INTER_LINEAR)
        if reference.size and float(np.nanmax(reference)) > 1.5:
            reference = reference / 255.0
        resized.append(reference)
    if not resized:
        return np.ones(current.shape[:2], bool)
    temporal_background = np.median(np.stack(resized), axis=0)
    difference = np.abs(current - temporal_background).mean(axis=2)
    return difference >= float(threshold)


def pose_guided_foreground_mask(image, boxes, keypoints=None, residual=None,
                                keypoint_confidence=0.25):
    """Turn person boxes and pose limbs into conservative GrabCut masks."""
    import cv2

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [height, width, 3]")
    if image.dtype != np.uint8:
        image = np.clip(
            image * (255.0 if image.size and image.max() <= 1.5 else 1.0),
            0, 255).astype(np.uint8)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if keypoints is None:
        keypoints = np.zeros((len(boxes), 17, 3), np.float32)
    keypoints = np.asarray(keypoints, np.float32)
    if keypoints.shape != (len(boxes), 17, 3):
        raise ValueError("keypoints must have shape [people, 17, 3]")
    if residual is None:
        residual = np.zeros(image.shape[:2], bool)
    residual = np.asarray(residual, bool)
    if residual.shape != image.shape[:2]:
        raise ValueError("residual mask has the wrong shape")

    height, width = image.shape[:2]
    output = np.zeros((height, width), bool)
    skeleton = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    )
    for box, pose in zip(boxes, keypoints):
        x0, y0, x1, y1 = box
        box_width = max(float(x1 - x0), 1.0)
        box_height = max(float(y1 - y0), 1.0)
        pad = int(round(0.04 * max(box_width, box_height)))
        x0 = max(0, int(np.floor(x0)) - pad)
        y0 = max(0, int(np.floor(y0)) - pad)
        x1 = min(width, int(np.ceil(x1)) + pad)
        y1 = min(height, int(np.ceil(y1)) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        inside = np.zeros((height, width), bool)
        inside[y0:y1, x0:x1] = True
        grabcut = np.full((height, width), cv2.GC_BGD, np.uint8)
        grabcut[inside] = cv2.GC_PR_BGD
        grabcut[inside & residual] = cv2.GC_PR_FGD

        seed = np.zeros((height, width), np.uint8)
        thickness = max(3, int(round(0.04 * max(box_width, box_height))))
        for first, second in skeleton:
            if pose[first, 2] < keypoint_confidence or \
                    pose[second, 2] < keypoint_confidence:
                continue
            cv2.line(
                seed, tuple(np.rint(pose[first, :2]).astype(int)),
                tuple(np.rint(pose[second, :2]).astype(int)), 1,
                thickness=thickness)
        for point in pose:
            if point[2] >= keypoint_confidence:
                cv2.circle(
                    seed, tuple(np.rint(point[:2]).astype(int)),
                    max(2, thickness // 2), 1, thickness=-1)
        seed = ((seed > 0) & inside).astype(np.uint8)
        if not seed.any():
            center = (
                int(round((x0 + x1) / 2.0)),
                int(round(y0 + 0.45 * (y1 - y0))),
            )
            axes = (
                max(2, int(round(0.12 * (x1 - x0)))),
                max(2, int(round(0.18 * (y1 - y0)))),
            )
            cv2.ellipse(seed, center, axes, 0, 0, 360, 1, thickness=-1)
            seed = ((seed > 0) & inside).astype(np.uint8)
        seed = seed > 0
        grabcut[seed] = cv2.GC_FGD
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                image, grabcut, None, background_model, foreground_model,
                3, cv2.GC_INIT_WITH_MASK)
            person = ((grabcut == cv2.GC_FGD)
                      | (grabcut == cv2.GC_PR_FGD)) & inside
        except cv2.error:
            person = (residual | seed) & inside
        output |= person
    return output


def regularize_component_disparity(disparity, mask, min_component_px=24,
                                    min_disparity=0.5):
    """Use one robust depth per foreground component to preserve silhouettes."""
    import cv2

    disparity = np.asarray(disparity, np.float32)
    mask = np.asarray(mask, bool)
    if disparity.shape != mask.shape:
        raise ValueError("disparity and mask shapes differ")
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    regularized = np.zeros_like(disparity)
    accepted = np.zeros_like(mask)
    components = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_px):
            continue
        component = labels == label
        values = disparity[
            component & np.isfinite(disparity)
            & (disparity > float(min_disparity))]
        if values.size < max(int(min_component_px), area // 3):
            continue
        median = float(np.median(values))
        regularized[component] = median
        accepted[component] = True
        components += 1
    return regularized, accepted, components


def select_stereo_replacements(dynamic, stereo, min_coverage=0.75,
                               min_component_px=24, association_px=8):
    """Accept only dynamic regions with a nearly complete stereo rendering."""
    import cv2

    dynamic = np.asarray(dynamic, bool)
    stereo = np.asarray(stereo, bool)
    if dynamic.shape != stereo.shape:
        raise ValueError("dynamic and stereo masks differ")
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dynamic.astype(np.uint8), connectivity=8)
    erase = np.zeros_like(dynamic)
    selected_stereo = np.zeros_like(stereo)
    accepted = 0
    radius = int(association_px)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_px):
            continue
        component = labels == label
        neighborhood = cv2.dilate(
            component.astype(np.uint8), kernel) > 0
        candidate = stereo & neighborhood
        coverage = float(candidate.sum()) / max(area, 1)
        if coverage < float(min_coverage):
            continue
        erase |= component
        selected_stereo |= candidate
        accepted += 1
    return erase, selected_stereo, accepted


def rectify_learned_pair(K_left, K_right, left_camera_to_world,
                         right_camera_to_world, size, alpha=0.0):
    """Build pinhole stereo maps from two image-estimated camera poses."""
    import cv2

    width, height = (int(value) for value in size)
    left_camera_to_world = np.asarray(left_camera_to_world, np.float64)
    right_camera_to_world = np.asarray(right_camera_to_world, np.float64)
    if left_camera_to_world.shape != (4, 4) or \
            right_camera_to_world.shape != (4, 4):
        raise ValueError("camera-to-world transforms must be 4x4")
    right_from_left = np.linalg.inv(right_camera_to_world) \
        @ left_camera_to_world
    rotation = right_from_left[:3, :3]
    translation = right_from_left[:3, 3].reshape(3, 1)
    baseline = float(np.linalg.norm(translation))
    if baseline <= 1e-8:
        raise ValueError("learned stereo cameras have zero baseline")
    distortion = np.zeros(5, np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        np.asarray(K_left, np.float64), distortion,
        np.asarray(K_right, np.float64), distortion,
        (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(alpha),
        newImageSize=(width, height))
    map_left = cv2.initUndistortRectifyMap(
        np.asarray(K_left, np.float64), distortion, R1, P1,
        (width, height), cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(
        np.asarray(K_right, np.float64), distortion, R2, P2,
        (width, height), cv2.CV_32FC1)
    return LearnedStereoRectification(
        map_left=map_left, map_right=map_right,
        P1=P1, P2=P2, Q=Q, R1=R1, R2=R2,
        left_camera_to_world=left_camera_to_world,
        baseline=baseline)


def rectified_world_points(disparity, rectification, mask=None,
                           min_disparity=0.5):
    """Turn left-view disparity into dense world points and validity."""
    import cv2

    disparity = np.asarray(disparity, np.float32)
    points_rectified = cv2.reprojectImageTo3D(
        disparity, np.asarray(rectification.Q, np.float32))
    points_left = points_rectified @ np.asarray(
        rectification.R1, np.float32)
    rotation = np.asarray(
        rectification.left_camera_to_world[:3, :3], np.float32)
    translation = np.asarray(
        rectification.left_camera_to_world[:3, 3], np.float32)
    points_world = points_left @ rotation.T + translation
    valid = (
        (disparity > float(min_disparity))
        & np.isfinite(points_world).all(axis=2)
        & (points_rectified[..., 2] > 0.0)
    )
    if mask is not None:
        valid &= np.asarray(mask, bool)
    return points_world.astype(np.float32), valid


def world_to_equirectangular(points, center, rig_camera_to_world,
                             height, width):
    """Project world points to ERP coordinates and radial range."""
    points = np.asarray(points, np.float32)
    center = np.asarray(center, np.float32)
    rig_camera_to_world = np.asarray(rig_camera_to_world, np.float32)
    direction_world = points - center
    radius = np.linalg.norm(direction_world, axis=-1)
    direction_rig = direction_world @ rig_camera_to_world
    direction_rig = np.divide(
        direction_rig, radius[..., None],
        out=np.zeros_like(direction_rig), where=radius[..., None] > 1e-8)
    longitude = np.arctan2(direction_rig[..., 0], direction_rig[..., 2])
    latitude = np.arcsin(np.clip(-direction_rig[..., 1], -1.0, 1.0))
    x = (longitude + np.pi) / (2.0 * np.pi) * int(width)
    y = (np.pi / 2.0 - latitude) / np.pi * int(height)
    return x.astype(np.float32), y.astype(np.float32), radius.astype(np.float32)


def splat_world_points(layers, center, rig_camera_to_world, height, width,
                       splat_radius=1, depth_tie=1e-4,
                       preferred_source=None):
    """Z-buffer current foreground points into one single-source ERP layer.

    ``layers`` contains ``(source_index, world_points, rgb, valid)`` tuples.
    RGB is copied from exactly one source after the nearest-depth test.
    """
    height, width = int(height), int(width)
    pixel_ids, ranges, colours, sources = [], [], [], []
    for source_index, points, rgb, valid in layers:
        points = np.asarray(points, np.float32)
        rgb = np.asarray(rgb)
        valid = np.asarray(valid, bool)
        if points.shape != (*valid.shape, 3) or rgb.shape != (*valid.shape, 3):
            raise ValueError("point, RGB and validity shapes differ")
        x, y, radius = world_to_equirectangular(
            points, center, rig_camera_to_world, height, width)
        keep = valid & np.isfinite(radius) & (radius > 1e-8) \
            & (y >= 0.0) & (y < height)
        if not keep.any():
            continue
        base_x = np.floor(x[keep]).astype(np.int32) % width
        base_y = np.floor(y[keep]).astype(np.int32)
        source_range = radius[keep]
        source_rgb = rgb[keep]
        for dy in range(-int(splat_radius), int(splat_radius) + 1):
            target_y = base_y + dy
            vertical = (target_y >= 0) & (target_y < height)
            if not vertical.any():
                continue
            for dx in range(-int(splat_radius), int(splat_radius) + 1):
                target_x = (base_x + dx) % width
                pixel_ids.append(
                    target_y[vertical] * width + target_x[vertical])
                ranges.append(source_range[vertical])
                colours.append(source_rgb[vertical])
                sources.append(np.full(
                    int(vertical.sum()), int(source_index), np.int16))

    output_rgb = np.zeros((height * width, 3), np.float32)
    output_range = np.full(height * width, np.inf, np.float32)
    output_source = np.full(height * width, -1, np.int16)
    if not pixel_ids:
        return EquirectangularLayer(
            output_rgb.reshape(height, width, 3),
            np.full((height, width), np.nan, np.float32),
            np.zeros((height, width), bool),
            output_source.reshape(height, width))

    pixel_ids = np.concatenate(pixel_ids)
    ranges = np.concatenate(ranges)
    colours = np.concatenate(colours).astype(np.float32)
    if colours.size and float(np.nanmax(colours)) > 1.5:
        colours /= 255.0
    sources = np.concatenate(sources)
    np.minimum.at(output_range, pixel_ids, ranges)
    eligible = ranges <= output_range[pixel_ids] + float(depth_tie)
    indices = np.flatnonzero(eligible)
    if preferred_source is None:
        preference_penalty = np.zeros(indices.size, np.int8)
    else:
        preferred_source = np.asarray(preferred_source)
        if preferred_source.shape != (height, width):
            raise ValueError("preferred source map has the wrong shape")
        desired = preferred_source.reshape(-1)[pixel_ids[indices]]
        preference_penalty = (sources[indices] != desired).astype(np.int8)
    order = np.lexsort((
        sources[indices], ranges[indices], preference_penalty,
        pixel_ids[indices]))
    indices = indices[order]
    ordered_pixels = pixel_ids[indices]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
    winners = indices[first]
    pixels = pixel_ids[winners]
    output_rgb[pixels] = colours[winners]
    output_range[pixels] = ranges[winners]
    output_source[pixels] = sources[winners]
    output_valid = np.isfinite(output_range)
    return EquirectangularLayer(
        output_rgb.reshape(height, width, 3),
        np.where(output_valid, output_range, np.nan).reshape(height, width),
        output_valid.reshape(height, width),
        output_source.reshape(height, width))
