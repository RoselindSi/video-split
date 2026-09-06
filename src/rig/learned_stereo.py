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
                       splat_radius=1, depth_tie=1e-4):
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
    order = np.lexsort((sources[indices], ranges[indices], pixel_ids[indices]))
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
