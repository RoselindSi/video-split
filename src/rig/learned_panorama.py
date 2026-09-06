"""Geometry helpers for rendering a learned scene into a cubemap panorama."""
from __future__ import annotations

import numpy as np


FACE_NAMES = ("front", "right", "back", "left", "up", "down")


def cube_face_rotations():
    """Return camera-to-rig rotations for a y-down, z-forward camera."""
    return np.asarray([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    ], dtype=np.float64)


def equirectangular_rays(height, width):
    """Create unit rays for an ERP image in rig coordinates."""
    x = np.arange(width, dtype=np.float64) + 0.5
    y = np.arange(height, dtype=np.float64) + 0.5
    longitude = x / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - y / height * np.pi
    longitude, latitude = np.meshgrid(longitude, latitude)
    cos_latitude = np.cos(latitude)
    return np.stack([
        cos_latitude * np.sin(longitude),
        -np.sin(latitude),
        cos_latitude * np.cos(longitude),
    ], axis=-1)


def classify_cube_rays(rays):
    """Map rig-space rays to a cube face and normalized face coordinates."""
    rays = np.asarray(rays, dtype=np.float64)
    if rays.shape[-1] != 3:
        raise ValueError("rays must have a final dimension of three")

    dominant_axis = np.argmax(np.abs(rays), axis=-1)
    faces = np.empty(dominant_axis.shape, dtype=np.int64)
    faces[(dominant_axis == 2) & (rays[..., 2] >= 0)] = 0
    faces[(dominant_axis == 0) & (rays[..., 0] >= 0)] = 1
    faces[(dominant_axis == 2) & (rays[..., 2] < 0)] = 2
    faces[(dominant_axis == 0) & (rays[..., 0] < 0)] = 3
    faces[(dominant_axis == 1) & (rays[..., 1] < 0)] = 4
    faces[(dominant_axis == 1) & (rays[..., 1] >= 0)] = 5

    rotations = cube_face_rotations()
    local = np.empty_like(rays)
    for face_index, rotation in enumerate(rotations):
        mask = faces == face_index
        local[mask] = rays[mask] @ rotation
    grid = local[..., :2] / local[..., 2:3]
    return faces, grid


def equirectangular_cube_lookup(height, width, face_fov_degrees=90.0):
    if not 90.0 <= face_fov_degrees < 180.0:
        raise ValueError("cube face FOV must be in [90, 180) degrees")
    faces, grid = classify_cube_rays(equirectangular_rays(height, width))
    overscan = np.tan(np.radians(face_fov_degrees) / 2.0)
    return faces, grid / overscan


def equirectangular_face_grids(height, width, face_fov_degrees=100.0,
                               feather_fraction=0.20):
    """Sampling grids and edge weights for every overlapping cube face.

    A 100-degree face overlaps its neighbour by about ten degrees.  Returning
    all six grids lets the learned static scene use that overlap instead of
    introducing a hard 45-degree cube boundary.  Physical current-frame RGB
    ownership remains separate and single-source.
    """
    if not 90.0 < face_fov_degrees < 180.0:
        raise ValueError("overlapping cube face FOV must be in (90, 180)")
    if not 0.0 < feather_fraction <= 1.0:
        raise ValueError("feather fraction must be in (0, 1]")
    rays = equirectangular_rays(height, width)
    rotations = cube_face_rotations()
    overscan = np.tan(np.radians(face_fov_degrees) / 2.0)
    grids, support, weights = [], [], []
    for rotation in rotations:
        local = rays @ rotation
        forward = local[..., 2]
        safe_forward = np.where(forward > 1e-9, forward, 1.0)
        grid = local[..., :2] / safe_forward[..., None] / overscan
        extent = np.max(np.abs(grid), axis=-1)
        valid = (forward > 1e-9) & (extent <= 1.0)
        edge_distance = np.clip((1.0 - extent) / feather_fraction,
                                0.0, 1.0)
        smooth = edge_distance * edge_distance * (3.0 - 2.0 * edge_distance)
        grids.append(grid.astype(np.float32))
        support.append(valid)
        weights.append(np.where(valid, smooth, 0.0).astype(np.float32))
    return np.stack(grids), np.stack(support), np.stack(weights)
