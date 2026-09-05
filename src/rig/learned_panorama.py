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


def equirectangular_cube_lookup(height, width):
    return classify_cube_rays(equirectangular_rays(height, width))
