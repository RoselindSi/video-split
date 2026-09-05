"""Project shared-centre neural renders into one spherical panorama."""
from __future__ import annotations

import numpy as np


def spherical_rays(width, height, hfov_deg, vfov_deg):
    """Return camera-space rays for a cropped equirectangular projection."""
    if width < 2 or height < 2:
        raise ValueError("panorama dimensions must be at least two pixels")
    if not (1.0 < hfov_deg < 179.0 and 1.0 < vfov_deg < 179.0):
        raise ValueError("panorama field of view must be between 1 and 179 degrees")
    yaw = np.linspace(-hfov_deg / 2.0, hfov_deg / 2.0, width)
    pitch = np.linspace(vfov_deg / 2.0, -vfov_deg / 2.0, height)
    yaw, pitch = np.meshgrid(np.radians(yaw), np.radians(pitch))
    cos_pitch = np.cos(pitch)
    return np.stack((
        np.sin(yaw) * cos_pitch,
        -np.sin(pitch),
        np.cos(yaw) * cos_pitch,
    ), axis=-1).astype(np.float32)


def camera_projection_map(rays, reference_c2w, camera_c2w, intrinsic,
                          image_shape):
    """Map reference-camera rays to normalized coordinates in one view."""
    source_h, source_w = (int(value) for value in image_shape)
    reference_r = np.asarray(reference_c2w, np.float32)[:3, :3]
    camera_r = np.asarray(camera_c2w, np.float32)[:3, :3]
    intrinsic = np.asarray(intrinsic, np.float32)
    world_rays = np.asarray(rays, np.float32) @ reference_r.T
    camera_rays = world_rays @ camera_r
    z = camera_rays[..., 2]
    safe_z = np.where(z > 1e-6, z, 1.0)
    x = intrinsic[0, 0] * camera_rays[..., 0] / safe_z + intrinsic[0, 2]
    y = intrinsic[1, 1] * camera_rays[..., 1] / safe_z + intrinsic[1, 2]
    grid = np.stack((
        2.0 * x / (source_w - 1.0) - 1.0,
        2.0 * y / (source_h - 1.0) - 1.0,
    ), axis=-1).astype(np.float32)
    grid[z <= 1e-6] = 2.0
    return grid, z.astype(np.float32)


def compose_panorama(colors, alphas, camera_c2w, intrinsics, reference_c2w,
                     output_shape=(336, 896), hfov_deg=135.0,
                     vfov_deg=60.0, feather=0.15):
    """Blend camera-major neural renders that share the same optical centre."""
    import torch
    import torch.nn.functional as F

    if colors.ndim != 5 or alphas.ndim != 4:
        raise ValueError("expected colors [camera,time,3,h,w] and alpha [camera,time,h,w]")
    cameras, times, _, source_h, source_w = colors.shape
    if alphas.shape[:2] != (cameras, times) or len(camera_c2w) != cameras:
        raise ValueError("camera/time dimensions do not agree")
    output_h, output_w = (int(value) for value in output_shape)
    rays = spherical_rays(output_w, output_h, hfov_deg, vfov_deg)
    sampled_colors = []
    sampled_alphas = []
    blend_weights = []
    for camera in range(cameras):
        grid, forward = camera_projection_map(
            rays, reference_c2w, camera_c2w[camera], intrinsics[camera],
            (source_h, source_w))
        grid = torch.from_numpy(grid).to(colors.device)[None].expand(times, -1, -1, -1)
        sampled_color = F.grid_sample(
            colors[camera], grid, mode="bilinear", padding_mode="zeros",
            align_corners=True)
        sampled_alpha = F.grid_sample(
            alphas[camera, :, None], grid, mode="bilinear",
            padding_mode="zeros", align_corners=True)
        edge = (1.0 - grid.abs().amax(dim=-1)).clamp(min=0.0)
        edge = (edge / max(float(feather), 1e-6)).clamp(max=1.0)[:, None]
        forward = torch.from_numpy(np.clip(forward, 0.0, 1.0)).to(colors.device)
        direction = forward.pow(4)[None, None]
        sampled_colors.append(sampled_color)
        sampled_alphas.append(sampled_alpha)
        blend_weights.append(sampled_alpha * edge * direction)

    sampled_colors = torch.stack(sampled_colors, dim=1)
    sampled_alphas = torch.stack(sampled_alphas, dim=1)
    blend_weights = torch.stack(blend_weights, dim=1)
    weight_sum = blend_weights.sum(dim=1).clamp(min=1e-6)
    panorama = (sampled_colors * blend_weights).sum(dim=1) / weight_sum
    coverage = sampled_alphas.amax(dim=1)[:, 0]
    overlap_count = (sampled_alphas[:, :, 0] >= 0.5).sum(dim=1)
    overlap_mask = overlap_count >= 2
    color_delta = (sampled_colors - panorama[:, None]).abs().mean(dim=2)
    valid = sampled_alphas[:, :, 0] >= 0.5
    disagreement = (color_delta * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    disagreement = torch.where(overlap_mask, disagreement, torch.zeros_like(disagreement))
    return panorama, coverage, overlap_count, disagreement
