"""Render a Self-Cali-GS model from one virtual center into an ERP image.

Run this file with both the video-split repository and the patched upstream
Self-Cali-GS checkout on PYTHONPATH. No hardware calibration file is read.
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.cameras import Camera
from src.rig.learned_panorama import (
    FACE_NAMES,
    cube_face_rotations,
    equirectangular_face_grids,
)
from src.rig.dynamic_panorama import (
    DynamicOwnershipConfig,
    DynamicSingleSourceCompositor,
    compose_gated_boundary_blend,
    compose_single_source,
    geometric_owner,
    regularize_semantic_owner,
)
from src.rig.learned_stereo import (
    camera_matrix_from_fov,
    combine_instance_masks,
    estimate_depth_scale,
    pose_guided_foreground_mask,
    rectified_camera_points,
    rectified_world_points,
    rectify_learned_pair,
    regularize_component_disparity,
    scale_camera_matrix,
    select_stereo_replacements,
    splat_world_points,
    temporal_change_mask,
)


CAMERA_NAME = re.compile(r"^cam([0-5])_t(\d+)$")


def cameras_at_time(scene, time_index):
    cameras = {}
    for camera in scene.getTrainCameras() + scene.getTestCameras():
        match = CAMERA_NAME.fullmatch(camera.image_name)
        if match and int(match.group(2)) == time_index:
            cameras[int(match.group(1))] = camera
    if sorted(cameras) != list(range(6)):
        raise RuntimeError(
            f"time {time_index} does not contain all six cameras: "
            f"{sorted(cameras)}")
    return [cameras[index] for index in range(6)]


def temporal_foreground_masks(scene, cameras, time_index, threshold,
                              max_references=3):
    """Find motion against nearby learned-input frames of each camera."""
    import cv2

    candidates = {index: [] for index in range(6)}
    for camera in scene.getTrainCameras() + scene.getTestCameras():
        match = CAMERA_NAME.fullmatch(camera.image_name)
        if not match:
            continue
        index, reference_time = (int(value) for value in match.groups())
        if reference_time != time_index:
            candidates[index].append((abs(reference_time - time_index), camera))
    masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for index, current_camera in enumerate(cameras):
        current = current_camera.original_image.permute(
            1, 2, 0).cpu().numpy()
        nearby = sorted(candidates[index], key=lambda item: item[0])[
            :int(max_references)]
        references = [
            camera.original_image.permute(1, 2, 0).cpu().numpy()
            for _, camera in nearby
        ]
        changed = temporal_change_mask(
            current, references, threshold=float(threshold))
        changed = cv2.morphologyEx(
            changed.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        masks[index] = changed
    return masks


def pose_foreground_masks(cameras, model_path, residual, confidence=0.15):
    """Detect complete people and refine their boxes into source-view masks."""
    from ultralytics import YOLO

    images = []
    for camera in cameras:
        image = camera.original_image.permute(1, 2, 0).cpu().numpy()
        images.append(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    model = YOLO(model_path)
    results = model.predict(
        [image[..., ::-1] for image in images], conf=float(confidence),
        verbose=False)
    masks = {}
    stats = []
    for index, (image, result) in enumerate(zip(images, results)):
        boxes = (result.boxes.xyxy.cpu().numpy()
                 if result.boxes is not None else np.zeros((0, 4)))
        keypoints = (result.keypoints.data.cpu().numpy()
                     if result.keypoints is not None
                     else np.zeros((len(boxes), 17, 3)))
        masks[index] = pose_guided_foreground_mask(
            image, boxes, keypoints, residual=residual[index])
        stats.append({
            "camera": cameras[index].image_name,
            "people": len(boxes),
            "mask_fraction": float(masks[index].mean()),
        })
    return masks, stats


def segmentation_foreground_masks(cameras, model_path, confidence=0.15):
    """Run instance segmentation and retain source-resolution person masks."""
    from ultralytics import YOLO
    from ultralytics.utils.ops import scale_masks

    images = []
    for camera in cameras:
        image = camera.original_image.permute(1, 2, 0).cpu().numpy()
        images.append(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    model = YOLO(model_path)
    results = model.predict(
        [image[..., ::-1] for image in images], conf=float(confidence),
        verbose=False)
    masks = {}
    stats = []
    for index, (image, result) in enumerate(zip(images, results)):
        classes = (result.boxes.cls.detach().cpu().numpy()
                   if result.boxes is not None else np.zeros(0))
        if result.masks is None or not len(classes):
            instance_masks = np.zeros((0, *image.shape[:2]), np.float32)
        else:
            scaled = scale_masks(
                result.masks.data[:, None], image.shape[:2],
                padding=True, mode="bilinear")[:, 0]
            instance_masks = scaled.detach().cpu().numpy()
        masks[index] = combine_instance_masks(
            instance_masks, classes, image.shape[:2], class_ids=(0,))
        stats.append({
            "camera": cameras[index].image_name,
            "instances": int(len(classes)),
            "people": int(np.count_nonzero(classes.astype(np.int64) == 0)),
            "mask_fraction": float(masks[index].mean()),
        })
    return masks, stats


def warp_native_masks(world_points, world_valid, cameras, alignment,
                      native_masks):
    """Inverse-project source masks only to locate the old wrong-depth layer."""
    points = world_points.permute(1, 2, 0)
    height, width = points.shape[:2]
    homogeneous = torch.cat(
        [points, torch.ones((height, width, 1), device=points.device)],
        dim=-1)
    warped = {}
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            world_view = camera.get_world_view_transform(
                alignment[0], alignment[1])
            xyz = (homogeneous @ world_view)[..., :3]
            z = xyz[..., 2]
            source_height, source_width = native_masks[index].shape
            focal_x = source_width / (
                2.0 * torch.tan(camera.learnable_fovx / 2.0))
            focal_y = source_height / (
                2.0 * torch.tan(camera.learnable_fovy / 2.0))
            u = focal_x * xyz[..., 0] / z.clamp_min(1e-6) \
                + source_width / 2.0
            v = focal_y * xyz[..., 1] / z.clamp_min(1e-6) \
                + source_height / 2.0
            grid = torch.stack([
                2.0 * u / max(source_width - 1, 1) - 1.0,
                2.0 * v / max(source_height - 1, 1) - 1.0,
            ], dim=-1)
            mask = torch.from_numpy(
                native_masks[index].astype(np.float32)).to(points.device)
            sampled = functional.grid_sample(
                mask[None, None], grid[None], mode="nearest",
                padding_mode="zeros", align_corners=True)[0, 0]
            inside = (
                world_valid & (z > 1e-5)
                & (u >= 0.0) & (u <= source_width - 1)
                & (v >= 0.0) & (v <= source_height - 1)
            )
            warped[index] = (sampled > 0.5).cpu().numpy() & \
                inside.cpu().numpy()
    return warped


def virtual_camera(
        reference, camera_to_world, center, face_size, face_fov_degrees):
    face_fov = math.radians(face_fov_degrees)
    focal = face_size / (2.0 * math.tan(face_fov / 2.0))
    intrinsic = np.asarray([
        [focal, 0.0, face_size / 2.0],
        [0.0, focal, face_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    world_to_camera_rotation = camera_to_world.T
    translation = -world_to_camera_rotation @ center
    return Camera(
        colmap_id=reference.colmap_id,
        R=camera_to_world,
        T=translation,
        intrinsic_matrix=intrinsic,
        FoVx=face_fov,
        FoVy=face_fov,
        focal_length_x=focal,
        focal_length_y=focal,
        image=reference.original_image_pil,
        gt_alpha_mask=None,
        fish_gt_image=reference.fish_gt_image_pil,
        image_name="virtual_panorama",
        uid=-1,
        data_device="cuda",
        ori_path=reference.ori_path,
        orig_fov_w=face_size,
        orig_fov_h=face_size,
        original_image_resolution=(3, face_size, face_size),
        fish_gt_image_resolution=(3, face_size, face_size),
    )


def compose_equirectangular(
        faces, face_alpha, height, width, face_fov_degrees):
    grids, support, edge_weights = equirectangular_face_grids(
        height, width, face_fov_degrees)
    device = faces[0].device
    channels = faces[0].shape[0]
    accumulated = torch.zeros((channels, height, width), device=device)
    total_weight = torch.zeros((1, height, width), device=device)
    panorama_alpha = torch.zeros((1, height, width), device=device)
    for face_index, (face, alpha) in enumerate(zip(faces, face_alpha)):
        grid = torch.from_numpy(grids[face_index]).to(
            device=device, dtype=torch.float32)
        sampled = functional.grid_sample(
            face.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
            padding_mode="zeros", align_corners=True)[0]
        sampled_alpha = functional.grid_sample(
            alpha.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
            padding_mode="zeros", align_corners=True)[0]
        geometry_weight = torch.from_numpy(edge_weights[face_index]).to(
            device=device, dtype=torch.float32)[None]
        face_support = torch.from_numpy(support[face_index]).to(
            device=device)[None]
        weight = geometry_weight * sampled_alpha * face_support
        accumulated += sampled * weight
        total_weight += weight
        panorama_alpha = torch.maximum(
            panorama_alpha, sampled_alpha * face_support)
    panorama = accumulated / total_weight.clamp_min(1e-6)
    panorama[:, total_weight[0] <= 1e-6] = 0.0
    return panorama, panorama_alpha


def surface_world_points(view, depth, alpha, alignment):
    """Convert alpha-weighted raster depth into one world point per pixel."""
    device = depth.device
    height, width = depth.shape[-2:]
    surface_depth = depth[:1] / alpha[:1].clamp_min(1e-6)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij")
    focal_x = width / (2.0 * torch.tan(view.learnable_fovx / 2.0))
    focal_y = height / (2.0 * torch.tan(view.learnable_fovy / 2.0))
    camera_points = torch.stack([
        (x - width / 2.0) * surface_depth[0] / focal_x,
        (y - height / 2.0) * surface_depth[0] / focal_y,
        surface_depth[0],
        torch.ones_like(surface_depth[0]),
    ], dim=-1)
    world_view = view.get_world_view_transform(alignment[0], alignment[1])
    camera_to_world = torch.linalg.inv(world_view.transpose(0, 1))
    world = camera_points @ camera_to_world.transpose(0, 1)
    return world[..., :3].permute(2, 0, 1)


def actual_camera_to_world(camera, alignment):
    world_view = camera.get_world_view_transform(
        alignment[0], alignment[1])
    return torch.linalg.inv(world_view.transpose(0, 1))


def _normalized_surface_depth(result):
    alpha = result["weights"][:1].clamp(0.0, 1.0)
    return result["depth"][:1] / alpha.clamp_min(1e-6), alpha


def warp_current_sources(world_points, world_valid, cameras, gaussians,
                         pipeline, background, shift, iteration, alignment,
                         foreground_threshold):
    """Project learned world surfaces into synchronized current RGB views."""
    points = world_points.permute(1, 2, 0)
    height, width = points.shape[:2]
    homogeneous = torch.cat(
        [points, torch.ones((height, width, 1), device=points.device)],
        dim=-1)
    warped, valid, cost, foreground = {}, {}, {}, {}
    native_foreground, native_depth, native_alpha = {}, {}, {}
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            world_view = camera.get_world_view_transform(
                alignment[0], alignment[1])
            camera_points = homogeneous @ world_view
            xyz = camera_points[..., :3]
            z = xyz[..., 2]
            source = camera.original_image.to(points.device)
            source_height, source_width = source.shape[-2:]
            focal_x = source_width / (
                2.0 * torch.tan(camera.learnable_fovx / 2.0))
            focal_y = source_height / (
                2.0 * torch.tan(camera.learnable_fovy / 2.0))
            u = focal_x * xyz[..., 0] / z.clamp_min(1e-6) \
                + source_width / 2.0
            v = focal_y * xyz[..., 1] / z.clamp_min(1e-6) \
                + source_height / 2.0
            grid = torch.stack([
                2.0 * u / max(source_width - 1, 1) - 1.0,
                2.0 * v / max(source_height - 1, 1) - 1.0,
            ], dim=-1)
            image = functional.grid_sample(
                source.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
                padding_mode="zeros", align_corners=True)[0]

            source_result = render(
                camera, gaussians, pipeline, background, 0, shift,
                iteration=iteration, hybrid=False,
                global_alignment=alignment)
            source_depth, source_alpha = _normalized_surface_depth(
                source_result)
            source_difference = torch.abs(
                source - source_result["render"].clamp(0.0, 1.0)
            ).mean(dim=0, keepdim=True)
            sampled_difference = functional.grid_sample(
                source_difference.unsqueeze(0), grid.unsqueeze(0),
                mode="bilinear", padding_mode="zeros",
                align_corners=True)[0, 0]
            sampled_depth = functional.grid_sample(
                source_depth.unsqueeze(0), grid.unsqueeze(0),
                mode="bilinear", padding_mode="zeros",
                align_corners=True)[0, 0]
            sampled_alpha = functional.grid_sample(
                source_alpha.unsqueeze(0), grid.unsqueeze(0),
                mode="bilinear", padding_mode="zeros",
                align_corners=True)[0, 0]
            behind_static_surface = (
                (sampled_alpha > 0.20)
                & (z > sampled_depth * 1.15 + 0.05)
            )
            inside = (
                world_valid & (z > 1e-5)
                & (u >= 0.0) & (u <= source_width - 1)
                & (v >= 0.0) & (v <= source_height - 1)
            )
            ray_length = torch.linalg.norm(xyz, dim=-1).clamp_min(1e-6)
            angle = torch.rad2deg(torch.acos(
                (z / ray_length).clamp(-1.0, 1.0)))
            # Raster depth is noisy around thin Gaussian surfaces. Treat a
            # likely occlusion as a strong preference against this camera,
            # not as an invalid pixel that fragments ownership into holes.
            angle = angle + behind_static_surface.float() * 45.0
            warped[index] = image.permute(1, 2, 0).cpu().numpy()
            valid[index] = inside.cpu().numpy()
            cost[index] = angle.cpu().numpy().astype(np.float32)
            foreground[index] = (
                sampled_difference >= foreground_threshold
            ).cpu().numpy().astype(np.float32)
            native_foreground[index] = (
                source_difference[0] >= foreground_threshold
            ).cpu().numpy()
            native_depth[index] = source_depth[0].cpu().numpy()
            native_alpha[index] = source_alpha[0].cpu().numpy()
    return (warped, valid, cost, foreground, native_foreground,
            native_depth, native_alpha)


def stereo_foreground_layer(
        cameras, native_foreground, temporal_foreground,
        semantic_foreground, native_depth, native_alpha, center,
        rig_camera_to_world, preferred_source,
        height, width, model_path, alignment, cuda_lib_dir=None,
        cudnn_lib_dir=None, splat_radius=1, scale_min_samples=2048,
        scale_max_p90=0.35):
    """Depth-place both colour views from each physical stereo pair."""
    import cv2

    from src.rig.fast_foundation_stereo import FastFoundationStereo

    model = FastFoundationStereo(
        model_path=model_path, cuda_lib_dir=cuda_lib_dir,
        cudnn_lib_dir=cudnn_lib_dir, require_cuda=True)
    layers = []
    pair_stats = []
    rectified_size = (model.target_w, model.target_h)
    for left_index, right_index in ((0, 1), (2, 3), (4, 5)):
        left_camera = cameras[left_index]
        right_camera = cameras[right_index]
        left = left_camera.original_image.permute(1, 2, 0).cpu().numpy()
        right = right_camera.original_image.permute(1, 2, 0).cpu().numpy()
        left = np.clip(left * 255.0, 0, 255).astype(np.uint8)
        right = np.clip(right * 255.0, 0, 255).astype(np.uint8)
        left_height, left_width = left.shape[:2]
        right_height, right_width = right.shape[:2]

        left_pose = actual_camera_to_world(
            left_camera, alignment).detach().cpu().numpy()
        right_pose = actual_camera_to_world(
            right_camera, alignment).detach().cpu().numpy()
        K_left = camera_matrix_from_fov(
            left_width, left_height,
            float(left_camera.learnable_fovx.detach().cpu()),
            float(left_camera.learnable_fovy.detach().cpu()))
        K_right = camera_matrix_from_fov(
            right_width, right_height,
            float(right_camera.learnable_fovx.detach().cpu()),
            float(right_camera.learnable_fovy.detach().cpu()))
        K_left = scale_camera_matrix(
            K_left, (left_width, left_height), rectified_size)
        K_right = scale_camera_matrix(
            K_right, (right_width, right_height), rectified_size)
        left = cv2.resize(left, rectified_size, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, rectified_size, interpolation=cv2.INTER_LINEAR)
        qualified_left = (
            (native_foreground[left_index]
             & temporal_foreground[left_index])
            | semantic_foreground[left_index]
        )
        qualified_right = (
            (native_foreground[right_index]
             & temporal_foreground[right_index])
            | semantic_foreground[right_index]
        )
        foreground_left_native = cv2.resize(
            qualified_left.astype(np.uint8), rectified_size,
            interpolation=cv2.INTER_NEAREST)
        foreground_right_native = cv2.resize(
            qualified_right.astype(np.uint8), rectified_size,
            interpolation=cv2.INTER_NEAREST)
        rectification = rectify_learned_pair(
            K_left, K_right, left_pose, right_pose,
            rectified_size)
        left_rectified = cv2.remap(
            left, *rectification.map_left, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        right_rectified = cv2.remap(
            right, *rectification.map_right, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        foreground_rectified = cv2.remap(
            foreground_left_native, *rectification.map_left,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT) > 0
        foreground_right_rectified = cv2.remap(
            foreground_right_native, *rectification.map_right,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT) > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foreground_rectified = cv2.morphologyEx(
            foreground_rectified.astype(np.uint8), cv2.MORPH_CLOSE,
            kernel) > 0
        foreground_right_rectified = cv2.morphologyEx(
            foreground_right_rectified.astype(np.uint8), cv2.MORPH_CLOSE,
            kernel) > 0

        raw_disparity = model.disparity(
            left_rectified[..., ::-1], right_rectified[..., ::-1])
        yy, xx = np.mgrid[
            0:rectified_size[1], 0:rectified_size[0]].astype(np.float32)
        right_x = xx - raw_disparity
        right_foreground_in_left = cv2.remap(
            foreground_right_rectified.astype(np.uint8), right_x, yy,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        combined_foreground = foreground_rectified | right_foreground_in_left
        static_points, static_stereo_valid = rectified_camera_points(
            raw_disparity, rectification)
        learned_depth = cv2.resize(
            native_depth[left_index], rectified_size,
            interpolation=cv2.INTER_LINEAR)
        learned_alpha = cv2.resize(
            native_alpha[left_index], rectified_size,
            interpolation=cv2.INTER_LINEAR)
        learned_depth = cv2.remap(
            learned_depth, *rectification.map_left, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        learned_alpha = cv2.remap(
            learned_alpha, *rectification.map_left, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        gray = cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY)
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        static_support = (
            static_stereo_valid & (learned_alpha > 0.50)
            & ~combined_foreground
        )
        if static_support.any():
            texture_threshold = float(np.median(gradient[static_support]))
            static_support &= gradient >= max(texture_threshold, 1.0)
        scale_estimate = estimate_depth_scale(
            static_points[..., 2], learned_depth, mask=static_support,
            min_samples=scale_min_samples)
        scale_applied = (
            scale_estimate.sample_count >= int(scale_min_samples)
            and scale_estimate.p90_relative_error <= float(scale_max_p90)
        )
        depth_scale = scale_estimate.scale if scale_applied else 1.0
        disparity, combined_foreground, component_count = \
            regularize_component_disparity(
                raw_disparity, combined_foreground, min_component_px=24)
        map_x, map_y = rectification.map_left
        left_support = (
            (map_x >= 0.0) & (map_x < rectified_size[0] - 1)
            & (map_y >= 0.0) & (map_y < rectified_size[1] - 1)
        )
        right_x = xx - disparity
        right_support = (right_x >= 0.0) \
            & (right_x < rectified_size[0] - 1)
        right_foreground_in_left = cv2.remap(
            foreground_right_rectified.astype(np.uint8), right_x, yy,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        right_colour_in_left = cv2.remap(
            right_rectified, right_x, yy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        points, point_valid = rectified_world_points(
            disparity, rectification,
            mask=combined_foreground & left_support,
            depth_scale=depth_scale)
        layers.append((
            left_index, points, left_rectified,
            point_valid & foreground_rectified))
        layers.append((
            right_index, points, right_colour_in_left,
            point_valid & right_foreground_in_left & right_support))
        pair_stats.append({
            "pair": [left_index, right_index],
            "native_sizes": [
                [left_width, left_height], [right_width, right_height]],
            "rectified_size": list(rectified_size),
            "baseline_learned_units": rectification.baseline,
            "depth_scale_fit": {
                "scale": scale_estimate.scale,
                "sample_count": scale_estimate.sample_count,
                "median_relative_error": (
                    scale_estimate.median_relative_error
                    if np.isfinite(scale_estimate.median_relative_error)
                    else None),
                "p90_relative_error": (
                    scale_estimate.p90_relative_error
                    if np.isfinite(scale_estimate.p90_relative_error)
                    else None),
                "applied": scale_applied,
            },
            "static_residual_fraction": float(
                native_foreground[left_index].mean()),
            "right_static_residual_fraction": float(
                native_foreground[right_index].mean()),
            "temporal_fraction": float(
                temporal_foreground[left_index].mean()),
            "right_temporal_fraction": float(
                temporal_foreground[right_index].mean()),
            "foreground_fraction": float(combined_foreground.mean()),
            "foreground_components": component_count,
            "valid_foreground_points": int(point_valid.sum()),
            "left_colour_points": int(
                (point_valid & foreground_rectified).sum()),
            "right_colour_points": int(
                (point_valid & right_foreground_in_left & right_support).sum()),
        })

    layer = splat_world_points(
        layers, center, rig_camera_to_world, height, width,
        splat_radius=splat_radius, preferred_source=preferred_source)
    return layer, {
        "providers": model.providers,
        "pairs": pair_stats,
        "layer_fraction": float(layer.valid.mean()),
        "scale_min_samples": int(scale_min_samples),
        "scale_max_p90_relative_error": float(scale_max_p90),
    }


def owner_diagnostic(owner):
    palette = np.asarray([
        [230, 25, 75], [60, 180, 75], [255, 225, 25],
        [0, 130, 200], [245, 130, 48], [145, 30, 180],
    ], np.uint8)
    output = np.zeros((*owner.shape, 3), np.uint8)
    for index, colour in enumerate(palette):
        output[owner == index] = colour
    return output


def crop_to_content(image, alpha, threshold=0.01, margin=8):
    valid = alpha[0] > threshold
    rows = torch.where(valid.any(dim=1))[0]
    columns = torch.where(valid.any(dim=0))[0]
    if not len(rows) or not len(columns):
        return image, alpha, [0, 0, image.shape[2], image.shape[1]]
    top = max(0, int(rows[0]) - margin)
    bottom = min(image.shape[1], int(rows[-1]) + margin + 1)
    left = max(0, int(columns[0]) - margin)
    right = min(image.shape[2], int(columns[-1]) + margin + 1)
    return (
        image[:, top:bottom, left:right],
        alpha[:, top:bottom, left:right],
        [left, top, right, bottom],
    )


def main():
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--face-size", type=int, default=768)
    parser.add_argument("--face-fov", type=float, default=100.0)
    parser.add_argument("--erp-width", type=int, default=2048)
    parser.add_argument("--dynamic-threshold", type=float, default=48.0)
    parser.add_argument("--foreground-threshold", type=float, default=72.0)
    parser.add_argument("--dynamic-close", type=int, default=8)
    parser.add_argument("--dynamic-dilate", type=int, default=3)
    parser.add_argument("--stereo-model")
    parser.add_argument("--stereo-cuda-lib-dir")
    parser.add_argument("--stereo-cudnn-lib-dir")
    parser.add_argument("--stereo-splat-radius", type=int, default=1)
    parser.add_argument("--stereo-scale-min-samples", type=int, default=2048)
    parser.add_argument("--stereo-scale-max-p90", type=float, default=0.35)
    parser.add_argument("--temporal-threshold", type=float, default=24.0)
    parser.add_argument("--stereo-replacement-coverage", type=float,
                        default=0.75)
    parser.add_argument("--stereo-association-px", type=int, default=8)
    parser.add_argument("--owner-smooth", type=int, default=8)
    parser.add_argument("--static-transition", type=int, default=12)
    parser.add_argument("--static-blend-temperature", type=float, default=6.0)
    parser.add_argument("--static-blend-gate", type=float, default=40.0)
    parser.add_argument("--pose-model")
    parser.add_argument("--pose-confidence", type=float, default=0.15)
    parser.add_argument("--segment-model")
    parser.add_argument("--segment-confidence", type=float, default=0.15)
    parser.add_argument("--semantic-owner-close", type=int, default=12)
    parser.add_argument("--semantic-owner-dilate", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.erp_width % 2:
        parser.error("--erp-width must be even")
    if not 90.0 <= args.face_fov < 180.0:
        parser.error("--face-fov must be in [90, 180) degrees")

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    scene = Scene(
        dataset, gaussians, load_iteration=args.iteration, shuffle=False,
        r_t_noise=[0.0, 0.0, 1.0])
    source_cameras = cameras_at_time(scene, args.time)
    camera_to_world = [
        camera.get_c2w()[:3, :3].detach().cpu().numpy()
        for camera in source_cameras
    ]
    centers = np.asarray([
        camera.get_c2w()[:3, 3].detach().cpu().numpy()
        for camera in source_cameras
    ])
    rig_center = centers.mean(axis=0)
    rig_orientation = camera_to_world[0]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    shift = torch.zeros(3, dtype=torch.float32, device="cuda")
    alignment = scene.getGlobalAlignment()
    face_images = []
    face_alpha = []
    face_world = []
    panorama_camera_to_world = None
    with torch.no_grad():
        for name, relative_rotation in zip(
                FACE_NAMES, cube_face_rotations()):
            view = virtual_camera(
                source_cameras[0], rig_orientation @ relative_rotation,
                rig_center, args.face_size, args.face_fov)
            result = render(
                view, gaussians, pipeline, background, 0, shift,
                iteration=scene.loaded_iter, hybrid=False,
                global_alignment=alignment)
            image = result["render"].clamp(0.0, 1.0)
            alpha = result["weights"].clamp(0.0, 1.0)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            face_images.append(image)
            face_alpha.append(alpha[:1])
            face_world.append(surface_world_points(
                view, result["depth"], alpha[:1], alignment))
            if name == "front":
                panorama_camera_to_world = actual_camera_to_world(
                    view, alignment).detach().cpu().numpy()
            torchvision.utils.save_image(image, output / f"face_{name}.png")
            torchvision.utils.save_image(alpha[:1], output / f"alpha_{name}.png")

    erp_height = args.erp_width // 2
    panorama, alpha = compose_equirectangular(
        face_images, face_alpha, erp_height, args.erp_width, args.face_fov)
    world_points, _ = compose_equirectangular(
        face_world, face_alpha, erp_height, args.erp_width, args.face_fov)
    world_valid = alpha[0] > 0.05
    (warped, valid, cost, foreground, native_foreground,
     native_depth, native_alpha) = warp_current_sources(
         world_points, world_valid, source_cameras, gaussians, pipeline,
         background, shift, scene.loaded_iter, alignment,
         args.foreground_threshold / 255.0)
    learned_background = panorama.permute(1, 2, 0).cpu().numpy()
    learned_valid = world_valid.cpu().numpy()
    base_owner = geometric_owner(valid, cost)
    base_image, _, _ = compose_single_source(
        warped, valid, base_owner, learned_background, learned_valid)
    compositor = DynamicSingleSourceCompositor(DynamicOwnershipConfig(
        disagreement=args.dynamic_threshold / 255.0,
        foreground_disagreement=0.5,
        close_px=args.dynamic_close,
        dilate_px=args.dynamic_dilate,
        owner_smooth_px=args.owner_smooth,
    ))
    dynamic_image, owner, dynamic_mask, dynamic_stats = compositor.render(
        warped, valid, cost, learned_background, learned_valid,
        foreground=foreground)
    semantic_stats = None
    semantic_foreground = {
        index: np.zeros(mask.shape, bool)
        for index, mask in native_foreground.items()
    }
    semantic_old = np.zeros(dynamic_mask.shape, bool)
    semantic_owner_stats = None
    semantic_method = None
    if args.segment_model:
        semantic_method = "instance-segmentation"
        semantic_foreground, semantic_stats = segmentation_foreground_masks(
            source_cameras, args.segment_model,
            confidence=args.segment_confidence)
    elif args.pose_model:
        semantic_method = "pose-grabcut"
        semantic_foreground, semantic_stats = pose_foreground_masks(
            source_cameras, args.pose_model, native_foreground,
            confidence=args.pose_confidence)
    if semantic_method:
        warped_semantic = warp_native_masks(
            world_points, world_valid, source_cameras, alignment,
            semantic_foreground)
        owner, semantic_old, semantic_owner_stats = \
            regularize_semantic_owner(
                owner, warped_semantic, valid, cost,
                close_px=args.semantic_owner_close,
                dilate_px=args.semantic_owner_dilate)
        dynamic_image, current, fallback = compose_single_source(
            warped, valid, owner, learned_background, learned_valid)
        dynamic_stats["current_fraction"] = float(current.mean())
        dynamic_stats["background_fallback_fraction"] = float(
            fallback.mean())
    protected_mask = dynamic_mask | semantic_old
    seam_image, transition_mask, gated_transition = \
        compose_gated_boundary_blend(
            warped, valid, cost, owner, learned_background, learned_valid,
            protected=protected_mask,
            temperature=args.static_blend_temperature,
            gate=args.static_blend_gate / 255.0,
            boundary_px=args.static_transition)
    stereo_stats = None
    depth_dynamic_image = seam_image
    stereo_mask = np.zeros(dynamic_mask.shape, bool)
    if args.stereo_model:
        if panorama_camera_to_world is None:
            raise RuntimeError("front panorama camera pose was not created")
        actual_center = panorama_camera_to_world[:3, 3]
        temporal_foreground = temporal_foreground_masks(
            scene, source_cameras, args.time,
            args.temporal_threshold / 255.0)
        stereo_layer, stereo_stats = stereo_foreground_layer(
            source_cameras, native_foreground, temporal_foreground,
            semantic_foreground, native_depth, native_alpha, actual_center,
            panorama_camera_to_world[:3, :3], owner,
            erp_height, args.erp_width, args.stereo_model, alignment,
            cuda_lib_dir=args.stereo_cuda_lib_dir,
            cudnn_lib_dir=args.stereo_cudnn_lib_dir,
            splat_radius=args.stereo_splat_radius,
            scale_min_samples=args.stereo_scale_min_samples,
            scale_max_p90=args.stereo_scale_max_p90)
        learned_range = np.linalg.norm(
            world_points.permute(1, 2, 0).cpu().numpy()
            - actual_center, axis=2)
        depth_visible = stereo_layer.valid & (
            ~learned_valid
            | (stereo_layer.range_m <= learned_range * 1.25 + 0.10)
        )
        erase, stereo_mask, accepted_components = select_stereo_replacements(
            protected_mask, depth_visible,
            min_coverage=args.stereo_replacement_coverage,
            association_px=args.stereo_association_px)
        depth_dynamic_image = seam_image.copy()
        erase &= learned_valid
        depth_dynamic_image[erase] = learned_background[erase]
        depth_dynamic_image[stereo_mask] = stereo_layer.rgb[stereo_mask]
        stereo_stats["visible_layer_fraction"] = float(stereo_mask.mean())
        stereo_stats["erased_wrong_depth_fraction"] = float(erase.mean())
        stereo_stats["accepted_dynamic_components"] = accepted_components
        stereo_stats["replacement_coverage"] = (
            args.stereo_replacement_coverage)
        stereo_stats["temporal_threshold_rgb_255"] = args.temporal_threshold
    base_tensor = torch.from_numpy(base_image).permute(2, 0, 1)
    dynamic_tensor = torch.from_numpy(dynamic_image).permute(2, 0, 1)
    seam_tensor = torch.from_numpy(seam_image).permute(2, 0, 1)
    depth_dynamic_tensor = torch.from_numpy(
        depth_dynamic_image).permute(2, 0, 1)
    owner_tensor = torch.from_numpy(
        owner_diagnostic(owner)).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(dynamic_mask.astype(np.float32))[None]
    stereo_mask_tensor = torch.from_numpy(
        stereo_mask.astype(np.float32))[None]
    semantic_mask_tensor = torch.from_numpy(
        semantic_old.astype(np.float32))[None]
    cropped, cropped_alpha, crop = crop_to_content(panorama, alpha)
    left, top, right, bottom = crop
    dynamic_cropped = dynamic_tensor[:, top:bottom, left:right]
    seam_cropped = seam_tensor[:, top:bottom, left:right]
    depth_dynamic_cropped = depth_dynamic_tensor[:, top:bottom, left:right]
    torchvision.utils.save_image(panorama, output / "panorama.png")
    torchvision.utils.save_image(alpha, output / "panorama_alpha.png")
    torchvision.utils.save_image(cropped, output / "panorama_cropped.png")
    torchvision.utils.save_image(
        cropped_alpha, output / "panorama_cropped_alpha.png")
    torchvision.utils.save_image(
        base_tensor, output / "panorama_current_hard.png")
    torchvision.utils.save_image(
        dynamic_tensor, output / "panorama_dynamic.png")
    torchvision.utils.save_image(
        dynamic_cropped, output / "panorama_dynamic_cropped.png")
    torchvision.utils.save_image(
        seam_tensor, output / "panorama_static_transition.png")
    torchvision.utils.save_image(
        seam_cropped, output / "panorama_static_transition_cropped.png")
    torchvision.utils.save_image(
        depth_dynamic_tensor, output / "panorama_dynamic_depth.png")
    torchvision.utils.save_image(
        depth_dynamic_cropped,
        output / "panorama_dynamic_depth_cropped.png")
    torchvision.utils.save_image(owner_tensor, output / "dynamic_owner.png")
    torchvision.utils.save_image(mask_tensor, output / "dynamic_mask.png")
    torchvision.utils.save_image(
        stereo_mask_tensor, output / "stereo_foreground_mask.png")
    torchvision.utils.save_image(
        semantic_mask_tensor, output / "semantic_foreground_mask.png")
    if semantic_method:
        for index, mask in semantic_foreground.items():
            source_mask = torch.from_numpy(mask.astype(np.float32))[None]
            torchvision.utils.save_image(
                source_mask, output / f"semantic_source_cam{index}.png")

    report = {
        "schema": "video-split.learned-panorama.v1",
        "uses_hardware_calibration": False,
        "composition": (
            "learned geometry with one physical source per dynamic region"
        ),
        "iteration": int(scene.loaded_iter),
        "time": args.time,
        "source_views": [camera.image_name for camera in source_cameras],
        "virtual_center": rig_center.tolist(),
        "face_size": args.face_size,
        "face_fov_degrees": args.face_fov,
        "erp_size": [args.erp_width, erp_height],
        "content_crop": crop,
        "valid_fraction": float((alpha > 0.01).float().mean().item()),
        "dynamic": {
            "threshold_rgb_255": args.dynamic_threshold,
            "foreground_threshold_rgb_255": args.foreground_threshold,
            "close_px": args.dynamic_close,
            "dilate_px": args.dynamic_dilate,
            **dynamic_stats,
        },
        "static_transition": {
            "owner_smooth_px": args.owner_smooth,
            "boundary_px": args.static_transition,
            "temperature_degrees": args.static_blend_temperature,
            "gate_rgb_255": args.static_blend_gate,
            "blended_fraction": float(transition_mask.mean()),
            "gated_fraction": float(gated_transition.mean()),
        },
        "semantic_foreground": {
            "method": semantic_method,
            "model": args.segment_model or args.pose_model,
            "confidence": (
                args.segment_confidence if args.segment_model
                else args.pose_confidence),
            "panorama_fraction": float(semantic_old.mean()),
            "sources": semantic_stats,
            "ownership": semantic_owner_stats,
        } if semantic_method else None,
        "source_valid_fraction": {
            source_cameras[index].image_name: float(mask.mean())
            for index, mask in valid.items()
        },
        "stereo_foreground": stereo_stats,
    }
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
