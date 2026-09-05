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
    equirectangular_cube_lookup,
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
    face_indices, grid = equirectangular_cube_lookup(
        height, width, face_fov_degrees)
    device = faces[0].device
    grid = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    indices = torch.from_numpy(face_indices).to(device=device)
    panorama = torch.zeros((3, height, width), device=device)
    panorama_alpha = torch.zeros((1, height, width), device=device)
    for face_index, (face, alpha) in enumerate(zip(faces, face_alpha)):
        sampled = functional.grid_sample(
            face.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
            padding_mode="zeros", align_corners=True)[0]
        sampled_alpha = functional.grid_sample(
            alpha.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
            padding_mode="zeros", align_corners=True)[0]
        mask = indices == face_index
        panorama[:, mask] = sampled[:, mask]
        panorama_alpha[:, mask] = sampled_alpha[:, mask]
    return panorama, panorama_alpha


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
            torchvision.utils.save_image(image, output / f"face_{name}.png")
            torchvision.utils.save_image(alpha[:1], output / f"alpha_{name}.png")

    erp_height = args.erp_width // 2
    panorama, alpha = compose_equirectangular(
        face_images, face_alpha, erp_height, args.erp_width, args.face_fov)
    cropped, cropped_alpha, crop = crop_to_content(panorama, alpha)
    torchvision.utils.save_image(panorama, output / "panorama.png")
    torchvision.utils.save_image(alpha, output / "panorama_alpha.png")
    torchvision.utils.save_image(cropped, output / "panorama_cropped.png")
    torchvision.utils.save_image(
        cropped_alpha, output / "panorama_cropped_alpha.png")

    report = {
        "schema": "video-split.learned-panorama.v1",
        "uses_hardware_calibration": False,
        "composition": "single learned 3D Gaussian scene at one virtual center",
        "iteration": int(scene.loaded_iter),
        "time": args.time,
        "source_views": [camera.image_name for camera in source_cameras],
        "virtual_center": rig_center.tolist(),
        "face_size": args.face_size,
        "face_fov_degrees": args.face_fov,
        "erp_size": [args.erp_width, erp_height],
        "content_crop": crop,
        "valid_fraction": float((alpha > 0.01).float().mean().item()),
    }
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
