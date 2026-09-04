"""Run an installed NoPo4D checkout and render one learned central view.

This file is executed directly, outside the video-split ``src`` package.  That
matters because upstream NoPo4D also calls its top-level package ``src``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


IMAGE_NAME = re.compile(r"cam(?P<camera>\d+)_t(?P<time>\d+)\.png")


def image_grid(image_dir, num_cameras):
    root = Path(image_dir)
    grid = {}
    for path in root.glob("*.png"):
        match = IMAGE_NAME.fullmatch(path.name)
        if match:
            grid[(int(match.group("camera")), int(match.group("time")))] = path
    if not grid:
        raise ValueError(f"no camN_tN.png inputs in {root}")
    times = sorted({time for _, time in grid})
    expected = [(camera, time) for camera in range(num_cameras)
                for time in range(len(times))]
    if times != list(range(len(times))) or list(sorted(grid)) != expected:
        raise ValueError(
            f"expected a complete {num_cameras}-camera grid, found "
            f"{sorted(grid)}")
    return [grid[key] for key in expected], len(times)


def wide_intrinsics(width, height, hfov_deg, vfov_deg):
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def central_camera(c2w, central_index):
    poses = np.asarray(c2w, dtype=np.float32)
    target = poses[central_index].copy()
    target[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    return target


def rotation_angle_deg(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def geometry_report(c2w):
    poses = np.asarray(c2w)
    return {
        "learned_camera_centres": poses[:, :3, 3].tolist(),
        "within_stereo_axis_deg": [
            rotation_angle_deg(poses[a], poses[b])
            for a, b in ((0, 1), (2, 3), (4, 5))
        ] if len(poses) == 6 else None,
        "neighbour_axis_deg": [
            rotation_angle_deg(poses[a], poses[b])
            for a, b in ((0, 2), (2, 4))
        ] if len(poses) == 6 else None,
    }


def save_tensor_frames(frames, root, prefix):
    from torchvision.transforms.functional import to_pil_image

    root.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        to_pil_image(frame.detach().float().cpu().clamp(0, 1)).save(
            root / f"{prefix}_{index:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nopo_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--da3_model", required=True)
    parser.add_argument("--num_cameras", type=int, default=6)
    parser.add_argument("--central_camera", type=int, default=2)
    parser.add_argument("--hfov", type=float, default=150.0)
    parser.add_argument("--vfov", type=float, default=90.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--render_inputs", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NOPO4D_DA3_MODEL"] = os.fspath(Path(args.da3_model).resolve())
    nopo_root = Path(args.nopo_root).resolve()
    sys.path.insert(0, os.fspath(nopo_root))

    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from src.model.nopo4d import NoPo4D

    paths, num_frames = image_grid(args.image_dir, args.num_cameras)
    images = torch.stack([
        pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0
        for path in paths
    ]).unsqueeze(0)
    _, _, _, height, width = images.shape
    device = torch.device("cuda")
    images = images.to(device)
    timestamps = torch.linspace(
        0, 1, num_frames, device=device).repeat(args.num_cameras).unsqueeze(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = NoPo4D.from_pretrained(
        os.fspath(Path(args.model).resolve())).to(device).eval()
    with torch.inference_mode():
        encoded = model(
            images=images, timestamps=timestamps,
            num_cameras=args.num_cameras)
        camera_c2w = encoded.camera_pose["extrinsic_c2w"][
            0, ::num_frames].float().cpu().numpy()
        target_pose = central_camera(camera_c2w, args.central_camera)
        target_k = wide_intrinsics(
            width, height, args.hfov, args.vfov)
        render_times = torch.linspace(0, 1, num_frames, device=device)[None]
        target_poses = torch.from_numpy(target_pose).to(device)[None, None]
        target_poses = target_poses.repeat(1, num_frames, 1, 1)
        target_intrinsics = torch.from_numpy(target_k).to(device)[None, None]
        target_intrinsics = target_intrinsics.repeat(1, num_frames, 1, 1)
        wide = model.render(
            gaussians=encoded.gaussians,
            extrinsics=target_poses,
            intrinsics=target_intrinsics,
            image_shape=(height, width),
            timestamps=render_times)
        if args.render_inputs:
            input_poses = encoded.camera_pose["extrinsic_c2w"]
            input_k = encoded.camera_pose["intrinsic"]
            reconstructed = model.render(
                gaussians=encoded.gaussians,
                extrinsics=input_poses,
                intrinsics=input_k,
                image_shape=(height, width),
                timestamps=timestamps)

    save_tensor_frames(wide.color[0], output_dir / "wide", "wide")
    save_tensor_frames(wide.alpha[0].unsqueeze(1).repeat(1, 3, 1, 1),
                       output_dir / "alpha", "alpha")
    if args.render_inputs:
        save_tensor_frames(reconstructed.color[0],
                           output_dir / "reconstruction", "view")

    report = {
        "schema": "video-split.nopo4d-validation.v1",
        "model": os.fspath(Path(args.model).resolve()),
        "da3_model": os.fspath(Path(args.da3_model).resolve()),
        "num_cameras": args.num_cameras,
        "num_frames": num_frames,
        "input_size": [width, height],
        "central_camera": args.central_camera,
        "virtual_hfov_deg": args.hfov,
        "virtual_vfov_deg": args.vfov,
        "mean_alpha": [float(value) for value in
                       wide.alpha[0].mean(dim=(-2, -1)).cpu()],
        "low_alpha_fraction": [float(value) for value in
                               (wide.alpha[0] < 0.5).float()
                               .mean(dim=(-2, -1)).cpu()],
        **geometry_report(camera_c2w),
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
