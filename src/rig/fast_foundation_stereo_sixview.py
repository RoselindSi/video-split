"""Render a synchronized six-camera clip with Fast-FoundationStereo depth.

Each physical stereo module contributes two images to correspondence. Only
the lower-numbered eye supplies RGB. The resulting three texture views are
reprojected with measured depth and retain one high-frequency owner per pixel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


DEFAULT_RECTIFIED_SIZE = (960, 720)


def videos_from_databag(databag):
    root = Path(databag)
    calibration = root / "calibration.yaml"
    videos = {key: root / f"{key}.mp4"
              for key in ("cam12", "cam34", "cam56")}
    missing = [os.fspath(path) for path in (calibration, *videos.values())
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing six-view input: " + ", ".join(missing))
    return calibration, {key: os.fspath(path) for key, path in videos.items()}


def build_renderer(rig, vcam, model_path, cuda_lib_dir=None,
                   cudnn_lib_dir=None, require_cuda=True,
                   rectified_size=DEFAULT_RECTIFIED_SIZE, depth_m=0.6,
                   disagreement_gate=40.0, mid_authority_deg=72.0):
    from src.rig.fast_foundation_stereo import FastFoundationStereoProvider
    from src.rig.panorama import DepthAwarePanorama

    provider = FastFoundationStereoProvider(
        model_path=model_path, cuda_lib_dir=cuda_lib_dir,
        cudnn_lib_dir=cudnn_lib_dir, require_cuda=require_cuda,
        rectified_size=rectified_size, owner_aligned=True,
        owner_depth_m=depth_m,
        owner_mid_authority_deg=mid_authority_deg)
    renderer = DepthAwarePanorama(
        rig, vcam, depth_m=depth_m, depth_provider=provider,
        texture_mode="module_left", use_depth=True,
        use_residual_flow=False, stabilize_depth=True,
        disagreement_gate=disagreement_gate,
        mid_authority_deg=mid_authority_deg)
    return renderer, provider


def _fit_samples(rig, videos, start, n, stride, count):
    from src.rig.seam_fix import ClipReader

    count = min(max(0, int(count)), int(n))
    if count == 0:
        return
    positions = np.unique(np.linspace(0, n - 1, count).round().astype(int))
    reader = ClipReader(rig, videos, start)
    previous = None
    try:
        for position in positions:
            skip = 0 if previous is None else \
                int((position - previous) * stride - 1)
            sources = reader.next(skip=max(0, skip))
            if not sources:
                break
            yield sources
            previous = int(position)
    finally:
        reader.close()


def _open_writer(path, fps, size):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        os.fspath(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
        tuple(int(v) for v in size))
    if not writer.isOpened():
        raise OSError(f"cannot create video {path}")
    return writer


def _label(image, text):
    import cv2

    out = image.copy()
    cv2.rectangle(out, (12, 12), (470, 58), (20, 20, 20), -1)
    cv2.putText(out, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def _distribution(rows, key):
    values = np.asarray([row[key] for row in rows
                         if np.isfinite(row.get(key, np.nan))], dtype=float)
    if not values.size:
        return {"median": None, "p90": None, "max": None}
    return {"median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values))}


def run(databag, model_path, out_path, comparison_path, metrics_path,
        start=3000, n=120, stride=1, fit_frames=6, output_fps=None,
        hfov=150.0, vfov=90.0, depth_m=0.6,
        disagreement_gate=40.0, rectified_size=DEFAULT_RECTIFIED_SIZE,
        mid_authority_deg=72.0, cuda_lib_dir=None, cudnn_lib_dir=None,
        require_cuda=True):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render as render_baseline
    from src.rig.seam_audit import seam_edge_ratio, skin_at_seam
    from src.rig.seam_fix import ClipReader, Prefetch

    calibration, videos = videos_from_databag(databag)
    rig = RigCalibration(os.fspath(calibration))
    vcam = VirtualWideCamera.from_rig(
        rig, hfov_deg=hfov, vfov_deg=vfov)
    renderer, provider = build_renderer(
        rig, vcam, model_path, cuda_lib_dir=cuda_lib_dir,
        cudnn_lib_dir=cudnn_lib_dir, require_cuda=require_cuda,
        rectified_size=rectified_size, depth_m=depth_m,
        disagreement_gate=disagreement_gate,
        mid_authority_deg=mid_authority_deg)

    fit_started = time.perf_counter()
    fit = renderer.fit(_fit_samples(
        rig, videos, start, n, stride, fit_frames))
    fit_seconds = time.perf_counter() - fit_started

    if output_fps is None:
        cap = cv2.VideoCapture(videos["cam12"])
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        output_fps = source_fps / stride
    else:
        source_fps = float(output_fps) * stride

    writer = _open_writer(out_path, output_fps, (vcam.width, vcam.height))
    comparison = (_open_writer(
        comparison_path, output_fps, (vcam.width, vcam.height * 2))
        if comparison_path else None)
    reader = Prefetch(ClipReader(rig, videos, start), skip=max(0, stride - 1))
    baseline_cache = {}
    rows = []
    first_owners = {}
    started = time.perf_counter()
    try:
        for index in range(int(n)):
            sources = reader.next()
            if not sources:
                break
            frame_number = int(start + index * stride)
            frame_started = time.perf_counter()
            baseline, baseline_owner, _, _ = render_baseline(
                rig, vcam, sources, depth_m, map_cache=baseline_cache)
            forward = provider.owned_rgbd(
                rig, vcam, sources, photometric=renderer.photo)
            rgb = forward.rgb.copy()
            rgb[~forward.valid] = baseline[~forward.valid]
            owner = provider.owner_map(rig, vcam)
            render_seconds = time.perf_counter() - frame_started
            writer.write(rgb)
            if comparison is not None:
                comparison.write(np.vstack((
                    _label(baseline, "constant-depth baseline"),
                    _label(rgb, "Fast-FoundationStereo forward six-view"))))

            first_owners.setdefault("baseline", baseline_owner.copy())
            first_owners.setdefault("ffs", owner.copy())
            base_edge = seam_edge_ratio(baseline, baseline_owner)
            ffs_edge = seam_edge_ratio(rgb, owner)
            _, _, skin_fraction = skin_at_seam(rgb, owner)
            rows.append({
                "frame": frame_number,
                "render_seconds": render_seconds,
                "depth_coverage": forward.measured_coverage(),
                "plane_fallback_fraction": float(np.mean(
                    forward.valid & ~forward.measured & (owner >= 0))),
                "output_fallback_fraction": float(np.mean(
                    (~forward.valid) & (owner >= 0))),
                "baseline_seam_ratio": float(base_edge["ratio"]),
                "baseline_seam_excess": float(base_edge["excess"]),
                "ffs_seam_ratio": float(ffs_edge["ratio"]),
                "ffs_seam_excess": float(ffs_edge["excess"]),
                "baseline_owner_change": float(np.mean(
                    baseline_owner != first_owners["baseline"])),
                "ffs_owner_change": float(np.mean(
                    owner != first_owners["ffs"])),
                "skin_on_ffs_seam_fraction": skin_fraction,
            })
            if (index + 1) % 10 == 0 or index == 0:
                elapsed = time.perf_counter() - started
                remaining = elapsed / (index + 1) * (n - index - 1)
                print(f"[{index + 1}/{n}] {elapsed:.1f}s, "
                      f"about {remaining:.1f}s left", flush=True)
    finally:
        reader.close()
        writer.release()
        if comparison is not None:
            comparison.release()

    summary_keys = (
        "render_seconds", "depth_coverage", "plane_fallback_fraction",
        "output_fallback_fraction",
        "baseline_seam_ratio", "baseline_seam_excess", "ffs_seam_ratio",
        "ffs_seam_excess", "baseline_owner_change", "ffs_owner_change",
        "skin_on_ffs_seam_fraction")
    result = {
        "schema": "video-split.ffs-six-view.v3",
        "databag": os.fspath(Path(databag).resolve()),
        "calibration": os.fspath(calibration),
        "videos": videos,
        "model": os.fspath(Path(model_path).resolve()),
        "providers": provider.stereo.providers,
        "start": int(start), "requested_frames": int(n),
        "written_frames": len(rows), "stride": int(stride),
        "source_fps": source_fps, "output_fps": float(output_fps),
        "output_size": [vcam.width, vcam.height],
        "rectified_size": list(rectified_size),
        "mid_authority_deg": float(mid_authority_deg),
        "texture_sources": [module.left.name for module in rig.modules],
        "geometry_sources": sorted(rig.cameras),
        "render_mode": "fixed-owner-forward-rgbd-with-source-plane-fill",
        "fit": {**fit, "seconds": fit_seconds},
        "summary": {key: _distribution(rows, key) for key in summary_keys},
        "per_frame": rows,
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"video -> {out_path}")
    if comparison_path:
        print(f"comparison -> {comparison_path}")
    print(f"metrics -> {metrics_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--databag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--comparison")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--start", type=int, default=3000)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fit_frames", type=int, default=6)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--hfov", type=float, default=150.0)
    parser.add_argument("--vfov", type=float, default=90.0)
    parser.add_argument("--depth_m", type=float, default=0.6)
    parser.add_argument("--gate", type=float, default=40.0)
    parser.add_argument("--mid_authority", type=float, default=72.0)
    parser.add_argument("--rectified_width", type=int, default=960)
    parser.add_argument("--rectified_height", type=int, default=720)
    parser.add_argument("--cuda_lib_dir")
    parser.add_argument("--cudnn_lib_dir")
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()
    run(
        args.databag, args.model, args.out, args.comparison, args.metrics,
        start=args.start, n=args.n, stride=args.stride,
        fit_frames=args.fit_frames, output_fps=args.fps,
        hfov=args.hfov, vfov=args.vfov, depth_m=args.depth_m,
        disagreement_gate=args.gate,
        rectified_size=(args.rectified_width, args.rectified_height),
        mid_authority_deg=args.mid_authority,
        cuda_lib_dir=args.cuda_lib_dir, cudnn_lib_dir=args.cudnn_lib_dir,
        require_cuda=not args.allow_cpu)


if __name__ == "__main__":
    main()
