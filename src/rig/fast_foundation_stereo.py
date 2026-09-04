"""Dense stereo depth with the official Fast-FoundationStereo ONNX model.

This module only replaces the correspondence stage in ``depth.py``.  The
input must already be an undistorted, horizontally rectified stereo pair.
The model predicts disparity; calibrated focal length and baseline turn that
into metric depth.

The right image is geometry evidence, not a second colour source.  Any RGB
renderer consuming this output must keep one texture owner per output pixel
and resolve visibility with depth.  Averaging the two input colours recreates
the StabStitch++ double-contour failure even when disparity is correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_DISPARITY = 0.5
MIN_DEPTH_M = 0.12
MAX_DEPTH_M = 8.0


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class FastFoundationStereo:
    """Small ONNXRuntime adapter with explicit colour and scale handling."""

    def __init__(self, model_path=None, providers=None, session=None,
                 cuda_lib_dir=None, cudnn_lib_dir=None,
                 require_cuda=False):
        if session is None:
            import onnxruntime as ort

            if cuda_lib_dir:
                ort.preload_dlls(
                    cuda=True, cudnn=False,
                    directory=os.fspath(cuda_lib_dir))
            if cudnn_lib_dir:
                ort.preload_dlls(
                    cuda=False, cudnn=True,
                    directory=os.fspath(cudnn_lib_dir))
            available = ort.get_available_providers()
            providers = providers or [
                name for name in ("CUDAExecutionProvider",
                                  "CPUExecutionProvider")
                if name in available
            ]
            if not providers:
                raise RuntimeError("ONNXRuntime has no usable execution provider")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                os.fspath(model_path), sess_options=options,
                providers=providers)
        self.session = session
        active_providers = self.providers
        if require_cuda and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider was requested but did not load; "
                f"active providers: {active_providers}")
        self.input_names = [item.name for item in session.get_inputs()]
        self.output_names = [item.name for item in session.get_outputs()]
        if set(self.input_names) != {"left_image", "right_image"}:
            raise ValueError(f"unexpected model inputs: {self.input_names}")
        if "disparity" not in self.output_names:
            raise ValueError(f"model has no disparity output: {self.output_names}")
        shape = session.get_inputs()[0].shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape[-2:]):
            raise ValueError(f"model needs a fixed NCHW image shape, got {shape}")
        self.target_h, self.target_w = shape[-2:]

    @property
    def providers(self):
        get = getattr(self.session, "get_providers", None)
        return list(get()) if get else []

    @staticmethod
    def _tensor(image_bgr, size):
        import cv2

        width, height = size
        resized = cv2.resize(image_bgr, (width, height),
                             interpolation=cv2.INTER_LINEAR)
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])

    def disparity(self, left_bgr, right_bgr):
        """Predict left-view disparity in pixels at the input resolution."""
        import cv2

        if left_bgr.shape != right_bgr.shape or left_bgr.ndim != 3:
            raise ValueError(
                f"rectified pair shapes differ: {left_bgr.shape}, "
                f"{right_bgr.shape}")
        height, width = left_bgr.shape[:2]
        feed = {
            "left_image": self._tensor(
                left_bgr, (self.target_w, self.target_h)),
            "right_image": self._tensor(
                right_bgr, (self.target_w, self.target_h)),
        }
        values = self.session.run(self.output_names, feed)
        outputs = dict(zip(self.output_names, values))
        disparity = np.asarray(outputs["disparity"], dtype=np.float32)
        disparity = disparity.reshape(self.target_h, self.target_w)
        disparity = np.maximum(disparity, 0)
        if (height, width) != (self.target_h, self.target_w):
            disparity = cv2.resize(disparity, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
            disparity *= width / self.target_w
        return disparity.astype(np.float32, copy=False)


class FastFoundationStereoProvider:
    """Reusable matcher plus rectification cache for wide RGBD rendering."""

    def __init__(self, model_path=None, providers=None, session=None,
                 cuda_lib_dir=None, cudnn_lib_dir=None,
                 require_cuda=False, stride=1, splat=2,
                 depth_tie_m=0.02, rectified_size=None,
                 owner_aligned=False, owner_depth_m=0.6,
                 owner_mid_authority_deg=72.0):
        self.stereo = FastFoundationStereo(
            model_path=model_path, providers=providers, session=session,
            cuda_lib_dir=cuda_lib_dir, cudnn_lib_dir=cudnn_lib_dir,
            require_cuda=require_cuda)
        self.stride = int(stride)
        self.splat = int(splat)
        self.depth_tie_m = float(depth_tie_m)
        self.rectified_size = (tuple(int(v) for v in rectified_size)
                               if rectified_size is not None else None)
        self.owner_aligned = bool(owner_aligned)
        self.owner_depth_m = float(owner_depth_m)
        self.owner_mid_authority_deg = float(owner_mid_authority_deg)
        self.rect_cache = {}
        self._fixed_owner = None

    def _prepare_rectification(self, rig):
        if self.rectified_size is None:
            return
        from src.rig.depth import rectify_maps

        for module in rig.modules:
            if module.name not in self.rect_cache:
                self.rect_cache[module.name] = rectify_maps(
                    rig, module, size=self.rectified_size)

    def __call__(self, rig, vcam, sources):
        """Return virtual-camera metric range for ``DepthAwarePanorama``."""
        from src.rig.wide_depth import (fixed_module_owner, wide_depth,
                                        wide_depth_owned)

        self._prepare_rectification(rig)
        if self.owner_aligned:
            if self._fixed_owner is None:
                self._fixed_owner = fixed_module_owner(
                    rig, vcam, depth_m=self.owner_depth_m,
                    mid_authority_deg=self.owner_mid_authority_deg)
            return wide_depth_owned(
                rig, vcam, sources, self._fixed_owner,
                rect_cache=self.rect_cache, stride=self.stride,
                splat=self.splat, matcher=self.stereo.disparity)
        return wide_depth(
            rig, vcam, sources, rect_cache=self.rect_cache,
            stride=self.stride, splat=self.splat,
            matcher=self.stereo.disparity)

    def rgbd(self, rig, vcam, sources):
        """Return the stricter forward-rendered, one-owner RGBD panorama."""
        from src.rig.wide_depth import wide_rgbd

        self._prepare_rectification(rig)
        return wide_rgbd(
            rig, vcam, sources, rect_cache=self.rect_cache,
            stride=self.stride, splat=self.splat,
            matcher=self.stereo.disparity,
            depth_tie_m=self.depth_tie_m)

    def owned_rgbd(self, rig, vcam, sources, photometric=None):
        """Forward RGBD using the same fixed module owner as the panorama."""
        from src.rig.wide_depth import wide_rgbd_owned

        self._prepare_rectification(rig)
        return wide_rgbd_owned(
            rig, vcam, sources, self.owner_map(rig, vcam),
            rect_cache=self.rect_cache, stride=self.stride,
            splat=self.splat, matcher=self.stereo.disparity,
            photometric=photometric, fallback_depth_m=self.owner_depth_m)

    def owner_map(self, rig, vcam):
        """Return the clip-constant texture owner used by direct RGBD."""
        from src.rig.wide_depth import fixed_module_owner

        if self._fixed_owner is None:
            self._fixed_owner = fixed_module_owner(
                rig, vcam, depth_m=self.owner_depth_m,
                mid_authority_deg=self.owner_mid_authority_deg)
        return self._fixed_owner


def disparity_to_depth(disparity, focal_px, baseline_m,
                       min_depth=MIN_DEPTH_M, max_depth=MAX_DEPTH_M):
    disparity = np.asarray(disparity, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = float(focal_px) * float(baseline_m) / disparity
    valid = ((disparity > MIN_DISPARITY) & np.isfinite(depth) &
             (depth >= min_depth) & (depth <= max_depth))
    return np.where(valid, depth, np.nan).astype(np.float32), valid


def _rectification_geometry(case_dir):
    manifest_path = Path(case_dir) / "manifest.json"
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    rect = manifest["rectification"]
    p1 = np.asarray(rect["P1"], dtype=float)
    p2 = np.asarray(rect["P2"], dtype=float)
    focal_px = float(p1[0, 0])
    baseline_m = abs(float(p2[0, 3]) / float(p2[0, 0]))
    return manifest, manifest_path, focal_px, baseline_m


def _photo_residual(left, right, disparity, valid):
    """Gradient residual after right-to-left reprojection; diagnostic only."""
    import cv2

    height, width = disparity.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx - disparity
    inside = valid & (map_x >= 0) & (map_x < width - 1)
    if inside.sum() < 100:
        return None
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warped = cv2.remap(right_gray, map_x, yy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    gx_left = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
    gx_warp = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
    residual = np.abs(gx_left[inside] - gx_warp[inside])
    return {
        "gradient_l1_median": float(np.median(residual)),
        "gradient_l1_p90": float(np.percentile(residual, 90)),
    }


def _depth_visual(depth, valid):
    import cv2

    inverse = np.zeros_like(depth, dtype=np.float32)
    inverse[valid] = 1.0 / depth[valid]
    if valid.any():
        lo, hi = np.percentile(inverse[valid], [2, 98])
        scaled = np.clip((inverse - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        scaled = inverse
    visual = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO)
    visual[~valid] = 0
    return visual


def run_case(case_dir, model_path, out_dir, limit=None, save_disparity=True,
             providers=None, cuda_lib_dir=None, cudnn_lib_dir=None,
             require_cuda=False):
    """Run a rectified pair directory and write depth diagnostics."""
    import cv2

    case_dir, out_dir = Path(case_dir), Path(out_dir)
    names1 = sorted((case_dir / "video1").glob("*.jpg"))
    names2 = sorted((case_dir / "video2").glob("*.jpg"))
    if not names1 or len(names1) != len(names2):
        raise ValueError("input video1/video2 frame counts are empty or unequal")
    if limit is not None:
        names1, names2 = names1[:limit], names2[:limit]
    manifest, manifest_path, focal_px, baseline_m = \
        _rectification_geometry(case_dir)
    model = FastFoundationStereo(
        model_path, providers=providers,
        cuda_lib_dir=cuda_lib_dir, cudnn_lib_dir=cudnn_lib_dir,
        require_cuda=require_cuda)
    out_dir.mkdir(parents=True, exist_ok=True)
    disparity_dir = out_dir / "disparity"
    if save_disparity:
        disparity_dir.mkdir(exist_ok=True)

    fps = float(manifest["frames"].get("fps", 30.0))
    writer = None
    rows = []
    try:
        for index, (name1, name2) in enumerate(zip(names1, names2)):
            left, right = cv2.imread(os.fspath(name1)), cv2.imread(os.fspath(name2))
            if left is None or right is None:
                raise OSError(f"cannot decode {name1} or {name2}")
            started = time.perf_counter()
            disparity = model.disparity(left, right)
            runtime = time.perf_counter() - started
            depth, valid = disparity_to_depth(
                disparity, focal_px, baseline_m)
            if save_disparity:
                np.save(disparity_dir / f"{index:06d}.npy", disparity)
            visual = _depth_visual(depth, valid)
            diagnostic = np.concatenate((left, visual), axis=1)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    os.fspath(out_dir / "left_and_depth.mp4"), fourcc, fps,
                    (diagnostic.shape[1], diagnostic.shape[0]))
                if not writer.isOpened():
                    raise OSError("cannot create left_and_depth.mp4")
            writer.write(diagnostic)
            valid_values = disparity[valid]
            row = {
                "frame": index,
                "runtime_s": runtime,
                "valid_fraction": float(valid.mean()),
                "disparity_median_px": (
                    float(np.median(valid_values)) if valid_values.size else None),
                "depth_median_m": (
                    float(np.nanmedian(depth)) if valid_values.size else None),
                "photo": _photo_residual(left, right, disparity, valid),
            }
            rows.append(row)
            print(f"{index + 1:4d}/{len(names1)}  {runtime * 1000:7.1f} ms  "
                  f"valid {row['valid_fraction']:6.1%}  "
                  f"depth {row['depth_median_m'] or float('nan'):.2f} m",
                  flush=True)
    finally:
        if writer is not None:
            writer.release()

    runtimes = np.asarray([row["runtime_s"] for row in rows])
    validity = np.asarray([row["valid_fraction"] for row in rows])
    result = {
        "schema": "video-split.fast-foundation-stereo.v1",
        "case": os.path.abspath(case_dir),
        "case_manifest_sha256": _sha256(manifest_path),
        "model": os.path.abspath(model_path),
        "model_sha256": _sha256(model_path),
        "model_input_size": [model.target_w, model.target_h],
        "providers": model.providers,
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "frames": len(rows),
        "summary": {
            "runtime_median_ms": float(np.median(runtimes) * 1000),
            "runtime_p90_ms": float(np.percentile(runtimes, 90) * 1000),
            "valid_fraction_median": float(np.median(validity)),
            "valid_fraction_min": float(np.min(validity)),
        },
        "per_frame": rows,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no_save_disparity", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cuda_lib_dir",
        help="directory containing CUDA shared libraries for ORT preload")
    parser.add_argument(
        "--cudnn_lib_dir",
        help="directory containing cuDNN shared libraries for ORT preload")
    args = parser.parse_args()
    providers = ["CPUExecutionProvider"] if args.cpu else None
    result = run_case(
        args.case, args.model, args.out, limit=args.limit,
        save_disparity=not args.no_save_disparity, providers=providers,
        cuda_lib_dir=args.cuda_lib_dir,
        cudnn_lib_dir=args.cudnn_lib_dir,
        require_cuda=not args.cpu)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nresult -> {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
