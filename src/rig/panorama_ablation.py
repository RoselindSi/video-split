"""A/B the old stitch against cumulative six-view panorama improvements.

The four variants isolate source count, image-derived depth and content-fitted
residual alignment.  Metrics are computed on the exact owner boundaries rather
than averaged over the large unchanged middle of the panorama.
"""
from __future__ import annotations

import json

import numpy as np


VARIANTS = (
    ("baseline-3view-plane", "baseline", False, False),
    ("6view-plane", "depth", False, False),
    ("6view-depth", "depth", True, False),
    ("6view-depth-flow", "depth", True, True),
)


def _sources_at(rig, videos, frame):
    from src.rig.render_wide import read_frame, split_halves
    out = {}
    for module in rig.modules:
        key = f"cam{module.left.name[-1]}{module.right.name[-1]}"
        if key not in videos:
            continue
        left, right = split_halves(read_frame(videos[key], frame))
        out[module.left.name], out[module.right.name] = left, right
    return out


def evaluate(rig, vcam, videos, frames, fit_frames=6, depth_m=0.6):
    from src.rig.panorama import DepthAwarePanorama
    from src.rig.render_wide import render
    from src.rig.seam_audit import seam_edge_ratio

    rows = []
    fit_at = frames[:max(0, min(int(fit_frames), len(frames)))]
    for name, mode, use_depth, use_flow in VARIANTS:
        renderer = None
        if mode == "depth":
            renderer = DepthAwarePanorama(
                rig, vcam, depth_m=depth_m, use_depth=use_depth,
                use_residual_flow=use_flow)
            renderer.fit(_sources_at(rig, videos, f) for f in fit_at)

        first_owner = None
        metrics = []
        cache = {}
        for frame in frames:
            sources = _sources_at(rig, videos, frame)
            if renderer is None:
                rgb, owner, _, _ = render(
                    rig, vcam, sources, depth_m, map_cache=cache)
                stats = {"depth_coverage": np.nan, "gated_frac": np.nan,
                         "n_views": 3}
            else:
                rgb, owner, stats, _ = renderer.render(sources)
            if first_owner is None:
                first_owner = owner.copy()
            edge = seam_edge_ratio(rgb, owner)
            metrics.append({
                "edge_ratio": edge["ratio"],
                "seam_excess": edge["excess"],
                "owner_changed": float((owner != first_owner).mean()),
                "gated_frac": float(stats.get("gated_frac", np.nan)),
                "depth_coverage": float(
                    stats.get("depth_coverage", np.nan)),
                "n_views": int(stats.get("n_views", 3)),
            })

        def med(key):
            values = np.array([m[key] for m in metrics], float)
            return float(np.nanmedian(values)) if np.isfinite(values).any() \
                else float("nan")

        rows.append({"variant": name, "frames": len(metrics),
                     "seam_edge_ratio_median": med("edge_ratio"),
                     "seam_excess_median": med("seam_excess"),
                     "owner_change_median": med("owner_changed"),
                     "gated_frac_median": med("gated_frac"),
                     "depth_coverage_median": med("depth_coverage"),
                     "n_views": max([m["n_views"] for m in metrics],
                                    default=0)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--databag")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fit_frames", type=int, default=6)
    ap.add_argument("--depth_m", type=float, default=0.6)
    ap.add_argument("--out_json")
    args = ap.parse_args()

    import os
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    if args.databag:
        calibration = os.path.join(args.databag, "calibration.yaml")
        videos = {k: os.path.join(args.databag, f"{k}.mp4")
                  for k in ("cam12", "cam34", "cam56")}
    else:
        if not args.calibration or not args.video:
            ap.error("give --databag, or --calibration and all three --video")
        calibration = args.calibration
        videos = dict(x.split("=", 1) for x in args.video)
    rig = RigCalibration(calibration)
    vcam = VirtualWideCamera.from_rig(rig)
    frames = list(range(args.start, args.start + args.n * args.stride,
                        args.stride))
    rows = evaluate(rig, vcam, videos, frames, args.fit_frames, args.depth_m)
    print(f"  {'variant':<24} {'views':>5} {'seam ratio':>11} "
          f"{'excess':>9} {'owner move':>11} {'gated':>9}")
    for row in rows:
        print(f"  {row['variant']:<24} {row['n_views']:>5d} "
              f"{row['seam_edge_ratio_median']:>11.3f} "
              f"{row['seam_excess_median']:>+9.3f} "
              f"{row['owner_change_median']:>11.3%} "
              f"{row['gated_frac_median']:>9.3%}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()

