"""Report what a feature cache ACTUALLY contains, per recording.

Written after a "10 fps" extraction silently produced ~1 fps: blur filtering
is ON BY DEFAULT in extract_features_recseg.py (--th_blur 100.0, no flag to
disable it -- the existing *_noblur_* caches were made by passing
--th_blur 0, since lap_var >= 0 always). At 10 fps most egocentric frames
are motion-blurred, so the filter drops them, and the survivors are exactly
the LOW-MOTION instants: a motion-dependent sampling bias, not a uniform
downsample. Any dynamics model trained on such a cache is being asked to
predict motion after the motion frames were removed.

Reports, per cache: nominal vs realized frame spacing, how non-uniform the
spacing is (blur filtering leaves gaps), coverage of a recording's span, and
which recording_ids are present -- so a coverage gap can be attributed to
"never extracted" vs "extracted but unusable".

Usage:
    python -m src.boundary.cache_diag \
        --cache /workspace/tr1/data_recseg/feat_10fps_full_noblur_multi.pt \
        --cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --gold data/gold/audit_188_gold_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def cache_stats(path, sample=40):
    cache = torch.load(path, weights_only=False)
    rows, dts, spans, gaps = [], [], [], []
    for rec in cache[:sample] if sample else cache:
        t = rec["times"]
        if len(t) < 3:
            continue
        d = np.diff(t.numpy())
        dts.append(float(np.median(d)))
        gaps.append(float(np.percentile(d, 95)))
        dur = float(rec.get("duration") or (t[-1] - t[0]))
        spans.append(len(t) / dur if dur > 0 else np.nan)
    ids = {r.get("recording_id") or r.get("video") for r in cache}
    return {
        "path": path, "n_recordings": len(cache), "ids": ids,
        "median_dt": float(np.median(dts)) if dts else float("nan"),
        "p95_gap": float(np.median(gaps)) if gaps else float("nan"),
        "realized_fps": float(np.median(spans)) if spans else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--sample", type=int, default=40,
                    help="recordings to measure spacing on (0 = all)")
    a = ap.parse_args()

    stats = [cache_stats(p, a.sample) for p in a.cache]
    print(f"{'cache':<52} {'#rec':>5} {'med dt':>8} {'p95 gap':>8} {'eff fps':>8}")
    for s in stats:
        print(f"{os.path.basename(s['path']):<52} {s['n_recordings']:>5} "
              f"{s['median_dt']:>8.3f} {s['p95_gap']:>8.3f} {s['realized_fps']:>8.2f}")

    for s in stats:
        dt, gap = s["median_dt"], s["p95_gap"]
        if not np.isfinite(dt):
            continue
        name = os.path.basename(s["path"])
        if gap > 3 * dt:
            print(f"\n!! {name}: p95 gap ({gap:.2f}s) >> median dt ({dt:.2f}s) -- frames "
                  f"were DROPPED non-uniformly. With extract_features_recseg.py's "
                  f"default --th_blur 100.0 this is blur filtering, which removes "
                  f"exactly the high-motion frames. Re-extract with --th_blur 0.")
        elif dt > 0.3:
            print(f"\n!! {name}: median dt {dt:.2f}s ~= {1/dt:.1f} fps effective. If this "
                  f"cache was extracted with --fps 10, frames were dropped; "
                  f"re-extract with --th_blur 0.")

    if os.path.exists(a.gold):
        gold = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
        dev = {g.get("recording_id") for g in gold if g.get("recording_id")}
        print(f"\ndev recordings in gold: {len(dev)}")
        for s in stats:
            have = len(dev & s["ids"])
            print(f"  {os.path.basename(s['path']):<52} covers {have}/{len(dev)} dev"
                  + ("" if have == len(dev) else f"  (MISSING {len(dev)-have})"))


if __name__ == "__main__":
    main()
