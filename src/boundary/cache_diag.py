"""Report what a feature cache ACTUALLY contains, per recording.

Written after a C1 run trained on 4 timesteps per 4 s window while its cache
held 40. Two failure modes motivate the two halves of this report:

  * COVERAGE -- extract_features_recseg.py's --data is not action="append",
    so passing it several times silently keeps only the last file. The
    resulting cache looked healthy but held one data file's recordings, and
    52 eval events scored NaN for want of a cache.
  * SPACING -- a cache's median can look right while individual recordings
    are far sparser. Any model that infers a window length from the data
    (predictive_continuity.infer_n_past) can be mis-configured by a single
    such recording, so sparse ones are listed by name, not just counted.
    Separately, blur/black filtering (--th_blur default 100.0, --th_black
    default 20.0) is applied unconditionally; when it bites hard it leaves a
    p95 gap much larger than the median dt, and the survivors are the
    LOW-MOTION frames -- a motion-dependent bias, not a uniform downsample.
    Pass --th_blur 0 --th_black 0 for a strictly uniform grid.

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


def cache_stats(path, sample=0):
    """Per-recording spacing for EVERY recording by default. A cache whose
    median looks right can still contain sparse recordings, and one of those
    is enough to mis-configure a model that infers its window from the data
    (that is how a 10 fps cache produced a 4-frame window)."""
    cache = torch.load(path, weights_only=False)
    dts, spans, gaps, per_rec = [], [], [], []
    for rec in (cache[:sample] if sample else cache):
        t = rec["times"]
        rid = rec.get("recording_id") or rec.get("video")
        if len(t) < 3:
            per_rec.append((float("inf"), rid, len(t)))
            continue
        d = np.diff(t.numpy())
        md = float(np.median(d))
        dts.append(md)
        gaps.append(float(np.percentile(d, 95)))
        dur = float(rec.get("duration") or (t[-1] - t[0]))
        spans.append(len(t) / dur if dur > 0 else np.nan)
        per_rec.append((md, rid, len(t)))
    ids = {r.get("recording_id") or r.get("video") for r in cache}
    med = float(np.median(dts)) if dts else float("nan")
    sparse = sorted((r for r in per_rec if r[0] > 2 * med), reverse=True)
    return {
        "path": path, "n_recordings": len(cache), "ids": ids,
        "median_dt": med,
        "p95_gap": float(np.median(gaps)) if gaps else float("nan"),
        "realized_fps": float(np.median(spans)) if spans else float("nan"),
        "max_dt": float(np.max(dts)) if dts else float("nan"),
        "sparse": sparse,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="recordings to measure spacing on (0 = all, the default)")
    a = ap.parse_args()

    stats = [cache_stats(p, a.sample) for p in a.cache]
    print(f"{'cache':<52} {'#rec':>5} {'med dt':>8} {'p95 gap':>8} {'max dt':>8} {'eff fps':>8}")
    for s in stats:
        print(f"{os.path.basename(s['path']):<52} {s['n_recordings']:>5} "
              f"{s['median_dt']:>8.3f} {s['p95_gap']:>8.3f} {s['max_dt']:>8.3f} "
              f"{s['realized_fps']:>8.2f}")

    for s in stats:
        dt, gap = s["median_dt"], s["p95_gap"]
        if not np.isfinite(dt):
            continue
        name = os.path.basename(s["path"])
        if gap > 3 * dt:
            print(f"\n!! {name}: p95 gap ({gap:.2f}s) >> median dt ({dt:.2f}s) -- frames "
                  f"were dropped non-uniformly. extract_features_recseg.py filters "
                  f"blurry (--th_blur, default 100.0) and dark (--th_black) frames "
                  f"unconditionally; pass --th_blur 0 --th_black 0 for a uniform grid.")
        if s["sparse"]:
            print(f"\n!! {name}: {len(s['sparse'])} recording(s) far sparser than the "
                  f"median ({dt:.3f}s). A model that infers its window length from the "
                  f"data can be mis-configured by these:")
            for md, rid, n in s["sparse"][:8]:
                print(f"     {rid}  dt={md:.3f}s  frames={n}")

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
