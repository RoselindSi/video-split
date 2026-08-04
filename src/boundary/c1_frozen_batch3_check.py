"""External diagnostic: run the FROZEN C1 predictor (from report_v4's saved
checkpoint, no retraining) against batch3's 168 clean binary events, and
cross-tab continuity surprise against P1's already-computed scores.

This is NOT a re-evaluation of C1 against a test set (batch3 was promoted to
development data on 2026-08-02, its labels are fully known) and it is NOT a
step in choosing whether to adopt C1 (that gate is already closed: DO NOT
ADOPT, decided on the frozen 145-event clean set). Its only purpose is
sample size: 145 events gave 37 same_action_internal_motion negatives and
only 2 of those fooled P1; batch3 gives up to 116, letting the "P1-correct
negatives already show elevated continuity surprise" finding from the
145-event subtype cross-tab be checked against ~3x more same-action
negatives, at the cost of losing the fine regrasp/direction_reversal/
periodic_repetition subtype breakdown (batch3 has no equivalent free-text
motion notes to tag against).

Reuses batch3_manifest.jsonl's PRECOMPUTED primary_score/secondary_score
(scored once at sampling time) instead of reloading the P1 artifacts, so
this script only needs the frozen continuity predictor and a 10 fps feature
cache for batch3's recordings.

Usage (server, after batch3's recordings have a 10 fps cache):
    python -m src.boundary.c1_frozen_batch3_check \
        --manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --pair_labels data/gold/batch3_pair_labels_v1.csv \
        --cont_feat_cache /workspace/tr1/data_recseg/feat_10fps_batch3_multi.pt \
        --predictor /workspace/tr1/results/hal/continuity_c1/report_v4.json.predictor.pt \
        --out /workspace/tr1/results/hal/continuity_c1/batch3_frozen_check.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import _auroc
from src.boundary.predictive_continuity import (
    ContinuityPredictor, continuity_features, CONT_NAMES,
)


def load_batch3_events(manifest_path, pair_labels_path, by_rid):
    """event_id -> {recording_id, t, y, candidate_type, primary_score,
    primary_provisional_keep, rec}, restricted to clean binary events."""
    manifest = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                manifest[m["event_id"]] = m
    labels = T.load_pair_labels(pair_labels_path)

    events = []
    n_no_cache, n_excluded = 0, 0
    for eid, m in manifest.items():
        lab = labels.get(eid)
        if lab is None or lab.get("pair_supervision") not in ("strong_separate", "strong_align"):
            n_excluded += 1
            continue
        rec = by_rid.get(m["recording_id"])
        if rec is None:
            n_no_cache += 1
            continue
        events.append({
            "event_id": eid, "recording_id": m["recording_id"], "t": float(m["t"]),
            "y": T.CLEAN_BINARY[lab["pair_supervision"]],
            "candidate_type": m.get("candidate_type", ""),
            "primary_score": m.get("primary_score"),
            "primary_provisional_keep": bool(m.get("primary_provisional_keep")),
            "rec": rec,
        })
    print(f"batch3 clean binary events: {len(events)}  "
          f"(excluded by supervision: {n_excluded}, no feature cache: {n_no_cache})")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pair_labels", default="data/gold/batch3_pair_labels_v1.csv")
    ap.add_argument("--cont_feat_cache", action="append", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.cont_feat_cache)
    events = load_batch3_events(a.manifest, a.pair_labels, by_rid)
    if len(events) < 10:
        raise SystemExit("too few scorable events -- check --cont_feat_cache covers "
                         "batch3's recordings")
    y = np.array([e["y"] for e in events], dtype=float)
    print(f"class balance: {int(y.sum())}+ / {int((1 - y).sum())}-")

    ckpt = torch.load(os.path.expanduser(a.predictor), weights_only=False)
    d_in = next(iter(by_rid.values()))["feats"].shape[1]
    model = ContinuityPredictor(d_in, ckpt["n_past"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(a.device).eval()
    print(f"loaded FROZEN predictor (n_past={ckpt['n_past']}) -- no training, no fitting")

    Xc, reasons, Xraw = continuity_features(model, events, a.device, cont_by_rid=by_rid)
    scorable = np.isfinite(Xc).all(1)
    print(f"continuity features: {scorable.sum()}/{len(events)} fully scorable")
    print("coverage breakdown:", dict(Counter(reasons)))

    standalone = {}
    for j, nm in enumerate(CONT_NAMES):
        m = np.isfinite(Xc[:, j])
        standalone[nm] = _auroc(y[m], Xc[m, j]) if len(set(y[m].tolist())) == 2 else float("nan")
    print("standalone AUROC on batch3 (descriptive only, no gate):",
          {k: round(v, 3) for k, v in standalone.items()})

    # P1's own already-computed verdict (from sampling time) vs continuity surprise,
    # same framing as the 145-event false-positive-rescue check but with a much
    # bigger same-action-negative sample.
    p1_keep = np.array([bool(e["primary_provisional_keep"]) for e in events])
    fp = scorable & p1_keep & (y == 0)          # P1 wrongly kept a same-action event
    tn = scorable & ~p1_keep & (y == 0)         # P1 correctly rejected it
    pos = scorable & (y == 1)
    j = CONT_NAMES.index("cont_efwd_z")
    col = Xc[:, j]
    print(f"\nP1 false positives on batch3 (kept a same-action event): {int(fp.sum())}")
    print(f"P1 true negatives (correctly rejected same-action):        {int(tn.sum())}")
    print(f"true positives (sharp_visible_transition):                 {int(pos.sum())}")
    def _report(mask, label):
        v = col[mask]
        v = v[np.isfinite(v)]
        if len(v):
            print(f"  {label}: n={len(v)} mean={v.mean():.3f} median={np.median(v):.3f}")
        else:
            print(f"  {label}: n=0")
    _report(fp, "forward-z, P1 false positives")
    _report(tn, "forward-z, P1 true negatives (same-action, P1 got it right)")
    _report(pos, "forward-z, true positives (sharp)")

    # by candidate_type -- batch3 has no finer motion-subtype tag (no free-text
    # notes like the original 145's audit), but gt_boundary vs raw_change_peak
    # is available and worth splitting, since raw_change_peak candidates are
    # generated from a naive change-detector and may skew toward exactly the
    # kind of local motion spikes (regrasp/reversal) the 145-event check flagged.
    by_type = defaultdict(list)
    for e, s, keep in zip(events, col, p1_keep):
        by_type[e["candidate_type"]].append((e["y"], s, keep))
    print("\nby candidate_type (same-action negatives only):")
    for ctype, rows in by_type.items():
        neg = [(s, k) for yv, s, k in rows if yv == 0 and np.isfinite(s)]
        if not neg:
            continue
        vals = np.array([s for s, k in neg])
        print(f"  {ctype}: n_negative={len(neg)} mean_fwd_z={vals.mean():.3f} "
              f"P1_keep_rate={np.mean([k for s, k in neg]):.3f}")

    if a.out:
        report = {
            "n_events": len(events), "n_scorable": int(scorable.sum()),
            "standalone_auroc": standalone,
            "p1_false_positive_z": {"n": int(fp.sum()), "mean": float(col[fp].mean()) if fp.sum() else None},
            "p1_true_negative_z": {"n": int(tn.sum()), "mean": float(col[tn].mean()) if tn.sum() else None},
            "true_positive_z": {"n": int(pos.sum()), "mean": float(col[pos].mean()) if pos.sum() else None},
        }
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
