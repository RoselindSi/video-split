"""HAL x VLM disagreement cross-tabulation + asymmetric routing (replaces
early fusion, which failed on the real 72-event set).

Why early fusion (hal_vlm_fusion.py's arm C: one logistic regression on
concatenated HAL + VLM-atomic features) was abandoned: on the full 63-event
usable set it made things WORSE than HAL-only alone on both axes
(valid_recall 0.846->0.722, motion_hard_negative_recall 0.400->0.200).
Confirmed reason: HAL and the VLM atomic auditor have OPPOSITE profiles,
not complementary-and-linearly-blendable ones --

  HAL-only (grouped, class-weighted LR on 5 cheap features): protects real
      boundaries well (valid_recall 0.846) but is only moderate at rejecting
      motion-only false ones (motion_hard_negative_recall 0.400).
  VLM atomic auditor (temporal_truth, the actual deployed auditor output,
      evaluated at full 72-event scale): the OPPOSITE profile -- catches
      17/20 (85%) motion-hard-negatives but now keeps only 12/43 (28%) real
      boundaries valid, having over-corrected toward "reject by default".

A single linear classifier averaging two oppositely-biased signals learns
neither one's strength -- confirmed empirically, not just argued. This
script instead treats the two as DISCRETE, separately-thresholded decisions
and cross-tabulates them against gold truth, to find:
  - an AGREEMENT cell precise enough to auto-act on
  - the disagreement cell (HAL says valid, VLM says spurious) that is the
    highest-value review queue -- it should contain both the VLM's wrongly-
    deleted real boundaries and HAL's wrongly-protected motion artifacts,
    and is exactly where a human's attention is worth the most.

HAL's own thresholds are NOT the old symmetric 0.25/0.75 band -- they are
picked per-side to target ~0.90 precision (a coverage/precision curve is
printed so you can see the actual tradeoff, not just the chosen operating
point).

Usage (after both a HAL feature cache and a full-coverage VLM pred jsonl
exist -- see hal_vlm_fusion.py's docstring for how each is produced):
    python -m src.boundary.hal_vlm_crosstab \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --vlm_pred /workspace/tr1/results/auditor/atomic_v4_full72_pred.jsonl \
        --out /workspace/tr1/results/hal/crosstab_72.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.hal_vlm_fusion import build_event_rows, grouped_loro_predict


def precision_coverage_curve(y, p, thresholds, side):
    """side='valid': precision/coverage of calling p>=thr 'valid'.
    side='spurious': precision/coverage of calling p<=thr 'spurious'."""
    mask = ~np.isnan(p)
    y_, p_ = y[mask], p[mask]
    rows = []
    for thr in thresholds:
        sel = (p_ >= thr) if side == "valid" else (p_ <= thr)
        n = int(sel.sum())
        if n == 0:
            rows.append({"thr": thr, "n": 0, "precision": None, "coverage": 0.0})
            continue
        target = 1 if side == "valid" else 0
        prec = float((y_[sel] == target).mean())
        rows.append({"thr": thr, "n": n, "precision": prec, "coverage": float(sel.mean())})
    return rows


def pick_threshold(curve, target_precision=0.90, min_n=3):
    """Loosest threshold (max coverage) that still clears target_precision
    with at least min_n samples; None if nothing qualifies."""
    qualifying = [r for r in curve if r["precision"] is not None
                 and r["precision"] >= target_precision and r["n"] >= min_n]
    if not qualifying:
        return None
    return max(qualifying, key=lambda r: r["coverage"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--vlm_pred", required=True)
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...)")
    ap.add_argument("--context", help="context jsonl (default: committed data/gold/...)")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--variance_half", type=float, default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--thr_valid", type=float, default=None,
                    help="override the auto-picked HAL 'valid' threshold")
    ap.add_argument("--thr_spurious", type=float, default=None,
                    help="override the auto-picked HAL 'spurious' threshold")
    ap.add_argument("--out")
    a = ap.parse_args()

    gold_path, ctx_path = S.default_gold_paths()
    gold = S.load_gold(a.gold or gold_path)
    ctx = S.load_context(a.context or ctx_path)
    by_rid = load_feature_caches(a.feat_cache)
    vlm_pred = {}
    with open(a.vlm_pred, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                vlm_pred[d["event_id"]] = d

    rows = build_event_rows(gold, ctx, by_rid, vlm_pred, short_half=a.short_half,
                            context_half=a.context_half, variance_half=a.variance_half)
    n_vlm = sum(r["has_vlm"] for r in rows)
    print(f"usable events: {len(rows)}  (VLM coverage: {n_vlm}/{len(rows)})")

    y = np.array([r["y"] for r in rows], dtype=float)
    groups = [r["recording_id"] for r in rows]
    X_hal = np.array([r["hal"] for r in rows], dtype=float)
    p_hal = grouped_loro_predict(X_hal, y, groups, l2=a.l2)

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    curve_valid = precision_coverage_curve(y, p_hal, thresholds, side="valid")
    curve_spur = precision_coverage_curve(y, p_hal, thresholds, side="spurious")

    print(f"\n-- HAL 'valid' precision/coverage curve (target precision={a.target_precision}) --")
    for r in curve_valid:
        print(f"  thr>={r['thr']:.2f}  n={r['n']:<3} precision={_fmt(r['precision'])}  coverage={r['coverage']:.3f}")
    print(f"\n-- HAL 'spurious' precision/coverage curve --")
    for r in curve_spur:
        print(f"  thr<={r['thr']:.2f}  n={r['n']:<3} precision={_fmt(r['precision'])}  coverage={r['coverage']:.3f}")

    pick_v = pick_threshold(curve_valid, a.target_precision)
    pick_s = pick_threshold(curve_spur, a.target_precision)
    thr_valid = a.thr_valid if a.thr_valid is not None else (pick_v["thr"] if pick_v else 0.95)
    thr_spur = a.thr_spurious if a.thr_spurious is not None else (pick_s["thr"] if pick_s else 0.05)

    def _thr_desc(pick):
        if pick is None:
            return "NO threshold cleared target precision -- defaulted, likely near-zero coverage"
        return f"auto, precision={_fmt(pick['precision'])} n={pick['n']}"

    print(f"\nchosen HAL thresholds: valid>={thr_valid:.2f} ({_thr_desc(pick_v)}), "
          f"spurious<={thr_spur:.2f} ({_thr_desc(pick_s)})")

    def hal_bucket(p):
        if np.isnan(p):
            return "review"
        if p >= thr_valid:
            return "valid"
        if p <= thr_spur:
            return "spurious"
        return "review"

    def vlm_bucket(tt):
        if tt == "valid":
            return "valid"
        if tt == "spurious":
            return "spurious"
        return "other"  # ambiguous/unresolved/no VLM coverage

    cells = {}
    for r, p in zip(rows, p_hal):
        hb, vb = hal_bucket(p), vlm_bucket(r["vlm_temporal_truth"])
        cells.setdefault((hb, vb), []).append(r)

    print("\n-- HAL x VLM cross-tab (rows: n / gold-valid / gold-spurious / valid-rate) --")
    hal_order = ["valid", "review", "spurious"]
    vlm_order = ["valid", "other", "spurious"]
    header = "HAL\\VLM".ljust(10) + "".join(v.ljust(22) for v in vlm_order)
    print(header)
    summary_cells = {}
    for hb in hal_order:
        line = hb.ljust(10)
        for vb in vlm_order:
            grp = cells.get((hb, vb), [])
            n = len(grp)
            n_valid = sum(r["y"] == 1 for r in grp)
            n_spur = n - n_valid
            rate = f"{n_valid/n:.2f}" if n else "n/a"
            line += f"n={n},v={n_valid},s={n_spur},r={rate}".ljust(22)
            summary_cells[f"{hb}|{vb}"] = {"n": n, "gold_valid": n_valid, "gold_spurious": n_spur,
                                          "valid_rate": (n_valid / n) if n else None,
                                          "event_ids": [r["event_id"] for r in grp]}
        print(line)

    disagreement = cells.get(("valid", "spurious"), [])
    print(f"\n-- highest-value review bucket: HAL=valid, VLM=spurious (n={len(disagreement)}) --")
    for r in disagreement:
        print(f"  {r['event_id']}  gold={'valid' if r['y'] == 1 else 'spurious'}")

    # --- frozen policy: selective boundary verifier, no auto-remove --------
    # HAL >= thr_valid -> provisional_keep (VLM attached only as an agree/
    # disagree flag, NEVER used to override -- the disagreement cell above
    # showed VLM's 'spurious' call is WRONG most of the time when HAL is
    # confidently 'valid': 5/6 gold-valid in this cross-tab). Everything else
    # -> review. There is deliberately NO provisional_remove branch: no HAL
    # low-score bucket cleared the target precision on the spurious side
    # (see the 'spurious' curve above), so a remove decision currently has no
    # trustworthy evidence behind it at all.
    keep_group = [r for r, p in zip(rows, p_hal) if hal_bucket(p) == "valid"]
    review_group = [r for r, p in zip(rows, p_hal) if hal_bucket(p) != "valid"]
    n_keep, n_review = len(keep_group), len(review_group)
    keep_valid = sum(r["y"] == 1 for r in keep_group)
    keep_precision = keep_valid / n_keep if n_keep else None
    keep_vlm_agree = sum(r["vlm_temporal_truth"] == "valid" for r in keep_group)
    keep_vlm_disagree = sum(r["vlm_temporal_truth"] == "spurious" for r in keep_group)
    print("\n-- FROZEN POLICY: selective boundary verifier (HAL>=%.2f -> provisional_keep, "
          "else -> review; VLM advisory-only, never overrides; no auto-remove) --" % thr_valid)
    print(f"  provisional_keep : n={n_keep}  precision={_fmt(keep_precision)}  "
          f"(coverage={n_keep/len(rows):.3f})  "
          f"[VLM agrees: {keep_vlm_agree}, VLM disagrees: {keep_vlm_disagree} -- disagreement NOT acted on]")
    print(f"  review           : n={n_review}  (coverage={n_review/len(rows):.3f})")
    print("  provisional_remove: NOT DEFINED -- no HAL threshold reaches target precision on "
          "the spurious side yet (see 'spurious' curve above)")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({
                "n_usable": len(rows), "n_vlm_coverage": n_vlm,
                "thr_valid": thr_valid, "thr_spurious": thr_spur,
                "curve_valid": curve_valid, "curve_spurious": curve_spur,
                "cells": summary_cells,
                "frozen_policy": {
                    "provisional_keep": {"n": n_keep, "precision": keep_precision,
                                        "coverage": n_keep / len(rows),
                                        "vlm_agree": keep_vlm_agree, "vlm_disagree": keep_vlm_disagree,
                                        "event_ids": [r["event_id"] for r in keep_group]},
                    "review": {"n": n_review, "coverage": n_review / len(rows)},
                    "provisional_remove": None,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")
        try:
            from src.eval.run_manifest import write_manifest
            write_manifest(a.out, input_paths=[gold_path, ctx_path, a.vlm_pred] + a.feat_cache)
        except Exception as e:
            print(f"[manifest] skipped ({e})")


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    main()
