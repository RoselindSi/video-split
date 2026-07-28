"""Held-out validation of the FROZEN HAL selective boundary verifier.

The policy under test was developed entirely on the original 72-event gold
set (25 recordings): fit a class-weighted logistic regression on 5 cheap HAL
temporal features, and auto-keep any candidate scoring >= 0.85. On its own
development data that reached 0.90 precision -- but at n=10, which is far too
small to deploy. This script answers the only question that matters next:

    does `HAL >= 0.85 -> provisional_keep` still hold ~0.90 precision on
    116 NEWLY audited events from 23 recordings the scorer has never seen?

This is a TEST, not another round of development. The script therefore
enforces the separation in code rather than trusting discipline:

  - The scorer is fit ONLY on rows with split == dev_original72. Test rows
    are never in any training matrix, and there is no cross-validation on
    the test half (LORO on the test set would NOT be held-out validation --
    it would be fitting on test recordings and reporting on test recordings).
  - The threshold is a frozen constant (--threshold, default 0.85). The
    script deliberately does NOT sweep thresholds on test or pick a
    best-performing one; --sweep_test_thresholds_UNSAFE exists only to make
    a post-hoc sweep visibly, namedly unsafe if someone runs it, and its
    output is labelled as not-a-validation-result.
  - Feature windows, imputation, standardization and class weighting all
    come from the dev fit; the test rows are transformed with the DEV
    statistics, never re-standardized on their own distribution.

Reported (per the review's requested table):
  provisional_keep n / precision / coverage, false keeps, review rate, and a
  Wilson 95% confidence interval on the keep precision (Wilson, not the
  normal approximation -- at n~10-30 and p near 0.9 the normal interval is
  badly wrong and can exceed 1.0). Plus a per-recording breakdown, because
  "0.90 precision" concentrated in 2-3 recordings would overstate how well
  this generalizes.

Verdict thresholds follow the review's own criteria: >=0.90 -> policy
validated; 0.80-0.90 -> ranking value only, not auto-keep; <0.80 -> the dev
result was development-set luck.

Usage:
    python -m src.boundary.hal_heldout_validate \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --out /workspace/tr1/results/hal/heldout_validation.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.hal_vlm_fusion import (
    HAL_FEATURE_NAMES, build_event_rows, fit_full_hal_model, score_hal_model,
)

DEV_SPLIT = "dev_original72"
TEST_SPLIT = "test_batch2"


def wilson_interval(k, n, z=1.96):
    """95% CI for a binomial proportion. Wilson score interval: correct at
    small n and at p near 0/1, unlike the normal approximation which can
    produce bounds outside [0,1] exactly where this experiment lives."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def evaluate_policy(rows, scores, threshold):
    """rows: list of dicts with 'y' (1=valid) and 'recording_id'.
    scores: parallel array of P(valid). Returns the frozen-policy metrics."""
    keep_idx = [i for i, s in enumerate(scores) if not np.isnan(s) and s >= threshold]
    n_total = len(rows)
    n_keep = len(keep_idx)
    keep_correct = sum(rows[i]["y"] == 1 for i in keep_idx)
    # keep the INDEX alongside the row: rows are plain dicts, so looking one
    # up later with list.index() would compare by value and could return the
    # wrong position if two events happen to have identical field values.
    false_keeps = [(i, rows[i]) for i in keep_idx if rows[i]["y"] == 0]
    precision = keep_correct / n_keep if n_keep else None
    lo, hi = wilson_interval(keep_correct, n_keep)
    return {
        "n_decisive": n_total,
        "provisional_keep_n": n_keep,
        "provisional_keep_precision": precision,
        "precision_ci95_wilson": [lo, hi],
        "coverage": n_keep / n_total if n_total else 0.0,
        "false_keeps_n": len(false_keeps),
        "review_n": n_total - n_keep,
        "review_rate": (n_total - n_keep) / n_total if n_total else 0.0,
        "_keep_idx": keep_idx,
        "_false_keeps": false_keeps,
    }


def verdict(precision, n_keep):
    if precision is None or n_keep == 0:
        return ("NO DECISION", "the frozen threshold auto-kept nothing on the test set -- "
                               "coverage is zero, so precision is undefined")
    if precision >= 0.90:
        return ("A: VALIDATED", "frozen policy holds on unseen recordings -- usable as a "
                                "one-sided auto-keep mechanism")
    if precision >= 0.80:
        return ("B: RANKING ONLY", "useful for lowering review priority, but NOT trustworthy "
                                    "as automatic keep")
    return ("C: FAILED", "the development-set 9/10 was largely luck -- do NOT re-pick a "
                          "threshold on this test set; analyse the false positives, change "
                          "the features, and validate on a NEW batch")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl",
                    help="merged gold jsonl carrying the `split` field")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="FROZEN decision threshold, chosen on the dev set before the test "
                         "labels existed. Changing this after seeing test results invalidates "
                         "the experiment.")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--variance_half", type=float, default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--sweep_test_thresholds_UNSAFE", action="store_true",
                    help="post-hoc threshold sweep ON THE TEST SET. This is not a validation "
                         "result and is labelled as such in the output; it exists only for "
                         "failure analysis after a verdict has already been recorded.")
    ap.add_argument("--out")
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    if not any("split" in g for g in gold):
        raise SystemExit(f"{a.gold} has no `split` field -- regenerate it with "
                         f"`python -m src.auditor.export_gold_v2 --merged ...`")
    by_rid = load_feature_caches(a.feat_cache)

    dev_gold = [g for g in gold if g.get("split") == DEV_SPLIT]
    test_gold = [g for g in gold if g.get("split") == TEST_SPLIT]
    dev_rows = build_event_rows(dev_gold, ctx, by_rid, {}, short_half=a.short_half,
                                context_half=a.context_half, variance_half=a.variance_half)
    test_rows = build_event_rows(test_gold, ctx, by_rid, {}, short_half=a.short_half,
                                 context_half=a.context_half, variance_half=a.variance_half)

    dev_recs = {r["recording_id"] for r in dev_rows}
    test_recs = {r["recording_id"] for r in test_rows}
    leak = dev_recs & test_recs
    print("=" * 78)
    print("HELD-OUT VALIDATION of the frozen HAL selective boundary verifier")
    print("=" * 78)
    print(f"  DEV  (fit here)     : {len(dev_rows)} decisive events / {len(dev_recs)} recordings")
    print(f"  TEST (evaluate here): {len(test_rows)} decisive events / {len(test_recs)} recordings")
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} recording(s) appear in BOTH splits {sorted(leak)} -- "
                         f"this is not a held-out test.")
    print(f"  recording overlap   : NONE (verified)")
    print(f"  frozen threshold    : {a.threshold} (chosen on DEV, before these labels existed)")
    if len(test_rows) == 0:
        raise SystemExit("ABORT: no usable test rows (label + HAL feature coverage). "
                         "Is the part_02 feature cache included in --feat_cache?")

    X_dev = np.array([r["hal"] for r in dev_rows], dtype=float)
    y_dev = np.array([r["y"] for r in dev_rows], dtype=float)
    model = fit_full_hal_model(X_dev, y_dev, l2=a.l2)
    print(f"\nfit on DEV only: {int(y_dev.sum())} valid / {int(len(y_dev) - y_dev.sum())} spurious")
    print("  feature weights (standardized):")
    for name, w in zip(HAL_FEATURE_NAMES, model["w"]):
        print(f"    {name:<26} {w:+.4f}")

    test_scores = np.array([score_hal_model(model, r["hal"]) for r in test_rows])
    dev_scores = np.array([score_hal_model(model, r["hal"]) for r in dev_rows])

    m_test = evaluate_policy(test_rows, test_scores, a.threshold)
    m_dev = evaluate_policy(dev_rows, dev_scores, a.threshold)

    print("\n" + "-" * 78)
    print("FROZEN POLICY on HELD-OUT TEST (the result)")
    print("-" * 78)
    lo, hi = m_test["precision_ci95_wilson"]
    print(f"  decisive rows                 {m_test['n_decisive']}")
    print(f"  provisional_keep n            {m_test['provisional_keep_n']}")
    print(f"  provisional_keep precision    {_fmt(m_test['provisional_keep_precision'])}"
          f"   95% CI [{_fmt(lo)}, {_fmt(hi)}]  (Wilson)")
    print(f"  coverage                      {_fmt(m_test['coverage'])}")
    print(f"  false keeps                   {m_test['false_keeps_n']}")
    print(f"  review rate                   {_fmt(m_test['review_rate'])}")
    print(f"  provisional_remove            NOT DEFINED (no trustworthy negative evidence)")

    print("\n  [reference only] same frozen model+threshold scored back on DEV "
          "(in-sample, NOT a validation number):")
    print(f"    keep n={m_dev['provisional_keep_n']}  precision={_fmt(m_dev['provisional_keep_precision'])}"
          f"  coverage={_fmt(m_dev['coverage'])}")

    # ---- per-recording concentration -------------------------------------
    keep_by_rec = Counter(test_rows[i]["recording_id"] for i in m_test["_keep_idx"])
    err_by_rec = Counter(r["recording_id"] for _, r in m_test["_false_keeps"])
    print("\n" + "-" * 78)
    print("PER-RECORDING distribution of auto-keeps (is 'precision' concentrated?)")
    print("-" * 78)
    print(f"  recordings producing >=1 provisional_keep: {len(keep_by_rec)} / {len(test_recs)}")
    if keep_by_rec:
        top = keep_by_rec.most_common()
        print(f"  keeps per recording: " + ", ".join(f"{r}={c}" for r, c in top))
        max_share = top[0][1] / m_test["provisional_keep_n"]
        print(f"  largest single recording's share of all keeps: {max_share:.1%}"
              f"{'   <-- concentrated, treat coverage with caution' if max_share > 0.4 else ''}")
    if err_by_rec:
        print(f"  false keeps by recording: " + ", ".join(f"{r}={c}" for r, c in err_by_rec.most_common()))
        if len(err_by_rec) == 1 and m_test["false_keeps_n"] > 1:
            print("    (all false keeps come from ONE recording -- look at it specifically "
                  "before concluding the feature set is broken)")

    # ---- every high-scoring false positive, for targeted analysis --------
    print("\n" + "-" * 78)
    print(f"FALSE KEEPS (HAL >= {a.threshold} but gold says spurious) -- inspect these clips")
    print("-" * 78)
    if not m_test["_false_keeps"]:
        print("  none")
    for i, r in m_test["_false_keeps"]:
        feats = ", ".join(f"{n}={_fmt(v, 5)}" for n, v in zip(HAL_FEATURE_NAMES, r["hal"]))
        print(f"  {r['event_id']}")
        print(f"      score={_fmt(test_scores[i])}  {feats}")

    v_code, v_text = verdict(m_test["provisional_keep_precision"], m_test["provisional_keep_n"])
    print("\n" + "=" * 78)
    print(f"VERDICT  {v_code}")
    print(f"  {v_text}")
    if m_test["provisional_keep_n"] and m_test["provisional_keep_n"] < 15:
        print(f"  CAVEAT: only {m_test['provisional_keep_n']} auto-keeps -- the CI above is wide; "
              f"treat the point estimate as provisional even if it clears 0.90.")
    print("=" * 78)

    if a.sweep_test_thresholds_UNSAFE:
        print("\n!! POST-HOC THRESHOLD SWEEP ON THE TEST SET -- NOT A VALIDATION RESULT !!")
        print("!! Any threshold picked from this table has been fitted to the test labels")
        print("!! and would need a fresh, unseen batch to validate. Failure analysis only.")
        for thr in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            m = evaluate_policy(test_rows, test_scores, thr)
            print(f"    thr>={thr:.2f}  n={m['provisional_keep_n']:<3} "
                  f"precision={_fmt(m['provisional_keep_precision'])}  "
                  f"coverage={_fmt(m['coverage'])}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        payload = {
            "threshold_frozen": a.threshold,
            "dev": {"n_decisive": len(dev_rows), "n_recordings": len(dev_recs),
                    "in_sample_keep_n": m_dev["provisional_keep_n"],
                    "in_sample_precision": m_dev["provisional_keep_precision"]},
            "test": {k: v for k, v in m_test.items() if not k.startswith("_")},
            "test_n_recordings": len(test_recs),
            "keeps_per_recording": dict(keep_by_rec),
            "false_keeps": [{"event_id": r["event_id"], "recording_id": r["recording_id"],
                            "score": float(test_scores[i]),
                            "features": dict(zip(HAL_FEATURE_NAMES, r["hal"]))}
                           for i, r in m_test["_false_keeps"]],
            "feature_weights": dict(zip(HAL_FEATURE_NAMES, [float(w) for w in model["w"]])),
            "verdict": v_code, "verdict_text": v_text,
        }
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2,
                      default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
        print(f"\nwrote {a.out}")
        try:
            from src.eval.run_manifest import write_manifest
            write_manifest(a.out, input_paths=[a.gold, a.context] + a.feat_cache,
                           extra={"threshold": a.threshold, "dev_split": DEV_SPLIT,
                                  "test_split": TEST_SPLIT})
        except Exception as e:
            print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
