"""Explain WHY the frozen HAL verifier failed held-out validation, before
changing anything about it.

Held-out result (batch2, 23 unseen recordings): `HAL >= 0.85 ->
provisional_keep` gave precision 0.767 (23/30), Wilson 95% CI
[0.591, 0.882] -- the 0.90 target sits ABOVE the interval, so this is a
rejection, not a near miss. Critically, several false keeps scored
0.965-1.000, i.e. the errors are inside the region the model is most
confident about, so raising the threshold cannot fix it. And
recording_000406 contributed 3 keeps, all 3 wrong -- a systematic
per-recording failure, not scattered noise.

This script does NOT re-pick a threshold, retrain, or touch the policy. It
produces the two tables needed to decide which of four causes is actually
responsible:
  (a) feature definitions don't capture "two stable, different states",
  (b) recording-level distribution shift / nuisance variation,
  (c) standardization / calibration (e.g. intercept dominating),
  (d) the frozen Qwen representation lacks the semantic separation at all.

Output 1 -- hal_feature_shift_dev_vs_test.csv
    Per feature, distribution (n, missing, median, IQR, mean, std) sliced by
    split (dev/test) and by gold truth (all/valid/spurious), plus per
    recording. Also: what fraction of TEST values fall outside the DEV
    observed range, and the mean standardized z (using the DEV mu/sigma the
    model actually applies) -- a large |mean z| on TEST means the model is
    extrapolating, which is (b)/(c), not (a).

Output 2 -- hal_false_keep_contributions.csv
    For every false keep, plus matched true keeps (same recording first,
    then nearest HAL score, then same source_category), a full decomposition
    of the decision: raw feature -> imputed -> standardized z -> weight x z
    contribution -> summed logit + intercept -> probability. This is what
    answers "why did context_change=0.00096 produce score=0.977": either one
    contribution term dominates (feature/scaling problem) or the intercept
    does (calibration problem).

    Each row also carries a RECORDING-LEVEL BASELINE for every feature:
    the median and MAD of that feature sampled on a regular time grid across
    the whole recording (not just at audited events), and the candidate's
    z_local against it. This directly tests the mentor's Z_local idea -- "is
    this candidate anomalous relative to its OWN recording's normal motion
    level?" -- and is the specific measurement that will confirm or rule out
    recording_000406 being a feature outlier.

Usage:
    python -m src.boundary.hal_failure_diagnostic \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --out_dir /workspace/tr1/results/hal/failure_diagnostic
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at
from src.boundary.hal_vlm_fusion import (
    HAL_FEATURE_NAMES, build_event_rows, fit_full_hal_model, score_hal_model,
)

DEV_SPLIT = "dev_original72"
TEST_SPLIT = "test_batch2"


def _stats(vals):
    """vals: list possibly containing None/NaN."""
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    n_missing = len(vals) - len(arr)
    if len(arr) == 0:
        return {"n": 0, "n_missing": n_missing, "median": None, "q25": None,
                "q75": None, "mean": None, "std": None, "min": None, "max": None}
    return {"n": len(arr), "n_missing": n_missing,
            "median": float(np.median(arr)), "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)), "mean": float(arr.mean()),
            "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def recording_baseline(rec, stride_s, short_half, context_half, variance_half,
                       max_samples=400):
    """Median/MAD of each HAL feature sampled on a regular grid across the
    WHOLE recording -- the recording's own 'normal' level of representation
    change, independent of which moments happened to be audited. This is the
    denominator the mentor's Z_local needs; without it, a candidate's raw
    feature value can only be compared against other recordings, which is
    exactly the comparison that breaks under per-recording nuisance shift."""
    times = rec["times"]
    t0, t1 = float(times[0]), float(times[-1])
    grid = np.arange(t0 + context_half, max(t0 + context_half, t1 - 2 * context_half), stride_s)
    if len(grid) > max_samples:
        grid = grid[np.linspace(0, len(grid) - 1, max_samples).astype(int)]
    acc = defaultdict(list)
    for t in grid:
        f = hal_features_at(rec["feats"], times, float(t), short_half=short_half,
                            context_half=context_half, variance_half=variance_half)
        for k, v in f.items():
            if v is not None:
                acc[k].append(float(v))
    out = {}
    for k in HAL_FEATURE_NAMES:
        vals = np.array(acc.get(k, []), dtype=float)
        if len(vals) == 0:
            out[k] = (None, None, 0)
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        out[k] = (med, mad, len(vals))
    return out


def match_true_keeps(false_keep, true_keeps, n_match=1):
    """Pick the most comparable true keeps for a given false keep, using the
    review's stated priority: same recording first, then closest HAL score,
    then same source_category. Returns up to n_match rows."""
    def sort_key(tk):
        return (
            0 if tk["recording_id"] == false_keep["recording_id"] else 1,
            0 if tk.get("source_category") == false_keep.get("source_category") else 1,
            abs(tk["score"] - false_keep["score"]),
        )
    return sorted(true_keeps, key=sort_key)[:n_match]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="the frozen threshold under investigation (NOT being re-tuned here)")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--variance_half", type=float, default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--baseline_stride_s", type=float, default=2.0,
                    help="time grid spacing for the per-recording baseline sampling")
    ap.add_argument("--n_match", type=int, default=2,
                    help="matched true keeps per false keep")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    os.makedirs(a.out_dir, exist_ok=True)

    cat_by_eid = {g["event_id"]: g.get("source_category") for g in gold}

    dev_gold = [g for g in gold if g.get("split") == DEV_SPLIT]
    test_gold = [g for g in gold if g.get("split") == TEST_SPLIT]
    fk = dict(short_half=a.short_half, context_half=a.context_half,
              variance_half=a.variance_half)
    dev_rows = build_event_rows(dev_gold, ctx, by_rid, {}, **fk)
    test_rows = build_event_rows(test_gold, ctx, by_rid, {}, **fk)
    for r in dev_rows + test_rows:
        r["source_category"] = cat_by_eid.get(r["event_id"])

    X_dev = np.array([r["hal"] for r in dev_rows], dtype=float)
    y_dev = np.array([r["y"] for r in dev_rows], dtype=float)
    model = fit_full_hal_model(X_dev, y_dev, l2=a.l2)
    for r in dev_rows:
        r["score"] = score_hal_model(model, r["hal"])
    for r in test_rows:
        r["score"] = score_hal_model(model, r["hal"])

    print(f"dev rows={len(dev_rows)}  test rows={len(test_rows)}")
    print("model (fit on DEV only):  intercept b = %+.4f" % model["b"])
    for name, w, mu, sd in zip(HAL_FEATURE_NAMES, model["w"], model["mu"], model["sigma"]):
        print(f"  {name:<26} w={w:+.4f}  dev_mu={mu:.6g}  dev_sigma={sd:.6g}")

    # ---------------- table 1: feature shift ------------------------------
    shift_path = os.path.join(a.out_dir, "hal_feature_shift_dev_vs_test.csv")
    dev_ranges = {}
    for j, name in enumerate(HAL_FEATURE_NAMES):
        s = _stats([r["hal"][j] for r in dev_rows])
        dev_ranges[name] = (s["min"], s["max"])

    rows_out = []
    for scope, recording_id, rowset_by_split in [
        ("overall", "", {"dev": dev_rows, "test": test_rows}),
    ] + [
        ("recording", rid, {"dev": [r for r in dev_rows if r["recording_id"] == rid],
                            "test": [r for r in test_rows if r["recording_id"] == rid]})
        for rid in sorted({r["recording_id"] for r in dev_rows + test_rows})
    ]:
        for split, rset in rowset_by_split.items():
            if not rset:
                continue
            for subset, sel in [("all", rset),
                                ("valid", [r for r in rset if r["y"] == 1]),
                                ("spurious", [r for r in rset if r["y"] == 0])]:
                if not sel:
                    continue
                for j, name in enumerate(HAL_FEATURE_NAMES):
                    vals = [r["hal"][j] for r in sel]
                    st = _stats(vals)
                    lo, hi = dev_ranges[name]
                    finite = [v for v in vals if v is not None and not np.isnan(v)]
                    if split == "test" and finite and lo is not None:
                        outside = sum(1 for v in finite if v < lo or v > hi) / len(finite)
                    else:
                        outside = None
                    mu, sd = model["mu"][j], model["sigma"][j]
                    mean_z = ((st["mean"] - mu) / sd) if st["mean"] is not None else None
                    rows_out.append({
                        "scope": scope, "recording_id": recording_id, "split": split,
                        "subset": subset, "feature": name, **st,
                        "dev_range_min": lo, "dev_range_max": hi,
                        "pct_outside_dev_range": outside,
                        "mean_z_using_dev_scaler": mean_z,
                    })
    with open(shift_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {shift_path}  ({len(rows_out)} rows)")

    print("\n-- overall DEV vs TEST (subset=all) --")
    for name in HAL_FEATURE_NAMES:
        d = next(r for r in rows_out if r["scope"] == "overall" and r["split"] == "dev"
                 and r["subset"] == "all" and r["feature"] == name)
        t = next((r for r in rows_out if r["scope"] == "overall" and r["split"] == "test"
                  and r["subset"] == "all" and r["feature"] == name), None)
        if t is None:
            continue
        print(f"  {name:<26} dev median={d['median']:.6g}  test median={t['median']:.6g}  "
              f"test mean_z={t['mean_z_using_dev_scaler']:+.3f}  "
              f"test outside dev range={t['pct_outside_dev_range']:.1%}")

    # ---------------- table 2: false-keep contributions -------------------
    keeps = [r for r in test_rows if r["score"] >= a.threshold]
    false_keeps = [r for r in keeps if r["y"] == 0]
    true_keeps = [r for r in keeps if r["y"] == 1]
    print(f"\ntest keeps at thr={a.threshold}: {len(keeps)}  "
          f"(false={len(false_keeps)}, true={len(true_keeps)})")

    need_baseline = {r["recording_id"] for r in false_keeps}
    for r in false_keeps:
        need_baseline.update(m["recording_id"] for m in match_true_keeps(r, true_keeps, a.n_match))
    print(f"computing per-recording baselines for {len(need_baseline)} recordings "
          f"(stride={a.baseline_stride_s}s)...")
    baselines = {}
    for rid in sorted(need_baseline):
        rec = by_rid.get(rid)
        if rec is not None:
            baselines[rid] = recording_baseline(rec, a.baseline_stride_s, a.short_half,
                                                a.context_half, a.variance_half)

    def decompose(r):
        x = np.array([np.nan if v is None else v for v in r["hal"]], dtype=float)
        x_imp = np.where(np.isnan(x), model["col_mean"], x)
        z = (x_imp - model["mu"]) / model["sigma"]
        contrib = model["w"] * z
        logit = float(contrib.sum() + model["b"])
        return x, x_imp, z, contrib, logit

    contrib_rows = []
    for gi, fkrow in enumerate(sorted(false_keeps, key=lambda r: -r["score"]), 1):
        group = [("false_keep", fkrow)] + [("matched_true_keep", m)
                                            for m in match_true_keeps(fkrow, true_keeps, a.n_match)]
        for role, r in group:
            x, x_imp, z, contrib, logit = decompose(r)
            base = baselines.get(r["recording_id"], {})
            row = {
                "group_id": gi, "role": role, "event_id": r["event_id"],
                "recording_id": r["recording_id"],
                "source_category": r.get("source_category"),
                "gold_temporal_truth": "valid" if r["y"] == 1 else "spurious",
                "score": round(r["score"], 6), "logit": round(logit, 6),
                "intercept": round(float(model["b"]), 6),
            }
            for j, name in enumerate(HAL_FEATURE_NAMES):
                med, mad, nb = base.get(name, (None, None, 0))
                row[f"raw_{name}"] = None if np.isnan(x[j]) else float(x[j])
                row[f"imputed_{name}"] = float(x_imp[j])
                row[f"z_{name}"] = float(z[j])
                row[f"contrib_{name}"] = float(contrib[j])
                row[f"recbase_median_{name}"] = med
                row[f"recbase_mad_{name}"] = mad
                row[f"zlocal_{name}"] = (float((x_imp[j] - med) / (mad + 1e-12))
                                         if med is not None and mad is not None else None)
            row["recbase_n_samples"] = base.get(HAL_FEATURE_NAMES[0], (None, None, 0))[2]
            contrib_rows.append(row)

    contrib_path = os.path.join(a.out_dir, "hal_false_keep_contributions.csv")
    if contrib_rows:
        with open(contrib_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(contrib_rows[0].keys()))
            w.writeheader()
            w.writerows(contrib_rows)
        print(f"wrote {contrib_path}  ({len(contrib_rows)} rows, "
              f"{len(false_keeps)} groups)")
    else:
        print("no false keeps at this threshold -- nothing to decompose")

    # readable summary of what drives each false keep
    print("\n-- what drives each FALSE keep (largest |contribution| first) --")
    for r in sorted(false_keeps, key=lambda r: -r["score"]):
        x, x_imp, z, contrib, logit = decompose(r)
        order = np.argsort(-np.abs(contrib))
        parts = ", ".join(f"{HAL_FEATURE_NAMES[j]}={contrib[j]:+.3f}(z={z[j]:+.2f})" for j in order)
        print(f"  {r['event_id']}  score={r['score']:.3f} logit={logit:+.3f} "
              f"intercept={model['b']:+.3f}")
        print(f"      {parts}")

    intercept_share = abs(model["b"]) / max(1e-9, abs(model["b"]) + float(np.mean(
        [np.abs(decompose(r)[3]).sum() for r in false_keeps]))) if false_keeps else None
    if intercept_share is not None:
        print(f"\n  intercept vs mean |feature contribution| share on false keeps: "
              f"{intercept_share:.1%}"
              f"{'  <-- intercept dominates: calibration problem, not feature problem' if intercept_share > 0.5 else ''}")

    summary = {
        "threshold": a.threshold,
        "n_dev": len(dev_rows), "n_test": len(test_rows),
        "n_keeps": len(keeps), "n_false_keeps": len(false_keeps),
        "intercept": float(model["b"]),
        "weights": dict(zip(HAL_FEATURE_NAMES, [float(w) for w in model["w"]])),
        "dev_mu": dict(zip(HAL_FEATURE_NAMES, [float(v) for v in model["mu"]])),
        "dev_sigma": dict(zip(HAL_FEATURE_NAMES, [float(v) for v in model["sigma"]])),
        "outputs": {"feature_shift_csv": shift_path,
                    "false_keep_contributions_csv": contrib_path if contrib_rows else None},
    }
    sum_path = os.path.join(a.out_dir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {sum_path}")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(sum_path, input_paths=[a.gold, a.context] + a.feat_cache,
                       extra={"threshold": a.threshold, "note": "diagnostic only, no retuning"})
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
