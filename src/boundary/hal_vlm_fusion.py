"""HAL-only vs VLM-atomic-only vs fusion, on the 72-event gold set, with
GROUPED (leave-one-recording-out) cross-validation -- never split by event,
since events from the same recording share context and would leak.

This is the experiment the HAL diagnostic (hal_diagnostic.py) was step 1 of.
That step showed context_change/change_persistence carry real (AUC~0.73,
not spectacular) signal for valid-vs-spurious; the visual auditor
(run_visual_auditor.py) on its own currently over-removes real boundaries
(hard-slice(1) recall as low as 30-40% on small subsets). The question here
is narrow and concrete: does COMBINING the two sources of evidence recover
more of the deleted real boundaries without giving back the motion-hard-
negative rejections the atomic VLM prompts already achieved?

Three arms, same grouped CV, same three decision thresholds:
  A. HAL-only    : 5 features from hal_features.py (short_change,
                   context_change, change_persistence, left/right_internal_
                   variance), logistic regression.
  B. VLM-only    : the atomic Pass A signals the auditor ALREADY produces
                   (semantic_action_changed, motion_change_without_semantic_
                   change, visual_evidence), numerically encoded, logistic
                   regression. NOTE: this is a v1 approximation of the
                   richer 5-field atomic scheme sketched in the design
                   review (transition_visible / object_state_changed /
                   interaction_target_changed aren't asked by prompts.py
                   yet) -- if fusion looks promising, adding those fields to
                   Pass A is a natural, cheap next step.
  C. Fusion      : both feature sets concatenated.

Target: temporal_truth == valid (1) vs spurious (0); ambiguous/unresolved
excluded (matches hal_diagnostic.py's own scoping).

Logistic regression is a plain from-scratch full-batch gradient descent (no
sklearn dependency) -- deliberately simple and inspectable given n~60-70.
Per-fold: mean-impute missing HAL values and standardize using ONLY the
training fold's statistics (no leakage from the held-out recording).

Metrics per arm, using the review-band decision rule (>=0.75 -> valid,
<=0.25 -> spurious, otherwise -> review, matching the mentor's proposed
thresholds):
  valid_recall, motion_hard_negative_recall (=spurious recall),
  balanced_accuracy, macro_f1, review_rate.

KNOWN GAP: a duration-stratified "fast-boundary false-removal rate" (does
fusion specifically save the <1s/1-2s real actions the atomic VLM currently
over-deletes) is NOT computed here -- it needs each event's underlying
segment duration, which the current gold/context export does not carry.
Would need to join back to the original recording segmentation JSON.

Usage (server, CPU-only after the HAL feature caches + a VLM pred jsonl
exist -- the VLM pred file may cover fewer than 72 events; arm A still runs
on the full HAL-covered set, arms B/C automatically restrict to whatever
subset actually has VLM predictions, with the coverage printed):
    python -m src.boundary.hal_vlm_fusion \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --vlm_pred /workspace/tr1/results/auditor/atomic_v4_8b_test_pred.jsonl \
        --out /workspace/tr1/results/hal/fusion_report.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at

HAL_FEATURE_NAMES = ["short_change", "context_change", "change_persistence",
                     "left_internal_variance", "right_internal_variance"]

_TRI = {"yes": 1.0, "no": 0.0, "unclear": 0.5, "clear": 1.0,
       "partial": 0.5, "insufficient": 0.0}


def _encode_vlm_atomic(pass_a):
    """pass_a: the `_pass_a` sub-dict from a run_visual_auditor.py pred
    record. Returns a fixed-length numeric vector, np.nan for missing."""
    if not pass_a:
        return [np.nan, np.nan, np.nan]
    return [
        _TRI.get(pass_a.get("semantic_action_changed"), np.nan),
        _TRI.get(pass_a.get("motion_change_without_semantic_change"), np.nan),
        _TRI.get(pass_a.get("visual_evidence"), np.nan),
    ]


# --- plain logistic regression (no sklearn) ---------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1.0, lr=0.2, iters=3000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / n + l2 * w / n
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def grouped_loro_predict(X, y, groups, l2=1.0, lr=0.2, iters=3000, min_train=6):
    """Leave-one-RECORDING-out (LORO) out-of-fold predicted P(valid).
    Skips a fold (leaves NaN) if the training data would be too small or
    single-class -- reported, not silently averaged over."""
    n = len(y)
    preds = np.full(n, np.nan)
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        test_mask = groups == g
        train_mask = ~test_mask
        y_train = y[train_mask]
        if train_mask.sum() < min_train or len(set(y_train.tolist())) < 2:
            continue
        Xtr, Xte = X[train_mask], X[test_mask]
        col_mean = np.nanmean(Xtr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        Xtr_i = np.where(np.isnan(Xtr), col_mean, Xtr)
        Xte_i = np.where(np.isnan(Xte), col_mean, Xte)
        mu, sigma = Xtr_i.mean(0), Xtr_i.std(0) + 1e-8
        w, b = fit_logreg((Xtr_i - mu) / sigma, y_train, l2=l2, lr=lr, iters=iters)
        preds[test_mask] = _sigmoid((Xte_i - mu) / sigma @ w + b)
    return preds


def arm_metrics(y, p, lo=0.25, hi=0.75):
    mask = ~np.isnan(p)
    n_excluded_fold = int((~mask).sum())
    y, p = y[mask], p[mask]
    decided = (p <= lo) | (p >= hi)
    n = len(y)
    if n == 0:
        return {"n": 0, "n_excluded_fold": n_excluded_fold}
    review_rate = float(1.0 - decided.mean())
    yd, pd_ = y[decided], p[decided]
    pred = (pd_ >= hi).astype(int)
    tp = int(((pred == 1) & (yd == 1)).sum())
    fn = int(((pred == 0) & (yd == 1)).sum())
    tn = int(((pred == 0) & (yd == 0)).sum())
    fp = int(((pred == 1) & (yd == 0)).sum())
    valid_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    spurious_recall = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_acc = (valid_recall + spurious_recall) / 2 if not (np.isnan(valid_recall) or np.isnan(spurious_recall)) else float("nan")
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    f1_1 = 2 * prec1 * valid_recall / (prec1 + valid_recall) if (prec1 + valid_recall) else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    f1_0 = 2 * prec0 * spurious_recall / (prec0 + spurious_recall) if (prec0 + spurious_recall) else 0.0
    macro_f1 = (f1_0 + f1_1) / 2
    return {
        "n": n, "n_excluded_fold": n_excluded_fold, "n_decided": int(decided.sum()),
        "review_rate": review_rate, "valid_recall": valid_recall,
        "motion_hard_negative_recall": spurious_recall,
        "balanced_accuracy": balanced_acc, "macro_f1": macro_f1,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--vlm_pred", required=True,
                    help="a run_visual_auditor.py --out jsonl (any coverage; "
                         "arms B/C auto-restrict to events present here)")
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...)")
    ap.add_argument("--context", help="context jsonl (default: committed data/gold/...)")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--variance_half", type=float, default=None)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--l2", type=float, default=1.0)
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

    rows = []
    for g in gold:
        truth = g.get("temporal_truth")
        if truth not in ("valid", "spurious"):
            continue
        eid = g["event_id"]
        c = ctx.get(eid, {})
        rid = g.get("recording_id") or c.get("recording_id")
        rec = by_rid.get(rid)
        if rec is None:
            continue
        t = c.get("pred_time")
        if t is None:
            t = c.get("gt_time")
        if t is None:
            continue
        hal = hal_features_at(rec["feats"], rec["times"], float(t),
                              short_half=a.short_half, context_half=a.context_half,
                              variance_half=a.variance_half)
        vlm_rec = vlm_pred.get(eid)
        pass_a = (vlm_rec or {}).get("_pass_a")
        rows.append({
            "event_id": eid, "recording_id": rid, "y": 1 if truth == "valid" else 0,
            "hal": [hal.get(k, np.nan) if hal.get(k) is not None else np.nan for k in HAL_FEATURE_NAMES],
            "vlm": _encode_vlm_atomic(pass_a),
            "has_vlm": pass_a is not None,
        })

    n_total = len(rows)
    n_vlm = sum(r["has_vlm"] for r in rows)
    print(f"usable events (label + HAL feature coverage): {n_total}")
    print(f"of those, events with a VLM Pass-A prediction available: {n_vlm}")
    if n_vlm < n_total:
        print(f"  (arm A runs on all {n_total}; arms B/C restrict to the {n_vlm} "
              f"with VLM coverage -- run the atomic auditor on more events for a "
              f"full-72 B/C comparison)")

    y_all = np.array([r["y"] for r in rows], dtype=float)
    groups_all = [r["recording_id"] for r in rows]
    X_hal_all = np.array([r["hal"] for r in rows], dtype=float)

    vlm_rows = [r for r in rows if r["has_vlm"]]
    y_vlm = np.array([r["y"] for r in vlm_rows], dtype=float)
    groups_vlm = [r["recording_id"] for r in vlm_rows]
    X_hal_vlm = np.array([r["hal"] for r in vlm_rows], dtype=float)
    X_vlm = np.array([r["vlm"] for r in vlm_rows], dtype=float)
    X_fusion = np.concatenate([X_hal_vlm, X_vlm], axis=1) if len(vlm_rows) else np.zeros((0, 8))

    results = {}
    print("\n=== A. HAL-only (n=%d, grouped LORO over %d recordings) ===" %
          (n_total, len(set(groups_all))))
    pA = grouped_loro_predict(X_hal_all, y_all, groups_all, l2=a.l2)
    mA = arm_metrics(y_all, pA, lo=a.lo, hi=a.hi)
    results["hal_only"] = mA
    for k, v in mA.items():
        print(f"  {k:<28} {_fmt(v)}")

    if len(vlm_rows) >= 8:
        print("\n=== B. VLM-atomic-only (n=%d, grouped LORO over %d recordings) ===" %
              (len(vlm_rows), len(set(groups_vlm))))
        pB = grouped_loro_predict(X_vlm, y_vlm, groups_vlm, l2=a.l2)
        mB = arm_metrics(y_vlm, pB, lo=a.lo, hi=a.hi)
        results["vlm_only"] = mB
        for k, v in mB.items():
            print(f"  {k:<28} {_fmt(v)}")

        print("\n=== C. Fusion (HAL + VLM atomic, n=%d) ===" % len(vlm_rows))
        pC = grouped_loro_predict(X_fusion, y_vlm, groups_vlm, l2=a.l2)
        mC = arm_metrics(y_vlm, pC, lo=a.lo, hi=a.hi)
        results["fusion"] = mC
        for k, v in mC.items():
            print(f"  {k:<28} {_fmt(v)}")

        print("\nThe question that matters: does Fusion's valid_recall exceed both "
              "HAL-only's and VLM-only's, WITHOUT motion_hard_negative_recall falling "
              "below VLM-only's (VLM-only already achieves whatever motion-rejection "
              "the atomic prompts got right) -- not whether Fusion's aggregate accuracy "
              "is marginally higher.")
    else:
        print(f"\n(skipping arms B/C -- only {len(vlm_rows)} events have a VLM "
              f"prediction; need at least 8 for a meaningful grouped fit. Run the "
              f"atomic auditor on more events and re-run this script.)")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"n_total": n_total, "n_vlm_covered": n_vlm, "results": results},
                     f, ensure_ascii=False, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
        print(f"\nwrote {a.out}")
        try:
            from src.eval.run_manifest import write_manifest
            write_manifest(a.out, input_paths=[gold_path, ctx_path, a.vlm_pred] + a.feat_cache)
        except Exception as e:
            print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
