"""Train on the question the deployment actually asks: may this be auto-kept?

Every model in this project was trained on sharp vs same-action over the 299
CLEAN events, and then applied to all 412 at inference. The taxonomy excludes
gradual, annotation_convention, camera, offscreen and ambiguous from
supervision, so the verifier was never given a reason to score them low -- and
14 of v4's 16 wrong auto-keeps are exactly those classes. That is the whole
18.3-point gap between the clean-binary precision of 0.967 and the
full-taxonomy 0.784.

So the target changes and NOTHING else does:

    positive  sharp_visible_transition                        175
    negative  same_action 124, gradual 37, annotation 35,
              camera 20, offscreen 18, ambiguous 3            237

WHAT THIS IS NOT. Not a better boundary representation and not evidence that
one exists. The features, PCA, scaler, architecture, L2, folds, reliability
definition and candidate generator are all unchanged; only the label the head
is fitted against moves. Reading a gain here as "the model got better at
finding boundaries" would be wrong -- it would mean the training objective
finally matches the deployment criterion.

Treating offscreen and ambiguous as negatives costs recall on events that
might be real boundaries. For a high-precision AUTO_KEEP that is an acceptable
conservative bias, and it is a bias, not a correction.

NESTED, NOT POOLED. The threshold is chosen inside the training recordings of
each outer fold and only then applied to that fold's held-out recordings. The
alternative -- pooling out-of-fold scores over all 412 and picking a threshold
that looks good on them -- is how every previous operating point in this
project was chosen, and it is why nested selection kept breaching afterwards.

ONE ARM. Global blocks, local blocks, the relative scalars and the two
reliability columns, fitted once. No second feature set, no fusion variants,
no architecture sweep: trying several and reporting the best would turn a
pre-registered test into a search, which is the failure mode this file is
supposed to close rather than repeat.

AUTO_REJECT STAYS OFF. This head answers keep-eligibility. A low score means
"not eligible for automatic acceptance", not "this is not a boundary", and
using it to reject would read a claim into it that it does not make.

PRE-REGISTERED SUCCESS CRITERIA, fixed before the run:

    full-taxonomy AUTO_KEEP precision >= 0.95
    AUTO_KEEP n >= 20
    one-error buffered:  TP / (TP + FP + 1) >= 0.95
    every outer fold produces some automation

The buffer is stated as a formula because "0.95 with one more error still
0.95" is ambiguous: 19/20 reads as a pass and 19/(19+1+1) = 0.905 does not.

Usage:
    python -m src.boundary.c3_keep_eligibility \
        --clean145 \
        --batch3_manifest .../batch3_manifest.jsonl \
        --batch3_pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --feat_cache ... --local_cache ... \
        --compare /workspace/tr1/results/hal/c3/policy_v4.json \
        --out /workspace/tr1/results/hal/c3/keep_eligibility.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from src.boundary.hal_features import load_feature_caches
from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.state_adapter import _auroc
from src.boundary.c3_local_eval import detect_coverage, detect_longest_gap_s
from src.boundary.c3_selective_policy import wilson
from src.boundary.frozen_scorer import gather, matrices
from src.boundary.c3_validity_gate import run_cv, cand_source

SHARP = "sharp_visible_transition"
CLEAN = ("sharp_visible_transition", "same_action_internal_motion")

MIN_PRECISION = 0.95
MIN_N = 20


def buffered(tp, fp):
    """Precision after one more false keep. Written out because 'still 0.95
    with one extra error' is ambiguous: 19/20 reads as a pass and
    19/(19+1+1) = 0.905 does not."""
    return tp / (tp + fp + 1) if (tp + fp + 1) else float("nan")


def pick_threshold(scores, y, target=MIN_PRECISION):
    """The deepest threshold whose prefix still satisfies the BUFFERED
    precision. Selected on inner-fold scores only; the caller never lets the
    outer fold's events influence it."""
    m = np.isfinite(scores)
    if m.sum() < 10 or len(set(y[m].tolist())) < 2:
        return None
    order = np.argsort(-scores[m])
    s, yy = scores[m][order], y[m][order]
    tp = fp = 0
    th = None
    for i in range(len(s)):
        tp += int(yy[i] == 1)
        fp += int(yy[i] == 0)
        if buffered(tp, fp) >= target:
            th = float(s[i])
    return th


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--clean145", action="store_true")
    ap.add_argument("--batch3_manifest", action="append", default=[])
    ap.add_argument("--batch3_pair_labels", action="append", default=[])
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="append", default=[],
                    help="policy result JSONs to put in the same table")
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    loc_rid = load_feature_caches(a.local_cache)
    sources = ([(None, None)] if a.clean145 else []) \
        + list(zip(a.batch3_manifest, a.batch3_pair_labels))
    if not sources:
        raise SystemExit("no event source")

    allev = []
    for man, pl in sources:
        sub = argparse.Namespace(**vars(a))
        sub.batch3_manifest, sub.batch3_pair_labels = man, pl
        ev_c, ev_x = gather(sub, by_rid, loc_rid)
        tag = "clean-145" if man is None else "batch3"
        for e in ev_c + ev_x:
            e["_source"] = tag
        allev += ev_c + ev_x
    ev, Lg, Rg, Ll, Rl, X_rel = matrices(allev, loc_rid)

    sub = [e.get("temporal_pair_subtype") or "" for e in ev]
    y = np.array([s == SHARP for s in sub], float)
    groups = [e["recording_id"] for e in ev]
    src = np.array([e.get("_source", "?") for e in ev])
    is_clean = np.array([s in CLEAN for s in sub])
    extra = np.array([[detect_coverage(loc_rid[e["recording_id"]], e["t"]),
                       detect_longest_gap_s(loc_rid[e["recording_id"]], e["t"])]
                      for e in ev])
    print(f"{len(ev)} events, {len(set(groups))} recordings, "
          f"{int(y.sum())} sharp / {int((1 - y).sum())} non-sharp "
          f"(base rate {y.mean():.3f})")
    print(f"  negatives: {dict(Counter(s for s, k in zip(sub, y == 0) if k))}")

    outer = stratified_grouped_folds(groups, y, 5, seed=a.seed)
    rows, oof = [], np.full(len(y), np.nan)
    print(f"\n{'=' * 74}\nNESTED: threshold chosen inside each fold's TRAINING "
          f"recordings\n{'=' * 74}")
    print(f"  {'fold':>4} {'n_test':>7} {'thr':>7} {'kept':>5} {'TP':>4} "
          f"{'FP':>4} {'prec':>6} {'buff':>6} {'cov':>6}")
    for fi, f in enumerate(outer):
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 5 or len(set(y[tr].tolist())) < 2:
            print(f"  {fi:>4}  skipped (fold too small)")
            continue
        # inner folds over the TRAINING recordings only
        gtr = [g for g, k in zip(groups, tr) if k]
        inner = stratified_grouped_folds(gtr, y[tr], 5, seed=a.seed + 1)
        s_in = run_cv(Lg[tr], Rg[tr], Ll[tr], Rl[tr], X_rel[tr], extra[tr],
                      y[tr], gtr, inner, a.pca_dim)
        th = pick_threshold(s_in, y[tr])
        # one model fitted on all training recordings, applied to held-out
        s_out = run_cv(Lg, Rg, Ll, Rl, X_rel, extra, y, groups, [f], a.pca_dim)
        oof[te] = s_out[te]
        if th is None:
            rows.append({"fold": fi, "n_test": int(te.sum()), "th": None,
                         "kept": 0, "tp": 0, "fp": 0})
            print(f"  {fi:>4} {int(te.sum()):>7}   none      0    0    0"
                  f"       -      -  0.000   <- no inner threshold reached the "
                  f"buffered target")
            continue
        keep = te & np.isfinite(oof) & (oof >= th)
        tp = int(y[keep].sum())
        fp = int(keep.sum()) - tp
        rows.append({"fold": fi, "n_test": int(te.sum()), "th": th,
                     "kept": int(keep.sum()), "tp": tp, "fp": fp,
                     "kept_ids": [ev[i]["event_id"] for i in np.nonzero(keep)[0]],
                     "fp_subtypes": [sub[i] for i in np.nonzero(keep)[0]
                                     if y[i] == 0]})
        p = tp / max(keep.sum(), 1)
        print(f"  {fi:>4} {int(te.sum()):>7} {th:>7.3f} {int(keep.sum()):>5} "
              f"{tp:>4} {fp:>4} {p:>6.3f} {buffered(tp, fp):>6.3f} "
              f"{keep.sum() / te.sum():>6.3f}")

    TP = sum(r["tp"] for r in rows)
    FP = sum(r["fp"] for r in rows)
    N = TP + FP
    prec = TP / N if N else float("nan")
    lo, hi = wilson(TP, N)
    buf = buffered(TP, FP)
    zero_folds = [r["fold"] for r in rows if r["kept"] == 0]
    m = np.isfinite(oof)
    au = _auroc(y[m], oof[m]) if len(set(y[m].tolist())) == 2 else float("nan")

    print(f"\n{'=' * 74}\nFULL TAXONOMY, pooled over the outer folds\n{'=' * 74}")
    print(f"  AUTO_KEEP {N} of {len(ev)} events (coverage {N / len(ev):.3f})")
    print(f"  precision {prec:.3f} [Wilson {lo:.3f}, {hi:.3f}]   "
          f"one-error buffered {buf:.3f}")
    print(f"  REVIEW rate {1 - N / len(ev):.3f}")
    print(f"  out-of-fold AUROC on the keep-eligibility target {au:.3f}")
    fps = Counter(s for r in rows for s in r.get("fp_subtypes", []))
    if fps:
        print(f"  the {FP} wrong keeps by subtype: {dict(fps)}")
    if zero_folds:
        print(f"  !! folds with NO automation at all: {zero_folds}. A pooled "
              f"precision hides this; a policy that automates nothing on two "
              f"of five recording groups is not one policy.")

    print(f"\n{'=' * 74}\nPRE-REGISTERED CRITERIA\n{'=' * 74}")
    checks = {
        f"full-taxonomy precision >= {MIN_PRECISION}": prec >= MIN_PRECISION,
        f"AUTO_KEEP n >= {MIN_N}": N >= MIN_N,
        f"one-error buffered >= {MIN_PRECISION}": buf >= MIN_PRECISION,
        "every outer fold automates something": not zero_folds,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    passed = all(checks.values())

    # clean subset: does the safety come from discrimination or from abstaining?
    ck = m & is_clean
    print(f"\n{'=' * 74}\nCLEAN SUBSET ({int(is_clean.sum())} events) -- is the "
          f"safety discrimination or abstention?\n{'=' * 74}")
    # the decision each event actually received: its own outer fold's
    # threshold, never a pooled one
    kept_clean = np.zeros(len(y), bool)
    for r in rows:
        if r["th"] is None:
            continue
        f = outer[r["fold"]]
        te = np.array([g in f for g in groups])
        kept_clean |= te & m & (oof >= r["th"])
    kc = kept_clean & is_clean
    P = int((y[ck] == 1).sum())
    Nn = int((y[ck] == 0).sum())
    tp_c = int(y[kc].sum())
    fp_c = int(kc.sum()) - tp_c
    print(f"  sharp kept {tp_c}/{P} (TPR {tp_c / max(P, 1):.3f})   "
          f"same-action kept {fp_c}/{Nn} (FPR {fp_c / max(Nn, 1):.3f})   "
          f"precision {tp_c / max(kc.sum(), 1):.3f}")
    print("  A high precision with a very low TPR means the head bought safety "
          "by abstaining, not by telling the classes apart.")

    print(f"\n{'=' * 74}\nBY SOURCE\n{'=' * 74}")
    print(f"  {'source':<12} {'n':>4} {'pi':>6} {'kept':>5} {'TP':>4} "
          f"{'FP':>4} {'prec':>6} {'cov':>6}")
    per_src = {}
    for t in sorted(set(src)):
        s_ = (src == t) & m
        # kept_clean is the union over outer folds of (this fold's test events
        # scoring above THAT fold's threshold), so restricting it to a source
        # is simply that source's share of the same decisions
        k_ = (src == t) & kept_clean
        tp_ = int(y[k_].sum())
        fp_ = int(k_.sum()) - tp_
        print(f"  {t:<12} {int(s_.sum()):>4} {y[s_].mean():>6.3f} "
              f"{int(k_.sum()):>5} {tp_:>4} {fp_:>4} "
              f"{tp_ / max(k_.sum(), 1):>6.3f} "
              f"{k_.sum() / max(s_.sum(), 1):>6.3f}")
        per_src[t] = {"n": int(s_.sum()), "prevalence": float(y[s_].mean()),
                      "kept": int(k_.sum()), "tp": tp_, "fp": fp_}

    print(f"\n{'=' * 74}\nAGAINST THE EXISTING POLICIES (full taxonomy)\n{'=' * 74}")
    print(f"  {'policy':<34} {'keep n':>7} {'precision':>10} {'review':>8}")
    print(f"  {'sharp-vs-all head (this run)':<34} {N:>7} {prec:>10.3f} "
          f"{1 - N / len(ev):>8.3f}")
    for p in a.compare:
        if not os.path.exists(p):
            print(f"  {os.path.basename(p):<34} (not found)")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for role, blk in (d.items() if isinstance(d, dict) else []):
            mm = blk.get("development") or blk.get("metrics") if isinstance(blk, dict) else None
            if not isinstance(mm, dict) or "n_all_auto_keep" not in mm:
                continue
            nk = mm["n_all_auto_keep"]
            fk = mm.get("full_false_keep_count", 0)
            print(f"  {os.path.basename(p)[:20] + ':' + role[:12]:<34} "
                  f"{nk:>7} {(nk - fk) / max(nk, 1):>10.3f} "
                  f"{mm.get('review_rate_all', float('nan')):>8.3f}")

    print(f"\n{'=' * 74}")
    if passed:
        print("  PASSES the pre-registered criteria. Next, and only this: fit "
              "the final model on all 412, freeze checkpoint, scaler, cache "
              "identities, threshold and\n  commit, run batch4 once, and do "
              "not retune on its result.")
    elif N and prec >= MIN_PRECISION:
        print("  A safe tail exists but it is not a deployable policy -- too "
              "small, unbuffered, or absent on some recording groups. This "
              "must not be reported as\n  a success; the honest statement is "
              "that a local safe region exists and does not generalise across "
              "recordings.")
    else:
        print("  Does not pass. Aligning the training target with the "
              "deployment criterion was not sufficient, which means the "
              "frozen representations cannot\n  separate keep-eligible sharp "
              "candidates from non-clean ones at this precision. That is a "
              "stronger case for a new observable than\n  anything measured so "
              "far -- but it is a research branch, not today's product path.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"n_events": len(ev), "base_rate": float(y.mean()),
                       "auroc": float(au), "auto_keep_n": N, "tp": TP, "fp": FP,
                       "precision": float(prec), "wilson": [lo, hi],
                       "buffered": float(buf), "zero_folds": zero_folds,
                       "criteria": {k: bool(v) for k, v in checks.items()},
                       "passed": bool(passed), "per_fold": rows,
                       "by_source": per_src,
                       "false_keep_subtypes": dict(fps)},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
