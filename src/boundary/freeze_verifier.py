"""Freeze the pairwise verifier for a genuine batch3 held-out test.

The development comparison passed its pre-registered gate: P1 reached AUROC
0.893 with coverage 0.604 at 90% precision (visual features only -- the
shortcut ablation showed the annotation-position features carried none of
the gain and in fact cost 0.006 AUROC, so they are dropped here), against
P0's 0.729 / 0.006, improving in 5/5 folds.

But coverage@0.90 read off pooled OOF predictions is chosen POST HOC on the
same predictions being scored -- it is an upper bound, not something that
will reproduce. That is exactly how the previous HAL policy came to claim
0.90 precision on dev and deliver 0.767 on held-out data. So this script
does two separate things and does not conflate them:

  1. NESTED estimate of what a frozen threshold actually delivers.
     Outer grouped folds; inside each outer TRAINING set, an inner grouped CV
     picks the probability threshold that hits --target_precision; that
     threshold is then applied, untouched, to the outer TEST fold. Precision
     and coverage are pooled over outer test folds. No threshold is ever
     selected using data it is then scored on, so this number is an honest
     forecast for batch3 -- and it is expected to be lower than 0.604.

  2. FROZEN artifact. The model is refit on ALL development events and the
     threshold is chosen by one inner grouped CV over the whole set. Model
     weights, PCA basis, scalers, feature list and the threshold are saved
     together, so batch3 runs a single fixed function with no fitting.

Everything the artifact needs to be reproduced is written into the JSON
sidecar: feature names, window scales, whether windows were clipped at
neighbouring annotated boundaries, PCA dimension, L2, fold seed.

IMPORTANT about window clipping: it uses the surrounding annotated segments
to bound each feature window. That is legitimate for a VERIFIER (the question
is "given the existing segmentation, is this candidate real"), and batch3
must supply the same annotation context. It would be circular in a detector
that is meant to find boundaries from scratch.

Usage (server):
    python -m src.boundary.freeze_verifier \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --clip_windows_at_neighbors \
        --out_dir /workspace/tr1/results/hal/frozen_verifier
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    build_matrices, stratified_grouped_folds, pair_block, pca_fit, pca_apply,
    _impute_scale_fit, _impute_scale_apply, REL_NAMES, SHORTCUT_GROUPS, SCALES,
)

DROP = [n for g in SHORTCUT_GROUPS.values() for n in g]
KEEP_REL = [n for n in REL_NAMES if n not in DROP]
KEEP_IDX = [REL_NAMES.index(n) for n in KEEP_REL]


def fit_fold(Ls, Rs, X_rel, y, tr, pca_dim, l2):
    """Fit the whole P1 pipeline on `tr` and return an apply() closure."""
    pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), pca_dim)
    Xr = X_rel[tr][:, KEEP_IDX]
    st_r = _impute_scale_fit(Xr)
    P = np.concatenate([pair_block(pca_apply(pca, Ls[tr]), pca_apply(pca, Rs[tr])),
                        _impute_scale_apply(st_r, Xr)], 1)
    stP = _impute_scale_fit(P)
    w, b = fit_logreg(_impute_scale_apply(stP, P), y[tr], l2=l2)

    def apply(sel):
        Xr_ = X_rel[sel][:, KEEP_IDX]
        P_ = np.concatenate([pair_block(pca_apply(pca, Ls[sel]), pca_apply(pca, Rs[sel])),
                             _impute_scale_apply(st_r, Xr_)], 1)
        return _sigmoid(_impute_scale_apply(stP, P_) @ w + b)

    artifact = {"pca_mu": pca["mu"], "pca_W": pca["W"], "rel_scaler": st_r,
                "pair_scaler": stP, "w": w, "b": float(b)}
    return apply, artifact


def threshold_for_precision(y, p, target, min_selected=10):
    """Loosest probability threshold whose selected set reaches `target`
    precision using at least `min_selected` items. Returns
    (threshold, coverage), or (None, 0.0) if unreachable.

    min_selected matters and is not cosmetic: without it, a single
    top-ranked positive gives precision 1.0 at k=1 and the function happily
    returns a threshold estimated from ONE sample. Inside the nested loop
    that threshold would then be applied to a whole outer fold. Caught by a
    synthetic test where a target was supposed to be unreachable and the
    function returned a k=1 threshold instead."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    best = None
    for k in range(min_selected, len(ys) + 1):
        if ys[:k].mean() >= target:
            best = (float(ps[k - 1]), k / len(ys))
    return best if best else (None, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--clip_windows_at_neighbors", action="store_true")
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_selected", type=int, default=10,
                    help="a threshold must select at least this many items; prevents "
                         "estimating an operating point from a handful of samples")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    X_v1, Ls, Rs, X_rel, keep = build_matrices(events, a.clip_windows_at_neighbors)
    events = [events[i] for i in keep]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = np.array([e["recording_id"] for e in events])
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"development events: {len(events)} ({int(y.sum())}+/{int(len(y)-y.sum())}-), "
          f"{len(set(groups))} recordings")
    print(f"features: {len(KEEP_REL)} relative ({KEEP_REL}) + PCA-{a.pca_dim} pair block")
    print(f"dropped as non-visual shortcuts: {DROP}")

    # ---- 1. nested estimate --------------------------------------------
    print(f"\n=== NESTED estimate (threshold chosen on inner folds only) ===")
    outer = stratified_grouped_folds(groups, y, a.folds, a.seed)
    sel_y, sel_p, n_total, thr_used = [], [], 0, []
    for fi, held in enumerate(outer):
        te = np.isin(groups, list(held)); tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        # inner CV on the training portion only
        inner_groups = groups[tr]
        inner = stratified_grouped_folds(inner_groups, y[tr], a.inner_folds, a.seed + fi)
        idx_tr = np.where(tr)[0]
        p_inner = np.full(int(tr.sum()), np.nan)
        for ih in inner:
            i_te = np.isin(inner_groups, list(ih)); i_tr = ~i_te
            if i_te.sum() == 0 or len(set(y[idx_tr[i_tr]].tolist())) < 2:
                continue
            ap_fn, _ = fit_fold(Ls, Rs, X_rel, y, idx_tr[i_tr], a.pca_dim, a.l2)
            p_inner[i_te] = ap_fn(idx_tr[i_te])
        m = ~np.isnan(p_inner)
        thr, inner_cov = threshold_for_precision(y[idx_tr][m], p_inner[m],
                                                 a.target_precision, a.min_selected)
        if thr is None:
            print(f"  fold {fi+1}: inner CV never reached {a.target_precision:.0%} precision "
                  f"-- no threshold, fold contributes nothing")
            continue
        ap_fn, _ = fit_fold(Ls, Rs, X_rel, y, tr, a.pca_dim, a.l2)
        p_te = ap_fn(te)
        chosen = p_te >= thr
        n_total += int(te.sum())
        thr_used.append(thr)
        sel_y.append(y[te][chosen]); sel_p.append(p_te[chosen])
        prec = y[te][chosen].mean() if chosen.any() else float("nan")
        print(f"  fold {fi+1}: inner thr={thr:.3f} (inner cov {inner_cov:.3f}) -> "
              f"outer selected {int(chosen.sum())}/{int(te.sum())}, precision={prec:.3f}")
    if sel_y:
        allsel = np.concatenate(sel_y)
        nested_prec = float(allsel.mean())
        nested_cov = float(len(allsel) / n_total)
        k, n = int(allsel.sum()), len(allsel)
        from src.boundary.hal_heldout_validate import wilson_interval
        lo, hi = wilson_interval(k, n)
        print(f"\n  NESTED precision = {nested_prec:.3f}  ({k}/{n}, Wilson 95% CI "
              f"[{lo:.3f}, {hi:.3f}])   coverage = {nested_cov:.3f}")
        print(f"  This -- not the 0.604 post-hoc figure -- is the forecast for batch3.")
        if nested_prec < a.target_precision:
            print(f"  !! below the {a.target_precision:.0%} target: the operating point does "
                  f"not survive honest threshold selection. Do NOT promise {a.target_precision:.0%} "
                  f"on batch3; either accept the lower precision or raise the target and "
                  f"lose coverage.")
    else:
        nested_prec = nested_cov = None
        print("  no fold produced a usable threshold")

    # ---- 2. frozen artifact ---------------------------------------------
    print(f"\n=== FROZEN artifact (refit on all {len(events)} development events) ===")
    all_idx = np.arange(len(events))
    inner = stratified_grouped_folds(groups, y, a.inner_folds, a.seed)
    p_oof = np.full(len(events), np.nan)
    for ih in inner:
        i_te = np.isin(groups, list(ih)); i_tr = ~i_te
        if i_te.sum() == 0 or len(set(y[i_tr].tolist())) < 2:
            continue
        ap_fn, _ = fit_fold(Ls, Rs, X_rel, y, i_tr, a.pca_dim, a.l2)
        p_oof[i_te] = ap_fn(i_te)
    m = ~np.isnan(p_oof)
    frozen_thr, frozen_cov = threshold_for_precision(y[m], p_oof[m], a.target_precision,
                                                     a.min_selected)
    if frozen_thr is None:
        raise SystemExit(f"no threshold reaches {a.target_precision:.0%} precision with at "
                         f"least {a.min_selected} selected items -- nothing to freeze")
    _, artifact = fit_fold(Ls, Rs, X_rel, y, all_idx, a.pca_dim, a.l2)
    print(f"  frozen threshold = {frozen_thr:.4f}  (OOF coverage at that threshold "
          f"{frozen_cov:.3f}, OOF AUROC {_auroc(y[m], p_oof[m]):.3f})")

    npz = os.path.join(a.out_dir, "frozen_verifier.npz")
    np.savez(npz, pca_mu=artifact["pca_mu"], pca_W=artifact["pca_W"],
             rel_cm=artifact["rel_scaler"]["cm"], rel_mu=artifact["rel_scaler"]["mu"],
             rel_sd=artifact["rel_scaler"]["sd"],
             pair_cm=artifact["pair_scaler"]["cm"], pair_mu=artifact["pair_scaler"]["mu"],
             pair_sd=artifact["pair_scaler"]["sd"], w=artifact["w"], b=artifact["b"],
             threshold=frozen_thr)
    cfg = {
        "threshold": frozen_thr, "target_precision": a.target_precision,
        "rel_features": KEEP_REL, "dropped_shortcut_features": DROP,
        "scales": SCALES, "pca_dim": a.pca_dim, "l2": a.l2, "seed": a.seed,
        "clip_windows_at_neighbors": bool(a.clip_windows_at_neighbors),
        "n_dev_events": len(events), "n_dev_recordings": int(len(set(groups))),
        "min_selected": a.min_selected,
        "nested_precision_forecast": nested_prec, "nested_coverage_forecast": nested_cov,
        "posthoc_oof_coverage_DO_NOT_QUOTE": frozen_cov,
        "batch3_requirements": [
            "recordings disjoint from all 47 development recordings",
            "same feature extraction params as the dev caches (pool multi, fps 2, no blur filter)",
            "annotated segments available (window clipping needs neighbouring boundaries)",
            "apply this artifact unchanged: no refitting, no threshold re-selection",
        ],
    }
    with open(os.path.join(a.out_dir, "frozen_verifier.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote {npz} + frozen_verifier.json")
    print("\n  batch3 protocol: apply this artifact ONCE, unchanged. Selecting a different "
          "threshold after seeing batch3 labels turns the test back into development.")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(npz, input_paths=[a.gold, a.context] + a.feat_cache,
                       extra={"threshold": frozen_thr, "nested_precision": nested_prec})
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
