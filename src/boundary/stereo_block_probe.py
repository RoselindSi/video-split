"""C3-1: does mixing the two stereo eyes into one representation hurt P1?

Answered WITHOUT re-extracting anything, because the existing caches already
contain per-eye vectors and nobody noticed.

extract_features_recseg.py's --pool multi concatenates five spatial pools of
each frame's patch grid:

    [ global | left | right | center | spatial_max ]     (5 x 1152 = 5760)

where left = g[:, :W//2] and right = g[:, W//2:]. The audit established that
these frames are two 640x480 cameras packed side by side, so those two pools
are not "left and right of the scene" -- they are the LEFT EYE and the RIGHT
EYE, already separated, already cached. Every P1 number in this project was
computed on all five blocks at once, i.e. on a representation that mixes two
viewpoints.

Three things this can distinguish, at zero GPU cost:

  global vs left vs right (all 1152-dim, so the comparison is not confounded
      by dimensionality): is a single eye better than the two-eye mean? If the
      mean is diluting the signal, one eye should win.
  left+right vs global: is keeping both eyes SEPARATE better than averaging
      them? That separates "two views are redundant" from "two views are
      useful but must not be averaged".
  all vs all-minus-center: the `center` pool is g[:, W//4 : 3W//4], which on a
      packed stereo frame straddles the seam -- it is the right half of the
      left eye spliced to the left half of the right eye, a picture of nothing
      that exists. This checks whether that chimera block is inert or harmful.

A hypothesis worth stating because it motivated the audit: if the two eyes are
averaged, then movement in DEPTH changes the disparity between them and
therefore changes the pooled feature, even when the action is unchanged.
regrasp_reposition and direction_reversal are exactly "position changed,
action did not", and are exactly the negatives P1 gets wrong. So the
watch-subtype false-positive rate is reported per arm, not just AUROC: an arm
that improves AUROC while leaving those untouched is not evidence for this
mechanism.

Nothing here is a pre-registered adoption gate. It is a diagnostic on frozen
cached features, run to decide whether a re-extraction with --crop left is
worth its GPU cost at all.

Usage (server, no GPU needed):
    python -m src.boundary.stereo_block_probe \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --same_action_subtype data/gold/same_action_subtype_v1.csv \
        --out /workspace/tr1/results/hal/c3/stereo_block_probe.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    stratified_grouped_folds, build_matrices,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid

# order fixed by pool_patch()'s torch.cat([glob, left, right, center, smax])
BLOCK_NAMES = ["global", "left", "right", "center", "spatial_max"]

ARMS = {
    "all (current P1 baseline)":      ["global", "left", "right", "center", "spatial_max"],
    "global only (two eyes mixed)":   ["global"],
    "left eye only":                  ["left"],
    "right eye only":                 ["right"],
    "both eyes, kept separate":       ["left", "right"],
    "all minus center (seam block)":  ["global", "left", "right", "spatial_max"],
}


def slice_blocks(by_rid, names, n_blocks=len(BLOCK_NAMES)):
    """Return a NEW cache dict whose feats keep only the named blocks.

    Slicing at the cache level (rather than inside one downstream consumer)
    keeps every consumer consistent: side_vectors, hal_features_at,
    relative_features and same_side_subcrops all read rec["feats"], so a
    partial slice applied in only one of them would silently mix feature
    spaces across the pipeline."""
    idx = [BLOCK_NAMES.index(n) for n in names]
    out = {}
    for rid, rec in by_rid.items():
        f = rec["feats"]
        if f.ndim != 2 or f.shape[1] % n_blocks:
            raise SystemExit(
                f"{rid}: feature dim {tuple(f.shape)} is not divisible into "
                f"{n_blocks} blocks -- this cache was probably not built with "
                f"--pool multi, and block slicing does not apply to it")
        d = f.shape[1] // n_blocks
        out[rid] = dict(rec, feats=np.concatenate([f[:, i * d:(i + 1) * d] for i in idx], 1)
                        if isinstance(f, np.ndarray)
                        else __import__("torch").cat([f[:, i * d:(i + 1) * d] for i in idx], 1))
    return out


def p1_eval(events, folds, groups, y, pca_dim=64, l2=5.0):
    """The project's standard P1 pipeline, refit per fold on the training split
    only -- identical to what slow_latent_c2/predictive_continuity use for
    their P1 arm, so numbers are comparable across all of them."""
    X_v1, Ls, Rs, X_rel, keep, crops = build_matrices(events, False)
    yk = y[keep]
    gk = [groups[i] for i in keep]
    oof = np.full(len(yk), np.nan)
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in gk])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(yk[tr].tolist())) < 2:
            continue
        pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), pca_dim)
        st_rel = _impute_scale_fit(X_rel[tr])
        def blk(m):
            return np.concatenate(
                [pair_block(pca_apply(pca, Ls[m]), pca_apply(pca, Rs[m])),
                 _impute_scale_apply(st_rel, X_rel[m])], 1)
        Ptr = blk(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), yk[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(stP, blk(te)) @ w + b)
    m = np.isfinite(oof)
    pooled = _auroc(yk[m], oof[m]) if len(set(yk[m].tolist())) == 2 else float("nan")
    for f in folds:
        te = np.array([g in f for g in gk]) & m
        if te.sum() >= 2 and len(set(yk[te].tolist())) == 2:
            per_fold.append(_auroc(yk[te], oof[te]))
    return pooled, per_fold, oof, keep, yk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--same_action_subtype", default="data/gold/same_action_subtype_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    d_total = next(iter(by_rid.values()))["feats"].shape[1]
    print(f"feature dim {d_total} = {len(BLOCK_NAMES)} x {d_total // len(BLOCK_NAMES)}  "
          f"blocks {BLOCK_NAMES}")
    if d_total % len(BLOCK_NAMES):
        raise SystemExit("cache is not a --pool multi cache; nothing to slice")

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    labels = T.load_pair_labels(a.pair_labels)

    sub_map = {}
    if os.path.exists(a.same_action_subtype):
        with open(a.same_action_subtype, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sub_map[r["event_id"]] = r["subtype"]
    watch = {"regrasp_reposition", "direction_reversal"}

    results = {}
    baseline_key = "all (current P1 baseline)"
    for arm, names in ARMS.items():
        sliced = slice_blocks(by_rid, names)
        events = T.apply_to_events(build_events(gold, ctx, sliced), labels)
        y = np.array([e["y"] for e in events], dtype=float)
        groups = [e["recording_id"] for e in events]
        folds = stratified_grouped_folds(groups, y, 5, seed=a.seed)
        pooled, per_fold, oof, keep, yk = p1_eval(events, folds, groups, y, a.pca_dim)
        kept_events = [events[i] for i in keep]

        cutoff = float(np.nanmedian(oof[yk == 1]))
        neg_watch = [i for i, e in enumerate(kept_events)
                     if yk[i] == 0 and sub_map.get(e["event_id"]) in watch
                     and np.isfinite(oof[i])]
        fp = sum(oof[i] >= cutoff for i in neg_watch)
        rate = fp / len(neg_watch) if neg_watch else float("nan")
        results[arm] = {"blocks": names, "dim": len(names) * (d_total // len(BLOCK_NAMES)),
                        "auroc": pooled, "per_fold": per_fold,
                        "watch_fp_rate": rate, "watch_fp": fp, "watch_n": len(neg_watch)}
        print(f"{arm:<34} dim {results[arm]['dim']:>5}  AUROC {pooled:.3f}  "
              f"per-fold {[round(x, 3) for x in per_fold]}  "
              f"watch FP {fp}/{len(neg_watch)}")

    base = results[baseline_key]
    print(f"\n=== vs the current baseline ({base['auroc']:.3f}) ===")
    for arm, r in results.items():
        if arm == baseline_key:
            continue
        print(f"  {arm:<34} {r['auroc'] - base['auroc']:+.3f}  "
              f"watch FP {r['watch_fp_rate'] - base['watch_fp_rate']:+.3f}")

    g, l, rr = (results["global only (two eyes mixed)"]["auroc"],
                results["left eye only"]["auroc"],
                results["right eye only"]["auroc"])
    # dimension is interpolated, never hardcoded: a sibling script printed a
    # stale constant long after the number it referred to had changed.
    bd = d_total // len(BLOCK_NAMES)
    print(f"\n=== reading (all three are {bd}-dim, so this is not a "
          f"dimensionality effect) ===")
    print(f"  two eyes mixed {g:.3f}   left eye {l:.3f}   right eye {rr:.3f}")
    print("  caveat, applies to every branch below: these per-eye blocks are a "
          "GLOBAL mean over one eye. A real --crop extraction would give that eye "
          "all five spatial pools computed WITHIN it, which is strictly richer "
          "than anything this probe can emulate -- so a null result here bounds "
          "the mixing effect, it does not rule out that cropping helps.")
    best_eye = max(l, rr)
    if best_eye - g >= 0.02:
        print("  -> a single eye beats the two-eye mean. Averaging two viewpoints "
              "is costing signal, and a re-extraction with --crop is worth its "
              "GPU cost. Check the watch-subtype column too: if it did NOT "
              "improve, the gain is not coming from the depth-disparity "
              "mechanism that motivated this.")
    elif g - best_eye >= 0.02:
        print("  -> the two-eye mean BEATS either eye alone. The second view is "
              "carrying real information (plausibly depth), not diluting. Do not "
              "re-extract with --crop; if anything the eyes deserve a better "
              "combination than a mean.")
    else:
        print("  -> no meaningful difference. The mixing hypothesis is not "
              "supported on this evidence, and a --crop re-extraction would be "
              "spending GPU on a difference this data cannot resolve. Note the "
              "limit: these are 1152-dim GLOBAL means per eye; a true --crop "
              "extraction would give a single eye all five spatial pools, which "
              "this cannot emulate.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"feature_dim": d_total, "block_names": BLOCK_NAMES,
                       "arms": results}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
