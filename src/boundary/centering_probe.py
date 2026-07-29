"""Does mean-centering the frozen features fix the degenerate geometry?

The collapse check on the RAW frozen Qwen features (not the adapter) found:

    effective rank 3.2 / 5760      pairwise cosine +0.998 +- 0.002
    per-dimension variance 3.4e-07  (centered effective rank 192.9)

Every frame, in every recording, points in almost the same direction. If
v_i = m + e_i with |m| >> |e_i|, then

    cos(v_i, v_j) ~ 1 - |e_i - e_j|^2 / (2 |m|^2)

so the real frame-to-frame variation is compressed by |m|^2 -- which is
exactly why every HAL distance came out at ~1e-4 (valid median 0.0003,
motion-HN 0.0001) and why a value like context_change=0.00096 could dominate
a decision. Two consequences worth separating:

  * dynamic range: the signal sits just above the float noise floor, and
    ratio features (sep = d(L,R) / (W_L + W_R)) divide two such tiny numbers.
  * cross-recording comparability: |m| differs per recording, so each
    recording's distances are scaled differently. That is a candidate
    mechanical explanation for the DEV->TEST shift (context_change mean z
    +0.78) which no amount of re-weighting could fix.

The centered effective rank of 192.9 says the variation structure IS there,
just buried. This script tests whether removing the mean recovers it, under
three treatments, and reports the SAME geometry diagnostics for each:

    none            L2-normalize only (what every experiment so far used)
    global          subtract the mean frame over ALL recordings, then
                    normalize -- removes the universal component
    per_recording   subtract each recording's OWN mean frame, then normalize
                    -- also removes per-recording scene identity, which is
                    nuisance for a WITHIN-recording boundary question and is
                    the treatment most likely to address the shift

Metric that decides it: distance-only AUROC (can d(L,R) alone rank valid
above motion-hard-negative). The uncentered baseline is 0.683. This is a
measurement, not a fix -- if centering does not move it, the frozen
representation genuinely lacks the distinction and no rescaling helps.

Usage (server):
    python -m src.boundary.centering_probe \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --out /workspace/tr1/results/hal/centering_probe.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, pool_window, _auroc
from src.boundary.adapter_diagnostics import effective_rank, lr_distance

TREATMENTS = ["none", "global", "per_recording"]


def apply_centering(feats, mode, global_mean=None):
    """feats: [T, D] float tensor. Returns L2-normalized [T, D]."""
    x = feats.float()
    if mode == "global" and global_mean is not None:
        x = x - global_mean
    elif mode == "per_recording":
        x = x - x.mean(0, keepdim=True)
    return F.normalize(x, dim=-1)


def geometry_report(events, embeds, y):
    d = np.array([lr_distance(*embeds[e["recording_id"]], e["t"]) or np.nan
                  for e in events], dtype=float)
    ok = ~np.isnan(d)
    dv, dn = d[ok & (y == 1)], d[ok & (y == 0)]
    X = torch.cat([v[0] for v in embeds.values()], 0).numpy()
    if len(X) > 6000:
        X = X[np.linspace(0, len(X) - 1, 6000).astype(int)]
    idx = np.random.RandomState(0).choice(len(X), size=min(1500, len(X)), replace=False)
    Xn = X[idx] / (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-12)
    cos = Xn @ Xn.T
    iu = np.triu_indices(len(Xn), k=1)
    return {
        "n": int(ok.sum()),
        "valid_median": float(np.median(dv)) if len(dv) else None,
        "motion_hn_median": float(np.median(dn)) if len(dn) else None,
        "distance_only_auroc": _auroc(y[ok], d[ok]),
        "effective_rank": effective_rank(X),
        "pairwise_cos_mean": float(cos[iu].mean()),
        "pairwise_cos_std": float(cos[iu].std()),
    }, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--out")
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    y = np.array([e["y"] for e in events], dtype=float)
    rids = sorted({e["recording_id"] for e in events})
    print(f"events: {len(events)} ({int(y.sum())} positive / {int(len(y)-y.sum())} motion-HN), "
          f"{len(rids)} recordings")

    # global mean over the recordings actually used
    tot, n = None, 0
    for rid in rids:
        f = by_rid[rid]["feats"].float()
        tot = f.sum(0) if tot is None else tot + f.sum(0)
        n += len(f)
    global_mean = tot / n
    print(f"global mean frame: norm={float(global_mean.norm()):.4f}  "
          f"(mean per-frame norm={float(torch.stack([by_rid[r]['feats'].float().norm(dim=-1).mean() for r in rids]).mean()):.4f})")

    results, dists = {}, {}
    for mode in TREATMENTS:
        embeds = {}
        for rid in rids:
            rec = by_rid[rid]
            embeds[rid] = (apply_centering(rec["feats"], mode, global_mean), rec["times"])
        rep, d = geometry_report(events, embeds, y)
        results[mode] = rep
        dists[mode] = d
        print(f"\n-- centering = {mode} --")
        print(f"  d(L,R): valid median={rep['valid_median']:.6f}  "
              f"motion-HN median={rep['motion_hn_median']:.6f}")
        print(f"  distance-only AUROC = {rep['distance_only_auroc']:.3f}")
        print(f"  effective rank = {rep['effective_rank']:.1f}  "
              f"pairwise cos = {rep['pairwise_cos_mean']:+.4f} +- {rep['pairwise_cos_std']:.4f}")

    base = results["none"]["distance_only_auroc"]
    print("\n=== verdict ===")
    best = max(TREATMENTS, key=lambda m: results[m]["distance_only_auroc"])
    gain = results[best]["distance_only_auroc"] - base
    print(f"  best treatment: {best}  (AUROC {results[best]['distance_only_auroc']:.3f} "
          f"vs uncentered {base:.3f}, gain {gain:+.3f})")
    if gain >= 0.03:
        print("  -> centering materially improves separability: the degenerate geometry was "
              "masking real signal. Re-run the feature-based arms with this treatment "
              "BEFORE drawing any further conclusion about representation limits.")
    elif gain > 0.0:
        print("  -> marginal gain: the compression was real but was not the binding "
              "constraint; the ranking was already monotone in the residual distance.")
    else:
        print("  -> no gain: the frozen representation genuinely does not encode this "
              "distinction, and rescaling cannot create it. This is evidence FOR needing "
              "different features (hand/contact state, motion), not better normalization.")
    print("  NOTE: distance-only AUROC is a single-feature probe, not the full model. A "
          "gain here is necessary, not sufficient -- it must survive the grouped-CV arms.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {a.out}")
        try:
            from src.eval.run_manifest import write_manifest
            write_manifest(a.out, input_paths=[a.gold, a.context] + a.feat_cache)
        except Exception as e:
            print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
