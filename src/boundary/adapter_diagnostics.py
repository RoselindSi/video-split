"""Diagnose WHY the contrastive state adapter made things worse, and surface
the audited pairs whose labels most contradict the raw geometry.

Nested-CV result that triggered this (all development numbers, 175 events /
47 recordings): v1 raw AUROC 0.702 -> v2 structural features on raw
embedding 0.608 -> v2 on the adapter embedding 0.576. Both new arms are
WORSE than the rejected baseline, so before writing any more model code we
need to know which of these is true:

  (a) the adapter never fit even its own training pairs   -> supervision or
      loss implementation is wrong, not a data-size problem;
  (b) it fit training pairs but not held-out recordings   -> cross-recording
      overfitting;
  (c) the embedding collapsed                             -> the contrastive
      objective degenerated (everything maps to one point / a low-rank
      subspace), which also explains a drop below the raw baseline;
  (d) the supervision itself is inconsistent              -> some audited
      "positive" pairs look identical in the raw geometry and some
      "motion_hard_negative" pairs look far apart; contrastive losses are
      much more sensitive to such pairs than a BCE head, which would explain
      why the adapter hurt while v1 (a plain logistic fit) tolerated them.

Three reports:

1. PAIR-DISTANCE GEOMETRY (raw vs adapter)
   median d(mu_L, mu_R) per class, and the AUROC obtainable from that single
   distance alone. The target geometry is d(valid) > d(motion_hard_negative).
   If the adapter makes the classes overlap MORE than raw does, the
   objective did not build the intended structure -- that is cause (a)/(c),
   and no amount of downstream feature engineering repairs it.

2. COLLAPSE CHECK (raw vs adapter)
   per-dimension variance, effective rank (exp of the entropy of the
   normalized singular-value spectrum), and the mean/std of pairwise cosine
   similarity across sampled frames. A sharp drop in effective rank or a
   pairwise-cosine mean near 1.0 on the adapter side is direct evidence of
   representation collapse.

3. CONTRADICTORY PAIRS (step 4 of the review's order)
   the audited events whose gold role disagrees most with the raw geometry:
   `positive` events with the SMALLEST raw left/right distance, and
   `motion_hard_negative` events with the LARGEST. These are exactly the
   pairs that would distort an embedding trained with push/pull losses. Each
   is written out with its media paths for a human to re-check whether the
   role should be `exclude`, or whether the corrected boundary window is
   off, or whether the window spans two actions.

Nothing here retrains or re-tunes anything -- it only measures the artifacts
that already exist.

Usage (server):
    python -m src.boundary.adapter_diagnostics \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --adapter /workspace/tr1/results/hal/state_adapter/state_adapter.pt \
        --media_csv /workspace/tr1/results/hal/batch2_media/audit_sample.csv \
        --out_dir /workspace/tr1/results/hal/state_adapter/diagnostics

(--adapter is optional: without it, reports 1 and 2 cover the raw embedding
only, and report 3 -- the supervision audit -- still works, since it is a
property of the raw geometry and the human labels.)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import (
    StateAdapter, build_events, pool_window, _auroc,
)


def lr_distance(emb, times, t, w=2.0):
    """d(mu_L, mu_R) with the SAME windows the contrastive loss uses, so the
    geometry reported here is the geometry the loss actually shaped."""
    cl = pool_window(emb, times, t - w, t)
    cr = pool_window(emb, times, t, t + w)
    if cl is None or cr is None:
        return None
    return float(1.0 - F.cosine_similarity(cl[None], cr[None]))


def effective_rank(X, max_rows=4000, center=False):
    """exp(entropy of the normalized singular-value spectrum): equals the true
    rank for a flat spectrum, falls toward 1 as energy concentrates in one
    direction.

    center=False by default, and that matters here. Mean-centering DEFEATS
    point-collapse detection: if every frame maps to the same vector plus
    tiny isotropic noise, centering subtracts exactly the collapsed direction
    and leaves full-rank noise, reporting a HIGH effective rank for a totally
    collapsed embedding (verified on synthetic data -- centered gave 62.9/64
    for a point-collapsed matrix). Since these embeddings are L2-normalized,
    the uncentered spectrum is the meaningful one: collapse shows up as one
    dominant singular value. The centered value is still reported alongside,
    because for NON-collapsed data it is the better measure of how many
    directions the variation actually uses."""
    if len(X) > max_rows:
        X = X[np.linspace(0, len(X) - 1, max_rows).astype(int)]
    M = X - X.mean(0, keepdims=True) if center else X
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_stats(emb_list, max_frames=6000):
    """emb_list: list of [T, D] tensors (one per recording, already
    normalized)."""
    X = torch.cat(emb_list, 0).numpy()
    if len(X) > max_frames:
        X = X[np.linspace(0, len(X) - 1, max_frames).astype(int)]
    norms = np.linalg.norm(X, axis=1)
    var = X.var(0)
    idx = np.random.RandomState(0).choice(len(X), size=min(1500, len(X)), replace=False)
    Xs = X[idx]
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
    cos = Xn @ Xn.T
    iu = np.triu_indices(len(Xn), k=1)
    return {
        "n_frames_sampled": int(len(X)), "dim": int(X.shape[1]),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "per_dim_var_mean": float(var.mean()), "per_dim_var_min": float(var.min()),
        "per_dim_var_max": float(var.max()),
        "effective_rank": effective_rank(X),
        "effective_rank_centered": effective_rank(X, center=True),
        "pairwise_cos_mean": float(cos[iu].mean()), "pairwise_cos_std": float(cos[iu].std()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--adapter", help="state_adapter.pt checkpoint (optional)")
    ap.add_argument("--media_csv", help="audit_sample.csv for clip paths in report 3")
    ap.add_argument("--top_k", type=int, default=15, help="contradictory pairs per class")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    os.makedirs(a.out_dir, exist_ok=True)
    y = np.array([e["y"] for e in events], dtype=float)
    print(f"events: {len(events)} ({int(y.sum())} positive / {int(len(y) - y.sum())} motion-HN) "
          f"across {len({e['recording_id'] for e in events})} recordings")

    model = None
    if a.adapter:
        ck = torch.load(a.adapter, map_location=a.device, weights_only=False)
        model = StateAdapter(ck["in_dim"]).to(a.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        print(f"loaded adapter {a.adapter}")
        print("  NOTE: this checkpoint was trained on ALL events except its early-stopping "
              "recordings, so distances below are largely IN-SAMPLE -- they measure the "
              "geometry the loss built, NOT generalization (that is the per-fold table).")

    embeds = {"raw": {}, "adapter": {}}
    for e in events:
        rid = e["recording_id"]
        if rid not in embeds["raw"]:
            embeds["raw"][rid] = (F.normalize(e["rec"]["feats"].float(), dim=-1), e["rec"]["times"])
            if model is not None:
                with torch.no_grad():
                    emb = model(e["rec"]["feats"].float().to(a.device)).cpu()
                embeds["adapter"][rid] = (emb, e["rec"]["times"])

    # ---- report 1: pair-distance geometry --------------------------------
    print("\n" + "=" * 74)
    print("1. PAIR-DISTANCE GEOMETRY  (want: d(valid) > d(motion_hard_negative))")
    print("=" * 74)
    geo = {}
    for space in (["raw", "adapter"] if model is not None else ["raw"]):
        d = np.array([lr_distance(*embeds[space][e["recording_id"]], e["t"])
                      if e["recording_id"] in embeds[space] else np.nan for e in events],
                     dtype=float)
        ok = ~np.isnan(d)
        dv, dn = d[ok & (y == 1)], d[ok & (y == 0)]
        auc = _auroc(y[ok], d[ok])
        geo[space] = {"n": int(ok.sum()),
                      "valid_median": float(np.median(dv)) if len(dv) else None,
                      "valid_iqr": [float(np.percentile(dv, 25)), float(np.percentile(dv, 75))] if len(dv) else None,
                      "motion_hn_median": float(np.median(dn)) if len(dn) else None,
                      "motion_hn_iqr": [float(np.percentile(dn, 25)), float(np.percentile(dn, 75))] if len(dn) else None,
                      "distance_only_auroc": auc}
        print(f"  {space:<8} valid median={geo[space]['valid_median']:.4f}  "
              f"motion-HN median={geo[space]['motion_hn_median']:.4f}  "
              f"distance-only AUROC={auc:.3f}")
        if space == "raw":
            raw_d = d
    if model is not None and geo["adapter"]["distance_only_auroc"] <= geo["raw"]["distance_only_auroc"]:
        print("  -> the adapter did NOT improve (or worsened) class separation even "
              "IN-SAMPLE: the contrastive objective failed to build the intended "
              "geometry, so this is cause (a)/(c), not a downstream feature problem.")

    # ---- report 2: collapse ----------------------------------------------
    print("\n" + "=" * 74)
    print("2. COLLAPSE CHECK")
    print("=" * 74)
    coll = {}
    for space in (["raw", "adapter"] if model is not None else ["raw"]):
        coll[space] = collapse_stats([v[0] for v in embeds[space].values()])
        c = coll[space]
        print(f"  {space:<8} eff_rank={c['effective_rank']:.1f}/{c['dim']} "
              f"(centered {c['effective_rank_centered']:.1f})  "
              f"per-dim var mean={c['per_dim_var_mean']:.2e}  "
              f"pairwise cos={c['pairwise_cos_mean']:+.3f}±{c['pairwise_cos_std']:.3f}")
    if model is not None:
        r, ad = coll["raw"], coll["adapter"]
        if ad["effective_rank"] < 0.25 * ad["dim"] or ad["pairwise_cos_mean"] > 0.9:
            print("  -> COLLAPSE: the adapter output occupies a very low-rank subspace / "
                  "all frames are near-parallel. The pull term dominated; fix with a "
                  "variance/decorrelation term or a much smaller pull weight.")
        else:
            print("  -> no strong collapse signature; the adapter's failure is more likely "
                  "inconsistent supervision (report 3) or too few recordings.")

    # ---- report 3: contradictory audited pairs ---------------------------
    print("\n" + "=" * 74)
    print(f"3. CONTRADICTORY AUDITED PAIRS (top {a.top_k} per class, by raw geometry)")
    print("=" * 74)
    media = {}
    if a.media_csv and os.path.exists(a.media_csv):
        with open(a.media_csv, newline="", encoding="utf-8", errors="replace") as f:
            media = {r["event_id"].strip(): r for r in csv.DictReader(f) if r.get("event_id")}

    rows = []
    ok = ~np.isnan(raw_d)
    pos_idx = [i for i in range(len(events)) if ok[i] and y[i] == 1]
    neg_idx = [i for i in range(len(events)) if ok[i] and y[i] == 0]
    pos_sorted = sorted(pos_idx, key=lambda i: raw_d[i])[:a.top_k]           # smallest distance
    neg_sorted = sorted(neg_idx, key=lambda i: -raw_d[i])[:a.top_k]          # largest distance
    for kind, idxs in [("positive_but_looks_identical", pos_sorted),
                       ("motion_hn_but_looks_different", neg_sorted)]:
        print(f"\n  -- {kind} --")
        for i in idxs:
            e = events[i]
            m = media.get(e["event_id"], {})
            print(f"    d={raw_d[i]:.4f}  {e['event_id']}")
            rows.append({
                "contradiction_kind": kind, "event_id": e["event_id"],
                "recording_id": e["recording_id"], "t": e["t"],
                "gold_role": "positive" if e["y"] == 1 else "motion_hard_negative",
                "raw_left_right_distance": round(float(raw_d[i]), 6),
                "clip_path": m.get("clip_path", ""),
                "contact_sheet_path": m.get("contact_sheet_path", ""),
                "score_plot_path": m.get("score_plot_path", ""),
                "prev_segment_label": m.get("prev_segment_label", ""),
                "next_segment_label": m.get("next_segment_label", ""),
                "containing_segment_label": m.get("containing_segment_label", ""),
                # blanks for the human pass
                "verdict_role_should_be": "", "window_spans_two_actions": "",
                "corrected_boundary_time_wrong": "", "only_annotation_convention": "",
                "unlabelled_subaction_present": "", "notes": "",
            })
    out_csv = os.path.join(a.out_dir, "contradictory_pairs_review.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {out_csv} ({len(rows)} rows, 6 blank verdict columns)")
    print("  read: many `positive_but_looks_identical` turning out to be annotation-"
          "convention-only splits, or many `motion_hn_but_looks_different` containing "
          "unlabelled subactions, means the CONTRASTIVE SUPERVISION is inconsistent -- "
          "those pairs must become `exclude` before any push/pull training is retried.")

    out = os.path.join(a.out_dir, "adapter_diagnostics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"pair_geometry": geo, "collapse": coll,
                   "contradictory_pairs_csv": out_csv}, f, indent=2)
    print(f"wrote {out}")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(out, input_paths=[a.gold, a.context] + a.feat_cache)
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
