"""Sample batch3: the confirmatory, deployment-distribution held-out test for
both frozen artifacts (primary no-clip, secondary clipped).

Per the review's protocol: >=30 recordings with ZERO overlap against every
recording used anywhere in development (audit_188_gold_v2.jsonl -- both the
original-72 and batch2 splits, 48 recordings total, regardless of which
split), sampled at RANDOM from the deployment distribution -- not stratified
by model score or difficulty the way batch2 was. batch2's stratified design
was right for DIAGNOSING failure modes; batch3's job is a single honest
precision/coverage number, which a stratified sample cannot give you no
matter how the strata are later reweighted.

WHY NAIVE CANDIDATE GENERATION, NOT THE TRAINED BOUNDARY HEAD: the trained
head's own saved logits (b2_logits.pt) cover only its ~30-recording val
split -- almost entirely already inside the 48 development recordings.
Re-running the trained head's inference on ~200 unused part_01 recordings
would need its checkpoint, which this session doesn't have confirmed access
to. So candidates here come from two sources that need ONLY the feature
cache (already available for every recording):

  gt_boundary       every annotated segment edge, away from the recording's
                    own start/end -- the real-transition side of the test.
  raw_change_peak   local maxima of context_change (hal_features.py; a
                    frozen-feature-only signal, no trained model) after a
                    percentile threshold + min-gap NMS -- a cheap, honest
                    proxy for "whatever a naive candidate generator would
                    flag", standing in for the false-positive side of the
                    test in the absence of the actual deployed decoder.
                    This is a real, disclosed limitation, not the literal
                    production candidate distribution -- say so plainly in
                    any report that uses this batch.

Both frozen artifacts are scored for every sampled event, but the scores are
written ONLY to the hidden manifest (batch3_manifest.jsonl) -- the CSV a
human labels from (batch3_blind_review.csv) carries no score, no candidate
type, and no derived category, matching the review's blind-review
requirement (independent temporal_truth first, no anchoring on a category or
confidence number).

Usage (server):
    python -m src.boundary.batch3_sample \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --primary_dir  /workspace/tr1/results/hal/frozen_verifier_primary_noclip \
        --secondary_dir /workspace/tr1/results/hal/frozen_verifier_secondary_clipped \
        --n_target 240 --min_recordings 30 --per_recording_cap 8 \
        --out_dir /workspace/tr1/results/hal/batch3
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at
from src.boundary.pairwise_verifier import (
    REL_NAMES, side_vectors, relative_features, pca_apply, pair_block, _impute_scale_apply,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def load_artifact(out_dir):
    cfg = json.load(open(os.path.join(out_dir, "frozen_verifier.json"), encoding="utf-8"))
    npz = np.load(os.path.join(out_dir, "frozen_verifier.npz"))
    return {
        "cfg": cfg,
        "pca": {"mu": npz["pca_mu"], "W": npz["pca_W"]},
        "rel_scaler": {"cm": npz["rel_cm"], "mu": npz["rel_mu"], "sd": npz["rel_sd"]},
        "pair_scaler": {"cm": npz["pair_cm"], "mu": npz["pair_mu"], "sd": npz["pair_sd"]},
        "w": npz["w"], "b": float(npz["b"]), "threshold": float(npz["threshold"]),
    }


def score_event(art, rec, t, other_times):
    """Replays freeze_verifier.py's fit_fold().apply() forward pass using a
    SAVED artifact instead of a freshly-fit one. Returns (score, scorable)."""
    clip = bool(art["cfg"]["clip_windows_at_neighbors"])
    l, r = side_vectors(rec, t, clip_neighbors=clip)
    if l is None:
        return None, False
    rel = relative_features(rec, t, other_times, clip_neighbors=clip)
    x_rel = np.array([[rel.get(k, np.nan) for k in art["cfg"]["rel_features"]]], dtype=float)
    zl = pca_apply(art["pca"], l.numpy()[None])
    zr = pca_apply(art["pca"], r.numpy()[None])
    rel_s = _impute_scale_apply(art["rel_scaler"], x_rel)
    P = np.concatenate([pair_block(zl, zr), rel_s], axis=1)
    P_s = _impute_scale_apply(art["pair_scaler"], P)
    score = float(_sigmoid(P_s @ art["w"] + art["b"])[0])
    return score, True


def gt_boundary_times(rec, min_from_edge=2.0):
    segs = rec.get("segments") or []
    times = rec["times"]
    t0, t1 = float(times[0]), float(times[-1])
    edges = set()
    for _name, s0, s1 in segs:
        for e in (float(s0), float(s1)):
            if t0 + min_from_edge < e < t1 - min_from_edge:
                edges.add(round(e, 2))
    return sorted(edges)


def naive_change_peaks(rec, grid_stride=1.0, half=2.0, min_gap=3.0, top_frac=0.15, max_grid=1200):
    """See module docstring -- a trained-model-free candidate generator."""
    times = rec["times"]
    t0, t1 = float(times[0]) + half, float(times[-1]) - half
    if t1 <= t0:
        return []
    grid = np.arange(t0, t1, grid_stride)
    if len(grid) > max_grid:
        grid = grid[np.linspace(0, len(grid) - 1, max_grid).astype(int)]
    vals = np.array([hal_features_at(rec["feats"], times, float(t), short_half=0.75,
                                     context_half=half).get("context_change") or 0.0
                     for t in grid])
    if not len(vals) or np.allclose(vals, vals[0]):
        return []
    thresh = np.quantile(vals, 1 - top_frac)
    order = np.argsort(-vals)
    picked = []
    for i in order:
        if vals[i] < thresh:
            break
        t = float(grid[i])
        if all(abs(t - p) >= min_gap for p in picked):
            picked.append(t)
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl",
                    help="defines the 48 recordings to EXCLUDE (all rows, both splits)")
    ap.add_argument("--exclude_manifest", action="append", default=[],
                    help="additional batch manifest(s) (e.g. batch3_manifest.jsonl) whose "
                         "recordings are ALSO excluded -- once a batch's labels are fully "
                         "known they graduate into development data and can no longer serve "
                         "as a blind test, so every later batch (batch4, batch5, ...) must "
                         "exclude every recording used by every earlier one, not just the "
                         "original 48 dev recordings.")
    ap.add_argument("--primary_dir", required=True)
    ap.add_argument("--secondary_dir", required=True)
    ap.add_argument("--n_target", type=int, default=240)
    ap.add_argument("--min_recordings", type=int, default=30)
    ap.add_argument("--per_recording_cap", type=int, default=8)
    ap.add_argument("--gt_frac", type=float, default=0.5,
                    help="target fraction of sampled events that are gt_boundary-anchored "
                         "(the rest are raw_change_peak-anchored). 0.5 is a neutral default, "
                         "not a claim about the true deployment ratio, which is unknown "
                         "without the trained head's own decode -- see module docstring.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    excluded = {g.get("recording_id") for g in gold if g.get("recording_id")}
    n_dev = len(excluded)
    for mp in a.exclude_manifest:
        with open(os.path.expanduser(mp), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    excluded.add(json.loads(line)["recording_id"])
    by_rid = load_feature_caches(a.feat_cache)
    new_recordings = sorted(set(by_rid) - excluded)
    print(f"excluding {n_dev} original development recordings"
          + (f" + {len(excluded) - n_dev} from {len(a.exclude_manifest)} prior batch "
             f"manifest(s)" if a.exclude_manifest else "")
          + f" = {len(excluded)} total")
    print(f"candidate pool: {len(new_recordings)} new recordings")
    if len(new_recordings) < a.min_recordings:
        raise SystemExit(f"only {len(new_recordings)} new recordings available -- need at "
                         f"least {a.min_recordings}")

    primary = load_artifact(a.primary_dir)
    secondary = load_artifact(a.secondary_dir)
    print(f"loaded primary  artifact: clip={primary['cfg']['clip_windows_at_neighbors']} "
          f"threshold={primary['threshold']:.4f}")
    print(f"loaded secondary artifact: clip={secondary['cfg']['clip_windows_at_neighbors']} "
          f"threshold={secondary['threshold']:.4f}")

    rng = random.Random(a.seed)
    pool_gt, pool_peak = [], []
    for rid in new_recordings:
        rec = by_rid[rid]
        for t in gt_boundary_times(rec):
            pool_gt.append((rid, t))
        for t in naive_change_peaks(rec):
            pool_peak.append((rid, t))
    print(f"raw candidate pool: {len(pool_gt)} gt_boundary, {len(pool_peak)} raw_change_peak "
          f"across {len(new_recordings)} recordings")

    rng.shuffle(pool_gt); rng.shuffle(pool_peak)
    n_gt_target = int(round(a.n_target * a.gt_frac))
    n_peak_target = a.n_target - n_gt_target

    per_rec = {}
    selected = []

    def take(pool, n, ctype):
        taken = 0
        for rid, t in pool:
            if taken >= n:
                break
            if per_rec.get(rid, 0) >= a.per_recording_cap:
                continue
            per_rec[rid] = per_rec.get(rid, 0) + 1
            selected.append({"recording_id": rid, "t": t, "candidate_type": ctype})
            taken += 1
        return taken

    got_gt = take(pool_gt, n_gt_target, "gt_boundary")
    got_peak = take(pool_peak, n_peak_target, "raw_change_peak")
    print(f"selected: {got_gt} gt_boundary + {got_peak} raw_change_peak = "
          f"{len(selected)} events across {len(per_rec)} recordings")
    if got_gt < n_gt_target or got_peak < n_peak_target:
        print(f"  !! pool exhausted before reaching target counts "
              f"(wanted {n_gt_target}/{n_peak_target}) -- raise --per_recording_cap or "
              f"widen the recording pool")
    if len(per_rec) < a.min_recordings:
        print(f"  !! only {len(per_rec)} recordings represented, below the "
              f"--min_recordings={a.min_recordings} target")

    os.makedirs(a.out_dir, exist_ok=True)
    manifest_rows, blind_rows = [], []
    other_times_by_rec = {}
    for e in selected:
        other_times_by_rec.setdefault(e["recording_id"], []).append(e["t"])

    for e in selected:
        rid, t, ctype = e["recording_id"], e["t"], e["candidate_type"]
        rec = by_rid[rid]
        others = [o for o in other_times_by_rec[rid] if abs(o - t) > 1e-6]
        p_score, p_scorable = score_event(primary, rec, t, others)
        s_score, s_scorable = score_event(secondary, rec, t, others)
        event_id = f"{rid}_batch3_{ctype}_t{t:.1f}"
        containing = next((s for s in (rec.get("segments") or []) if s[1] <= t <= s[2]), None)
        idx = sorted(rec.get("segments") or [], key=lambda s: s[1])
        prev_label = next((s[0] for s in reversed(idx) if s[2] <= t), None)
        next_label = next((s[0] for s in idx if s[1] >= t), None)
        manifest_rows.append({
            "event_id": event_id, "recording_id": rid, "t": round(t, 3),
            "candidate_type": ctype,
            "primary_score": p_score, "primary_scorable": p_scorable,
            "primary_provisional_keep": bool(p_scorable and p_score >= primary["threshold"]),
            "secondary_score": s_score, "secondary_scorable": s_scorable,
            "secondary_provisional_keep": bool(s_scorable and s_score >= secondary["threshold"]),
        })
        # BLIND sheet: context only, no score, no type, no derived decision.
        blind_rows.append({
            "event_id": event_id, "recording_id": rid, "t": round(t, 3),
            "containing_segment_label": containing[0] if containing else "",
            "prev_segment_label": prev_label or "", "next_segment_label": next_label or "",
            "temporal_truth": "", "notes": "",
            "clip_path": "", "contact_sheet_path": "",
        })

    with open(os.path.join(a.out_dir, "batch3_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import csv
    with open(os.path.join(a.out_dir, "batch3_blind_review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        w.writeheader(); w.writerows(blind_rows)

    print(f"\nwrote {len(manifest_rows)} rows to "
          f"{os.path.join(a.out_dir, 'batch3_manifest.jsonl')} (HIDDEN -- do not show to reviewer)")
    print(f"wrote {len(blind_rows)} rows to "
          f"{os.path.join(a.out_dir, 'batch3_blind_review.csv')} (BLIND -- fill temporal_truth "
          f"while watching the clip, no score/category visible)")
    print("next: python -m src.boundary.render_batch3_media ... to generate clip+contact_sheet "
          "(no score_plot -- showing a probability curve would break blind review)")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(os.path.join(a.out_dir, "batch3_manifest.jsonl"),
                       input_paths=[a.gold] + a.feat_cache +
                       [os.path.join(a.primary_dir, "frozen_verifier.npz"),
                        os.path.join(a.secondary_dir, "frozen_verifier.npz")])
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
