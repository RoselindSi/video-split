"""Select a NEW, stratified ~100-150 event batch for human audit, isolated
from the 24 recordings already in the frozen Gold v2 set -- the held-out
validation the mentor asked for before trusting `HAL>=0.85 -> provisional_
keep` (fit on the original 72-event gold set; 0.90 precision but n=10,
too small to deploy on unseen recordings without a genuine held-out check).

Reuses the EXISTING boundary_error_audit.py `predictions.jsonl` (already
covers all ~220 recordings; the original 72-event gold set was itself a
stratified subsample drawn from it) -- flattens it into candidates the same
way render_audit_media.py does, excludes the 24 gold recording_ids, scores
every remaining candidate with a HAL model FULL-FIT on all 63 usable gold
events (no held-out fold -- there is no fold structure left once you're
scoring genuinely new recordings), and stratified-samples into two arms:

  DEPLOYMENT_DISTRIBUTION (report as your real coverage/precision estimate):
    - random       : uniform draw over ALL remaining candidates
    - hal_high     : HAL P(valid) >= 0.85 -- exactly the rule being deployed
    - hal_near     : 0.75 <= HAL P(valid) < 0.85 -- the near-miss band

  STRESS_TEST (report separately -- deliberately targeted, NOT representative
  of the true population mix; use only to look for failure modes):
    - fast_action_like       : GT-anchored candidate whose relevant segment
                                duration < 2.0s (heuristic proxy for "brief
                                real action" -- not a guarantee)
    - repetitive_motion_like : FP-anchored candidate inside a segment longer
                                than 6.0s (heuristic proxy for "probably a
                                multi-cycle action like wiping/folding" --
                                not a guarantee)
    - ambiguous_like         : GT boundary with another GT boundary within
                                1.0s (heuristic proxy for annotation-
                                ambiguity zones)

Each candidate is assigned to exactly ONE stratum (priority order above --
hal_high beats hal_near beats fast_action_like beats repetitive_motion_like
beats ambiguous_like beats random), with a per-recording cap so no single
recording dominates a stratum.

Output: (1) a predictions.jsonl-SHAPED subset (only selected events, same
nested per-recording gt_boundaries/predicted_peaks structure) -- feed this
directly into the EXISTING, unmodified render_audit_media.py as --predictions
to generate clip/contact-sheet/score-plot media exactly as before; pass
generously large --n_per_category/--per_video_cap so its OWN internal
sampling doesn't drop anything you already selected here. (2) a manifest
jsonl carrying event_id -> stratum/arm/hal_score, since render_audit_media's
own output CSV doesn't know about these new tags -- join on event_id when
you report deployment-distribution vs stress-test numbers separately.

Usage:
    python -m src.boundary.hal_audit_batch_select \
        --predictions /workspace/tr1/results/boundary/error_audit/predictions.jsonl \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --out /workspace/tr1/results/hal/batch2_predictions.jsonl \
        --out_manifest /workspace/tr1/results/hal/batch2_manifest.jsonl

Then (reusing the existing script unchanged):
    python -m src.boundary.render_audit_media \
        --predictions /workspace/tr1/results/hal/batch2_predictions.jsonl \
        --logits <SAME --logits path used for the original batch> \
        --data <SAME recseg json used for the original batch> \
        --out_dir /workspace/tr1/results/hal/batch2_media \
        --n_per_category 500 --per_video_cap 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at
from src.boundary.hal_vlm_fusion import (
    HAL_FEATURE_NAMES, build_event_rows, fit_full_hal_model, score_hal_model,
)

FAST_ACTION_MAX_S = 2.0
REPETITIVE_MIN_S = 6.0
AMBIGUOUS_GAP_S = 1.0


def _segment_duration_at(segments, t):
    """segments: list of (label, start, end). Duration of the segment
    containing t, or the nearest one if t falls in an annotation gap. None
    if there are no segments at all."""
    if not segments:
        return None
    containing = [s for s in segments if s[1] <= t <= s[2]]
    if containing:
        s = containing[0]
        return float(s[2] - s[1])
    nearest = min(segments, key=lambda s: min(abs(s[1] - t), abs(s[2] - t)))
    return float(nearest[2] - nearest[1])


def flatten_candidates(preds, exclude_recordings):
    """Same flattening render_audit_media.py does (status/category ->
    (recording_id, center_time, extra_dict)), skipping excluded recordings."""
    cands = []
    for rec in preds:
        rid = rec["recording_id"]
        if rid in exclude_recordings:
            continue
        for g in rec.get("gt_boundaries", []):
            cat = f"missed_{g['signal']}" if g["status"] == "missed" else g["status"]
            cands.append({"recording_id": rid, "center": g["gt_time"], "category": cat,
                         "anchor": "gt", "extra": g})
        for p in rec.get("predicted_peaks", []):
            cands.append({"recording_id": rid, "center": p["pred_time"], "category": p["status"],
                         "anchor": "pred", "extra": p})
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="existing boundary_error_audit.py predictions.jsonl")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...) -- defines excluded recordings")
    ap.add_argument("--context", help="context jsonl (default: committed data/gold/...) -- for fitting the HAL model")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--variance_half", type=float, default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--n_random", type=int, default=40)
    ap.add_argument("--n_hal_high", type=int, default=25)
    ap.add_argument("--n_hal_near", type=int, default=20)
    ap.add_argument("--n_fast_action", type=int, default=15)
    ap.add_argument("--n_repetitive", type=int, default=15)
    ap.add_argument("--n_ambiguous", type=int, default=5)
    ap.add_argument("--per_recording_cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="predictions.jsonl-shaped output (selected events only)")
    ap.add_argument("--out_manifest", required=True, help="event_id -> stratum/arm/hal_score jsonl")
    a = ap.parse_args()

    gold_path, ctx_path = S.default_gold_paths()
    gold = S.load_gold(a.gold or gold_path)
    ctx = S.load_context(a.context or ctx_path)
    by_rid = load_feature_caches(a.feat_cache)
    exclude_recordings = {g.get("recording_id") or ctx.get(g["event_id"], {}).get("recording_id") for g in gold}
    exclude_recordings.discard(None)
    print(f"excluding {len(exclude_recordings)} already-audited recordings")

    fit_rows = build_event_rows(gold, ctx, by_rid, {}, short_half=a.short_half,
                                context_half=a.context_half, variance_half=a.variance_half)
    import numpy as np
    X_fit = np.array([r["hal"] for r in fit_rows], dtype=float)
    y_fit = np.array([r["y"] for r in fit_rows], dtype=float)
    model = fit_full_hal_model(X_fit, y_fit, l2=a.l2)
    print(f"fit HAL model on {len(fit_rows)} gold events "
          f"({int(y_fit.sum())} valid / {int(len(y_fit) - y_fit.sum())} spurious)")

    preds = [json.loads(l) for l in open(a.predictions)]
    cands = flatten_candidates(preds, exclude_recordings)
    print(f"candidate pool after exclusion: {len(cands)} events across "
          f"{len(set(c['recording_id'] for c in cands))} recordings")

    for c in cands:
        rid, t = c["recording_id"], c["center"]
        rec = by_rid.get(rid)
        if rec is None:
            c["hal_score"] = None
            c["duration"] = None
            continue
        hal = hal_features_at(rec["feats"], rec["times"], float(t),
                              short_half=a.short_half, context_half=a.context_half,
                              variance_half=a.variance_half)
        c["hal_score"] = score_hal_model(model, [hal.get(k) for k in HAL_FEATURE_NAMES])
        c["duration"] = _segment_duration_at(rec.get("segments", []), float(t))

    scored = [c for c in cands if c["hal_score"] is not None]
    print(f"scored (had a matching feature-cache recording): {len(scored)}/{len(cands)}")

    def is_hal_high(c):
        return c["hal_score"] >= 0.85

    def is_hal_near(c):
        return 0.75 <= c["hal_score"] < 0.85

    def is_fast_action(c):
        return c["anchor"] == "gt" and c["duration"] is not None and c["duration"] < FAST_ACTION_MAX_S

    def is_repetitive(c):
        return c["anchor"] == "pred" and c["duration"] is not None and c["duration"] > REPETITIVE_MIN_S

    def is_ambiguous(c):
        gap = c["extra"].get("nearest_next_gap_s")
        return c["anchor"] == "gt" and gap is not None and gap < AMBIGUOUS_GAP_S

    rng = random.Random(a.seed)
    rng.shuffle(scored)
    used_ids = set()
    per_rec_count = Counter()
    selected = []

    def take(pool_filter, n, stratum, arm):
        taken = 0
        for c in pool_filter_iter(pool_filter):
            if taken >= n:
                break
            key = (c["recording_id"], c["center"])
            if key in used_ids or per_rec_count[c["recording_id"]] >= a.per_recording_cap:
                continue
            used_ids.add(key)
            per_rec_count[c["recording_id"]] += 1
            c["sample_stratum"] = stratum
            c["sample_arm"] = arm
            selected.append(c)
            taken += 1
        return taken

    def pool_filter_iter(f):
        for c in scored:
            if (c["recording_id"], c["center"]) not in used_ids and f(c):
                yield c

    n1 = take(is_hal_high, a.n_hal_high, "hal_high", "deployment_distribution")
    n2 = take(is_hal_near, a.n_hal_near, "hal_near", "deployment_distribution")
    n3 = take(is_fast_action, a.n_fast_action, "fast_action_like", "stress_test")
    n4 = take(is_repetitive, a.n_repetitive, "repetitive_motion_like", "stress_test")
    n5 = take(is_ambiguous, a.n_ambiguous, "ambiguous_like", "stress_test")
    n6 = take(lambda c: True, a.n_random, "random", "deployment_distribution")

    print(f"\nselected: hal_high={n1} hal_near={n2} fast_action_like={n3} "
          f"repetitive_motion_like={n4} ambiguous_like={n5} random={n6}  "
          f"(total={len(selected)})")
    for name, target, got in [("hal_high", a.n_hal_high, n1), ("hal_near", a.n_hal_near, n2),
                              ("fast_action_like", a.n_fast_action, n3),
                              ("repetitive_motion_like", a.n_repetitive, n4),
                              ("ambiguous_like", a.n_ambiguous, n5),
                              ("random", a.n_random, n6)]:
        if got < target:
            print(f"  !! {name}: only found {got}/{target} -- pool exhausted, "
                  f"widen the heuristic thresholds or raise --per_recording_cap")

    # rebuild the predictions.jsonl-shaped output: only selected events,
    # grouped back by recording_id into gt_boundaries/predicted_peaks
    by_recording_out = defaultdict(lambda: {"gt_boundaries": [], "predicted_peaks": []})
    manifest = []
    for c in selected:
        rid = c["recording_id"]
        if c["anchor"] == "gt":
            by_recording_out[rid]["gt_boundaries"].append(c["extra"])
        else:
            by_recording_out[rid]["predicted_peaks"].append(c["extra"])
        event_id = f"{rid}_{c['category']}_t{c['center']:.1f}"
        manifest.append({"event_id": event_id, "recording_id": rid,
                         "category": c["category"], "center": c["center"],
                         "sample_stratum": c["sample_stratum"], "sample_arm": c["sample_arm"],
                         "hal_score": round(c["hal_score"], 4), "duration": c["duration"]})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for rid, d in by_recording_out.items():
            f.write(json.dumps({"recording_id": rid, **d}, ensure_ascii=False) + "\n")
    with open(a.out_manifest, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nwrote {a.out} ({sum(len(d['gt_boundaries']) + len(d['predicted_peaks']) for d in by_recording_out.values())} events, "
          f"{len(by_recording_out)} recordings)")
    print(f"wrote {a.out_manifest}")

    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(a.out, input_paths=[a.predictions, gold_path, ctx_path] + a.feat_cache)
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
