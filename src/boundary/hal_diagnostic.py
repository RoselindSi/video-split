"""Step 1 of the HAL proposal (per the mentor design): validate the cheap
features on the 72 audited gold events BEFORE training anything. Pure
diagnostic -- no classifier, no contrastive loss, just: do short_change /
context_change / change_persistence / internal_variance actually separate
real boundaries from motion-induced false ones, in data we've already hand-
labeled the ground truth for?

If they don't separate cleanly here, they won't magically work once wired
into a classifier or a contrastive loss -- this is the honest place to find
that out, at near-zero cost (CPU-only, no VLM, no GPU).

Two class distinctions are checked (both meaningful, for different reasons):
  temporal_truth: valid vs spurious
      -- the visual auditor's own target distinction (this session's whole
      focus). If HAL features separate this cleanly and the auditor
      struggles, that's a strong argument for routing "easy" cases to HAL
      and only "uncertain" ones to the (expensive, currently unreliable) VLM.
  boundary_contrastive_role: positive vs motion_hard_negative
      -- the actual training-signal distinction a future reranker/dense
      scorer would need. This is the one the mentor's cascade design cares
      about most directly.

Usage (server, after extract_features_recseg.py has produced a cache
containing the 72 audited recordings -- point at whichever file(s) actually
have them, train and/or val):
    python -m src.boundary.hal_diagnostic \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --out /workspace/tr1/results/hal/hal_diagnostic_72.json
"""
from __future__ import annotations

import argparse
import json
import os

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at

FEATURES = ["short_change", "context_change", "change_persistence",
            "left_internal_variance", "right_internal_variance"]


def _auc(pos_vals, neg_vals):
    """Rank-based AUC (P(random positive's value > random negative's)),
    average-rank tie handling. None if either class is empty."""
    if not pos_vals or not neg_vals:
        return None
    vals = sorted([(v, 1) for v in pos_vals] + [(v, 0) for v in neg_vals], key=lambda x: x[0])
    n = len(vals)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, c) in zip(ranks, vals) if c == 1)
    n_pos, n_neg = len(pos_vals), len(neg_vals)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {"n": 0}
    n = len(vals)
    return {"n": n, "mean": sum(vals) / n, "median": vals[n // 2],
            "min": vals[0], "max": vals[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="one or more extract_features_recseg.py .pt caches; repeat for train+val")
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...)")
    ap.add_argument("--context", help="context jsonl (default: committed data/gold/...)")
    ap.add_argument("--short_half", type=float, default=0.75)
    ap.add_argument("--context_half", type=float, default=3.0)
    ap.add_argument("--out", help="write JSON summary here")
    a = ap.parse_args()

    gold_path, ctx_path = S.default_gold_paths()
    gold = S.load_gold(a.gold or gold_path)
    ctx = S.load_context(a.context or ctx_path)

    print(f"loading {len(a.feat_cache)} feature cache(s)...")
    by_rid = load_feature_caches(a.feat_cache)
    print(f"  {len(by_rid)} recordings indexed")

    rows = []
    missing_recording = 0
    for g in gold:
        eid = g["event_id"]
        c = ctx.get(eid, {})
        rid = g.get("recording_id") or c.get("recording_id")
        rec = by_rid.get(rid)
        if rec is None:
            missing_recording += 1
            continue
        t = c.get("pred_time")
        if t is None:
            t = c.get("gt_time")
        if t is None:
            continue
        feats_at_t = hal_features_at(rec["feats"], rec["times"], float(t),
                                     short_half=a.short_half, context_half=a.context_half)
        rows.append({
            "event_id": eid, "recording_id": rid, "t": t,
            "temporal_truth": g.get("temporal_truth"),
            "boundary_contrastive_role": g.get("boundary_contrastive_role"),
            **feats_at_t,
        })

    print(f"computed HAL features for {len(rows)}/{len(gold)} events "
          f"({missing_recording} skipped -- recording not in --feat_cache)")

    def _class_vals(rows, label_field, pos_val, neg_val, feature):
        pos = [r[feature] for r in rows if r.get(label_field) == pos_val and r[feature] is not None]
        neg = [r[feature] for r in rows if r.get(label_field) == neg_val and r[feature] is not None]
        return pos, neg

    summary = {"n_gold": len(gold), "n_computed": len(rows),
              "n_missing_recording": missing_recording, "comparisons": {}}

    for label_field, pos_val, neg_val, tag in [
        ("temporal_truth", "valid", "spurious", "temporal_truth: valid(+) vs spurious(-)"),
        ("boundary_contrastive_role", "positive", "motion_hard_negative",
         "boundary_contrastive_role: positive(+) vs motion_hard_negative(-)"),
    ]:
        print(f"\n=== {tag} ===")
        comp = {}
        for feat in FEATURES:
            pos, neg = _class_vals(rows, label_field, pos_val, neg_val, feat)
            auc = _auc(pos, neg)
            ps, ns = _stats(pos), _stats(neg)
            comp[feat] = {"auc_higher_favors_positive": auc, "positive": ps, "negative": ns}
            auc_str = f"{auc:.3f}" if auc is not None else "n/a"
            print(f"  {feat:<26} AUC={auc_str}  "
                  f"pos(n={ps.get('n',0)}) mean={ps.get('mean', float('nan')):.4f}  "
                  f"neg(n={ns.get('n',0)}) mean={ns.get('mean', float('nan')):.4f}")
        summary["comparisons"][label_field] = comp

    print("\nReading the AUC: 0.5 = no separation (useless), 1.0 = perfectly separates with "
          "positive always higher, 0.0 = perfectly separates with negative always higher "
          "(still useful -- just flip the sign). Values near 0.5 for a feature mean it does "
          "NOT carry the signal the HAL proposal expects, for THIS distinction -- a real "
          "finding, not a bug to fix.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")

    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(a.out or "/tmp/hal_diagnostic.json", input_paths=[gold_path, ctx_path] + a.feat_cache)
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
