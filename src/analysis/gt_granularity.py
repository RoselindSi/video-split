"""Quantify GT naming-granularity problems (pure analysis, no model).

The per-segment naming eval showed the model collapses many GT segments of one
recording into a single name (e.g. 8 mug-washing sub-steps -> "Wash the mug").
But some of that is NOT the model's fault: the GT itself is often cyclic /
mechanically enumerated (e.g. "Paper fold-unfold cycle" repeated, "Slide bottle
left / back / forward ...", "iteration 1/2/3"), so a coarser prediction is
arguably correct and the metric over-penalizes it.

This script splits the naming difficulty into:
  (A) GT-granularity artifacts -- GT is cyclic/enumerated OR adjacent GT names
      are near-duplicates (the model SHOULD be allowed to give one coarse name);
  (B) genuine distinctions the model missed (adjacent GT names semantically
      distinct AND on the same object -> needs state/motion evidence).

Three orthogonal measures:
  1. cyclic/enumeration cue rate  : regex over GT names (cycle, iteration, pass,
     ordinals, "back and forth", "-unfold", "1st/2nd/3rd" ...).
  2. adjacent-GT self-similarity  : embedding sim between consecutive GT names
     within a recording (high = GT itself barely varies = granularity too fine).
  3. cross with model dup         : among recordings where the model repeats a
     lot (pred_uniq low), how much of that is explained by high adjacent GT
     self-similarity (bucket into A vs B).

Usage (server):
    python -m src.analysis.gt_granularity \
        --persegment_jsonl /tmp/naming_ablation_local.jsonl
"""
import argparse, json, re, statistics
from collections import defaultdict

try:
    from src.seg_rewards import _default_sim_fn
except ImportError:
    from src.rewards.seg_rewards import _default_sim_fn

CYCLIC_RE = re.compile(
    r"\bcycle\b|\biteration\b|\bpass\b|\brepeat|back[- ]and[- ]forth|"
    r"-unfold|-retract|-uncoil|\b\d+(st|nd|rd|th)\b|\bfirst\b|\bsecond\b|"
    r"\bthird\b|\bfourth\b|\bfifth\b|\bsixth\b|\bseventh\b|\beighth\b|"
    r"\b(starts|ends) here\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persegment_jsonl", required=True,
                    help="output of eval_naming_persegment.py (has gt_name, "
                         "pred_name, emb_sim, recording_id, segment_idx)")
    ap.add_argument("--adj_sim_thresh", type=float, default=0.8,
                    help="adjacent GT names with sim >= this are near-duplicates")
    ap.add_argument("--pred_dup_thresh", type=float, default=0.5,
                    help="recordings with pred_uniq < this are model-repetitive")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.persegment_jsonl)]
    sim = _default_sim_fn()

    # ---- 1. cyclic/enumeration cue rate over all GT names ----
    all_gt = [r["gt_name"] for r in recs]
    cyclic = [g for g in all_gt if CYCLIC_RE.search(g)]
    print(f"=== 1. cyclic/enumeration cue rate ===")
    print(f"  {len(cyclic)}/{len(all_gt)} GT names have a cyclic/enum cue "
          f"({len(cyclic)/len(all_gt):.1%})")
    print(f"  examples: {[g for g in cyclic[:5]]}\n")

    # ---- 2. adjacent-GT self-similarity within each recording ----
    by_rec = defaultdict(list)
    for r in recs:
        by_rec[r["recording_id"]].append(r)
    adj_sims_all = []
    rec_adj = {}
    for rid, segs in by_rec.items():
        segs.sort(key=lambda r: r["segment_idx"])
        gts = [s["gt_name"] for s in segs]
        adj = [sim(gts[i], [gts[i + 1]])[0] for i in range(len(gts) - 1)]
        rec_adj[rid] = statistics.mean(adj) if adj else 0.0
        adj_sims_all += adj
    print(f"=== 2. adjacent-GT self-similarity (high = GT itself barely varies) ===")
    print(f"  mean adjacent-GT sim across all pairs: {statistics.mean(adj_sims_all):.3f}")
    near_dup = sum(1 for s in adj_sims_all if s >= a.adj_sim_thresh)
    print(f"  adjacent GT pairs that are near-duplicates (>= {a.adj_sim_thresh}): "
          f"{near_dup}/{len(adj_sims_all)} ({near_dup/len(adj_sims_all):.1%})\n")

    # ---- 3. cross model-repetition with GT self-similarity: A vs B ----
    print(f"=== 3. model-repetitive recordings: is it GT's fault (A) or model's (B)? ===")
    print(f"{'recording':20s} {'pred_uniq':>9} {'gt_uniq':>8} {'adjGTsim':>9} "
          f"{'cyc%':>6} {'verdict':>10}")
    A = B = 0
    for rid, segs in sorted(by_rec.items()):
        preds = [s["pred_name"] for s in segs]
        gts = [s["gt_name"] for s in segs]
        pred_uniq = len(set(preds)) / len(preds)
        if pred_uniq >= a.pred_dup_thresh:
            continue                              # model not repetitive here
        gt_uniq = len(set(gts)) / len(gts)
        adjgt = rec_adj[rid]
        cyc = sum(1 for g in gts if CYCLIC_RE.search(g)) / len(gts)
        # A = GT-granularity artifact: GT near-dup adjacent OR heavily cyclic
        is_A = adjgt >= a.adj_sim_thresh or cyc >= 0.5
        verdict = "A(GT-fine)" if is_A else "B(model)"
        A += is_A; B += (not is_A)
        print(f"{rid:20s} {pred_uniq:9.0%} {gt_uniq:8.0%} {adjgt:9.3f} "
              f"{cyc:6.0%} {verdict:>10}")
    print(f"\n  A (GT granularity too fine, model's coarse name defensible): {A}")
    print(f"  B (GT genuinely distinct, model missed it -> needs state/motion): {B}")
    print("\n-> A share = how much of the 'repetition' failure is a metric/label "
          "artifact vs a real model capability gap (B).")


if __name__ == "__main__":
    main()
