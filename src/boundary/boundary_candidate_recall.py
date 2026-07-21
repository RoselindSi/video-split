"""B-final (2) -- candidate recall vs proposal budget, and raw local-peak
rank distribution near each GT boundary. Answers the question
boundary_error_audit.py's fixed-threshold decoder can't: if you decoupled
"how many candidates do we keep" from a fixed probability threshold, how
many candidates would you need to reach high recall? This is the actual
question for a two-stage design (cheap high-recall proposal stage + a
second-stage reranker/classifier), separate from "is the current single-
stage threshold decoder good" (already answered: no).

No model calls -- reads a saved logits file (train_head_multi.py
--save_logits).

1. CANDIDATE RECALL vs BUDGET: candidates = ALL local maxima of the
   probability curve (not threshold-filtered). For budgets expressed as
   density (peaks per 10s of video) and as a fixed absolute count (top-K),
   selects the top-K candidates per video by score (with min_gap NMS applied
   within the candidate pool itself, so budget isn't wasted on duplicate
   peaks 0.1s apart), and reports recall@{0.5,1.0}s and the resulting
   candidates/GT ratio -- this is a proper one-to-one-matched recall at each
   budget, not the naive "nearest distance" rescue-rate from
   boundary_error_audit.py.

2. RAW LOCAL-PEAK RANK DISTRIBUTION: for each GT boundary, finds the best
   local maximum within +-1s, then reports its RANK by score both among ALL
   local maxima in that recording, and restricted to local maxima within a
   10s window centered on the GT (rank1 / rank2-3 / rank4-5 / rank>5 /
   no_candidate). If most true boundaries' candidates sit in the top few
   ranks even globally, a reranker has a lot to work with; if they're
   consistently ranked low even in a tight local window, the raw scorer
   isn't discriminative enough locally and no reranker downstream will fix it.

Usage (server):
    python -m src.boundary.boundary_candidate_recall \
        --logits /workspace/tr1/results/boundary/b2_logits.pt \
        --out_dir /workspace/tr1/results/boundary/error_audit \
        --min_gap 1.0 --tol 0.5
"""
import argparse, json, os, statistics
from collections import Counter

import torch

from src.boundary.decode_sweep import bf1
from src.boundary.boundary_error_audit import all_local_maxima


def select_topk_with_nms(idx_by_score, prob, times, k, min_gap):
    kept = []
    for i in idx_by_score:
        if len(kept) >= k:
            break
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.logits)
    data = torch.load(a.logits, weights_only=False)
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------------- candidate recall vs budget ----------------
    budgets_per_10s = [0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"\n=== candidate recall vs proposal budget (candidates = ALL local "
          f"maxima, NOT threshold-filtered; min_gap={a.min_gap}s NMS applied "
          f"within the candidate pool) ===")
    print(f"{'budget (peaks/10s)':22s} {'recall@0.5':>11s} {'recall@1.0':>11s} "
          f"{'mean K':>8s} {'K/GT ratio':>11s}")
    for budget in budgets_per_10s:
        r05, r10, ks, ratios = [], [], [], []
        for v in data:
            prob, times, gts = v["prob"], v["times"], v["gt"]
            duration = times[-1] - times[0] if len(times) > 1 else 0
            k = max(1, round(budget * duration / 10))
            local_idx = all_local_maxima(prob, times)
            ranked = sorted(local_idx, key=lambda i: -prob[i])
            preds = select_topk_with_nms(ranked, prob, times, k, a.min_gap)
            r05.append(bf1(preds, gts, 0.5)[2]); r10.append(bf1(preds, gts, 1.0)[2])
            ks.append(len(preds)); ratios.append(len(preds) / max(len(gts), 1))
        print(f"{budget:22.1f} {statistics.mean(r05):11.1%} {statistics.mean(r10):11.1%} "
              f"{statistics.mean(ks):8.1f} {statistics.mean(ratios):11.2f}")

    # all local maxima, no budget cap (upper bound -- every real local max, incl. tiny ones)
    r05, r10, ks, ratios = [], [], [], []
    for v in data:
        prob, times, gts = v["prob"], v["times"], v["gt"]
        local_idx = all_local_maxima(prob, times)
        ranked = sorted(local_idx, key=lambda i: -prob[i])
        preds = select_topk_with_nms(ranked, prob, times, len(local_idx), a.min_gap)
        r05.append(bf1(preds, gts, 0.5)[2]); r10.append(bf1(preds, gts, 1.0)[2])
        ks.append(len(preds)); ratios.append(len(preds) / max(len(gts), 1))
    print(f"{'all local maxima':22s} {statistics.mean(r05):11.1%} {statistics.mean(r10):11.1%} "
          f"{statistics.mean(ks):8.1f} {statistics.mean(ratios):11.2f}")
    print("read: this is the CEILING recall achievable by ANY proposal stage "
          "built on this scorer (every local max, no budget limit, only "
          "min_gap NMS). If even this is well below ~95%, the raw scorer "
          "itself is missing signal for a meaningful fraction of boundaries "
          "-- no amount of reranking or budget tuning downstream can recover "
          "those; that's a representation problem, not a proposal-budget one.")

    # ---------------- raw local-peak rank distribution ----------------
    global_bins = Counter(); local10s_bins = Counter()
    n_gt = 0

    def bin_rank(rank):
        if rank is None:
            return "no_candidate"
        if rank == 1:
            return "rank1"
        if rank <= 3:
            return "rank2-3"
        if rank <= 5:
            return "rank4-5"
        return "rank>5"

    for v in data:
        prob, times, gts = v["prob"], v["times"], v["gt"]
        local_idx = all_local_maxima(prob, times)
        global_ranked = sorted(local_idx, key=lambda i: -prob[i])
        global_rank_of = {i: r + 1 for r, i in enumerate(global_ranked)}
        for g in gts:
            n_gt += 1
            near = [i for i in local_idx if abs(times[i] - g) <= 1.0]
            best = max(near, key=lambda i: prob[i]) if near else None
            global_bins[bin_rank(global_rank_of[best] if best is not None else None)] += 1
            if best is not None:
                window10 = [i for i in local_idx if abs(times[i] - g) <= 5.0]
                window10_ranked = sorted(window10, key=lambda i: -prob[i])
                local_rank = window10_ranked.index(best) + 1
                local10s_bins[bin_rank(local_rank)] += 1
            else:
                local10s_bins["no_candidate"] += 1

    print(f"\n=== raw local-peak rank near each GT boundary (n_gt={n_gt}) ===")
    order = ["rank1", "rank2-3", "rank4-5", "rank>5", "no_candidate"]
    print("rank of best local max within +-1s of GT, among ALL local maxima in the recording:")
    for k in order:
        print(f"  {k:14s} {global_bins[k]:5d}  {global_bins[k]/max(n_gt,1):.1%}")
    print("rank of best local max within +-1s of GT, among local maxima within a 10s window:")
    for k in order:
        print(f"  {k:14s} {local10s_bins[k]:5d}  {local10s_bins[k]/max(n_gt,1):.1%}")
    print("read: if most GTs' best local candidate is rank1-3 even GLOBALLY "
          "(whole recording), a reranker/second-stage classifier has an easy "
          "job -- the raw scorer already isolates the right region, just "
          "doesn't call it a boundary. If rank is consistently >5 even in the "
          "tight 10s window, the local score itself doesn't distinguish the "
          "true boundary from nearby noise -- representation needs to improve, "
          "not just the decision layer on top of it.")

    out_path = os.path.join(a.out_dir, "candidate_recall_summary.json")
    with open(out_path, "w") as f:
        json.dump({"budgets_per_10s": budgets_per_10s,
                   "global_rank_bins": dict(global_bins),
                   "local10s_rank_bins": dict(local10s_bins), "n_gt": n_gt}, f, indent=2)
    write_manifest(out_path, input_paths=[a.logits], extra={"min_gap": a.min_gap, "tol": a.tol})
    print(f"\nwrote summary -> {out_path}")


if __name__ == "__main__":
    main()
