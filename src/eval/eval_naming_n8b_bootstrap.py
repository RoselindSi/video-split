"""N8 follow-up -- paired bootstrap CI on the whole_segment vs contrastive
deltas from N8 (n=25 compound items, too small to trust a point estimate:
0.640 -> 0.672 AUC could easily be noise). No model calls -- reads
eval_naming_n8_windowed.py's saved per-candidate scores.

Resamples by RECORDING (not by segment or candidate) with replacement --
candidates within a segment, and segments within a recording, are not
independent (same video, same annotator, same object-family), so segment- or
candidate-level bootstrap would understate the true variance.

Reports 95% CI (percentile method) for: negative-margin-rate delta,
Recall@1 delta, mean-margin delta, pairwise-AUC delta (contrastive - whole).

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n8b_bootstrap --jsonl /tmp/n8_windowed.jsonl --n_boot 2000
"""
import argparse, json, random
from collections import defaultdict


def compute_metrics(items):
    """items: list of (secondary_gt, scores, primary_letter). Returns dict or
    None if too few usable compound items in this resample."""
    margins, ranks = [], []
    pooled_pos, pooled_neg = [], []
    for secondary_gt, scores, primary_letter in items:
        non_primary = {l: s for l, s in scores.items() if l != primary_letter}
        pos = {l: s for l, s in non_primary.items() if l in secondary_gt}
        neg = {l: s for l, s in non_primary.items() if l not in secondary_gt}
        if not pos or not neg:
            continue
        margins.append(max(pos.values()) - max(neg.values()))
        pooled_pos += list(pos.values()); pooled_neg += list(neg.values())
        ranked = sorted(non_primary, key=lambda l: -non_primary[l])
        best_pos_letter = max(pos, key=pos.get)
        ranks.append(ranked.index(best_pos_letter) + 1)
    if not margins:
        return None
    neg_rate = sum(m < 0 for m in margins) / len(margins)
    recall1 = sum(r <= 1 for r in ranks) / len(ranks)
    mean_margin = sum(margins) / len(margins)
    wins = sum(1 for p in pooled_pos for n in pooled_neg if p > n)
    ties = sum(1 for p in pooled_pos for n in pooled_neg if p == n)
    auc = (wins + 0.5 * ties) / max(len(pooled_pos) * len(pooled_neg), 1)
    return {"neg_rate": neg_rate, "recall1": recall1, "mean_margin": mean_margin, "auc": auc}


def percentile_ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    n = len(s)
    return s[int(n * lo / 100)], s[min(int(n * hi / 100), n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="output of eval_naming_n8_windowed.py")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.jsonl)
    recs = [json.loads(l) for l in open(a.jsonl)]

    by_recording = defaultdict(list)
    for r in recs:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        by_recording[r["recording_id"]].append({
            "whole": (secondary_gt, r["whole_scores"], r["primary_letter"]),
            "contrastive": (secondary_gt, r["contrastive_scores"], r["primary_letter"]),
        })
    recording_ids = list(by_recording)
    print(f"n_compound_items={len(recs)}  n_recordings={len(recording_ids)}")

    rng = random.Random(a.seed)
    deltas = {"neg_rate": [], "recall1": [], "mean_margin": [], "auc": []}
    skipped = 0
    for _ in range(a.n_boot):
        sampled_recordings = [rng.choice(recording_ids) for _ in recording_ids]
        whole_items = [it["whole"] for rid in sampled_recordings for it in by_recording[rid]]
        contrastive_items = [it["contrastive"] for rid in sampled_recordings for it in by_recording[rid]]
        m_whole = compute_metrics(whole_items)
        m_contrastive = compute_metrics(contrastive_items)
        if m_whole is None or m_contrastive is None:
            skipped += 1
            continue
        for k in deltas:
            deltas[k].append(m_contrastive[k] - m_whole[k])

    print(f"\n==== N8 paired bootstrap (recording-level resampling, "
          f"n_boot={a.n_boot}, usable={a.n_boot - skipped}) ====")
    point = compute_metrics([it["whole"] for its in by_recording.values() for it in its])
    point_c = compute_metrics([it["contrastive"] for its in by_recording.values() for it in its])
    for k, label in [("neg_rate", "negative-margin rate"), ("recall1", "Recall@1"),
                     ("mean_margin", "mean margin"), ("auc", "pairwise AUC")]:
        if not deltas[k]:
            continue
        lo, hi = percentile_ci(deltas[k])
        point_delta = point_c[k] - point[k]
        sig = "SIGNIFICANT (95% CI excludes 0)" if (lo > 0 or hi < 0) else "not significant (95% CI includes 0)"
        print(f"{label:24s} delta={point_delta:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
