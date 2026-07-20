"""N7 offline follow-up (1): is the secondary-scorer's failure a RANKING
problem or a THRESHOLD/CALIBRATION problem, and is the reported 50%
oracle_count exact-set number actually about compound recognition or just
free single-action credit? No model calls -- reads eval_naming_n7_scored.py's
saved per-candidate scores.

Two things this answers cheaply, using ONLY the true primary (consistent
with N7's own PR-AUC pool definition):

  1. Margin diagnosis: for each COMPOUND item (has >=1 true secondary verb),
     m = max score among true secondary candidates - max score among true
     negative candidates (both pools exclude the true primary). m<0 means
     the best true secondary verb scores BELOW the best pure distractor --
     that's a ranking failure no threshold can fix. m>0 but small means
     ranking is right-but-fragile -- threshold choice matters a lot.
     Also reports: mean/median rank of the best true secondary among all
     non-primary candidates (rank 1 = top), Recall@{1,2,3}, and a pooled
     pairwise ranking AUC (P(score(true secondary) > score(true negative)),
     distinct from N7's PR-AUC which integrates precision/recall instead).

  2. Per-GT-cardinality breakdown (1 / 2 / 3+) of exact-set accuracy for all
     three N7 decoders, so "50% exact" from oracle_count_secondary can't be
     silently read as compound-recognition success when most of that 50% is
     single-action items trivially getting an empty prediction right.

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n7b_rank_diagnosis --jsonl /tmp/n7_scored.jsonl
"""
import argparse, json, statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="output of eval_naming_n7_scored.py")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.jsonl)
    recs = [json.loads(l) for l in open(a.jsonl)]

    # ---------------- 1. margin / rank diagnosis (compound items only) ----------------
    margins, ranks, recall_at = [], [], {1: 0, 2: 0, 3: 0}
    pooled_pos, pooled_neg = [], []
    n_compound = 0
    for r in recs:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        if not secondary_gt:
            continue
        n_compound += 1
        non_primary = {l: s for l, s in r["scores"].items() if l != r["primary_letter"]}
        pos_scores = {l: s for l, s in non_primary.items() if l in secondary_gt}
        neg_scores = {l: s for l, s in non_primary.items() if l not in secondary_gt}
        if not pos_scores or not neg_scores:
            continue
        best_pos = max(pos_scores.values()); best_neg = max(neg_scores.values())
        margins.append(best_pos - best_neg)
        pooled_pos += list(pos_scores.values()); pooled_neg += list(neg_scores.values())
        ranked = sorted(non_primary, key=lambda l: -non_primary[l])
        best_pos_letter = max(pos_scores, key=pos_scores.get)
        rank = ranked.index(best_pos_letter) + 1
        ranks.append(rank)
        for k in recall_at:
            recall_at[k] += int(rank <= k)

    print(f"\n=== margin/rank diagnosis (n_compound={n_compound}, "
          f"usable={len(margins)}) ===")
    if margins:
        neg_margin = sum(m < 0 for m in margins)
        print(f"margin (best true-secondary score - best pure-distractor score):")
        print(f"  mean={statistics.mean(margins):.2f}  median={statistics.median(margins):.2f}  "
              f"negative-margin items: {neg_margin}/{len(margins)} = {neg_margin/len(margins):.1%}")
        print(f"  -> {'>50% negative margin: this IS a ranking failure, not just calibration.' if neg_margin/len(margins) > 0.5 else 'majority positive margin: ranking has signal, failure is more threshold/calibration.'}")
        print(f"best-true-secondary rank among non-primary candidates: "
              f"mean={statistics.mean(ranks):.2f}  median={statistics.median(ranks)}")
        for k, c in recall_at.items():
            print(f"  Recall@{k}: {c}/{len(margins)} = {c/len(margins):.1%}")
        # pooled pairwise ranking AUC = P(score(true secondary) > score(true negative))
        wins = sum(1 for p in pooled_pos for n in pooled_neg if p > n)
        ties = sum(1 for p in pooled_pos for n in pooled_neg if p == n)
        total = len(pooled_pos) * len(pooled_neg)
        pairwise_auc = (wins + 0.5 * ties) / max(total, 1)
        print(f"pooled pairwise ranking AUC (P(true-secondary score > true-negative "
              f"score), distinct from N7's PR-AUC): {pairwise_auc:.3f}")

    # ---------------- 2. per-cardinality breakdown for all 3 decoders ----------------
    pool_scores, pool_labels = [], []
    for r in recs:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        for l, s in r["scores"].items():
            if l != r["primary_letter"]:
                pool_scores.append(s); pool_labels.append(int(l in secondary_gt))
    best_tau, best_f1 = None, -1
    for tau in [x / 20 for x in range(-40, 41)]:
        tp = fp = fn = 0
        for r in recs:
            secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
            for l, s in r["scores"].items():
                if l == r["primary_letter"]:
                    continue
                pred_pos = s > tau
                tp += pred_pos and (l in secondary_gt)
                fp += pred_pos and (l not in secondary_gt)
                fn += (not pred_pos) and (l in secondary_gt)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau

    def decode(r, primary_mode, secondary_mode):
        letters = list(r["scores"].keys())
        ranked = sorted(letters, key=lambda l: -r["scores"][l])
        primary = ranked[0] if primary_mode == "predicted" else r["primary_letter"]
        rest = [l for l in ranked if l != primary]
        if secondary_mode == "threshold":
            secondary = [l for l in rest if r["scores"][l] > best_tau]
        else:  # oracle_count
            k = len(set(r["gt_letters"]) - {r["primary_letter"]})
            secondary = rest[:k]
        return {primary} | set(secondary)

    configs = [("predicted+threshold", "predicted", "threshold"),
              ("oracle_primary+threshold", "oracle", "threshold"),
              ("predicted+oracle_count", "predicted", "oracle_count")]
    by_card = defaultdict(lambda: defaultdict(list))
    for r in recs:
        card = len(r["gt_letters"])
        bucket = 1 if card == 1 else (2 if card == 2 else 3)
        for label, pmode, smode in configs:
            pred = decode(r, pmode, smode)
            by_card[label][bucket].append(int(pred == set(r["gt_letters"])))

    print(f"\n=== per-GT-cardinality exact-set accuracy (tau={best_tau:.2f}) ===")
    print(f"{'decoder':28s} {'n=1':>14s} {'n=2':>14s} {'n=3+':>14s} {'compound-only (2+3+)':>22s}")
    for label, pmode, smode in configs:
        cells = []
        for bucket in (1, 2, 3):
            vals = by_card[label][bucket]
            cells.append(f"{sum(vals)}/{len(vals)}={sum(vals)/max(len(vals),1):.0%}" if vals else "n/a")
        compound_vals = by_card[label][2] + by_card[label][3]
        compound = f"{sum(compound_vals)}/{len(compound_vals)}={sum(compound_vals)/max(len(compound_vals),1):.1%}" if compound_vals else "n/a"
        print(f"{label:28s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s} {compound:>22s}")
    print("\nread: compound-only column is the number that matters -- if overall "
          "exact looked good mostly from n=1 buckets, this column will be visibly "
          "lower, exposing it.")


if __name__ == "__main__":
    main()
