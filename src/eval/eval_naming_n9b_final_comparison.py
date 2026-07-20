"""N9 final step (1) -- apples-to-apples comparison table. Re-decodes the
INDEPENDENT yes/no scorer (N7's original whole-segment scores, already
computed for all 84 items) through the SAME recording-grouped 5-fold CV
machinery N9 used for the contrastive scorer, instead of comparing against
the old same-set threshold numbers. windowed_max is reported compound-only
(n=25) since it was never scored on the 59 atomic items -- noted explicitly,
not silently extrapolated.

No model calls -- reads eval_naming_n7_scored.py (independent whole-segment
scores + labels for all 84), eval_naming_n9_full_contrastive.py (contrastive
scores for all 84), and eval_naming_n8_windowed.py (windowed_max scores,
compound n=25 only).

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n9b_final_comparison \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --n8_jsonl /workspace/tr1/results/naming/n8_windowed.jsonl \
        --n_folds 5
"""
import argparse, json, random

from src.boundary.decode_sweep import pr_auc


def grouped_cv_decode(records, score_key, n_folds=5, seed=0):
    """records: list of dicts with recording_id, primary_letter, gt_letters,
    and record[score_key] = {letter: score}. Returns oof_pred dict keyed by
    (recording_id, segment_idx) -> predicted secondary set, and per-fold taus."""
    rng = random.Random(seed)
    recording_ids = sorted({r["recording_id"] for r in records})
    rng.shuffle(recording_ids)
    folds = [recording_ids[i::n_folds] for i in range(n_folds)]
    fold_of = {rid: fi for fi, fold in enumerate(folds) for rid in fold}

    def candidates_of(subset):
        out = []
        for r in subset:
            secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
            for l, s in r[score_key].items():
                out.append((s, int(l in secondary_gt)))
        return out

    def best_f1_threshold(pairs):
        if not pairs:
            return 0.0
        best_tau, best_f1 = 0.0, -1
        for tau in sorted({s for s, _ in pairs}):
            tp = sum(1 for s, l in pairs if s > tau and l == 1)
            fp = sum(1 for s, l in pairs if s > tau and l == 0)
            fn = sum(1 for s, l in pairs if s <= tau and l == 1)
            p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        return best_tau

    oof_pred, taus = {}, []
    for fi in range(n_folds):
        train = [r for r in records if fold_of[r["recording_id"]] != fi]
        test = [r for r in records if fold_of[r["recording_id"]] == fi]
        tau = best_f1_threshold(candidates_of(train))
        taus.append(tau)
        for r in test:
            pred = {l for l, s in r[score_key].items() if s > tau}
            oof_pred[(r["recording_id"], r["segment_idx"])] = pred
    return oof_pred, taus


def full_report(label, records, score_key, n_folds, seed):
    oof_pred, taus = grouped_cv_decode(records, score_key, n_folds, seed)
    all_scores = [s for r in records for s in r[score_key].values()]
    all_labels = [int(l in (set(r["gt_letters"]) - {r["primary_letter"]}))
                  for r in records for l in r[score_key]]
    auc = pr_auc(all_scores, all_labels)
    pos = [s for s, l in zip(all_scores, all_labels) if l == 1]
    neg = [s for s, l in zip(all_scores, all_labels) if l == 0]
    wins = sum(1 for p in pos for n in neg if p > n); ties = sum(1 for p in pos for n in neg if p == n)
    auroc = (wins + 0.5 * ties) / max(len(pos) * len(neg), 1)

    tp = fp = fn = 0; n_atomic = 0; atomic_fp = 0; empty_correct = 0
    exact_total = correct_exact = 0; compound_exact_total = compound_exact_correct = 0
    card_mae_sum = 0
    for r in records:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        pred = oof_pred[(r["recording_id"], r["segment_idx"])]
        tp += len(pred & secondary_gt); fp += len(pred - secondary_gt); fn += len(secondary_gt - pred)
        card_mae_sum += abs(len(pred) - len(secondary_gt))
        if not secondary_gt:
            n_atomic += 1; atomic_fp += len(pred); empty_correct += int(len(pred) == 0)
        full_pred = {r["primary_letter"]} | pred
        exact_total += 1; correct_exact += int(full_pred == set(r["gt_letters"]))
        if secondary_gt:
            compound_exact_total += 1
            compound_exact_correct += int(full_pred == set(r["gt_letters"]))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    recall_at = {1: 0, 2: 0, 3: 0}; n_compound = 0
    for r in records:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        if not secondary_gt:
            continue
        n_compound += 1
        ranked = sorted(r[score_key], key=lambda l: -r[score_key][l])
        best_rank = min(ranked.index(l) + 1 for l in secondary_gt if l in ranked)
        for k in recall_at:
            recall_at[k] += int(best_rank <= k)

    return {"label": label, "n": len(records), "auc": auc, "auroc": auroc,
            "f1": f1, "empty_acc": empty_correct / max(n_atomic, 1),
            "fp_per_atomic": atomic_fp / max(n_atomic, 1),
            "full_exact": correct_exact / exact_total,
            "compound_exact": compound_exact_correct / max(compound_exact_total, 1),
            "card_mae": card_mae_sum / len(records), "compound_recall1": recall_at[1] / max(n_compound, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--n8_jsonl", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    n7 = {(r["recording_id"], r["segment_idx"]): r for r in
          (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in
          (json.loads(l) for l in open(a.n9_jsonl))}
    n8 = [json.loads(l) for l in open(a.n8_jsonl)]

    independent_recs = [{"recording_id": rid, "segment_idx": sid, "primary_letter": r["primary_letter"],
                         "gt_letters": r["gt_letters"], "ind_scores": r["scores"]}
                        for (rid, sid), r in n7.items()]
    contrastive_recs = [{"recording_id": rid, "segment_idx": sid, "primary_letter": r["primary_letter"],
                         "gt_letters": r["gt_letters"], "con_scores": r["contrastive_scores"]}
                        for (rid, sid), r in n9.items()]

    rows = [full_report("independent yes/no (whole-segment, N7)", independent_recs, "ind_scores", a.n_folds, a.seed),
           full_report("contrastive (primary-only vs primary+V, N9)", contrastive_recs, "con_scores", a.n_folds, a.seed)]

    # windowed_max: compound-only (n=25), never scored on atomic -- report
    # compound-only ranking diagnostics, not decoded set metrics
    print(f"{'method':45s} {'AUROC':>7s} {'PR-AUC':>7s} {'sec-F1':>7s} "
          f"{'empty-acc':>10s} {'FP/atomic':>10s} {'full-exact':>11s} "
          f"{'compound-exact':>15s} {'card-MAE':>9s} {'recall@1':>9s}")
    for row in rows:
        print(f"{row['label']:45s} {row['auroc']:7.3f} {row['auc']:7.3f} {row['f1']*100:6.1f}% "
              f"{row['empty_acc']*100:9.1f}% {row['fp_per_atomic']:10.2f} {row['full_exact']*100:10.1f}% "
              f"{row['compound_exact']*100:14.1f}% {row['card_mae']:9.2f} {row['compound_recall1']*100:8.1f}%")

    print(f"\nwindowed_max: COMPOUND-ONLY (n=25) -- never scored on the 59 atomic "
          f"items, so full-set/atomic-FP/empty-acc rows are not comparable and "
          f"deliberately omitted here rather than extrapolated. See N8's own "
          f"log for its compound-only Recall@1/2/3 (36%/56%/76%) and pairwise "
          f"AUC (0.617) if a reference point is needed -- do not re-run more "
          f"window configurations per the decision to stop tuning this axis.")
    print(f"\nglobal gate + scorer: NOT rebuilt as a unified decoder here -- "
          f"N7e (prior+gate LOOCV AUROC delta +0.003) and N7f (matched pairs) "
          f"already established the gate carries no independent signal beyond "
          f"object/verb prior, so combining it with any scorer would not be "
          f"expected to help and isn't worth new GPU time under the decision "
          f"to converge.")


if __name__ == "__main__":
    main()
