"""N9 -- extend the contrastive scorer (N8's clear winner: negative-margin
72%->52%, Recall@1 28%->48%, pairwise AUC 0.640->0.672 on compound-only)
to the FULL 84-item benchmark (59 atomic + 25 compound), and test the
question N8 couldn't answer: on a truly atomic segment, does the contrastive
scorer correctly reject ALL secondary candidates?

Reuses N8's already-computed contrastive scores for the 25 compound items
(no need to re-score); only the 59 atomic items need new model calls here.

Threshold is selected via RECORDING-grouped k-fold cross-validation, not a
same-set sweep (the earlier N7/N7c same-set threshold caveat) -- each fold's
threshold is fit on the OTHER folds' recordings and applied to held-out
recordings, so no item's own recording influences the threshold used to
decode it. Reports out-of-fold aggregate metrics only.

Full metric set (per the N7/N9 follow-up discussion): secondary PR-AUC,
AUROC, secondary precision/recall/F1 (at the grouped-CV threshold),
empty-secondary accuracy, false positives per ATOMIC segment, compound
Recall@{1,2,3}, full-set exact accuracy, compound-only exact accuracy,
cardinality MAE.

Usage (server):
    python -m src.eval.eval_naming_n9_full_contrastive \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --n7_jsonl /tmp/n7_scored.jsonl --n8_jsonl /tmp/n8_windowed.jsonl \
        --out /tmp/n9_full_contrastive.jsonl --n_folds 5
"""
import argparse, json, os, random

import torch
from decord import VideoReader
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.eval.eval_naming_n7_scored import resolve_first_token_ids, YES_SURFACES, NO_SURFACES
from src.eval.eval_naming_n8_windowed import score_contrastive, AB_A_SURFACES, AB_B_SURFACES
from src.boundary.decode_sweep import pr_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--n7_jsonl", required=True, help="output of eval_naming_n7_scored.py (all 84 items)")
    ap.add_argument("--n8_jsonl", required=True, help="output of eval_naming_n8_windowed.py (25 compound, has contrastive_scores already)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.n7_jsonl); print_manifest_if_exists(a.n8_jsonl)
    n7_items = [json.loads(l) for l in open(a.n7_jsonl)]
    n8_items = {(r["recording_id"], r["segment_idx"]): r["contrastive_scores"]
               for r in (json.loads(l) for l in open(a.n8_jsonl))}

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    a_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_A_SURFACES))
    b_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_B_SURFACES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    vr_cache = {}
    records = []
    n_scored = 0
    for it in n7_items:
        key = (it["recording_id"], it["segment_idx"])
        letters = "ABCDEF"[:len(it["options"])]
        primary_verb = it["options"][letters.index(it["primary_letter"])]
        if key in n8_items:
            contrastive_scores = n8_items[key]
        else:
            rid = it["recording_id"]
            if rid not in vr_cache:
                vr_cache[rid] = VideoReader(it["video"], num_threads=1)
            vr = vr_cache[rid]
            frames = [vr[i].asnumpy() for i in it["frame_indices"]]
            contrastive_scores = {}
            for l, verb in zip(letters, it["options"]):
                if l == it["primary_letter"]:
                    continue
                contrastive_scores[l] = score_contrastive(proc, model, frames, primary_verb, verb,
                                                           it["object"], a_ids, b_ids)
            n_scored += 1
        rec = {"recording_id": it["recording_id"], "segment_idx": it["segment_idx"],
               "object": it["object"], "primary_letter": it["primary_letter"],
               "gt_letters": it["gt_letters"], "contrastive_scores": contrastive_scores}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        records.append(rec)
    print(f"reused {len(records) - n_scored} compound items' scores from N8, "
          f"newly scored {n_scored} atomic items")

    # ---------------- recording-grouped k-fold CV ----------------
    rng = random.Random(a.seed)
    recording_ids = sorted({r["recording_id"] for r in records})
    rng.shuffle(recording_ids)
    folds = [recording_ids[i::a.n_folds] for i in range(a.n_folds)]
    fold_of = {rid: fi for fi, fold in enumerate(folds) for rid in fold}

    def candidates_of(rec_subset):
        """flatten to (score, is_secondary) pairs across all non-primary candidates."""
        out = []
        for r in rec_subset:
            secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
            for l, s in r["contrastive_scores"].items():
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

    oof_pred = {}  # (recording_id, segment_idx) -> predicted secondary set
    fold_taus = []
    for fi in range(a.n_folds):
        train_recs = [r for r in records if fold_of[r["recording_id"]] != fi]
        test_recs = [r for r in records if fold_of[r["recording_id"]] == fi]
        tau = best_f1_threshold(candidates_of(train_recs))
        fold_taus.append(tau)
        for r in test_recs:
            pred = {l for l, s in r["contrastive_scores"].items() if s > tau}
            oof_pred[(r["recording_id"], r["segment_idx"])] = pred
    print(f"per-fold thresholds: {[round(t,2) for t in fold_taus]}")

    # ---------------- full metric suite (out-of-fold) ----------------
    all_scores = [s for r in records for s in r["contrastive_scores"].values()]
    all_labels = [int(l in (set(r["gt_letters"]) - {r["primary_letter"]}))
                  for r in records for l in r["contrastive_scores"]]
    auc = pr_auc(all_scores, all_labels)
    pos = [s for s, l in zip(all_scores, all_labels) if l == 1]
    neg = [s for s, l in zip(all_scores, all_labels) if l == 0]
    wins = sum(1 for p in pos for n in neg if p > n); ties = sum(1 for p in pos for n in neg if p == n)
    auroc = (wins + 0.5 * ties) / max(len(pos) * len(neg), 1)

    tp = fp = fn = 0
    atomic_fp_total = 0; n_atomic = 0; empty_correct = 0
    exact_total = correct_exact = 0
    compound_exact_total = compound_exact_correct = 0
    card_mae_sum = 0
    for r in records:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        pred = oof_pred[(r["recording_id"], r["segment_idx"])]
        tp += len(pred & secondary_gt); fp += len(pred - secondary_gt); fn += len(secondary_gt - pred)
        card_mae_sum += abs(len(pred) - len(secondary_gt))
        if not secondary_gt:
            n_atomic += 1
            atomic_fp_total += len(pred)
            empty_correct += int(len(pred) == 0)
        full_pred = {r["primary_letter"]} | pred
        full_gt = set(r["gt_letters"])
        exact_total += 1; correct_exact += int(full_pred == full_gt)
        if secondary_gt:
            compound_exact_total += 1
            compound_exact_correct += int(full_pred == full_gt)

    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    # compound Recall@k from contrastive scores directly (ranking metric,
    # independent of threshold)
    recall_at = {1: 0, 2: 0, 3: 0}; n_compound = 0
    for r in records:
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        if not secondary_gt:
            continue
        n_compound += 1
        ranked = sorted(r["contrastive_scores"], key=lambda l: -r["contrastive_scores"][l])
        best_secondary_rank = min(ranked.index(l) + 1 for l in secondary_gt if l in ranked)
        for k in recall_at:
            recall_at[k] += int(best_secondary_rank <= k)

    print(f"\n==== N9 full-benchmark contrastive (n={len(records)}, "
          f"atomic={n_atomic}, compound={n_compound}, grouped {a.n_folds}-fold CV) ====")
    print(f"secondary PR-AUC: {auc:.3f}   AUROC: {auroc:.3f}")
    print(f"secondary precision/recall/F1 (out-of-fold): {prec:.1%} / {rec:.1%} / {f1:.1%}")
    print(f"empty-secondary accuracy (atomic segments, n={n_atomic}): {empty_correct}/{n_atomic} = {empty_correct/max(n_atomic,1):.1%}")
    print(f"false positives per atomic segment: {atomic_fp_total/max(n_atomic,1):.2f}")
    print(f"compound Recall@1/2/3: {recall_at[1]}/{n_compound}={recall_at[1]/max(n_compound,1):.1%}  "
          f"{recall_at[2]}/{n_compound}={recall_at[2]/max(n_compound,1):.1%}  "
          f"{recall_at[3]}/{n_compound}={recall_at[3]/max(n_compound,1):.1%}")
    print(f"full-set exact accuracy: {correct_exact}/{exact_total} = {correct_exact/exact_total:.1%}")
    print(f"compound-only exact accuracy: {compound_exact_correct}/{compound_exact_total} = "
          f"{compound_exact_correct/max(compound_exact_total,1):.1%}")
    print(f"cardinality MAE: {card_mae_sum/len(records):.2f}")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=[a.n7_jsonl, a.n8_jsonl],
                   extra={"n": len(records), "auroc": auroc, "pr_auc": auc,
                          "atomic_fp_per_segment": atomic_fp_total / max(n_atomic, 1)})


if __name__ == "__main__":
    main()
