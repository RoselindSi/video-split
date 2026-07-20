"""N8 -- secondary evidence LOCALIZATION, replacing the abandoned global
has_secondary gate (N7e: prior+gate LOOCV AUROC delta only +0.003 over
object/primary-verb prior alone; N7f matched pairs confirmed gate score is
mostly prior, not per-clip visual evidence).

The N7b margin diagnosis found a majority of compound items have NEGATIVE
margin (best true-secondary score < best pure-distractor score) when scored
on the WHOLE 16-frame segment at once. Hypothesis: short secondary actions
(e.g. "unbox + inspect + repack" -- inspect/repack are often brief tail
actions) get diluted when the whole segment is scored as one unit, dominated
by whichever action occupies the most frames (usually the primary). This
tests that directly by comparing THREE scoring methods on the SAME N7
candidate items (reused from n7_scored.jsonl for exact frame parity):

  whole_segment : N7's original method -- one forward pass over all 16 frames
  windowed_max  : split the 16 frames into 4 windows of 4, score the
                  candidate independently in EACH window, take the max --
                  a brief secondary action confined to one window shouldn't
                  get diluted by the other 12 frames showing the primary
  contrastive   : instead of independent "does V occur, yes/no" (which N7b
                  showed picks up "V is a plausible action" bias), ask a
                  forced A/B choice: "A: only PRIMARY occurs. B: PRIMARY AND
                  V both occur." scored via logit(B)-logit(A) at the next
                  token -- directly targets "does V add anything beyond
                  primary" rather than "is V plausible in isolation"

Reports, for each method: negative-margin rate, mean/median rank of the best
true secondary among non-primary candidates, Recall@{1,2,3}, and pooled
pairwise ranking AUC -- same diagnostics N7b used for whole_segment, so the
three methods are read off the same table.

Usage (server):
    python -m src.eval.eval_naming_n8_windowed \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --prev_jsonl /tmp/n7_scored.jsonl --out /tmp/n8_windowed.jsonl
"""
import argparse, json, os, statistics

import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.eval.eval_naming_n7_scored import resolve_first_token_ids, YES_SURFACES, NO_SURFACES


AB_A_SURFACES = ["A", " A"]
AB_B_SURFACES = ["B", " B"]


def _forward_score(proc, model, content_msg, pos_ids, neg_ids):
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]
    pos_logit = torch.logsumexp(logits[pos_ids], dim=0)
    neg_logit = torch.logsumexp(logits[neg_ids], dim=0)
    return (pos_logit - neg_logit).item()


def score_whole(proc, model, frames, verb, obj, yes_ids, no_ids):
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. Does the action \"{verb} {obj}\" occur "
        "SOMEWHERE in this clip? Answer YES or NO.")})
    return _forward_score(proc, model, content_msg, yes_ids, no_ids)


def score_windowed_max(proc, model, frames, verb, obj, yes_ids, no_ids, n_windows=4):
    win = max(len(frames) // n_windows, 1)
    scores = []
    for w in range(0, len(frames), win):
        sub = frames[w:w + win]
        if not sub:
            continue
        scores.append(score_whole(proc, model, sub, verb, obj, yes_ids, no_ids))
    return max(scores) if scores else float("-inf")


def score_contrastive(proc, model, frames, primary_verb, verb, obj, a_ids, b_ids):
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. The main action is \"{primary_verb} {obj}\". "
        "Which is better supported by the clip?\n"
        f"A: only \"{primary_verb} {obj}\" occurs, nothing else.\n"
        f"B: \"{primary_verb} {obj}\" occurs, AND \"{verb} {obj}\" also occurs.\n"
        "Answer with exactly one letter: A or B.")})
    return _forward_score(proc, model, content_msg, b_ids, a_ids)


def margin_rank_report(label, items):
    """items: list of (secondary_gt:set[letter], scores:dict[letter->float],
    primary_letter). Same diagnostics as N7b, for one scoring method."""
    margins, ranks, recall_at = [], [], {1: 0, 2: 0, 3: 0}
    pooled_pos, pooled_neg = [], []
    for secondary_gt, scores, primary_letter in items:
        if not secondary_gt:
            continue
        non_primary = {l: s for l, s in scores.items() if l != primary_letter}
        pos = {l: s for l, s in non_primary.items() if l in secondary_gt}
        neg = {l: s for l, s in non_primary.items() if l not in secondary_gt}
        if not pos or not neg:
            continue
        margins.append(max(pos.values()) - max(neg.values()))
        pooled_pos += list(pos.values()); pooled_neg += list(neg.values())
        ranked = sorted(non_primary, key=lambda l: -non_primary[l])
        best_pos_letter = max(pos, key=pos.get)
        rank = ranked.index(best_pos_letter) + 1
        ranks.append(rank)
        for k in recall_at:
            recall_at[k] += int(rank <= k)
    print(f"\n-- {label} (n_compound_usable={len(margins)}) --")
    if not margins:
        print("  no usable compound items"); return
    neg_rate = sum(m < 0 for m in margins) / len(margins)
    print(f"  negative-margin rate: {neg_rate:.1%}  mean margin: {statistics.mean(margins):.2f}")
    print(f"  best-secondary rank: mean={statistics.mean(ranks):.2f} median={statistics.median(ranks)}")
    for k, c in recall_at.items():
        print(f"  Recall@{k}: {c}/{len(margins)} = {c/len(margins):.1%}")
    wins = sum(1 for p in pooled_pos for n in pooled_neg if p > n)
    ties = sum(1 for p in pooled_pos for n in pooled_neg if p == n)
    auc = (wins + 0.5 * ties) / max(len(pooled_pos) * len(pooled_neg), 1)
    print(f"  pooled pairwise ranking AUC: {auc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--prev_jsonl", required=True, help="output of eval_naming_n7_scored.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=4)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.prev_jsonl)
    items = [json.loads(l) for l in open(a.prev_jsonl)]

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    yes_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, YES_SURFACES))
    no_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, NO_SURFACES))
    a_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_A_SURFACES))
    b_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_B_SURFACES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    vr_cache = {}
    whole_items, windowed_items, contrastive_items = [], [], []

    for it in items:
        secondary_gt = set(it["gt_letters"]) - {it["primary_letter"]}
        if not secondary_gt:
            continue  # N8 only needs compound items -- diagnostics are compound-only anyway
        rid = it["recording_id"]
        if rid not in vr_cache:
            vr_cache[rid] = VideoReader(it["video"], num_threads=1)
        vr = vr_cache[rid]
        letters = "ABCDEF"[:len(it["options"])]
        primary_verb = it["options"][letters.index(it["primary_letter"])]
        frames = [vr[i].asnumpy() for i in it["frame_indices"]]

        whole_scores, windowed_scores, contrastive_scores = {}, {}, {}
        for l, verb in zip(letters, it["options"]):
            if l == it["primary_letter"]:
                continue
            whole_scores[l] = score_whole(proc, model, frames, verb, it["object"], yes_ids, no_ids)
            windowed_scores[l] = score_windowed_max(proc, model, frames, verb, it["object"],
                                                     yes_ids, no_ids, a.n_windows)
            contrastive_scores[l] = score_contrastive(proc, model, frames, primary_verb, verb,
                                                       it["object"], a_ids, b_ids)

        rec = {"recording_id": rid, "segment_idx": it["segment_idx"], "object": it["object"],
               "primary_verb": primary_verb, "primary_letter": it["primary_letter"],
               "gt_letters": it["gt_letters"], "whole_scores": whole_scores,
               "windowed_scores": windowed_scores, "contrastive_scores": contrastive_scores}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print(f"{rid} seg{it['segment_idx']} obj='{it['object']}' primary={primary_verb} "
              f"secondary_gt={sorted(secondary_gt)}  whole={ {k: round(v,2) for k,v in whole_scores.items()} }  "
              f"windowed={ {k: round(v,2) for k,v in windowed_scores.items()} }")

        whole_items.append((secondary_gt, whole_scores, it["primary_letter"]))
        windowed_items.append((secondary_gt, windowed_scores, it["primary_letter"]))
        contrastive_items.append((secondary_gt, contrastive_scores, it["primary_letter"]))

    print(f"\n==== N8 secondary evidence localization (n_compound={len(whole_items)}) ====")
    margin_rank_report("whole_segment (N7 baseline, re-scored here for exact parity)", whole_items)
    margin_rank_report(f"windowed_max ({a.n_windows} windows)", windowed_items)
    margin_rank_report("contrastive (A: primary only / B: primary+V)", contrastive_items)

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=[a.prev_jsonl], extra={"n_compound": len(whole_items)})


if __name__ == "__main__":
    main()
