"""N7 (was called N6.1 option 1+3) -- independently-scored candidate verbs,
decoupling primary selection from optional secondary detection.

Why this instead of list-generation multi-select (N6) or count-then-truncate
(N6.1/N6.1-iter2, abandoned -- count exact 5.2%, MAE 2.69, 89.7% over-count,
and worse than a trivial constant-K=2 baseline given the benchmark's own
91.4% GT=2 skew): free-form list generation conflates "is this action really
here" with "how many things should I list", and both empirically collapsed
(list generation over-selects; count prediction failed outright). Independent
per-candidate scoring asks ONE well-posed question per candidate ("does verb
V occur, yes or no") and reads the answer off the model's own next-token
logit margin -- log P(Yes) - log P(No) -- not a free-text generation to
parse, so there's no list-length bias to begin with.

Scoring: forward pass only (no generation loop) per candidate -- cheap.
score(v) = logsumexp(logits[yes_token_ids]) - logsumexp(logits[no_token_ids])
at the next-token position after "... Answer YES or NO." yes/no token ids are
resolved once at startup by tokenizing several surface variants (" Yes",
"Yes", "YES", ...) and taking their first sub-token, so tokenizer quirks
don't silently break scoring for one string form.

Five decoders on the SAME per-candidate scores (see the comparison table
printed at the end):
  predicted_primary_thresh_secondary : primary=argmax(score); secondary =
      {v != primary : score(v) > tau} (tau from a same-set sweep, CAVEAT:
      not a held-out calibration split -- N3's clean benchmark should
      recalibrate this once it exists)
  oracle_primary_thresh_secondary    : primary = TRUE primary (isolates
      secondary scorer/threshold quality from primary-selection error
      propagation)
  predicted_primary_oracle_count     : primary=argmax(score); secondary =
      top (TRUE secondary count) non-primary candidates by score (isolates
      whether the RANKING captures true secondary verbs, independent of
      threshold calibration)
  fixed_k2 / oracle_k / untruncated-list : reference points reused from
      N6/N6.1-iter3 for a same-script comparison against list generation.

Usage (server):
    python -m src.eval.eval_naming_n7_scored \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --pool_data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/n7_scored.jsonl --max_per_video 3
"""
import argparse, json, os, random
from collections import defaultdict

import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.eval.eval_naming_hard_negative_v2 import (
    primary_verb_and_object, build_object_verb_pool,
)
from src.eval.eval_naming_n5_sampling import sample_uniform
from src.eval.eval_naming_n6b_cardinality import score_set
from src.boundary.decode_sweep import pr_auc

try:
    from src.seg_rewards import _as_segs
except ImportError:
    from src.rewards.seg_rewards import _as_segs

YES_SURFACES = ["Yes", " Yes", "YES", " YES", "yes", " yes"]
NO_SURFACES = ["No", " No", "NO", " NO", "no", " no"]


def resolve_first_token_ids(tokenizer, surfaces):
    ids = set()
    for s in surfaces:
        enc = tokenizer(s, add_special_tokens=False).input_ids
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def score_candidate(proc, model, frames, verb, obj, yes_ids, no_ids):
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. Does the action \"{verb} {obj}\" occur "
        "SOMEWHERE in this clip? Answer YES or NO.")})
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]
    yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(logits[no_ids], dim=0)
    return (yes_logit - no_logit).item()


def primary_stats(rows):
    n = len(rows)
    acc = sum(r["primary_correct"] for r in rows) / max(n, 1)
    top2 = sum(r["primary_top2"] for r in rows) / max(n, 1)
    print(f"  primary accuracy: {acc:.1%}   primary top-2 inclusion: {top2:.1%}")


def secondary_stats(label, rows):
    n = len(rows)
    tp = sum(r["sec_tp"] for r in rows)
    pred_n = sum(r["sec_pred_n"] for r in rows)
    gt_n = sum(r["sec_gt_n"] for r in rows)
    prec = tp / max(pred_n, 1); rec = tp / max(gt_n, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    empty_gt = [r for r in rows if r["sec_gt_n"] == 0]
    fp_per_seg = sum(max(r["sec_pred_n"] - r["sec_tp"], 0) for r in rows) / max(n, 1)
    tail = (f"empty-secondary acc (n={len(empty_gt)}): "
           f"{sum(r['sec_pred_n'] == 0 for r in empty_gt) / len(empty_gt):.1%}"
           if empty_gt else "(no empty-secondary GT items in this set)")
    print(f"  [{label}] secondary P/R/F1: {prec:.1%} / {rec:.1%} / {f1:.1%}  "
          f"FP/segment: {fp_per_seg:.2f}  {tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--pool_data", nargs="+", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_video", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = build_object_verb_pool(a.pool_data)
    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed)

    yes_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, YES_SURFACES))
    no_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, NO_SURFACES))
    print(f"yes_token_ids={yes_ids.tolist()}  no_token_ids={no_ids.tolist()}")

    rows = json.load(open(a.target_data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    records = []

    for r in rows:
        gts = _as_segs(r["solution"])
        candidates = []
        for i, (name, s, e) in enumerate(gts):
            verbs, obj = primary_verb_and_object(name)
            if len(verbs) < 1 or obj is None:
                continue
            pool_others = sorted(pool[obj] - set(verbs))
            if len(pool_others) < 2:
                continue
            candidates.append((i, name, s, e, verbs, obj, pool_others))
        if not candidates:
            continue
        rng.shuffle(candidates)
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        picked = 0
        for i, name, s, e, verbs, obj, pool_others in candidates:
            if picked >= a.max_per_video:
                break
            n_distractors = min(len(pool_others), 6 - len(verbs))
            if n_distractors < 1:
                continue
            distractors = rng.sample(pool_others, n_distractors)
            options = verbs + distractors
            rng.shuffle(options)
            letters = "ABCDEF"[:len(options)]
            gt_letters = [letters[options.index(v)] for v in verbs]
            true_primary = letters[options.index(verbs[0])]

            frames, fidx = sample_uniform(vr, vfps, s, e, 16)
            scores = {l: score_candidate(proc, model, frames, v, obj, yes_ids, no_ids)
                      for l, v in zip(letters, options)}

            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": i, "start": s, "end": e, "gt_name": name,
                   "object": obj, "gt_verbs": verbs, "options": options,
                   "gt_letters": gt_letters, "primary_letter": true_primary,
                   "frame_indices": fidx, "scores": scores}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            records.append(rec)
            print(f"{r.get('recording_id')} seg{i} obj='{obj}' gt_letters={gt_letters} "
                  f"primary={true_primary} scores={ {k: round(v,2) for k,v in scores.items()} }")
            picked += 1
        del vr

    # ---------------- threshold sweep for secondary (same-set, NOT held out) ----------------
    pool_scores, pool_labels = [], []
    for rec in records:
        secondary_gt = set(rec["gt_letters"]) - {rec["primary_letter"]}
        for l in rec["scores"]:
            if l == rec["primary_letter"]:
                continue
            pool_scores.append(rec["scores"][l])
            pool_labels.append(int(l in secondary_gt))
    auc = pr_auc(pool_scores, pool_labels)
    print(f"\nsecondary-scorer PR-AUC (positive=true secondary verb, pool "
          f"excludes each item's own TRUE primary candidate): {auc:.3f}")
    print("CAVEAT: this and the threshold below are fit on the SAME 58 items "
          "being evaluated -- optimistic, not a held-out estimate. Recalibrate "
          "once N3's clean benchmark exists.")
    best_tau, best_f1 = None, -1
    for tau in [x / 20 for x in range(-40, 41)]:
        tp = fp = fn = 0
        for rec in records:
            secondary_gt = set(rec["gt_letters"]) - {rec["primary_letter"]}
            for l in rec["scores"]:
                if l == rec["primary_letter"]:
                    continue
                pred_pos = rec["scores"][l] > tau
                is_pos = l in secondary_gt
                tp += pred_pos and is_pos
                fp += pred_pos and not is_pos
                fn += (not pred_pos) and is_pos
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    print(f"best same-set threshold tau={best_tau:.2f} (secondary F1={best_f1:.3f})")

    # ---------------- build the 6 decoders ----------------
    def decode(rec, primary_mode, secondary_mode, tau=best_tau):
        letters = list(rec["scores"].keys())
        ranked = sorted(letters, key=lambda l: -rec["scores"][l])
        if primary_mode == "predicted":
            primary = ranked[0]
        else:  # oracle
            primary = rec["primary_letter"]
        top2 = set(ranked[:2])
        rest = [l for l in ranked if l != primary]
        if secondary_mode == "threshold":
            secondary = [l for l in rest if rec["scores"][l] > tau]
        elif secondary_mode == "oracle_count":
            k = len(set(rec["gt_letters"]) - {rec["primary_letter"]})
            secondary = rest[:k]
        else:
            secondary = []
        return primary, set(secondary), (rec["primary_letter"] in top2)

    configs = [
        ("predicted_primary + threshold_secondary", "predicted", "threshold"),
        ("oracle_primary + threshold_secondary", "oracle", "threshold"),
        ("predicted_primary + oracle_count_secondary", "predicted", "oracle_count"),
    ]
    print(f"\n==== N7 independently-scored candidates (n={len(records)}) ====")
    for label, pmode, smode in configs:
        primary_rows, secondary_rows, full_rows = [], [], []
        for rec in records:
            primary, secondary, top2 = decode(rec, pmode, smode)
            gt_secondary = set(rec["gt_letters"]) - {rec["primary_letter"]}
            primary_rows.append({"primary_correct": primary == rec["primary_letter"],
                                 "primary_top2": top2})
            secondary_rows.append({"sec_tp": len(secondary & gt_secondary),
                                   "sec_pred_n": len(secondary), "sec_gt_n": len(gt_secondary)})
            full_pred = {primary} | secondary
            full_rows.append(score_set(full_pred, rec["gt_letters"], rec["primary_letter"]))
        print(f"\n-- {label} --")
        primary_stats(primary_rows)
        secondary_stats(label, secondary_rows)
        n = len(full_rows)
        exact = sum(r["exact"] for r in full_rows) / max(n, 1)
        jac = sum(r["jaccard"] for r in full_rows) / max(n, 1)
        tp_s = sum(r["tp"] for r in full_rows); pn = sum(r["pred_n"] for r in full_rows)
        gn = sum(r["gt_n"] for r in full_rows)
        mp = tp_s / max(pn, 1); mr = tp_s / max(gn, 1)
        mf1 = 2 * mp * mr / max(mp + mr, 1e-9)
        cmae = sum(abs(r["pred_n"] - r["gt_n"]) for r in full_rows) / max(n, 1)
        print(f"  full-set: exact={exact:.1%}  jaccard={jac:.1%}  "
              f"micro P/R/F1={mp:.1%}/{mr:.1%}/{mf1:.1%}  cardinality MAE={cmae:.2f}")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=list(a.pool_data) + [a.target_data],
                   extra={"n_done": len(records), "secondary_pr_auc": auc, "best_tau": best_tau})


if __name__ == "__main__":
    main()
