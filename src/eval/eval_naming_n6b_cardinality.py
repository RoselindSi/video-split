"""N6.1 -- selection-calibration fix for N6's compound multi-select.

N6 showed the model isn't blind to secondary actions (secondary recall
64.1%, vs ~0% under free generation) -- it OVER-selects (pred cardinality
mean 2.66 vs GT mean 2.10, exact-set only 12.1%, set precision ~58.4%). This
is option (2) from that diagnosis: predict how many actions are shown FIRST
(a separate, cheap query against the SAME frames, not conditioned on the
candidate list -- so it isn't anchored by seeing 6 plausible-looking verbs),
then truncate the multi-select answer to that many letters, keeping them in
GENERATION ORDER (the order the model listed them in is used as a weak
confidence proxy -- we don't have per-candidate scores without a 6x-cost
independent-scoring pass, which is option (1), saved for later if this
doesn't fix enough of the gap).

Uses uniform16 frame sampling (N5's frozen conclusion: explicit before/
during/after sampling did not beat plain uniform coverage), not N6's BDA.

Reports the FULL metric set requested after N6 (not just exact/secondary-
recall/false-positive-rate, which conflated a few different things):
  primary accuracy (single-choice), primary-inclusion recall (in the
  multi-select prediction), secondary precision/recall/F1, full-set micro
  precision/recall/F1, mean Jaccard, exact-set accuracy, predicted vs GT
  cardinality mean + MAE. Run twice in one pass -- BEFORE truncation (same
  question as plain N6, for a same-frames comparison point) and AFTER
  truncation -- so the effect of cardinality-based truncation is isolated.

Usage (server):
    python -m src.eval.eval_naming_n6b_cardinality \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --pool_data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/n6b_cardinality.jsonl --max_per_video 3
"""
import argparse, json, os, random, re
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

try:
    from src.seg_rewards import _as_segs
except ImportError:
    from src.rewards.seg_rewards import _as_segs

LETTERS_RE = re.compile(r"\b([A-F])\b", re.I)
DIGIT_RE = re.compile(r"\b([1-6])\b")


def _generate(proc, model, content_msg, max_new_tokens=5):
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def ask_cardinality(proc, model, frames, obj):
    """NOT shown the candidate verb list -- asked purely from the frames, so
    the count isn't anchored by seeing N plausible options."""
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. How many DISTINCT actions occur in this clip? "
        "Answer with a single digit.")})
    out = _generate(proc, model, content_msg, max_new_tokens=3)
    m = DIGIT_RE.search(out)
    return (int(m.group(1)) if m else None), out


def ask_multi(proc, model, frames, options, obj):
    letters = "ABCDEF"[:len(options)]
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order (before, during, after a "
        f"short clip) of a person acting on the {obj}. This clip may show "
        "MORE THAN ONE action in sequence -- select ALL options that apply, "
        f"listing the MOST confident action first.\n{opts_str}\n"
        "Answer with the letters of ALL actions shown, separated by commas, "
        "most confident first (e.g. \"B,D\"). If only one action applies, "
        "give just that letter.")})
    return _generate(proc, model, content_msg, max_new_tokens=20)


def ordered_letters(raw):
    """Letters in the order they were generated (used as a confidence proxy
    for truncation), de-duplicated, keeping first occurrence."""
    out = []
    for m in LETTERS_RE.findall(raw):
        u = m.upper()
        if u not in out:
            out.append(u)
    return out


def score_set(pred_letters, gt_letters, primary_letter):
    """One item's contribution to every metric in the requested set."""
    pred, gt = set(pred_letters), set(gt_letters)
    tp = len(pred & gt)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gt), 1)
    jaccard = tp / max(len(pred | gt), 1)
    exact = (pred == gt)
    primary_included = primary_letter in pred
    secondary_gt = gt - {primary_letter}
    secondary_pred_hit = len(secondary_gt & pred)
    return {"tp": tp, "pred_n": len(pred), "gt_n": len(gt), "precision": precision,
            "recall": recall, "jaccard": jaccard, "exact": exact,
            "primary_included": primary_included, "secondary_gt_n": len(secondary_gt),
            "secondary_hit": secondary_pred_hit}


def aggregate_report(label, rows, primary_correct_single):
    n = len(rows)
    tp_sum = sum(r["tp"] for r in rows)
    pred_sum = sum(r["pred_n"] for r in rows)
    gt_sum = sum(r["gt_n"] for r in rows)
    sec_gt_sum = sum(r["secondary_gt_n"] for r in rows)
    sec_hit_sum = sum(r["secondary_hit"] for r in rows)
    micro_p = tp_sum / max(pred_sum, 1)
    micro_r = tp_sum / max(gt_sum, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
    sec_recall = sec_hit_sum / max(sec_gt_sum, 1)
    cardinality_mae = sum(abs(r["pred_n"] - r["gt_n"]) for r in rows) / max(n, 1)
    print(f"\n---- {label} (n={n}) ----")
    print(f"  single-choice primary accuracy: {primary_correct_single}/{n} = "
          f"{primary_correct_single/max(n,1):.1%}")
    print(f"  primary-inclusion recall (in multi-select set): "
          f"{sum(r['primary_included'] for r in rows)}/{n} = "
          f"{sum(r['primary_included'] for r in rows)/max(n,1):.1%}")
    print(f"  secondary recall: {sec_hit_sum}/{sec_gt_sum} = {sec_recall:.1%}")
    print(f"  exact verb-set accuracy: {sum(r['exact'] for r in rows)}/{n} = "
          f"{sum(r['exact'] for r in rows)/max(n,1):.1%}")
    print(f"  full-set micro precision/recall/F1: {micro_p:.1%} / {micro_r:.1%} / {micro_f1:.1%}")
    print(f"  mean Jaccard: {sum(r['jaccard'] for r in rows)/max(n,1):.1%}")
    print(f"  cardinality: GT mean={gt_sum/max(n,1):.2f}  pred mean={pred_sum/max(n,1):.2f}  "
          f"MAE={cardinality_mae:.2f}")


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

    rows = json.load(open(a.target_data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")

    before_rows, after_rows = [], []
    primary_correct = 0

    for r in rows:
        gts = _as_segs(r["solution"])
        candidates = []
        for i, (name, s, e) in enumerate(gts):
            verbs, obj = primary_verb_and_object(name)
            if len(verbs) < 2 or obj is None:
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
            primary_letter = letters[options.index(verbs[0])]

            frames, fidx = sample_uniform(vr, vfps, s, e, 16)

            card_pred, card_raw = ask_cardinality(proc, model, frames, obj)
            ms_out = ask_multi(proc, model, frames, options, obj)
            ordered = ordered_letters(ms_out)

            sc_content = [{"type": "image", "image": Image.fromarray(f),
                          "max_pixels": 768 * 28 * 28} for f in frames]
            opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
            sc_content.append({"type": "text", "text": (
                "The images are frames in temporal order of a short clip of a "
                f"person acting on the {obj}. Which SINGLE option best "
                f"describes the MAIN action shown?\n{opts_str}\n"
                f"Answer with exactly one letter: {'/'.join(letters)}.")})
            sc_raw = _generate(proc, model, sc_content, max_new_tokens=5)
            sc_m = re.search(r"\b([A-F])\b", sc_raw, re.I)
            sc_pred = sc_m.group(1).upper() if sc_m else "?"
            sc_correct = (sc_pred == primary_letter)
            primary_correct += int(sc_correct)

            before = score_set(ordered, gt_letters, primary_letter)
            k = card_pred if card_pred else len(ordered)
            k = max(1, min(k, len(options)))
            truncated = ordered[:k]
            after = score_set(truncated, gt_letters, primary_letter)
            before_rows.append(before); after_rows.append(after)

            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": i, "start": s, "end": e, "gt_name": name,
                   "object": obj, "gt_verbs": verbs, "options": options,
                   "gt_letters": gt_letters, "primary_letter": primary_letter,
                   "frame_indices": fidx, "cardinality_pred": card_pred,
                   "cardinality_raw": card_raw, "multi_select_ordered": ordered,
                   "multi_select_truncated": truncated, "single_choice_pred": sc_pred,
                   "single_choice_correct": sc_correct, "raw_multi": ms_out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{i} obj='{obj}' gt_verbs={verbs} "
                  f"gt_letters={gt_letters} card_pred={card_pred} "
                  f"multi_ordered={ordered} -> truncated={truncated}")
            picked += 1
        del vr

    print(f"\n==== N6.1 cardinality-truncated multi-select vs untruncated (n={len(before_rows)}) ====")
    aggregate_report("BEFORE truncation (same as plain N6 multi-select)", before_rows, primary_correct)
    aggregate_report("AFTER cardinality-based truncation", after_rows, primary_correct)

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=list(a.pool_data) + [a.target_data],
                   extra={"n_done": len(before_rows)})


if __name__ == "__main__":
    main()
