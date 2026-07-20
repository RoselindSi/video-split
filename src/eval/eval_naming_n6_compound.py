"""N6 -- compound-verb discrimination: does multi-select recover the secondary
verb that single-choice / free-generation both miss?

For every GT segment with >=2 canonical verbs (post VERB_NORM + object-
conditioned CONTEXTUAL_VERB_NORM fold -- same definition as N4/N2), builds
TWO questions on the SAME frames/candidates:
  (a) single-choice: "which ONE verb best describes this" (only the primary
      verb counts as correct -- this is the old N4-style question, kept as a
      baseline for direct comparison)
  (b) multi-select: "select ALL verbs that apply" over the same candidate set
      (all GT verbs are valid options, plus distractors up to 6 total)

Candidates = ALL GT verbs (2-3, so the model CAN get full credit) + distractor
verbs from the object's global pool (GENERIC_VERBS excluded), total capped at
6, order shuffled, lettered A-F.

Reports what actually matters for the compound problem: not just exact-set
accuracy, but secondary-verb recall SPECIFICALLY (of the non-primary GT
verbs, how many did multi-select actually pick), and false-positive rate
(extra verbs selected that aren't in GT), compared directly against the
single-choice primary-only baseline on the identical items.

Usage (server):
    python -m src.eval.eval_naming_n6_compound \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --pool_data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/n6_compound.jsonl --max_per_video 3
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

try:
    from eval_naming_persegment import sample_transition_frames
except ImportError:
    try:
        from src.eval.eval_naming_persegment import sample_transition_frames
    except ImportError:
        from eval_naming_persegment import sample_transition_frames

try:
    from src.seg_rewards import _as_segs
except ImportError:
    from src.rewards.seg_rewards import _as_segs

LETTERS_RE = re.compile(r"\b([A-F])\b", re.I)
MAX_CHOICES = 6


def ask_single(proc, model, frames, options, obj):
    letters = "ABCDEF"[:len(options)]
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order (before, during, after a "
        f"short clip) of a person acting on the {obj}. Which SINGLE option "
        f"best describes the MAIN action shown?\n{opts_str}\n"
        f"Answer with exactly one letter: {'/'.join(letters)}.")})
    return _generate(proc, model, content_msg)


def ask_multi(proc, model, frames, options, obj):
    letters = "ABCDEF"[:len(options)]
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order (before, during, after a "
        f"short clip) of a person acting on the {obj}. This clip may show "
        "MORE THAN ONE action in sequence -- select ALL options that "
        f"apply.\n{opts_str}\n"
        "Answer with the letters of ALL actions shown, separated by commas "
        "(e.g. \"B,D\"). If only one action applies, give just that letter.")})
    return _generate(proc, model, content_msg, max_new_tokens=20)


def _generate(proc, model, content_msg, max_new_tokens=5):
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--pool_data", nargs="+", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context_s", type=float, default=1.0)
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
    n_done = 0
    sc_primary_correct = 0
    ms_exact_set = 0
    sec_hit = sec_total = fp_total = pred_total = 0

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
            n_distractors = min(len(pool_others), MAX_CHOICES - len(verbs))
            if n_distractors < 1:
                continue
            distractors = rng.sample(pool_others, n_distractors)
            options = verbs + distractors
            rng.shuffle(options)
            letters = "ABCDEF"[:len(options)]
            gt_letters = {letters[options.index(v)] for v in verbs}
            primary_letter = letters[options.index(verbs[0])]

            frames, fidx = sample_transition_frames(
                vr, vfps, s, e, a.context_s, 4, 8, 4)

            sc_out = ask_single(proc, model, frames, options, obj)
            sc_m = re.search(r"\b([A-F])\b", sc_out, re.I)
            sc_pred = sc_m.group(1).upper() if sc_m else "?"
            sc_correct = (sc_pred == primary_letter)

            ms_out = ask_multi(proc, model, frames, options, obj)
            ms_pred_letters = {m.upper() for m in LETTERS_RE.findall(ms_out)}
            ms_correct_set = (ms_pred_letters == gt_letters)
            secondary_letters = gt_letters - {primary_letter}
            hit = len(secondary_letters & ms_pred_letters)
            fp = len(ms_pred_letters - gt_letters)

            n_done += 1
            sc_primary_correct += int(sc_correct)
            ms_exact_set += int(ms_correct_set)
            sec_total += len(secondary_letters); sec_hit += hit
            fp_total += fp; pred_total += len(ms_pred_letters)
            picked += 1

            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": i, "start": s, "end": e, "gt_name": name, "object": obj,
                   "gt_verbs": verbs, "options": options, "frame_indices": fidx,
                   "gt_letters": sorted(gt_letters), "primary_letter": primary_letter,
                   "single_choice_pred": sc_pred, "single_choice_correct": sc_correct,
                   "multi_select_pred_letters": sorted(ms_pred_letters),
                   "multi_select_exact": ms_correct_set,
                   "secondary_hit": hit, "secondary_total": len(secondary_letters),
                   "false_positive": fp, "raw_single": sc_out, "raw_multi": ms_out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{i} obj='{obj}' gt_verbs={verbs} "
                  f"gt_letters={sorted(gt_letters)} single={sc_pred}"
                  f"({'OK' if sc_correct else 'WRONG'}) "
                  f"multi={sorted(ms_pred_letters)}({'EXACT' if ms_correct_set else 'partial/wrong'})")
        del vr

    print(f"\n==== N6 compound: single-choice primary vs multi-select full set (n={n_done}) ====")
    print(f"single-choice PRIMARY-verb accuracy: {sc_primary_correct}/{n_done} "
          f"= {sc_primary_correct/max(n_done,1):.1%}")
    print(f"multi-select EXACT verb-set accuracy: {ms_exact_set}/{n_done} "
          f"= {ms_exact_set/max(n_done,1):.1%}")
    print(f"multi-select SECONDARY-verb recall (the number that actually "
          f"matters -- this is what N0's v3 schema got 0% on in free "
          f"generation): {sec_hit}/{sec_total} = {sec_hit/max(sec_total,1):.1%}")
    print(f"multi-select false-positive rate (extra wrong verbs selected): "
          f"{fp_total}/{pred_total} = {fp_total/max(pred_total,1):.1%}")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=list(a.pool_data) + [a.target_data],
                   extra={"n_done": n_done, "single_choice_accuracy": sc_primary_correct / max(n_done, 1),
                          "secondary_recall": sec_hit / max(sec_total, 1)})


if __name__ == "__main__":
    main()
