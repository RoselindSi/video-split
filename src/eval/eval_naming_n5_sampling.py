"""N5 -- input-sampling ablation on the EXACT SAME 81 (or however many) items
from N4's hard-negative benchmark. Holds the question (object, options,
correct_letter) fixed and only varies what frames the model sees, isolating
whether the model's verb discrimination depends on temporal sampling/order:

  uniform16   : 16 frames evenly spaced across [start, end] only (no before/
                after context) -- does it need the transition context at all?
  bda         : the ORIGINAL before/during/after 4+8+4 sampling N4 already
                used -- reused via the prior jsonl's pred_letter, NOT
                re-run (saves compute, and guarantees it's identical).
  single      : one frame at the segment midpoint -- lower bound: is there
                even single-frame-visible information for these verbs.
  reversed16  : the SAME bda 16 frames, temporal order reversed. If the model
                actually uses temporal order, this should make it MORE likely
                to pick the (wrong) verb that is the correct verb's inverse
                (that's the key metric here, not just accuracy).
  shuffled16  : the SAME bda 16 frames, order randomly permuted. Should hurt
                any verb whose meaning depends on direction (retrieve/flip/
                slide/extend/retract/...) more than order-insensitive verbs.

The prompt text is held IDENTICAL across variants ("frames in temporal
order...") even for reversed/shuffled -- we want to know if the model's
ANSWER changes when the true order is scrambled, not whether it can detect
scrambling. Only the single-frame variant drops the "temporal order" claim
since it's meaningless with n=1.

Usage (server):
    python -m src.eval.eval_naming_n5_sampling \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --prev_jsonl /tmp/naming_hard_negative_v2.jsonl \
        --out_dir /tmp/n5 --variants uniform16 single reversed16 shuffled16
"""
import argparse, json, os, random, re

import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

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

from src.analysis.build_ontology import STRICT_INVERSE, CONTEXTUAL_INVERSE

LETTER_RE = re.compile(r"\b([A-D])\b", re.I)
DIR_SENSITIVE_VERBS = {"retrieve", "flip", "slide", "extend", "retract",
                       "remove", "seat", "insert", "fold", "unfold",
                       "coil", "uncoil", "open", "close"}


def sample_uniform(vr, vfps, s, e, n=16):
    n_frames = len(vr)
    ts = [s] if n <= 1 else [s + (e - s) * i / (n - 1) for i in range(n)]
    idx = [min(max(0, int(t * vfps)), n_frames - 1) for t in ts]
    return [vr[i].asnumpy() for i in idx], idx


def get_frames(vr, vfps, s, e, variant, bda_frames, bda_idx, context_s):
    if variant == "uniform16":
        return sample_uniform(vr, vfps, s, e, 16)
    if variant == "single":
        return sample_uniform(vr, vfps, s, e, 1)
    if variant == "reversed16":
        return list(reversed(bda_frames)), list(reversed(bda_idx))
    if variant == "shuffled16":
        order = list(range(len(bda_frames)))
        random.Random(0).shuffle(order)
        return [bda_frames[i] for i in order], [bda_idx[i] for i in order]
    raise ValueError(variant)


def ask(proc, model, frames, options, obj, claim_temporal_order):
    letters = "ABCD"[:len(options)]
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
    if claim_temporal_order and len(frames) > 1:
        lead = ("The images are frames in temporal order (before, during, after "
                f"a short clip) of a person acting on the {obj}.")
    else:
        lead = f"The image shows a person acting on the {obj}."
    content_msg.append({"type": "text", "text": (
        f"{lead} Which option best describes the action shown?\n{opts_str}\n"
        f"Answer with exactly one letter: {'/'.join(letters)}.")})
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=5, do_sample=False)
    out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = LETTER_RE.search(out)
    return (m.group(1).upper() if m else "?"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--prev_jsonl", required=True,
                     help="output of eval_naming_hard_negative_v2.py -- reused "
                          "for the exact items/options/correct_letter, and for "
                          "the 'bda' variant's answer (not re-inferenced)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--context_s", type=float, default=1.0)
    ap.add_argument("--variants", nargs="+",
                     default=["uniform16", "single", "reversed16", "shuffled16"])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.prev_jsonl)
    items = [json.loads(l) for l in open(a.prev_jsonl)]
    rows = {r.get("recording_id"): r for r in json.load(open(a.target_data))}

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()

    results = {v: [] for v in ["bda"] + a.variants}
    vr_cache = {}
    for it in items:
        rid = it["recording_id"]
        r = rows.get(rid)
        if r is None:
            continue
        gts = _as_segs(r["solution"])
        name, s, e = gts[it["segment_idx"]]
        if rid not in vr_cache:
            vr_cache[rid] = VideoReader(r["video"], num_threads=1)
        vr = vr_cache[rid]
        vfps = vr.get_avg_fps()

        bda_frames, bda_idx = sample_transition_frames(vr, vfps, s, e, a.context_s, 4, 8, 4)
        results["bda"].append({**it, "variant": "bda", "pred_letter": it["pred_letter"],
                                "correct": it["correct"], "frame_indices": bda_idx})

        for variant in a.variants:
            frames, idx = get_frames(vr, vfps, s, e, variant, bda_frames, bda_idx, a.context_s)
            claim_order = variant not in ("single",)
            pred_letter, raw = ask(proc, model, frames, it["options"], it["object"], claim_order)
            correct = (pred_letter == it["correct_letter"])
            results[variant].append({**it, "variant": variant, "pred_letter": pred_letter,
                                     "correct": correct, "frame_indices": idx, "raw": raw})
            print(f"[{variant}] {rid} seg{it['segment_idx']} obj='{it['object']}' "
                  f"correct_verb={it['correct_verb']} pred={pred_letter}"
                  f"({'OK' if correct else 'WRONG'})")

    with open(os.path.join(a.out_dir, "n5_results.jsonl"), "w") as f:
        for variant, recs in results.items():
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def inverse_of(correct_verb, options):
        inv_set = set(STRICT_INVERSE.get(correct_verb, [])) | set(CONTEXTUAL_INVERSE.get(correct_verb, []))
        return next((o for o in options if o in inv_set), None)

    print(f"\n==== N5 sampling ablation (n={len(items)} items per variant) ====")
    print(f"{'variant':12s} {'overall':>9s} {'inverse-pair acc':>17s} "
          f"{'picked-inverse rate':>20s} {'dir-sensitive verbs':>20s}")
    for variant, recs in results.items():
        overall = sum(r["correct"] for r in recs) / max(len(recs), 1)
        inv = [r for r in recs if r.get("has_inverse_distractor")]
        inv_acc = sum(r["correct"] for r in inv) / max(len(inv), 1) if inv else float("nan")
        # of the inverse-pair items, how often did the pred letter equal the
        # SPECIFIC inverse-verb option (not just "any wrong answer")
        picked_inv = 0
        for r in inv:
            iv = inverse_of(r["correct_verb"], r["options"])
            pl = r["pred_letter"]
            if iv and pl in "ABCD" and r["options"][ord(pl) - 65] == iv:
                picked_inv += 1
        picked_inv_rate = picked_inv / max(len(inv), 1) if inv else float("nan")
        dsv = [r for r in recs if r["correct_verb"] in DIR_SENSITIVE_VERBS]
        dsv_acc = sum(r["correct"] for r in dsv) / max(len(dsv), 1) if dsv else float("nan")
        print(f"{variant:12s} {overall:9.1%} {inv_acc:17.1%} {picked_inv_rate:20.1%} "
              f"{dsv_acc:20.1%} (n_inv={len(inv)}, n_dsv={len(dsv)})")

    print("\nread: if 'bda' (real before/during/after) clearly beats 'uniform16' "
          "and 'single', temporal transition sampling is doing real work -- keep "
          "it. If 'reversed16' inverse-pair accuracy DROPS vs 'bda' (model more "
          "often picks the inverse verb when frames are reversed), the model IS "
          "using true temporal order for direction, not guessing. If "
          "'shuffled16' hurts dir-sensitive verbs specifically (vs less on "
          "direction-insensitive verbs), order matters for direction "
          "specifically, not just 'more frames = better'.")

    from src.eval.run_manifest import write_manifest
    out_stub = os.path.join(a.out_dir, "n5_results.jsonl")
    write_manifest(out_stub, input_paths=[a.target_data, a.prev_jsonl],
                   extra={"n_items": len(items), "variants": a.variants,
                          "note": "input_files includes prev_jsonl's content "
                                  "hash -- compare this against another N5 "
                                  "run's manifest before treating their "
                                  "numbers as the same underlying questions "
                                  "(see run_manifest.check_comparable)."})


if __name__ == "__main__":
    main()
