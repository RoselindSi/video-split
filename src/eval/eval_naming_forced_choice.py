"""P2 -- forced-choice inverse-verb benchmark: the clean, direct test of
whether Qwen3-VL uses motion direction.

The earlier frame-REVERSAL probe conflated two things: order-sensitivity and
distribution-shift grounding failure (reversed video is out-of-distribution --
8.1% of cases hallucinated an entirely different object). This benchmark avoids
that: clips are shown in NORMAL forward order (in-distribution), and for each
GT segment whose verb belongs to a known reversible pair (remove<->insert,
open<->close, fold<->unfold, coil<->uncoil, extend<->retract, fill<->empty,
...), the model is asked a forced two-choice question: does the clip show
<verb><object> or its opposite <inverse_verb><object>? Options are randomly
assigned to A/B per item to cancel position bias.

This isolates "does the model discriminate direction" from confounds in free
generation (verb vocabulary choice, JSON formatting, narrative prose) -- it
only has to output one letter. Accuracy vs the 50% chance baseline is the
answer to "does it use motion direction".

Usage (server):
    python eval_naming_forced_choice.py \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/naming_forced_choice.jsonl --max_segments_per_video 15
"""
import argparse, json, os, random, re
import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from src.seg_rewards import _as_segs
except ImportError:
    from src.rewards.seg_rewards import _as_segs

try:
    from eval_naming_persegment import sample_transition_frames
except ImportError:
    try:
        from src.eval.eval_naming_persegment import sample_transition_frames
    except ImportError:
        from eval_naming_persegment import sample_transition_frames

try:
    from eval_naming_decoupled import content
except ImportError:
    try:
        from src.eval.eval_naming_decoupled import content
    except ImportError:
        from eval_naming_decoupled import content

# canonical single-word inverse map -- used to CONSTRUCT the two phrase
# choices from the GT verb (distinct from the fuzzy cluster-based INVERSE_PAIRS
# in naming_transition_report.py, which SCORES free-text against a set).
INVERSE_WORD = {
    "remove": "insert", "insert": "remove", "take": "insert", "extract": "insert",
    "unpack": "repack", "unbox": "repack", "repack": "unpack", "pack": "unpack",
    "open": "close", "close": "open", "unwrap": "wrap", "wrap": "unwrap",
    "fold": "unfold", "unfold": "fold",
    "coil": "uncoil", "uncoil": "coil",
    "extend": "retract", "retract": "extend",
    "fill": "empty", "empty": "fill",
    "tighten": "loosen", "loosen": "tighten",
    "screw": "unscrew", "unscrew": "screw",
    "attach": "detach", "detach": "attach", "install": "uninstall", "mount": "detach",
    "pick": "put", "put": "pick", "grab": "put", "place": "remove",
}

LETTER_RE = re.compile(r"\b([AB])\b", re.I)


def find_verb_and_object(gt_name):
    toks = gt_name.lower().split()
    v = next((t for t in toks[:3] if t in INVERSE_WORD), None)
    if v is None:
        return None, None
    obj_words = content(gt_name)
    obj = " ".join(sorted(obj_words, key=lambda w: gt_name.lower().find(w)))[:40]
    return v, (obj or "the object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_before", type=int, default=4)
    ap.add_argument("--n_during", type=int, default=8)
    ap.add_argument("--n_after", type=int, default=4)
    ap.add_argument("--context_s", type=float, default=1.0)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--max_segments_per_video", type=int, default=0,
                    help="cap of REVERSIBLE-verb segments sampled per video")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed)

    rows = json.load(open(a.data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    n_done = n_correct = 0
    for r in rows:
        gts = _as_segs(r["solution"])
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        idx_pool = list(range(len(gts)))
        rng.shuffle(idx_pool)
        picked = 0
        for si in idx_pool:
            if a.max_segments_per_video and picked >= a.max_segments_per_video:
                break
            name, s, e = gts[si]
            v, obj = find_verb_and_object(name)
            if v is None:
                continue
            inv = INVERSE_WORD[v]
            correct_phrase, wrong_phrase = f"{v} {obj}", f"{inv} {obj}"
            if rng.random() < 0.5:
                opt_a, opt_b, correct_letter = correct_phrase, wrong_phrase, "A"
            else:
                opt_a, opt_b, correct_letter = wrong_phrase, correct_phrase, "B"

            frames, fidx = sample_transition_frames(
                vr, vfps, s, e, a.context_s, a.n_before, a.n_during, a.n_after)
            content_msg = [{"type": "image", "image": Image.fromarray(f),
                            "max_pixels": a.max_pixels} for f in frames]
            content_msg.append({"type": "text", "text": (
                "The images are frames in temporal order (before, during, after "
                "a short clip). Which better describes the action shown?\n"
                f"A: {opt_a}\nB: {opt_b}\n"
                "Answer with exactly one letter: A or B.")})
            msgs = [{"role": "user", "content": content_msg}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=5, do_sample=False)
            out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
            m = LETTER_RE.search(out)
            pred_letter = m.group(1).upper() if m else "?"
            correct = (pred_letter == correct_letter)
            n_done += 1; n_correct += int(correct); picked += 1
            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": si, "gt_name": name, "verb": v, "inverse": inv,
                   "opt_a": opt_a, "opt_b": opt_b, "correct_letter": correct_letter,
                   "pred_letter": pred_letter, "correct": correct, "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{si} v={v} A='{opt_a}' B='{opt_b}' "
                  f"correct={correct_letter} pred={pred_letter} "
                  f"{'OK' if correct else 'WRONG'}")
        del vr
    print(f"\n==== FORCED-CHOICE (n={n_done}) ====")
    print(f"accuracy: {n_correct}/{n_done} = {n_correct/max(n_done,1):.1%}  "
          f"(chance = 50%)")


if __name__ == "__main__":
    main()
