"""P2 -- same-object hard-negative multiple-choice benchmark.

Instead of hand-writing per-object candidate-verb lists (won't generalize),
this builds hard negatives DATA-DRIVEN: within one recording, group GT segments
whose OBJECT overlaps heavily (Jaccard on content words) but whose VERB
differs. If a group has >=3 distinct verbs, sample a target segment and ask a
forced multi-choice question among those real, observed verbs (options
shuffled). This tests "does the model discriminate ACTION given the SAME
object is clearly in view" without the confound of free-generation format.

Usage (server):
    python eval_naming_hard_negative.py \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/naming_hard_negative.jsonl --max_per_video 3
"""
import argparse, json, os, random, re, statistics
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

_WORD = re.compile(r"[a-zA-Z]+")
tok = lambda s: [w.lower() for w in _WORD.findall(s)]
STOP = {"the", "a", "an", "and", "or", "to", "of", "into", "onto", "on", "in",
        "with", "from", "for", "at", "by", "up", "down", "out", "off", "over",
        "then", "all", "it", "its", "this", "that", "these", "those", "each",
        "again", "first", "second", "third", "starts", "ends", "here", "step",
        "cycle", "iteration"}
ORD_RE = re.compile(r"\b\d+(st|nd|rd|th)\b|\bfirst\b|\bsecond\b|\bthird\b", re.I)
LETTER_RE = re.compile(r"\b([A-E])\b", re.I)


def verb_of(name):
    toks = tok(name)
    return toks[0] if toks else ""


def object_words(name):
    v = verb_of(name)
    return {w for w in tok(name) if w not in STOP and w != v and len(w) > 2}


def group_by_object(gts, jaccard_thresh=0.4):
    """gts: list of (name, start, end). Returns groups of segment indices whose
    object words overlap >= jaccard_thresh but verbs differ, i.e. real
    same-object/different-action clusters observed in this recording's own GT."""
    n = len(gts)
    objs = [object_words(g[0]) for g in gts]
    verbs = [verb_of(g[0]) for g in gts]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        if not objs[i]:
            continue
        for j in range(i + 1, n):
            if not objs[j] or verbs[i] == verbs[j]:
                continue
            jac = len(objs[i] & objs[j]) / max(len(objs[i] | objs[j]), 1)
            if jac >= jaccard_thresh:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [idxs for idxs in groups.values() if len({verbs[i] for i in idxs}) >= 3]


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
    ap.add_argument("--max_choices", type=int, default=4)
    ap.add_argument("--max_per_video", type=int, default=3,
                    help="cap of hard-negative items sampled per recording")
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
        groups = group_by_object(gts)
        if not groups:
            continue
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        picked = 0
        rng.shuffle(groups)
        for idxs in groups:
            if picked >= a.max_per_video:
                break
            distinct_verbs = sorted({verb_of(gts[i][0]) for i in idxs})
            target_i = rng.choice(idxs)
            name, s, e = gts[target_i]
            correct_verb = verb_of(name)
            obj_phrase = " ".join(sorted(object_words(name),
                                         key=lambda w: name.lower().find(w)))[:40]
            distractors = [v for v in distinct_verbs if v != correct_verb]
            rng.shuffle(distractors)
            options = [correct_verb] + distractors[:a.max_choices - 1]
            rng.shuffle(options)
            letters = "ABCDE"[:len(options)]
            correct_letter = letters[options.index(correct_verb)]

            frames, fidx = sample_transition_frames(
                vr, vfps, s, e, a.context_s, a.n_before, a.n_during, a.n_after)
            content_msg = [{"type": "image", "image": Image.fromarray(f),
                            "max_pixels": a.max_pixels} for f in frames]
            opts_str = "\n".join(f"{l}: {v} {obj_phrase}" for l, v in zip(letters, options))
            content_msg.append({"type": "text", "text": (
                "The images are frames in temporal order (before, during, after "
                f"a short clip) of a person acting on the {obj_phrase}. Which "
                f"option best describes the action shown?\n{opts_str}\n"
                f"Answer with exactly one letter: {'/'.join(letters)}.")})
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
                   "segment_idx": target_i, "gt_name": name, "correct_verb": correct_verb,
                   "options": options, "correct_letter": correct_letter,
                   "pred_letter": pred_letter, "correct": correct,
                   "n_choices": len(options), "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{target_i} obj='{obj_phrase}' "
                  f"correct_verb={correct_verb} options={options} "
                  f"pred={pred_letter}({'OK' if correct else 'WRONG'})")
        del vr
    chance = statistics.mean([1 / json.loads(l)["n_choices"]
                              for l in open(a.out)]) if n_done else 0.0
    print(f"\n==== HARD-NEGATIVE MULTI-CHOICE (n={n_done}) ====")
    print(f"accuracy: {n_correct}/{n_done} = {n_correct/max(n_done,1):.1%}  "
          f"(mean chance level given per-item choice count = {chance:.1%})")


if __name__ == "__main__":
    main()
