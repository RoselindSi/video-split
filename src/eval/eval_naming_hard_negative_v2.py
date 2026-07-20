"""N4 -- clean, FIXED 4-choice same-object hard-negative benchmark, rebuilt on
the frozen ontology (build_ontology.py) instead of the old script's raw-token
verb_of()/Jaccard grouping (which let inflection dupes like scrub/scrubbing
count as "different verbs", and left third/paper/re-type stopword junk as
"objects").

Differences from eval_naming_hard_negative.py (kept for reference, not used
here):
  - candidate verbs come from CANONICAL_VERBS post VERB_NORM + the object-
    conditioned CONTEXTUAL_VERB_NORM fold, so replace/reinstall/install don't
    masquerade as 3 different "observed verbs" on sink strainer.
  - GENERIC_VERBS (manipulate/present/display/adjust/move/arrange) are never
    used as distractors -- a generic verb isn't a meaningful wrong answer.
  - grouping is by CANONICAL OBJECT (exact match via extract_object), pooled
    across the WHOLE dataset (not just one recording), so the distractor pool
    reflects everything actually observed on that object, not one video's
    local Jaccard cluster.
  - FIXED 4 choices (chance = exactly 25%, not a variable ~28% average);
    objects with < 4 distinct legal verbs in their global pool are skipped
    rather than padded.
  - each item is tagged with whether a distractor is a known INVERSE (strict
    or contextual) of the correct verb, and whether the target segment's GT
    has >=2 verbs (compound), for the required per-slice reporting.

Usage (server):
    python -m src.eval.eval_naming_hard_negative_v2 \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --pool_data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/naming_hard_negative_v2.jsonl --max_per_video 3
"""
import argparse, json, os, random, statistics
from collections import Counter, defaultdict

import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.analysis.build_ontology import (
    extract_verbs, extract_object, CANONICAL_VERBS, GENERIC_VERBS,
    STRICT_INVERSE, CONTEXTUAL_INVERSE, CONTEXTUAL_VERB_NORM,
)

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

import re
LETTER_RE = re.compile(r"\b([A-D])\b", re.I)


def dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def primary_verb_and_object(name):
    """Ordered canonical verbs (post VERB_NORM), canonical object, folded via
    CONTEXTUAL_VERB_NORM using that object (so replace/reinstall/install on
    sink strainer all become 'seat' -- one verb, not three)."""
    verbs = dedup(extract_verbs(name))
    obj, _, _, _ = extract_object(name)
    folded = dedup(CONTEXTUAL_VERB_NORM.get((v, obj or ""), v) for v in verbs)
    return folded, obj


def build_object_verb_pool(paths):
    """object -> set of legal (non-generic) canonical verbs observed on it,
    anywhere in the given data. This is the GLOBAL candidate pool a
    same-object hard-negative question draws distractors from."""
    pool = defaultdict(set)
    for p in paths:
        for r in json.load(open(p)):
            for seg in r.get("solution", []):
                if not (seg and isinstance(seg[0], str)):
                    continue
                verbs, obj = primary_verb_and_object(seg[0])
                if not obj:
                    continue
                for v in verbs:
                    if v not in GENERIC_VERBS:
                        pool[obj].add(v)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--pool_data", nargs="+", required=True,
                     help="data files to build the global object->verb pool from "
                          "(train+val recommended for a richer distractor pool)")
    ap.add_argument("--target_data", required=True,
                     help="data file to actually sample benchmark items from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_before", type=int, default=4)
    ap.add_argument("--n_during", type=int, default=8)
    ap.add_argument("--n_after", type=int, default=4)
    ap.add_argument("--context_s", type=float, default=1.0)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--max_per_video", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = build_object_verb_pool(a.pool_data)
    eligible_objects = {o for o, vs in pool.items() if len(vs) >= 4}
    print(f"objects with >=4 legal distinct verbs (usable for fixed 4-choice): "
          f"{len(eligible_objects)}/{len(pool)}")

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed)

    rows = json.load(open(a.target_data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    n_done = n_correct = 0
    n_inverse_distractor = n_inverse_wrong = 0
    n_compound = n_compound_wrong = 0

    for r in rows:
        gts = _as_segs(r["solution"])
        candidates = []
        for i, (name, s, e) in enumerate(gts):
            verbs, obj = primary_verb_and_object(name)
            if not verbs or obj not in eligible_objects:
                continue
            correct_verb = verbs[0]
            if correct_verb not in pool[obj]:
                continue
            candidates.append((i, name, s, e, correct_verb, obj, len(verbs) >= 2))
        if not candidates:
            continue
        rng.shuffle(candidates)
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        picked = 0
        for i, name, s, e, correct_verb, obj, is_compound in candidates:
            if picked >= a.max_per_video:
                break
            pool_others = sorted(pool[obj] - {correct_verb})
            if len(pool_others) < 3:
                continue
            # force-include a real observed inverse of correct_verb as one of
            # the 3 distractors whenever the object's pool has one available --
            # otherwise whether we get an inverse-pair item at all is down to
            # random.sample() luck, which starves the direction-discrimination
            # analysis of samples (N5 audit: n_inv=6 out of 81, useless power).
            # Still only drawn from REAL observed (object,verb) pairs, nothing
            # synthetic.
            inv_candidates = [v for v in pool_others
                              if v in STRICT_INVERSE.get(correct_verb, [])
                              or v in CONTEXTUAL_INVERSE.get(correct_verb, [])]
            if inv_candidates:
                forced = rng.choice(inv_candidates)
                rest = [v for v in pool_others if v != forced]
                distractors = [forced] + rng.sample(rest, min(2, len(rest)))
                if len(distractors) < 3:
                    continue
            else:
                distractors = rng.sample(pool_others, 3)
            options = [correct_verb] + distractors
            rng.shuffle(options)
            letters = "ABCD"
            correct_letter = letters[options.index(correct_verb)]
            has_inverse_distractor = any(
                dv in STRICT_INVERSE.get(correct_verb, [])
                or dv in CONTEXTUAL_INVERSE.get(correct_verb, [])
                for dv in distractors)

            frames, fidx = sample_transition_frames(
                vr, vfps, s, e, a.context_s, a.n_before, a.n_during, a.n_after)
            content_msg = [{"type": "image", "image": Image.fromarray(f),
                            "max_pixels": a.max_pixels} for f in frames]
            opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
            content_msg.append({"type": "text", "text": (
                "The images are frames in temporal order (before, during, after "
                f"a short clip) of a person acting on the {obj}. Which option "
                f"best describes the action shown?\n{opts_str}\n"
                f"Answer with exactly one letter: A/B/C/D.")})
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
            if has_inverse_distractor:
                n_inverse_distractor += 1
                n_inverse_wrong += int(not correct)
            if is_compound:
                n_compound += 1
                n_compound_wrong += int(not correct)
            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": i, "start": s, "end": e, "gt_name": name, "object": obj,
                   "correct_verb": correct_verb, "options": options,
                   "correct_letter": correct_letter, "pred_letter": pred_letter,
                   "correct": correct, "has_inverse_distractor": has_inverse_distractor,
                   "is_compound": is_compound, "frame_indices": fidx, "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{i} obj='{obj}' "
                  f"correct_verb={correct_verb} options={options} "
                  f"pred={pred_letter}({'OK' if correct else 'WRONG'})")
        del vr

    print(f"\n==== N4 fixed 4-choice hard-negative (n={n_done}, chance=25.0%) ====")
    print(f"overall accuracy: {n_correct}/{n_done} = {n_correct/max(n_done,1):.1%}")
    if n_inverse_distractor:
        print(f"items with an inverse-pair distractor: {n_inverse_distractor} "
              f"(wrong: {n_inverse_wrong}, acc="
              f"{1-n_inverse_wrong/n_inverse_distractor:.1%}) -- direction "
              f"discrimination specifically")
    if n_compound:
        print(f"items whose GT segment is compound (>=2 verbs, scored on "
              f"primary only here): {n_compound} (wrong: {n_compound_wrong}, "
              f"acc={1-n_compound_wrong/n_compound:.1%})")

    per_obj = defaultdict(lambda: [0, 0])
    per_verb = defaultdict(lambda: [0, 0])
    for line in open(a.out):
        rec = json.loads(line)
        per_obj[rec["object"]][0] += 1; per_obj[rec["object"]][1] += int(rec["correct"])
        per_verb[rec["correct_verb"]][0] += 1; per_verb[rec["correct_verb"]][1] += int(rec["correct"])
    print("\nper-object accuracy:")
    for o, (n, c) in sorted(per_obj.items(), key=lambda kv: -kv[1][0]):
        print(f"  {o:20s} {c}/{n} = {c/n:.1%}")
    print("\nper-verb accuracy:")
    for v, (n, c) in sorted(per_verb.items(), key=lambda kv: -kv[1][0]):
        print(f"  {v:12s} {c}/{n} = {c/n:.1%}")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=list(a.pool_data) + [a.target_data],
                   extra={"n_done": n_done, "overall_accuracy": n_correct / max(n_done, 1)})


if __name__ == "__main__":
    main()
