"""P1 -- true naming-only baseline: per-segment independent naming.

The whole-video decoupled eval (eval_naming_decoupled.py) asks the model to name
ALL segments of a recording in one generation call, matched back to GT by
POSITION. On dense recordings (mean 57 segs/video, up to 355) this fails for two
reasons unrelated to naming quality: index misalignment when the predicted line
count drifts, and decode-time degeneration (long structured-list greedy decoding
locking into repeating one line verbatim). Per-segment independent naming
sidesteps both (one segment in, one name out, no count to guess).

On the CLEAN per-segment baseline, a new dominant failure emerged: within one
recording, many genuinely different repeated/cyclic sub-steps get the SAME
prediction (e.g. 8 different mug-washing sub-steps -> "Wash the mug" every
time) -- a granularity/context problem, not object misrecognition (mean_sim
0.53, only ~14% clearly wrong). This script now supports:

  --n_frames / --context_s     : P1 sampling ablation (more frames / more
                                  boundary context per segment)
  --context_mode local|procedure|neighbor
                                : P2 context ablation.
      local     = only the segment's own clip (current baseline)
      procedure = + a single frame from the recording's overall midpoint as a
                  "what broad task is this" hint (may OVER-anchor predictions
                  to the macro task, which is the suspected cause of the
                  repetition pattern -- test this hypothesis)
      neighbor  = + one frame from just before the segment and one from just
                  after, with an EXPLICIT instruction to describe what CHANGED
                  relative to neighbors, not to summarize the whole task

Usage (server):
    python eval_naming_persegment.py \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/naming_persegment_v2.jsonl \
        --max_segments_per_video 10 --n_frames 6 --context_mode local
"""
import argparse, json, os, random, re
import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from src.seg_rewards import _as_segs, _default_sim_fn
except ImportError:
    from src.rewards.seg_rewards import _as_segs, _default_sim_fn

NAME_RE = re.compile(r"<name>(.*?)</name>", re.S | re.I)

BASE_INSTRUCTION = (
    "Give this clip a short name = an imperative verb + the specific object "
    "being acted on (e.g. \"Slide the water bottle\", \"Open the notebook\"). "
    "Name the actual object; do NOT use generic words like 'object' or 'item'. "
    "Output exactly one line:\n<seg><name>NAME</name></seg>")

# Structured state-transition output: forces the model to reason about what
# CHANGED (before -> after) rather than concatenating object + task family.
# Targets the dominant failure (right object, wrong verb: "remove strainer" ->
# "wash strainer") by making the object-state delta the thing it must produce.
CONTROLLED_VERBS = (
    "open, close, remove, insert, replace, unpack, repack, fold, unfold, coil, "
    "uncoil, extend, retract, fill, empty, tighten, loosen, attach, detach, "
    "pick up, put down, wipe, clean, rinse, scrub, inspect, rotate, flip, "
    "slide, press, pour, adjust, wrap, unwrap")

# v2 (P0 fix): the v1 schema let the model write long NARRATIVE sentences for
# state_before/state_after ("dirty, held over sink with water running from
# faucet...") which crowded out precise verb choice (e.g. "reach for" instead
# of a controlled verb) -- verb_acc/obj_f1 came out WORSE than free-text.
# v2: short schema, canonical_name placed early (survives truncation), an
# explicit controlled-verb list, and an 8-word cap on before/after so the model
# can't hide behind prose.
STRUCTURED_INSTRUCTION = (
    "Focus ONLY on THIS clip; do NOT summarize the overall procedure. Pick the "
    "verb from this list if it fits: " + CONTROLLED_VERBS + ". "
    "Output exactly one JSON object with SHORT values (a few words each, not "
    "full sentences):\n"
    "{\"verb\": \"<one controlled verb>\", \"object\": \"<short noun phrase>\", "
    "\"canonical_name\": \"<verb + object>\", "
    "\"before\": \"<state before, max 8 words>\", "
    "\"after\": \"<state after, max 8 words>\"}")

# JSON name key can drift; grab canonical_name, else fall back to verb+object.
JSON_NAME_RE = re.compile(r'"canonical_name"\s*:\s*"([^"]*)"', re.I)
JSON_VERB_RE = re.compile(r'"verb"\s*:\s*"([^"]*)"', re.I)
JSON_OBJ_RE = re.compile(r'"object"\s*:\s*"([^"]*)"', re.I)

PROMPTS = {
    "local": (
        "The images above show frames from ONE short clip of a person doing a "
        "task, sampled evenly from start to end.\n" + BASE_INSTRUCTION),
    "procedure": (
        "The first image is a reference frame from elsewhere in the same "
        "recording, showing the overall task being performed. The remaining "
        "images are frames from ONE short clip within that recording, sampled "
        "evenly from start to end.\n" + BASE_INSTRUCTION),
    "neighbor": (
        "The images below show, in order: one frame from JUST BEFORE a clip, "
        "then frames from THIS clip sampled evenly start to end, then one frame "
        "from JUST AFTER the clip. Focus ONLY on this clip and describe what "
        "action/change happens in it relative to its immediate neighbors -- do "
        "NOT summarize the overall task the recording is about, since many "
        "different clips in this recording show different sub-steps of the same "
        "broader task.\n" + BASE_INSTRUCTION),
}


def _even(lo_i, hi_i, k, nmax):
    if hi_i <= lo_i:
        hi_i = min(lo_i + 1, nmax - 1)
    return [min(max(lo_i + round(i * (hi_i - lo_i) / max(k - 1, 1)), 0), nmax - 1)
            for i in range(k)]


def sample_clip_frames(vr, vfps, start, end, context_s, n_frames):
    lo_i, hi_i = int(max(0.0, start - context_s) * vfps), int((end + context_s) * vfps)
    idxs = sorted(set(_even(lo_i, hi_i, n_frames, len(vr))))
    return vr.get_batch(idxs).asnumpy(), idxs


def sample_transition_frames(vr, vfps, start, end, ctx_s, n_before, n_during, n_after):
    """before-during-after sampling so the model sees state_before -> action ->
    state_after. n_frames total = n_before + n_during + n_after."""
    n = len(vr)
    s_i, e_i = int(start * vfps), int(end * vfps)
    before = _even(int(max(0.0, start - ctx_s) * vfps), s_i, n_before, n)
    during = _even(s_i, e_i, n_during, n)
    after = _even(e_i, int((end + ctx_s) * vfps), n_after, n)
    idxs = before + during + after                 # KEEP order (temporal), no sort/dedup
    idxs = [min(max(i, 0), n - 1) for i in idxs]
    return vr.get_batch(idxs).asnumpy(), idxs


def build_content(vr, vfps, n, start, end, context_s, n_frames, max_pixels, mode,
                  structured=False, reverse=False):
    extra_idx = {}
    if mode == "transition":
        nb = max(1, n_frames // 4)
        na = max(1, n_frames // 4)
        nd = max(1, n_frames - nb - na)
        clip_frames, fidx = sample_transition_frames(
            vr, vfps, start, end, context_s, nb, nd, na)
        extra_idx = {"n_before": nb, "n_during": nd, "n_after": na}
    else:
        clip_frames, fidx = sample_clip_frames(vr, vfps, start, end, context_s, n_frames)

    frames = list(clip_frames)
    if reverse:
        frames = frames[::-1]

    content = []
    if mode == "procedure":
        mid_i = n // 2
        content.append({"type": "image", "image": Image.fromarray(vr[mid_i].asnumpy()),
                        "max_pixels": max_pixels})
        extra_idx["procedure_ref"] = mid_i
    for f in frames:
        content.append({"type": "image", "image": Image.fromarray(f), "max_pixels": max_pixels})

    if structured:
        instr = STRUCTURED_INSTRUCTION
    elif mode == "transition":
        instr = ("The images are frames in temporal order: first a few from just "
                 "BEFORE the clip, then the clip itself, then a few from just "
                 "AFTER. " + BASE_INSTRUCTION)
    else:
        instr = PROMPTS[mode]
    content.append({"type": "text", "text": instr})
    return content, fidx, extra_idx


def parse_prediction(out, structured):
    if structured:
        m = JSON_NAME_RE.search(out)
        if m and m.group(1).strip():
            return m.group(1).strip()
        v = JSON_VERB_RE.search(out); o = JSON_OBJ_RE.search(out)
        if v and o:
            return f"{v.group(1).strip()} {o.group(1).strip()}".strip()
        return out.strip()
    m = NAME_RE.search(out)
    return m.group(1).strip() if m else out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=6, help="frames sampled per segment clip")
    ap.add_argument("--context_s", type=float, default=1.0,
                    help="extra seconds of boundary context included in the clip window")
    ap.add_argument("--context_mode",
                    choices=["local", "procedure", "neighbor", "transition"],
                    default="local",
                    help="transition = before/during/after sampling so the model "
                         "sees state_before -> action -> state_after")
    ap.add_argument("--structured", action="store_true",
                    help="output verb/object/state_before/state_after JSON instead "
                         "of a free-text name (targets right-object/wrong-verb)")
    ap.add_argument("--reverse", action="store_true",
                    help="feed frames in REVERSE temporal order -- order-sensitivity "
                         "control: a model using motion direction should change its "
                         "answer on direction-reversible actions (remove<->replace)")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="one name/JSON, not a list -- keep small; raised "
                         "automatically for --structured")
    ap.add_argument("--max_segments_per_video", type=int, default=0,
                    help="0 = all; else randomly subsample per recording (dense "
                         "videos have hundreds of segments)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    sim = _default_sim_fn()
    rng = random.Random(a.seed)

    rows = json.load(open(a.data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    n_done = 0
    for r in rows:
        gts = _as_segs(r["solution"])
        vr = VideoReader(r["video"], num_threads=1)
        vfps = vr.get_avg_fps()
        n = len(vr)
        idx_pool = list(range(len(gts)))
        if a.max_segments_per_video and len(idx_pool) > a.max_segments_per_video:
            idx_pool = sorted(rng.sample(idx_pool, a.max_segments_per_video))
        mnt = max(a.max_new_tokens, 160) if a.structured else a.max_new_tokens
        for si in idx_pool:
            name, s, e = gts[si]
            content, fidx, extra_idx = build_content(
                vr, vfps, n, s, e, a.context_s, a.n_frames, a.max_pixels,
                a.context_mode, structured=a.structured, reverse=a.reverse)
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=mnt,
                                     do_sample=False, repetition_penalty=1.3)
            out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
            pred = parse_prediction(out, a.structured)
            es = sim(pred, [name])[0]
            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": si, "start": s, "end": e, "gt_name": name,
                   "pred_name": pred, "emb_sim": es, "frame_idx": fidx,
                   "extra_idx": extra_idx, "context_mode": a.context_mode,
                   "structured": a.structured, "reverse": a.reverse,
                   "n_frames": a.n_frames, "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{si} gt='{name}' pred='{pred}' sim={es:.2f}")
            n_done += 1
        del vr
    print(f"\nwrote {n_done} segment-level naming predictions -> {a.out}")


if __name__ == "__main__":
    main()
