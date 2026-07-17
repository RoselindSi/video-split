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


def sample_clip_frames(vr, vfps, start, end, context_s, n_frames):
    lo = max(0.0, start - context_s)
    hi = end + context_s
    lo_i, hi_i = int(lo * vfps), min(int(hi * vfps), len(vr) - 1)
    if hi_i <= lo_i:
        hi_i = min(lo_i + 1, len(vr) - 1)
    idxs = [lo_i + round(i * (hi_i - lo_i) / max(n_frames - 1, 1)) for i in range(n_frames)]
    idxs = sorted(set(min(max(i, 0), len(vr) - 1) for i in idxs))
    return vr.get_batch(idxs).asnumpy(), idxs


def build_content(vr, vfps, n, start, end, context_s, n_frames, max_pixels, mode):
    clip_frames, fidx = sample_clip_frames(vr, vfps, start, end, context_s, n_frames)
    content = []
    extra_idx = {}
    if mode == "procedure":
        mid_i = n // 2
        ref = vr[mid_i].asnumpy()
        content.append({"type": "image", "image": Image.fromarray(ref), "max_pixels": max_pixels})
        extra_idx["procedure_ref"] = mid_i
    elif mode == "neighbor":
        before_i = max(0, int((start - 3.0) * vfps))
        before = vr[before_i].asnumpy()
        content.append({"type": "image", "image": Image.fromarray(before), "max_pixels": max_pixels})
        extra_idx["before"] = before_i
    for f in clip_frames:
        content.append({"type": "image", "image": Image.fromarray(f), "max_pixels": max_pixels})
    if mode == "neighbor":
        after_i = min(n - 1, int((end + 3.0) * vfps))
        after = vr[after_i].asnumpy()
        content.append({"type": "image", "image": Image.fromarray(after), "max_pixels": max_pixels})
        extra_idx["after"] = after_i
    content.append({"type": "text", "text": PROMPTS[mode]})
    return content, fidx, extra_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=6, help="frames sampled per segment clip")
    ap.add_argument("--context_s", type=float, default=1.0,
                    help="extra seconds of boundary context included in the clip window")
    ap.add_argument("--context_mode", choices=["local", "procedure", "neighbor"],
                    default="local")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="ONE name, not a list -- keep this small so decode "
                         "degeneration (long-list repetition) cannot happen")
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
        for si in idx_pool:
            name, s, e = gts[si]
            content, fidx, extra_idx = build_content(
                vr, vfps, n, s, e, a.context_s, a.n_frames, a.max_pixels, a.context_mode)
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, repetition_penalty=1.3)
            out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
            m = NAME_RE.search(out)
            pred = m.group(1).strip() if m else out.strip()
            es = sim(pred, [name])[0]
            rec = {"video": r["video"], "recording_id": r.get("recording_id"),
                   "segment_idx": si, "start": s, "end": e, "gt_name": name,
                   "pred_name": pred, "emb_sim": es, "frame_idx": fidx,
                   "extra_idx": extra_idx, "context_mode": a.context_mode,
                   "n_frames": a.n_frames, "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{si} gt='{name}' pred='{pred}' sim={es:.2f}")
            n_done += 1
        del vr
    print(f"\nwrote {n_done} segment-level naming predictions -> {a.out}")


if __name__ == "__main__":
    main()
