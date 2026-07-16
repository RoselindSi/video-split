"""P1 -- true naming-only baseline: per-segment independent naming.

The whole-video decoupled eval (eval_naming_decoupled.py) asks the model to name
ALL segments of a recording in one generation call, matched back to GT by
POSITION. On dense recordings (mean 57 segs/video, up to 355) this fails for two
independent reasons that have nothing to do with naming quality:
  1. Index misalignment: any skipped/extra line shifts every later (pred, gt)
     pair out of correspondence.
  2. Decode-time degeneration: long structured-list greedy generation can lock
     into repeating the same line verbatim until max_new_tokens truncates it
     (observed directly: 36 identical "Rotate the blender lid" predictions on a
     147-segment recording).

This script sidesteps BOTH: for each GT segment independently, crop a short
local clip [start-context_s, end+context_s] + a couple of frames for global
procedure context, and ask for ONE structured name. One segment in, one name
out -- no count to guess, no long list to generate, no positional ambiguity.

Usage (server):
    python eval_naming_persegment.py \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/naming_persegment.jsonl --max_segments_per_video 10
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


def build_prompt():
    return ("The images above show frames from ONE short clip of a person doing "
            "a task, sampled evenly from start to end. Give this single clip a "
            "short name = an imperative verb + the specific object being acted "
            "on (e.g. \"Slide the water bottle\", \"Open the notebook\"). Name "
            "the actual object; do NOT use generic words like 'object' or "
            "'item'. Output exactly one line:\n<seg><name>NAME</name></seg>")


def sample_clip_frames(vr, vfps, start, end, context_s, n_frames):
    lo = max(0.0, start - context_s)
    hi = end + context_s
    lo_i, hi_i = int(lo * vfps), min(int(hi * vfps), len(vr) - 1)
    if hi_i <= lo_i:
        hi_i = min(lo_i + 1, len(vr) - 1)
    idxs = [lo_i + round(i * (hi_i - lo_i) / max(n_frames - 1, 1)) for i in range(n_frames)]
    idxs = sorted(set(min(max(i, 0), len(vr) - 1) for i in idxs))
    return vr.get_batch(idxs).asnumpy(), idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=6, help="frames sampled per segment clip")
    ap.add_argument("--context_s", type=float, default=1.0)
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
        idx_pool = list(range(len(gts)))
        if a.max_segments_per_video and len(idx_pool) > a.max_segments_per_video:
            idx_pool = sorted(rng.sample(idx_pool, a.max_segments_per_video))
        for si in idx_pool:
            name, s, e = gts[si]
            frames, fidx = sample_clip_frames(vr, vfps, s, e, a.context_s, a.n_frames)
            content = [{"type": "image", "image": Image.fromarray(f),
                       "max_pixels": a.max_pixels} for f in frames]
            content.append({"type": "text", "text": build_prompt()})
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
                   "pred_name": pred, "emb_sim": es, "frame_idx": fidx, "raw": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{r.get('recording_id')} seg{si} gt='{name}' pred='{pred}' sim={es:.2f}")
            n_done += 1
        del vr
    print(f"\nwrote {n_done} segment-level naming predictions -> {a.out}")


if __name__ == "__main__":
    main()
