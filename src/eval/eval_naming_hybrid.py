"""A6 test — hybrid-resolution decoupled naming eval.

Motivation: uniform 4x resolution on the whole video gives a small naming gain
at 4x cost (see naming-and-10fps-plan.md). Qwen's video tensor is a single
uniform grid (can't vary resolution frame-by-frame within one video), so instead
we keep the VIDEO at cheap base resolution and append ~27% of frames (motion-peak
KEYFRAMES, pure-CV, no model) as separate HIGH-RES IMAGE crops with timestamps.
Cost ~= 1x video + small keyframe budget, far below a uniform 4x video.

Keyframe detection mirrors src/preprocess/analyze_frames.py (duplicated inline
here to keep this eval self-contained / robust to server sync state).

Import shim: server (flat `src/seg_rewards.py`) vs repo (nested `src/rewards/`).

Usage (server):
    python eval_naming_hybrid.py --model_base /workspace/tr1/ckpts/<model> \
        --out /tmp/naming_hybrid_<model>.jsonl --keyframe_pixels_mult 4
"""
import argparse, json, os, re, statistics
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

try:                                            # server flat layout
    from src.seg_rewards import _as_segs, _default_sim_fn
except ImportError:                             # repo nested layout
    from src.rewards.seg_rewards import _as_segs, _default_sim_fn

try:
    from src.eval.eval_naming_decoupled import (
        verb_match, obj_f1, is_generic, build_prompt, NAME_RE)
except ImportError:
    from eval_naming_decoupled import (
        verb_match, obj_f1, is_generic, build_prompt, NAME_RE)


# ---- pure-CV keyframe detection (mirrors src/preprocess/analyze_frames.py) ----
def gray(f):
    return f.astype(np.float32).mean(axis=2)


def lap_var(g):
    l = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
         + g[1:-1, :-2] + g[1:-1, 2:])
    return float(l.var())


def find_keyframes(path, base_fps, th_blur, th_key, max_keys):
    from decord import VideoReader
    vr = VideoReader(path)
    n = len(vr); vfps = vr.get_avg_fps()
    step = max(1, int(round(vfps / base_fps)))
    idxs = list(range(0, n, step))
    frames = vr.get_batch(idxs).asnumpy()
    grays = [gray(f) for f in frames]
    blur = [lap_var(g) for g in grays]
    motion = [0.0] + [float(np.abs(grays[i] - grays[i - 1]).mean())
                      for i in range(1, len(grays))]
    ts = [idxs[i] / vfps for i in range(len(idxs))]
    keys = []
    for i in range(len(motion)):
        if blur[i] < th_blur:
            continue
        if (motion[i] >= th_key
                and (i == 0 or motion[i] >= motion[i - 1])
                and (i == len(motion) - 1 or motion[i] >= motion[i + 1])):
            keys.append(i)
    if len(keys) > max_keys:                    # keep strongest peaks only
        keys = sorted(keys, key=lambda i: -motion[i])[:max_keys]
        keys.sort()
    return [(ts[i], frames[i]) for i in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", default="/workspace/tr1/data_handtask/train_multiseg_val.json")
    ap.add_argument("--out", default="logs/eval_naming_hybrid.jsonl")
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28,
                    help="base video resolution budget (kept at 1x)")
    ap.add_argument("--keyframe_pixels_mult", type=float, default=4.0,
                    help="per-frame pixel budget multiplier for keyframe image crops")
    ap.add_argument("--th_blur", type=float, default=100.0)
    ap.add_argument("--th_key", type=float, default=8.0)
    ap.add_argument("--max_keys", type=int, default=8,
                    help="cap keyframes/video to bound extra token cost")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    sim = _default_sim_fn()

    # per-frame budget at base video resolution, ~120 frames/video at 2fps/60s
    base_per_frame = a.total_pixels / 120
    key_max_pixels = int(base_per_frame * a.keyframe_pixels_mult)

    rows = json.load(open(a.data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w"); agg = []
    for r in rows:
        gts = _as_segs(r["solution"])
        keys = find_keyframes(r["video"], base_fps=2.0, th_blur=a.th_blur,
                              th_key=a.th_key, max_keys=a.max_keys)

        content = [{"type": "video", "video": r["video"], "total_pixels": a.total_pixels}]
        for t, frame in keys:
            content.append({"type": "image", "image": Image.fromarray(frame),
                            "max_pixels": key_max_pixels})
        note = ""
        if keys:
            ts_str = ", ".join(f"{t:.1f}s" for t, _ in keys)
            note = (f"\n\nThe {len(keys)} images above are close-up frames at times "
                    f"{ts_str}, in that order, for identifying small objects/details.")
        content.append({"type": "text", "text": build_prompt(gts) + note})

        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        if isinstance(vkw.get("fps"), list):
            vkw["fps"] = vkw["fps"][0]
        fps_val = vkw.get("fps", 2.0)
        nf = vids[0].shape[0] if hasattr(vids[0], "shape") else len(vids[0])
        vmeta = [{"fps": float(fps_val), "total_num_frames": int(nf),
                  "duration": float(nf) / float(fps_val)}]
        inp = proc(text=[text], images=imgs, videos=vids, video_metadata=vmeta,
                   return_tensors="pt").to("cuda")
        n_tokens = int(inp["input_ids"].shape[1])
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        pred_names = [m.strip() for m in NAME_RE.findall(out)]
        gt_names = [g[0] for g in gts]
        k = min(len(pred_names), len(gt_names))
        vm = [verb_match(pred_names[i], gt_names[i]) for i in range(k)]
        of = [obj_f1(pred_names[i], gt_names[i]) for i in range(k)]
        gr = [is_generic(pred_names[i]) for i in range(k)]
        es = [sim(pred_names[i], [gt_names[i]])[0] for i in range(k)]
        m = {"n_gt": len(gt_names), "n_pred": len(pred_names), "n_keys": len(keys),
             "prompt_tokens": n_tokens,
             "count_match": 1.0 if len(pred_names) == len(gt_names) else 0.0,
             "verb_acc": statistics.mean(vm) if vm else 0.0,
             "obj_f1": statistics.mean(of) if of else 0.0,
             "generic_rate": statistics.mean(gr) if gr else 0.0,
             "emb_sim": statistics.mean(es) if es else 0.0}
        agg.append(m)
        print(os.path.basename(r["video"]), "gt", m["n_gt"], "keys", m["n_keys"],
              "tok", m["prompt_tokens"], "verb", round(m["verb_acc"], 2),
              "obj", round(m["obj_f1"], 2), "sim", round(m["emb_sim"], 2))
        fout.write(json.dumps({"video": r["video"], **m, "pred_names": pred_names,
                               "gt_names": gt_names, "raw": out}) + "\n")
        fout.flush()
    print("\n==== NAMING HYBRID (n=%d, keyframe_mult=%.1fx) ====" %
          (len(agg), a.keyframe_pixels_mult))
    for k in ["count_match", "verb_acc", "obj_f1", "generic_rate", "emb_sim",
              "n_keys", "prompt_tokens"]:
        print(k.ljust(14), round(statistics.mean([m[k] for m in agg]), 2))


if __name__ == "__main__":
    main()
