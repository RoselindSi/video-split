"""E2 feature extraction for long recordings (human_ego_recording_segmentation).

Long videos (10-16 min @10fps) can't go through the ViT in one pass, so we:
  1. sample candidate frames at --fps,
  2. P0 filter (pure-CV): drop black + blurry frames (always), static frames
     (optional --filter_static),
  3. encode each KEPT frame independently through the frozen Qwen3-VL ViT (image
     mode), spatially mean-pool last_hidden_state -> one vector per frame,
  4. cache the continuous [N, D] sequence + each frame's REAL timestamp.

Per-frame (image) encoding makes chunking seamless: frames are encoded
independently, so batching is only a memory convenience -- the concatenated
feature sequence has no window seams. Temporal modeling is left to the head.

Note: with heavy --filter_static the timestamps become non-uniform; the head's
1D conv assumes ~uniform spacing, so conservative filtering (black+blur only) is
the default. See train_head for the non-uniform caveat.

Usage (server):
    python -m src.boundary.extract_features_recseg \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /workspace/tr1/data_recseg/feat_val.pt --fps 2
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def gray(f):
    return f.astype(np.float32).mean(axis=2)


def lap_var(g):
    l = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
         + g[1:-1, :-2] + g[1:-1, 2:])
    return float(l.var())


def encode_frames(frames, proc, model, max_pixels):
    """frames: list of HxWx3 uint8 -> [n, D] pooled per-frame features."""
    content = [{"type": "image", "image": Image.fromarray(f), "max_pixels": max_pixels}
               for f in frames]
    content.append({"type": "text", "text": "x"})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    grid = inp["image_grid_thw"]
    with torch.no_grad():
        out = model.model.visual(inp["pixel_values"].to(model.dtype), grid_thw=grid)
    lhs = out.last_hidden_state.float()
    feats, off = [], 0
    for gr in grid.tolist():
        p = gr[0] * gr[1] * gr[2]
        feats.append(lhs[off:off + p].mean(0))
        off += p
    return torch.stack(feats).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--crop", choices=["none", "left", "right"], default="none",
                    help="packed_mid_stereo is two side-by-side cameras; crop to one "
                         "half to un-dilute the hand-action signal (boundary features)")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--enc_batch", type=int, default=48, help="frames per ViT call")
    ap.add_argument("--dec_chunk", type=int, default=200, help="frames per decode")
    ap.add_argument("--th_black", type=float, default=20.0)
    ap.add_argument("--th_blur", type=float, default=100.0)
    ap.add_argument("--filter_static", action="store_true")
    ap.add_argument("--th_static", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()

    from decord import VideoReader
    rows = json.load(open(a.data))
    if a.limit:
        rows = rows[:a.limit]
    cache = []
    for ri, r in enumerate(rows):
        vr = VideoReader(r["video"])
        n = len(vr); vfps = vr.get_avg_fps()
        step = max(1, int(round(vfps / a.fps)))
        cand = list(range(0, n, step))

        kept_frames, kept_times = [], []
        prev_g = None
        n_black = n_blur = n_static = 0
        for c0 in range(0, len(cand), a.dec_chunk):
            chunk_idx = cand[c0:c0 + a.dec_chunk]
            frames = vr.get_batch(chunk_idx).asnumpy()
            if a.crop != "none":
                W = frames.shape[2]
                frames = frames[:, :, :W // 2] if a.crop == "left" else frames[:, :, W // 2:]
            for j, f in enumerate(frames):
                g = gray(f)
                if float(g.mean()) < a.th_black:
                    n_black += 1; continue
                if lap_var(g) < a.th_blur:
                    n_blur += 1; continue
                if a.filter_static and prev_g is not None:
                    if float(np.abs(g - prev_g).mean()) < a.th_static:
                        n_static += 1; continue
                prev_g = g
                kept_frames.append(f)
                kept_times.append(chunk_idx[j] / vfps)

        # encode kept frames in ViT-sized sub-batches
        feats = []
        for b0 in range(0, len(kept_frames), a.enc_batch):
            feats.append(encode_frames(kept_frames[b0:b0 + a.enc_batch],
                                       proc, model, a.max_pixels))
            torch.cuda.empty_cache()
        feats = torch.cat(feats, 0) if feats else torch.zeros(0)
        times = torch.tensor(kept_times)
        segs = [(s[0], float(s[1]), float(s[2])) for s in r["solution"]]
        cache.append({"video": r["video"], "recording_id": r.get("recording_id"),
                      "feats": feats, "times": times,
                      "duration": float(r["duration"]), "segments": segs})
        print(f"[{ri+1}/{len(rows)}] {r.get('recording_id')} "
              f"cand {len(cand)} kept {len(kept_frames)} "
              f"(black {n_black} blur {n_blur} static {n_static}) "
              f"feats {tuple(feats.shape)} segs {len(segs)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(cache, a.out)
    print(f"\nwrote {len(cache)} recordings -> {a.out}")


if __name__ == "__main__":
    main()
