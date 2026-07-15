"""E2 step 1 — extract frozen Qwen3-VL ViT per-timestep features.

Runs each video through the (frozen) Qwen3-VL vision tower, spatially mean-pools
`last_hidden_state` per temporal token -> a [T, D] sequence, and caches it with
timestamps + GT segments. This is the input to the supervised temporal boundary
head (train_head.py). One-time offline pass; the LM is never run.

Notes from probing Qwen3-VL-8B:
  - visual = model.model.visual (Qwen3VLVisionModel), call .visual(pixels, grid_thw=grid)
  - out.last_hidden_state : [T*H*W, 1152] at PATCH resolution (H,W from grid_thw)
  - spatial_merge_size=2, temporal_patch_size=2; at 2fps/60s -> T=60 (1s/token)
  - out.deepstack_features : 3 x [T*(H/2)*(W/2), 4096] (richer; ablation, not used here)
  - total_pixels caps frame count (fps>2 currently no-op) -> temporal res ~1s for now

Usage (server):
    python -m src.boundary.extract_features \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_handtask/train_multiseg_train.json \
        --out /tmp/feat_train.pt
"""
import argparse, json, os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


def as_segs(sol):
    if isinstance(sol, str):
        sol = json.loads(sol)
    return [(x[0], float(x[1]), float(x[2])) for x in sol]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = json.load(open(a.data))
    if a.limit:
        rows = rows[:a.limit]
    cache = []
    for r in rows:
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": r["video"], "fps": a.fps,
             "total_pixels": a.total_pixels},
            {"type": "text", "text": "x"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        fv = vkw.get("fps"); fv = fv[0] if isinstance(fv, list) else fv
        nf = vids[0].shape[0] if hasattr(vids[0], "shape") else len(vids[0])
        vmeta = [{"fps": float(fv), "total_num_frames": int(nf),
                  "duration": float(nf) / float(fv)}]
        inp = proc(text=[text], images=imgs, videos=vids, video_metadata=vmeta,
                   return_tensors="pt").to("cuda")
        grid = inp["video_grid_thw"]
        T, H, W = grid[0].tolist()
        with torch.no_grad():
            out = model.model.visual(inp["pixel_values_videos"].to(model.dtype),
                                     grid_thw=grid)
        lhs = out.last_hidden_state                 # [T*H*W, D]
        D = lhs.shape[-1]
        feats = lhs.view(T, H * W, D).float().mean(1).cpu()   # [T, D]
        dur = float(r.get("duration") or (nf / fv))
        times = torch.tensor([(i + 0.5) * dur / T for i in range(T)])
        cache.append({"video": r["video"],
                      "recording_id": r.get("recording_id"),
                      "feats": feats, "times": times, "duration": dur,
                      "segments": as_segs(r["solution"])})
        print(os.path.basename(r["video"]), "T", T, "D", D,
              "dur", round(dur, 1), "segs", len(cache[-1]["segments"]))
        del out, lhs, feats, inp
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(cache, a.out)
    print(f"\nwrote {len(cache)} videos -> {a.out}")


if __name__ == "__main__":
    main()
