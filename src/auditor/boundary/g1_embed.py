"""Frozen segment embeddings from Qwen3-VL's own vision tower.

WHY THIS ENCODER AND NOT A STANDALONE CLIP. G1 asks whether the interaction
graph carries predictive information that a visual likelihood does not
already have. The visual likelihood should therefore come from the same
perception the rest of the study uses; borrowing a different encoder would
make a negative result ambiguous between "the graph adds nothing" and "this
encoder happens to be strong here".

THE LANGUAGE MODEL IS NEVER LOADED. Only the vision tower runs, so nothing in
G1 depends on prompt length, instruction following, dictionary formatting or
any of the effects E2 turned out to be measuring. That is the entire point of
the probe: if the graph carries signal, it should show up without a single
token of prompt.

ONE VECTOR PER SEGMENT, mean-pooled over its sampled frames. Segments run
from 0.3 s to 53 s, so a fixed frame budget would sample a 53 s segment far
more sparsely than a 3 s one; frames are taken at a fixed rate and capped, and
the count is recorded so a later analysis can check whether similarity tracks
it.
"""
from __future__ import annotations

import argparse
import json
import os

FPS = 1.0
MAX_FRAMES = 16
MODEL = "/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="E1a 的 sample_frozen.json，含 population")
    ap.add_argument("--clips", required=True, help="E1a 切好的逐段片段")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    pop = json.load(open(a.sample, encoding="utf-8"))["population"]
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    tower = model.model.visual
    print("视觉塔已加载；语言模型不参与计算", flush=True)

    import decord  # noqa: F401  仅用于确认可用
    from decord import VideoReader

    out = {}
    for i, r in enumerate(pop):
        p = os.path.join(a.clips, r["segment_id"].replace("#", "_s") + ".mp4")
        if not os.path.exists(p):
            continue
        vr = VideoReader(p)
        native = vr.get_avg_fps() or 30.0
        step = max(1, int(round(native / a.fps)))
        idx = list(range(0, len(vr), step))[:a.max_frames] or [0]
        frames = vr.get_batch(idx).asnumpy()
        inputs = proc.image_processor(images=list(frames), return_tensors="pt")
        with torch.no_grad():
            feats = tower(inputs["pixel_values"].to("cuda:0", torch.bfloat16),
                          grid_thw=inputs["image_grid_thw"].to("cuda:0"))
            # The tower returns BaseModelOutputWithDeepstackFeatures, not a
            # bare tensor; take the merged patch states and pool over every
            # patch of every frame.
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state
            elif isinstance(feats, (tuple, list)):
                feats = feats[0]
            v = feats.float().reshape(-1, feats.shape[-1]).mean(0)
            v = v / v.norm().clamp_min(1e-6)
        out[r["segment_id"]] = {"vec": v.cpu().numpy().tolist(),
                                "n_frames": len(idx),
                                "duration_s": r["duration_s"]}
        if i % 40 == 0:
            print("  %d/%d" % (i, len(pop)), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        ids=np.array(sorted(out)),
        vecs=np.stack([out[k]["vec"] for k in sorted(out)]).astype("float32"),
        n_frames=np.array([out[k]["n_frames"] for k in sorted(out)]),
    )
    print("-> %s  %d 段，维度 %d"
          % (a.out, len(out), len(next(iter(out.values()))["vec"])))


if __name__ == "__main__":
    main()
