"""Cache pooled Qwen vision features for the Handness V2 manifest.

Only the Qwen vision tower is loaded. The language model is never constructed,
and the resulting feature cache can be reused by every head experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def token_counts(grid_thw, spatial_merge_size):
    divisor = spatial_merge_size ** 2
    counts = []
    for temporal, height, width in grid_thw:
        tokens = int(temporal) * int(height) * int(width)
        if tokens % divisor:
            raise ValueError(f"grid {temporal, height, width} is not merge-aligned")
        counts.append(tokens // divisor)
    return counts


def load_vision_tower(model_path, device):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3VLVisionConfig, Qwen3VLVisionModel

    config = json.load(open(os.path.join(model_path, "config.json"), encoding="utf-8"))
    vision_config = Qwen3VLVisionConfig(**config["vision_config"])
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3VLVisionModel(vision_config)
    finally:
        torch.set_default_dtype(original_dtype)
    expected = set(model.state_dict())
    weights_path = os.path.join(model_path, "model.safetensors")
    state = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.visual."):
                state[key.removeprefix("model.visual.")] = handle.get_tensor(key)
    missing = expected - set(state)
    extra = set(state) - expected
    if missing or extra:
        raise RuntimeError(f"vision checkpoint mismatch: missing={len(missing)} extra={len(extra)}")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=torch.bfloat16).eval(), vision_config


def feature_kind(image_field):
    return f"qwen_pooler_mean_max_l2:{image_field}"


def save_cache(path, model_path, rows, features, image_field="crop_path"):
    import torch

    payload = {
        "model": model_path,
        "feature_kind": feature_kind(image_field),
        "item_ids": [row["item_id"] for row in rows],
        "features": torch.stack(features).to(torch.float16),
    }
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="/workspace/tr1/ckpts/Qwen3-VL-Embedding-2B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_pixels", type=int, default=256 * 256)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-field", choices=("crop_path", "full_path"),
                        default="crop_path")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    if not 0 <= args.shard < args.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")
    all_rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    rows = all_rows[args.shard::args.nshards]
    if args.limit:
        rows = rows[:args.limit]

    done_rows, done_features = [], []
    if os.path.exists(args.out):
        old = torch.load(args.out, map_location="cpu", weights_only=False)
        expected_kind = feature_kind(args.image_field)
        if old.get("feature_kind") not in {
                expected_kind,
                "qwen_pooler_mean_max_l2" if args.image_field == "crop_path" else ""}:
            raise ValueError(
                f"cache feature kind {old.get('feature_kind')!r} != {expected_kind!r}")
        old_map = dict(zip(old["item_ids"], old["features"]))
        for row in rows:
            if row["item_id"] in old_map:
                done_rows.append(row)
                done_features.append(old_map[row["item_id"]])
        print(f"resume {len(done_rows)}/{len(rows)}", flush=True)
    else:
        old_map = {}

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tower, vision_config = load_vision_tower(args.model, args.device)
    patch_size = int(processor.image_processor.patch_size)
    merge = int(vision_config.spatial_merge_size)

    pending = [row for row in rows if row["item_id"] not in old_map]
    for start in range(0, len(pending), args.batch):
        batch_rows = pending[start:start + args.batch]
        images = [Image.open(row[args.image_field]).convert("RGB")
                  for row in batch_rows]
        content = [{"type": "image", "image": image,
                    "min_pixels": args.max_pixels, "max_pixels": args.max_pixels}
                   for image in images]
        content.append({"type": "text", "text": "x"})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages, image_patch_size=patch_size)
        inputs = processor(text=[text], images=image_inputs,
                           return_tensors="pt").to(args.device)
        grid = inputs["image_grid_thw"]
        with torch.no_grad():
            output = tower(inputs["pixel_values"].to(torch.bfloat16), grid_thw=grid)
        counts = token_counts(grid.tolist(), merge)
        if sum(counts) != len(output.pooler_output):
            raise RuntimeError(
                f"pooled token mismatch: {sum(counts)} != {len(output.pooler_output)}")
        offset = 0
        for row, count in zip(batch_rows, counts):
            tokens = output.pooler_output[offset:offset + count].float()
            feature = torch.cat((tokens.mean(0), tokens.max(0).values))
            done_rows.append(row)
            done_features.append(F.normalize(feature, dim=0).cpu())
            offset += count
        batch_number = start // args.batch + 1
        if batch_number % args.checkpoint_every == 0:
            save_cache(args.out, args.model, done_rows, done_features,
                       args.image_field)
        print(f"{len(done_rows)}/{len(rows)}", flush=True)
        del output, inputs
        torch.cuda.empty_cache()

    order = {row["item_id"]: index for index, row in enumerate(rows)}
    paired = sorted(zip(done_rows, done_features), key=lambda pair: order[pair[0]["item_id"]])
    save_cache(args.out, args.model,
               [row for row, _ in paired], [feature for _, feature in paired],
               args.image_field)
    print(f"wrote {len(paired)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
