#!/usr/bin/env python3
"""Concatenate crop and context Qwen feature caches by item ID."""
from __future__ import annotations

import argparse
import os


def load_set(paths):
    import torch

    features = {}
    kinds = []
    models = []
    order = []
    for path in paths:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        kinds.append(cache.get("feature_kind", "unknown"))
        models.append(cache.get("model", ""))
        for item_id, feature in zip(cache["item_ids"], cache["features"]):
            if item_id in features:
                raise ValueError(f"duplicate feature for {item_id}")
            features[item_id] = feature.float()
            order.append(item_id)
    return features, order, kinds, models


def combine(left_paths, right_paths, output):
    import torch

    output = os.fspath(output)
    left, order, left_kinds, left_models = load_set(left_paths)
    right, _, right_kinds, right_models = load_set(right_paths)
    if set(left) != set(right):
        missing_left = len(set(right) - set(left))
        missing_right = len(set(left) - set(right))
        raise ValueError(
            f"cache item mismatch: missing_left={missing_left} missing_right={missing_right}")
    features = [torch.cat((left[item_id], right[item_id])) for item_id in order]
    payload = {
        "model": "+".join(sorted(set(left_models + right_models))),
        "feature_kind": "concat(" + ";".join(left_kinds + right_kinds) + ")",
        "item_ids": order,
        "features": torch.stack(features).to(torch.float16),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    tmp = output + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, output)
    return len(order), features[0].numel() if features else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", action="append", required=True)
    parser.add_argument("--right", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    count, width = combine(args.left, args.right, args.out)
    print(f"{count} items x {width} features -> {args.out}")


if __name__ == "__main__":
    main()
