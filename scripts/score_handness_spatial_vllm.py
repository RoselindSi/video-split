#!/usr/bin/env python3
"""Ask a local Qwen teacher for hand presence and a crop-local hand box."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os

from scripts.score_handness_vllm import data_url, post_json


QUESTION = (
    "The first image gives scene context. The second image is the square "
    "candidate crop. Decide whether any visible pixels in the second image "
    "belong to a human hand or a gloved hand. An arm, elbow, knee, leg, cloth, "
    "tool, food, or machine part without visible hand pixels is not a hand. "
    "If hand pixels are visible, give one tight bounding box around all visible "
    "hand pixels in the SECOND image only. Coordinates are integers from 0 to "
    "1000 as [left, top, right, bottom]. Exclude arm and objects where possible. "
    "Reply with JSON only, exactly one of: "
    '{"hand":true,"bbox":[left,top,right,bottom]} or '
    '{"hand":false,"bbox":null}.')


def request_payload(full_path, crop_path, model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(full_path)}},
            {"type": "image_url", "image_url": {"url": data_url(crop_path)}},
            {"type": "text", "text": QUESTION},
        ]}],
        "max_tokens": 120,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_spatial_answer(content):
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict) or type(value.get("hand")) is not bool:
        raise ValueError("hand must be a JSON boolean")
    bbox = value.get("bbox")
    if not value["hand"]:
        if bbox is not None:
            raise ValueError("bbox must be null when hand is false")
        return False, None
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(type(number) not in (int, float) for number in bbox)):
        raise ValueError("bbox must contain four numbers")
    bbox = [int(round(number)) for number in bbox]
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError(f"invalid normalized bbox: {bbox}")
    return True, bbox


def score_one(row, url, model):
    response = post_json(
        url, request_payload(row["full_path"], row["crop_path"], model))
    content = response["choices"][0]["message"]["content"]
    result = {key: row.get(key, "") for key in (
        "item_id", "rec", "frame", "canonical_tid", "human_label",
        "full_path", "crop_path")}
    result["raw_answer"] = content
    try:
        hand, bbox = parse_spatial_answer(content)
        result.update({"parse_ok": True, "teacher_hand": hand,
                       "teacher_bbox": bbox, "parse_error": ""})
    except (ValueError, json.JSONDecodeError) as error:
        result.update({"parse_ok": False, "teacher_hand": None,
                       "teacher_bbox": None, "parse_error": str(error)})
    return result


def score(index_path, output, endpoint, model, limit=0, workers=1):
    rows = list(csv.DictReader(open(index_path, encoding="utf-8-sig")))
    if limit:
        rows = rows[:limit]
    done = set()
    if os.path.exists(output):
        done = {json.loads(line)["item_id"] for line in open(output)
                if line.strip()}
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    pending = [row for row in rows if row["item_id"] not in done]
    with open(output, "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, workers)) as executor:
            results = executor.map(
                lambda row: score_one(row, url, model), pending)
            for index, result in enumerate(results, 1):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if index == 1 or index % 25 == 0:
                    print(f"{index}/{len(pending)} parse_ok={result['parse_ok']}",
                          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18998")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    score(args.index, args.out, args.endpoint, args.model, args.limit,
          args.workers)


if __name__ == "__main__":
    main()
