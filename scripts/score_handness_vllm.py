#!/usr/bin/env python3
"""Score Handness views through an already-running OpenAI-compatible vLLM."""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import mimetypes
import os
import urllib.error
import urllib.request

from src.semhand.handness import PREFIX, QUESTION


TRUE_ID = 804
FALSE_ID = 867


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    encoded = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_payload(full_path, crop_path, model):
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(full_path)}},
                {"type": "image_url", "image_url": {"url": data_url(crop_path)}},
                {"type": "text", "text": QUESTION},
            ]},
            {"role": "assistant", "content": PREFIX},
        ],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "logit_bias": {str(TRUE_ID): 100, str(FALSE_ID): 100},
    }


def hand_probability(choice):
    entries = choice["logprobs"]["content"][0]["top_logprobs"]
    exact = {entry["token"]: float(entry["logprob"]) for entry in entries}
    true_logprob = exact.get(" true")
    false_logprob = exact.get(" false")
    if true_logprob is None or false_logprob is None:
        stripped = collections.defaultdict(list)
        for entry in entries:
            stripped[entry["token"].strip()].append(float(entry["logprob"]))
        if not stripped["true"] or not stripped["false"]:
            raise ValueError(f"true/false missing from top logprobs: {entries}")
        true_logprob = max(stripped["true"])
        false_logprob = max(stripped["false"])
    peak = max(true_logprob, false_logprob)
    true = math.exp(true_logprob - peak)
    false = math.exp(false_logprob - peak)
    return true / (true + false), true_logprob - false_logprob


def post_json(url, payload, timeout=300):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {error.code}: {body}") from error


def score(index_path, views_dir, output, endpoint, model, limit=0):
    rows = list(csv.DictReader(open(index_path, encoding="utf-8-sig")))
    if limit:
        rows = rows[:limit]
    done = set()
    if os.path.exists(output):
        done = {json.loads(line)["stem"] for line in open(output, encoding="utf-8")
                if line.strip()}
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    with open(output, "a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            stem = row.get("stem") or row.get("item_id")
            if stem in done:
                continue
            full = row.get("full_path") or os.path.join(views_dir, stem + "_full.jpg")
            crop = row.get("crop_path") or os.path.join(views_dir, stem + "_crop.jpg")
            response = post_json(url, request_payload(full, crop, model))
            choice = response["choices"][0]
            probability, margin = hand_probability(choice)
            result = {key: row.get(key, "") for key in
                      ("stem", "item_id", "rec", "frame", "conf", "tid",
                       "canonical_tid", "n_claimed")}
            result["stem"] = stem
            result.update({"p": probability, "margin": margin,
                           "token": choice["logprobs"]["content"][0]["token"],
                           "top_logprobs": choice["logprobs"]["content"][0]["top_logprobs"]})
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0:
                print(f"{index}/{len(rows)} p={probability:.4f}", flush=True)
    return len(rows), len(done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18998")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    count, done = score(args.index, args.views, args.out, args.endpoint,
                        args.model, args.limit)
    print(f"scored {count - done}/{count}; resumed {done}")


if __name__ == "__main__":
    main()
