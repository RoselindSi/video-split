"""Ask Qwen3.8-27B which numbered boxes are the camera wearer's hands.

Runs in `/workspace/lvs/.venv` (transformers 5.11). Resumable: ids already in
the answers file are skipped, so an interrupted run continues rather than
paying the 27B load and every earlier answer again.

A FRAME WITH NO BOXES IS NOT ASKED. It is written straight to the answers as
empty -- there is nothing to choose among, and a model asked to pick from
nothing will sometimes pick anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

PROMPT = (
    "This image is from a camera worn on the head of a factory worker (the "
    "camera wearer). Each detected hand is outlined by a coloured box with a "
    "number. Other people's hands may also be visible, and a box may "
    "occasionally contain something that is not a hand. A person has at most "
    "two hands.\n"
    "Which numbered boxes are the camera wearer's own hands? Which boxes are "
    "not hands at all?\n"
    "Reply with JSON only, no explanation: "
    '{"wearer": [numbers], "not_hand": [numbers]}'
)


def parse(text, n_boxes):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    def ints(v):
        out = []
        for x in (v if isinstance(v, list) else []):
            try:
                x = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= x <= n_boxes and x not in out:
                out.append(x)
        return out
    return {"wearer": ints(d.get("wearer")), "not_hand": ints(d.get("not_hand"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pixels", type=int, default=1600 * 900)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    items = []
    for m in a.manifest:
        items.extend(json.loads(line) for line in open(m) if line.strip())
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(items)} 帧，已答 {len(done)}，本次 {len(todo)}", flush=True)
    if not todo:
        return

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa").eval()
    print(f"模型加载 {time.time() - t0:.0f}s", flush=True)

    with open(a.out, "a") as fout:
        for i, it in enumerate(todo):
            n = len(it["boxes"])
            if n == 0:
                fout.write(json.dumps({"id": it["id"], "raw": "", "parsed": {"wearer": [], "not_hand": []},
                                       "ok": True, "asked": False, "sec": 0.0}) + "\n")
                continue
            t = time.time()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": it["image"], "min_pixels": 3136,
                 "max_pixels": a.max_pixels},
                {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            images, _videos = process_vision_info(
                messages, image_patch_size=int(processor.image_processor.patch_size))
            inputs = processor(text=[text], images=images, padding=True,
                               return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)[0]
            parsed = parse(raw, n)
            fout.write(json.dumps({"id": it["id"], "raw": raw, "parsed": parsed,
                                   "ok": parsed is not None, "asked": True,
                                   "sec": round(time.time() - t, 2)}) + "\n")
            fout.flush()
            if i % 20 == 0:
                print(f"  {i + 1}/{len(todo)}  {time.time() - t:.1f}s  {raw[:80]!r}", flush=True)
    print(f"完成，用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
