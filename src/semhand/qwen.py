"""Qwen3.8-27B in the semantic-layer experiment: describe every hand, then score.

Runs in `/workspace/lvs/.venv`. ONE LOAD PER GPU FOR EVERYTHING. The weights
come off shared storage at about 50 MB/s, so a worker loads once and works
through its shard in order:

    describe   bank + fresh hands                -> qwen/desc_<shard>.jsonl
    Q0         fresh hands, zero-shot            -> qwen/Q0_<shard>.jsonl
    (waits for retrieval.json, which needs every shard's descriptions)
    Q2, Q2s, Q1, Q2v                             -> qwen/<arm>_<shard>.jsonl

Every output is appended per hand and resumed by id.

WHAT QWEN SEES, FOR EVERY ARM AND EVERY TEMPLATE. The clean panorama resized
to 1280x704 with one green box, then a 256x256 crop centred on that box (1.6x
its longer side). Fixed sizes on purpose: the processor does not resize, and
two hands always cost the same number of tokens, so descriptions can be
generated in batches without padding.

THE SCORE IS A PROBABILITY, NOT A PARSE. The prompt ends with the assistant
turn already opened at `{"wearer":`; P(self) is the softmax of the next-token
logits for ` true` against ` false`. Nothing is generated, nothing can fail to
parse, and the mass the two tokens hold is kept so a model that wanted to say
something else is visible.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time

from src.semhand import K, QWEN, QWEN_ARMS

FRAME_WH = (1280, 704)
CROP = 256
BOX_RGB = (0, 230, 0)
ANSWER_PREFIX = '{"wearer":'
TRUE_ID, FALSE_ID = 804, 867           # " true", " false"; checked at load

DESCRIBE = (
    "This image is from a camera worn on the head of a factory worker. One hand "
    "is outlined by a green box; the second image is a zoomed crop around that "
    "box. Describe only what that hand is doing. Do not say or guess whose hand "
    "it is.\nReply with JSON only:\n"
    '{"hand_side": "left" | "right" | "unclear", '
    '"interaction": "<what the hand is doing, a short phrase>", '
    '"object": "<object in contact, or none>", '
    '"grasp": "<grasp or hand pose, a short phrase>", '
    '"occlusion": "<what hides part of the hand, or none>", '
    '"ambiguous_regions": "<anything that makes the hand hard to see, or none>"}')

INTRO = ("These images come from a camera worn on the head of a factory worker "
         "(the camera wearer). In each case one hand is outlined by a green box, "
         "and the image after it is a zoomed crop around that box. Other people's "
         "hands may also be visible. A person has at most two hands.")
QUESTION = ("Is the outlined hand one of the camera wearer's own hands? Reply with "
            'JSON only: {"wearer": true} or {"wearer": false}')

LEAK_RE = re.compile(r"\b(wearer|my|mine|own|another person|other person|someone"
                     r"|colleague|co-?worker|operator|second person|first[- ]person|"
                     r"person's|user)\b", re.I)


# ------------------------------------------------------------------ items

def load_items(root):
    """-> {id: item}. Fresh ids are `rec|frame|tid`; bank ids are package stems."""
    items = {}
    for p in sorted(glob.glob(os.path.join(root, "fresh", "*", "index.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["status"] == "ok":
                items[f"{r['rec']}|{r['frame']}|{r['tid']}"] = {
                    "kind": "fresh", "image": r["image"], "databag": r["databag"],
                    "box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")]}
    for p in sorted(glob.glob(os.path.join(root, "pkg", "index_*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["status"] == "ok":
                items[r["stem"]] = {
                    "kind": "bank", "image": r["image"], "databag": r["databag"],
                    "y": int(r["y"]), "box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")]}
    return items


def views(item):
    """-> (frame with the box, crop) as PIL images of fixed size."""
    from PIL import Image, ImageDraw
    img = Image.open(item["image"]).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = item["box"]
    sx, sy = FRAME_WH[0] / W, FRAME_WH[1] / H
    frame = img.resize(FRAME_WH, Image.BICUBIC)
    ImageDraw.Draw(frame).rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                                    outline=BOX_RGB, width=4)
    side = max(x1 - x0, y1 - y0) * 1.6
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    crop = img.crop((int(cx - side / 2), int(cy - side / 2),
                     int(cx + side / 2), int(cy + side / 2))).resize((CROP, CROP), Image.BICUBIC)
    return frame, crop


def image_parts(item):
    frame, crop = views(item)
    n_f, n_c = FRAME_WH[0] * FRAME_WH[1], CROP * CROP
    return [{"type": "image", "image": frame, "min_pixels": n_f, "max_pixels": n_f},
            {"type": "image", "image": crop, "min_pixels": n_c, "max_pixels": n_c}]


def parse_desc(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        return None
    side = str(d.get("hand_side", "")).lower()
    d["hand_side"] = "left" if "left" in side else "right" if "right" in side else "unclear"
    return d


def sem_text(rec):
    """The description as one line of text; the raw reply if it did not parse."""
    d = rec.get("parsed")
    if not d:
        return re.sub(r"\s+", " ", rec.get("raw", "")).strip()[:400]
    return (f"{d.get('hand_side', 'unclear')} hand; interaction: {d.get('interaction', '')}; "
            f"object: {d.get('object', '')}; grasp: {d.get('grasp', '')}; "
            f"occlusion: {d.get('occlusion', '')}; ambiguous: {d.get('ambiguous_regions', '')}")


def read_jsonl(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d
    return out


# ------------------------------------------------------------------ prompts

def content_for(arm, qid, items, descs, retrieval):
    q = items[qid]
    parts = [{"type": "text", "text": INTRO}]
    if arm in ("Q2", "Q2s", "Q2v"):
        mode = {"Q2": "semantic", "Q2s": "shuffled", "Q2v": "visual"}[arm]
        tids = retrieval[qid][mode][:K]
        parts.append({"type": "text", "text": "Labelled examples from other recordings:"})
        for j, t in enumerate(tids):
            parts.append({"type": "text", "text": f"Example {j + 1}:"})
            parts.extend(image_parts(items[t]))
            ans = "true" if items[t]["y"] == 1 else "false"
            parts.append({"type": "text", "text": f"Description: {sem_text(descs[t])}\n"
                                                  f'Answer: {{"wearer": {ans}}}'})
        parts.append({"type": "text", "text": "Now the case to decide:"})
    parts.extend(image_parts(q))
    if arm != "Q0":
        parts.append({"type": "text", "text": f"Description: {sem_text(descs[qid])}"})
    parts.append({"type": "text", "text": QUESTION})
    return [{"role": "user", "content": parts}]


# ------------------------------------------------------------------ model

class Qwen:
    def __init__(self, path, tiny=False):
        import torch
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        t0 = time.time()
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        if tiny:
            # CODE PATH ONLY. Same architecture, processor and template, four
            # text layers and two vision blocks, random weights: it exercises
            # every call this file makes without the fifty-minute load, and
            # its answers mean nothing.
            cfg = AutoConfig.from_pretrained(path)
            cfg.text_config.num_hidden_layers = 4
            cfg.text_config.layer_types = cfg.text_config.layer_types[:4]
            cfg.vision_config.depth = 2
            cfg.vision_config.deepstack_visual_indexes = [0]
            with torch.device("cuda"):
                self.model = AutoModelForImageTextToText.from_config(
                    cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                path, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
                attn_implementation="sdpa").eval()
        self.ps = int(self.proc.image_processor.patch_size)
        print(f"模型加载 {time.time() - t0:.0f}s", flush=True)
        # THE ANSWER TOKENS, CHECKED IN CONTEXT. ` true` is one token on its
        # own; what matters is that it is still the next token after the
        # prefix once the chat template sits in front of it.
        msgs = [{"role": "user", "content": [{"type": "text", "text": QUESTION}]}]
        head = self.template(msgs)
        tok = self.proc.tokenizer
        a = tok.encode(head + ANSWER_PREFIX, add_special_tokens=False)
        for word, tid in ((" true", TRUE_ID), (" false", FALSE_ID)):
            b = tok.encode(head + ANSWER_PREFIX + word + "}", add_special_tokens=False)
            assert b[:len(a)] == a and b[len(a)] == tid, (word, b[len(a) - 2:])

    def template(self, msgs):
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def inputs(self, convs, suffix=""):
        from qwen_vl_utils import process_vision_info
        texts = [self.template(c) + suffix for c in convs]
        images, _ = process_vision_info(convs, image_patch_size=self.ps)
        return self.proc(text=texts, images=images, padding=True,
                         return_tensors="pt").to(self.model.device)

    def describe(self, convs, max_new_tokens=160):
        x = self.inputs(convs)
        lens = x["attention_mask"].sum(1)
        if len(convs) > 1 and int(lens.min()) != int(lens.max()):
            return [r for c in convs for r in self.describe([c], max_new_tokens)]
        with self.torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)

    def score(self, conv):
        x = self.inputs([conv], suffix=ANSWER_PREFIX)
        with self.torch.no_grad():
            try:
                logits = self.model(**x, logits_to_keep=1).logits[0, -1].float()
            except TypeError:
                logits = self.model(**x).logits[0, -1].float()
        pr = self.torch.softmax(logits, 0)
        two = self.torch.softmax(logits[[TRUE_ID, FALSE_ID]], 0)
        return {"p": float(two[0]), "margin": float(logits[TRUE_ID] - logits[FALSE_ID]),
                "mass": float(pr[TRUE_ID] + pr[FALSE_ID]), "tokens": int(x["input_ids"].shape[1])}


# ------------------------------------------------------------------ worker

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/semhand")
    ap.add_argument("--out", default=None, help="default <root>/qwen")
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--arms", default=",".join(QWEN_ARMS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N bank + N fresh")
    ap.add_argument("--retrieval", default=None, help="default <root>/retrieval.json")
    ap.add_argument("--tiny", action="store_true", help="random 4-layer model, code path only")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "qwen")
    os.makedirs(out, exist_ok=True)

    items = load_items(a.root)
    ids = sorted(items)
    if a.limit:
        ids = ([i for i in ids if items[i]["kind"] == "bank"][:a.limit]
               + [i for i in ids if items[i]["kind"] == "fresh"][:a.limit])
    mine = ids[a.shard::a.nshards]
    print(f"条目 {len(items)}，本分片 {len(mine)}", flush=True)
    q = Qwen(a.model, tiny=a.tiny)

    # ---- describe ------------------------------------------------------
    path = os.path.join(out, f"desc_{a.shard}.jsonl")
    done = set(read_jsonl(path))
    todo = [i for i in mine if i not in done]
    t0 = time.time()
    with open(path, "a") as fh:
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]
            convs = [[{"role": "user", "content": image_parts(items[i])
                       + [{"type": "text", "text": DESCRIBE}]}] for i in chunk]
            t = time.time()
            for i, raw in zip(chunk, q.describe(convs)):
                d = parse_desc(raw)
                fh.write(json.dumps({"id": i, "raw": raw, "parsed": d, "ok": d is not None,
                                     "leak": bool(LEAK_RE.search(raw))}) + "\n")
            fh.flush()
            if s % (a.batch * 10) == 0:
                print(f"  describe {s + len(chunk)}/{len(todo)}  {time.time() - t:.1f}s/批  "
                      f"{raw[:100]!r}", flush=True)
    print(f"describe 完成 {len(todo)} 条，{time.time() - t0:.0f}s", flush=True)
    open(os.path.join(out, f"desc_done_{a.shard}"), "w").close()

    # ---- arms ----------------------------------------------------------
    fresh = [i for i in mine if items[i]["kind"] == "fresh"]
    descs, retrieval = None, None
    for arm in a.arms.split(","):
        if arm != "Q0":
            rp = a.retrieval or os.path.join(a.root, "retrieval.json")
            while not os.path.exists(rp):
                time.sleep(60)
            if retrieval is None:
                retrieval = json.load(open(rp))
                descs = read_jsonl(os.path.join(out, "desc_*.jsonl"))
        path = os.path.join(out, f"{arm}_{a.shard}.jsonl")
        done = set(read_jsonl(path))
        todo = [i for i in fresh if i not in done]
        if arm != "Q0":
            missing = [i for i in todo if i not in retrieval or i not in descs]
            if missing:
                print(f"  {arm}: {len(missing)} 条没有描述或检索，不打分", flush=True)
            todo = [i for i in todo if i in retrieval and i in descs]
        t0 = time.time()
        with open(path, "a") as fh:
            for n, i in enumerate(todo):
                t = time.time()
                r = q.score(content_for(arm, i, items, descs, retrieval))
                fh.write(json.dumps(dict(r, id=i, sec=round(time.time() - t, 2))) + "\n")
                fh.flush()
                if n % 50 == 0:
                    print(f"  {arm} {n + 1}/{len(todo)}  {time.time() - t:.1f}s  "
                          f"p {r['p']:.3f} mass {r['mass']:.3f} tokens {r['tokens']}", flush=True)
        print(f"{arm} 完成 {len(todo)} 条，{time.time() - t0:.0f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
