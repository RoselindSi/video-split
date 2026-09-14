"""Qwen3-VL-Embedding-2B vectors for every described hand, then the retrieval file.

Runs in `/workspace/lvs/.venv`, after every Qwen shard has written its
descriptions.

    text   the description line (`qwen.sem_text`)      -> semantic retrieval
    image  the 256x256 crop Qwen was shown             -> visual retrieval

ONE RETRIEVAL FILE FOR BOTH FAMILIES. For each hand -- bank hands too, since
the V1 arms train on them -- up to N_CANDIDATES bank hands, best first, under
three orderings. Candidates never come from the query's own databag, and are
restricted to Qwen's coarse class (the same hand side) when the query has a
side and enough candidates share it; `shuffled` is a seeded permutation of that
same filtered pool, so the null differs from `semantic` only in the ordering.
Consumers take the first K after removing anything their split forbids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from src.semhand import EMBEDDER, N_CANDIDATES
from src.semhand.qwen import load_items, read_jsonl, sem_text, views

TEXT_INS = "Represent what this hand is doing, for retrieving hands doing similar things."
IMAGE_INS = "Represent this image of a hand, for retrieving similar hands."


def embed(a):
    import torch
    sys.path.insert(0, os.path.join(EMBEDDER, "scripts"))
    from qwen3_vl_embedding import Qwen3VLEmbedder
    torch.set_num_threads(4)
    items = load_items(a.root)
    descs = read_jsonl(os.path.join(a.qwen, "desc_*.jsonl"))
    ids = sorted(i for i in descs if i in items)
    print(f"描述 {len(descs)}，可嵌入 {len(ids)}", flush=True)
    emb = Qwen3VLEmbedder(EMBEDDER, dtype=torch.bfloat16, attn_implementation="sdpa")
    text, image = [], []
    for s in range(0, len(ids), a.batch):
        chunk = ids[s:s + a.batch]
        text.append(emb.process([{"text": sem_text(descs[i]), "instruction": TEXT_INS}
                                 for i in chunk]).float().cpu().numpy())
        image.append(emb.process([{"image": views(items[i])[1], "instruction": IMAGE_INS}
                                  for i in chunk]).float().cpu().numpy())
        if s % (a.batch * 20) == 0:
            print(f"  {s + len(chunk)}/{len(ids)}", flush=True)
    np.savez(a.embed, ids=np.array(ids), text=np.concatenate(text), image=np.concatenate(image))
    print(f"-> {a.embed}", flush=True)


def retrieve(a):
    items = load_items(a.root)
    descs = read_jsonl(os.path.join(a.qwen, "desc_*.jsonl"))
    E = np.load(a.embed)
    at = {i: n for n, i in enumerate(E["ids"].tolist())}
    T = E["text"] / np.linalg.norm(E["text"], axis=1, keepdims=True)
    I = E["image"] / np.linalg.norm(E["image"], axis=1, keepdims=True)
    side = lambda i: (descs[i].get("parsed") or {}).get("hand_side", "unclear")
    bank = [i for i in sorted(at) if items[i]["kind"] == "bank"]
    b_at = np.array([at[i] for i in bank])
    b_bag = np.array([items[i]["databag"] for i in bank])
    b_side = np.array([side(i) for i in bank])
    b_y = np.array([items[i]["y"] for i in bank])
    out, agree = {}, {"semantic": [], "visual": [], "shuffled": []}
    for q in sorted(at):
        mask = b_bag != items[q]["databag"]
        s = side(q)
        if s in ("left", "right") and (mask & (b_side == s)).sum() >= N_CANDIDATES:
            mask &= b_side == s
        cand = np.where(mask)[0]
        rng = np.random.default_rng(int(hashlib.md5(q.encode()).hexdigest()[:8], 16))
        order = {"semantic": cand[np.argsort(-(T[b_at[cand]] @ T[at[q]]), kind="stable")],
                 "visual": cand[np.argsort(-(I[b_at[cand]] @ I[at[q]]), kind="stable")],
                 "shuffled": rng.permutation(cand)}
        out[q] = {m: [bank[j] for j in o[:N_CANDIDATES]] for m, o in order.items()}
        if items[q]["kind"] == "bank":
            for m, o in order.items():
                agree[m].append(float(np.mean(b_y[o[:2]] == items[q]["y"])))
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, a.out)
    print(f"检索 {len(out)} 条（bank {len(bank)}） -> {a.out}")
    print("bank 查询：前 2 个模板与查询同标签的比例 " + "  ".join(
        f"{m} {np.mean(v):.3f}" for m, v in agree.items() if v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/semhand")
    ap.add_argument("--qwen", default=None, help="default <root>/qwen")
    ap.add_argument("--embed", default=None, help="default <root>/embed.npz")
    ap.add_argument("--out", default=None, help="default <root>/retrieval.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip_embed", action="store_true")
    a = ap.parse_args()
    a.qwen = a.qwen or os.path.join(a.root, "qwen")
    a.embed = a.embed or os.path.join(a.root, "embed.npz")
    a.out = a.out or os.path.join(a.root, "retrieval.json")
    if not a.skip_embed:
        embed(a)
    retrieve(a)


if __name__ == "__main__":
    main()
