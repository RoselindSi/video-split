"""Descriptive report: the GPT-6 second-teacher students beside S_wide on a set of zone batches.

Read at the post-processing the ablation chose for S_wide (geom_w 0.25, no
cap) and raw. This is not the replacement decision -- that is made on batches
7-9 by the rule in `distil_train.py` -- it is what the arms look like on 4-6.
Also prints how often Q1 agrees with GPT-6 on GPT-6's hands.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

from src.semhand.distil_ablate import load, name, show

ARMS = ("S_wide", "S_wide_g6", "S_wide_g6and")


def agreement(root):
    q1 = {}
    for f in glob.glob(os.path.join(root, "qwen", "Q1_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            q1[d["id"]] = int(d["p"] >= 0.5)
    c = collections.Counter()
    for p in glob.glob(os.path.join(root, "fresh", "*", "index.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            k = f"{r['rec']}|{r['frame']}|{r['tid']}"
            if k in q1:
                c[(r["gpt6"], "owner" if q1[k] else "other")] += 1
    n = sum(c.values())
    agree = c[("owner", "owner")] + c[("other", "other")]
    print(f"Q1 对 GPT-6 的手：{n} 只，一致 {agree}（{agree / n:.1%}）；"
          f"GPT-6 owner→Q1 other {c[('owner', 'other')]}，GPT-6 other→Q1 owner {c[('other', 'owner')]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", action="append", required=True)
    ap.add_argument("--g6_root", default="/workspace/g6teach")
    a = ap.parse_args()
    agreement(a.g6_root)
    tab, _, _ = load(a.batch, ARMS)
    keep = {k: v for k, v in tab.items()
            if k[0] == "V1 deployed" or (k[0] in ARMS and k[1] in ("raw", (0.25, None), (0.5, 2)))}
    show(keep, "第四+五+六批（描述性）")


if __name__ == "__main__":
    main()
