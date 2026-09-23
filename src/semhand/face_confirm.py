"""Tighten the interval on the one region of small boxes the model is right about.

WHAT THE LAST SHEET FOUND BY ACCIDENT. Asked about small boxes -- the ones the
size cap never reaches -- face-ness looked unusable: its not-a-face verdict was
right 74 times in 100, and a gate there would have dropped a quarter of the
real faces. That number was an aggregate over every score it gave. Split by
score it is a different picture: all nine errors sit at p >= 0.095, and below
0.05 it is 20 for 20.

That is the opposite of hand-ness, whose most confident rejection on small
boxes (0.0005) was a real hand and where no threshold recovered anything. The
two questions do not fail the same way, which is worth knowing on its own.

WHY A SECOND SHEET RATHER THAN ACTING ON 20 FOR 20. The Wilson lower bound on
20 of 20 is 84%, and those 20 came from six recordings. Acting on it would be
uncovering a face about one time in six at worst, which is the failure this
whole line exists to prevent. Sixty more, all correct, puts the bound near
95%; sixty more with three errors says the region is not what it looks like
and costs one sheet to learn.

TWO BANDS, BECAUSE THE BOUNDARY IS ALSO UNKNOWN. Sixty at p <= 0.05, where the
decision would be taken, and fifteen between 0.05 and 0.20, where the previous
sheet put precision at 85% on eleven boxes. If the second band comes back
clean the threshold can move up and take more of the mosaic with it; if it
comes back dirty, 0.05 is a boundary and not an arbitrary line.

ALREADY-JUDGED EPISODES ARE EXCLUDED. Re-showing them would re-count evidence
already in the numerator and narrow the interval without adding anything.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import shutil

BANDS = [("p<=0.05", 0.0, 0.05, 60), ("0.05<p<=0.20", 0.05, 0.20, 15)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="/workspace/faceness.jsonl")
    ap.add_argument("--views", default="/workspace/face_views")
    ap.add_argument("--cap", type=float, default=0.18)
    ap.add_argument("--exclude", default="",
                    help="a labels CSV whose stems were judged already")
    ap.add_argument("--max_per_rec", type=int, default=8)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = {r["stem"]: r
           for r in csv.DictReader(open(os.path.join(a.views, "index.csv")))}
    done = set()
    if a.exclude and os.path.exists(a.exclude):
        done = {r["stem"] for r in csv.DictReader(open(a.exclude))}
        print("已判过 %d 段，排除" % len(done))

    rows = []
    for line in open(a.scores):
        d = json.loads(line)
        if d["stem"] in idx and d["stem"] not in done and d["w_frac"] <= a.cap:
            rows.append({**d, "n_frames": idx[d["stem"]]["n_frames"]})
    print("小框里没判过的 %d 段" % len(rows))

    rnd = random.Random(a.seed)
    pick = []
    for name, lo, hi, n in BANDS:
        sub = [r for r in rows if lo <= r["p"] <= hi] if lo == 0 else \
              [r for r in rows if lo < r["p"] <= hi]
        rnd.shuffle(sub)
        seen = collections.Counter()
        take = []
        for r in sub:
            if seen[r["rec"]] >= a.max_per_rec:
                continue
            seen[r["rec"]] += 1
            take.append(r)
            if len(take) >= n:
                break
        print("  %-14s 可选 %3d  抽 %d（%d 条录像）"
              % (name, len(sub), len(take), len(seen)))
        pick += [(name, r) for r in take]
    rnd.shuffle(pick)

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out, key = [], []
    for band, r in pick:
        f = os.path.join(a.views, r["stem"] + "_full.jpg")
        c = os.path.join(a.views, r["stem"] + "_crop.jpg")
        if not (os.path.exists(f) and os.path.exists(c)):
            continue
        shutil.copy(f, os.path.join(a.out, "context", r["stem"] + ".jpg"))
        shutil.copy(c, os.path.join(a.out, "crops", r["stem"] + ".jpg"))
        out.append({"stem": r["stem"], "frame": int(r["frame"]), "hand": 0,
                    "conf": 0.0, "w_frac": round(float(r["w_frac"]), 4),
                    "h_frac": 0.0, "cx_frac": 0.0, "cy_frac": 0.0,
                    "model": "face_confirm", "label": "", "label_mode": ""})
        key.append({"stem": r["stem"], "band": band,
                    "p_face": round(float(r["p"]), 6),
                    "w_frac": round(float(r["w_frac"]), 4),
                    "n_frames": r["n_frames"], "rec": r["rec"]})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：红框里的东西是不是一张脸或一个头（侧脸、后脑勺都算）？\n"
                 "是 -> owner   不是 -> nothand   看不清 -> unsure\n"
                 "全部是小框（<=0.18 宽），模型都说「不是脸」，分数在 "
                 "%s_key.csv 里。\n" % a.out.rstrip("/"))
    print("-> %s (%d)  分数另存 %s_key.csv" % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
