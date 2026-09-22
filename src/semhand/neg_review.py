"""Check the automatic negatives before they become a class.

WHY THIS STEP EXISTS AND IS NOT OPTIONAL. The negatives are the probe's own
output. Training on them without looking is exactly what makes a model
inherit its teacher's errors, and this project has a fresh example: 2,508
boxes another line labelled `not_hand` turned out to be 97% real hands, and
nothing caught it until 65 of them went to a person. The probe earned 40 of
40 on OUR population, which is the reason to use it and not a reason to skip
the check -- that was 40 boxes from six recordings, and this harvest is a
hundred recordings the pipeline has never run on.

STRATIFIED BY SIZE, BECAUSE SIZE IS THE KNOWN FAULT LINE. The probe is right
at 246 px and wrong at 25. The harvest floor is 150 px, so every negative
here should be inside the good regime -- but "should be" is a prediction and
the bands are what tests it. A band where the error rate climbs says where to
move the floor.

AND BY RECORDING, so one scene with an unusual object cannot supply half the
class and half the check at once.

CONTROLS ARE INCLUDED AND THEY ARE NOT PADDING. Boxes the probe accepted go
in the same shuffled sheet. Without them the reader sees only candidate
non-hands and drifts toward saying yes; with them, agreement on the easy
cells is visible and a disagreement there is a signal about the sheet.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import shutil


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * f))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="/workspace/neg_probe.jsonl")
    ap.add_argument("--views", default="/workspace/neg_views")
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--n_neg", type=int, default=100)
    ap.add_argument("--n_pos", type=int, default=40)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # DEDUPED BY STEM, because the shards are also concatenated into the
    # base name and the glob matches both -- which counted every box twice
    # and would have let one box appear twice in a sample of 140.
    files = sorted(glob.glob(a.scores + "*"))
    rows, seen_stem = [], set()
    for p in files:
        for line in open(p):
            d = json.loads(line)
            if d["stem"] in seen_stem:
                continue
            seen_stem.add(d["stem"])
            # The index is a CSV, so everything came through as text. Casting
            # here rather than at the source keeps the scored files a faithful
            # copy of what was asked, and a silent string comparison in a size
            # band would put boxes in the wrong stratum without raising.
            d["w_px"] = float(d["w_px"])
            d["conf"] = float(d["conf"])
            d["frame"] = int(d["frame"])
            rows.append(d)
    print("打了分的框 %d 个（%d 个分片文件）" % (len(rows), len(files)))
    if not rows:
        return
    neg = [r for r in rows if r["p"] <= a.thr]
    pos = [r for r in rows if r["p"] > 0.60]
    print("  判非手(p<=%.2f) %d（%.1f%%）  判是手(p>0.60) %d  中间 %d"
          % (a.thr, len(neg), 100.0 * len(neg) / len(rows), len(pos),
             len(rows) - len(neg) - len(pos)))
    print("  p 中位 %.3f   非手的框宽中位 %.0f px   是手的 %.0f px"
          % (q([r["p"] for r in rows], .5),
             q([r["w_px"] for r in neg], .5), q([r["w_px"] for r in pos], .5)))

    print("\n按框宽分带：")
    print("  %-14s %8s %8s %8s" % ("宽(px)", "框数", "判非手", "比例"))
    bands = [(150, 200), (200, 260), (260, 340), (340, 10000)]
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r["w_px"] < hi]
        if not sub:
            continue
        k = sum(1 for r in sub if r["p"] <= a.thr)
        print("  %-14s %8d %8d %7.1f%%"
              % ("%d-%d" % (lo, hi) if hi < 10000 else "%d+" % lo,
                 len(sub), k, 100.0 * k / len(sub)))

    by_rec = collections.Counter(r["rec"] for r in neg)
    print("\n非手最多的 6 条录像：")
    for rec, n in by_rec.most_common(6):
        tot = sum(1 for r in rows if r["rec"] == rec)
        print("  %-30s %3d / %3d = %.0f%%" % (rec, n, tot, 100.0 * n / tot))
    print("  非手分布在 %d 条录像上（共 %d 条）"
          % (len(by_rec), len({r["rec"] for r in rows})))

    if not a.out:
        return

    # stratified draw: size band x recording, so no scene and no band dominates
    rnd = random.Random(a.seed)
    per_band = max(1, a.n_neg // len(bands))
    pick = []
    for lo, hi in bands:
        sub = [r for r in neg if lo <= r["w_px"] < hi]
        rnd.shuffle(sub)
        seen = collections.Counter()
        take = []
        for r in sub:
            if seen[r["rec"]] >= 2:
                continue
            seen[r["rec"]] += 1
            take.append(r)
            if len(take) >= per_band:
                break
        print("  抽样 %s: %d" % ("%d-%d" % (lo, hi), len(take)))
        pick += [("neg", r) for r in take]
    ctrl = rnd.sample(pos, min(a.n_pos, len(pos)))
    pick += [("pos", r) for r in ctrl]
    rnd.shuffle(pick)

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out, key = [], []
    for g, r in pick:
        f = os.path.join(a.views, r["stem"] + "_full.jpg")
        c = os.path.join(a.views, r["stem"] + "_crop.jpg")
        if not (os.path.exists(f) and os.path.exists(c)):
            continue
        shutil.copy(f, os.path.join(a.out, "context", r["stem"] + ".jpg"))
        shutil.copy(c, os.path.join(a.out, "crops", r["stem"] + ".jpg"))
        out.append({"stem": r["stem"], "frame": r["frame"], "hand": 0,
                    "conf": r["conf"], "w_frac": 0.0, "h_frac": 0.0,
                    "cx_frac": 0.0, "cy_frac": 0.0, "model": "neg_harvest",
                    "label": "", "label_mode": ""})
        key.append({"stem": r["stem"], "group": g, "p_hand": round(r["p"], 4),
                    "w_px": r["w_px"], "rec": r["rec"], "det_conf": r["conf"]})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：绿框里的东西是不是一只手（戴手套也算）？\n"
                 "是手 -> owner   不是手 -> nothand   看不清 -> unsure\n"
                 "模型的答案在 %s_key.csv，不在这张表里。\n" % a.out.rstrip("/"))
    print("-> %s (%d)  答案另存 %s_key.csv" % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
