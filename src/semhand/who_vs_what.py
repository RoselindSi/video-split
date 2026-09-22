"""Does the shipped ownership classifier already know a hand from a bench.

IT WAS NEVER ASKED. The student answers whose hand this is, on a population
of detector boxes that are mostly hands, against labels that are `owner` or
`other` and never `not a hand`. So a machine part reaching it comes back with
an ownership probability like anything else, and if that probability is high
the part is delivered as the wearer's hand. Whether that actually happens is
a measurement, not a deduction -- a classifier can pick up a correlate of
hand-ness for free -- and this is the measurement.

THE COMPARISON IS AGAINST THE DETECTOR'S OWN SCORE, which was already tried
for this job and failed: 53.3% balanced accuracy on 88 boxes, with the
non-hands scoring HIGHER than the hands. If `p_owner` separates no better,
then nothing already in the pipeline answers the question and the semantic
layer is not duplicating anything.

THE LABEL HERE IS THE PROBE'S, NOT A PERSON'S, so this says whether the two
machine signals agree, not which is right. That is the whole question being
asked, and it does not need a gold set -- but it also cannot produce one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def disjoint(items, thr):
    items = sorted(items, key=lambda it: -(it["box"][2] - it["box"][0])
                   * (it["box"][3] - it["box"][1]))
    keep = []
    for it in items:
        if all(iou(it["box"], k["box"]) < thr for k in keep):
            keep.append(it)
    return keep


def auc(pos, neg):
    """Mann-Whitney, ties at half. -> P(a random pos scores above a random neg)"""
    if not pos or not neg:
        return float("nan")
    xs = sorted((v, 1) for v in pos)
    xs += sorted((v, 0) for v in neg)
    xs.sort()
    r, i, n = {}, 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1][0] == xs[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r.setdefault(k, rank)
        i = j + 1
    s = sum(r[k] for k in range(n) if xs[k][1] == 1)
    np_, nn = len(pos), len(neg)
    return (s - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * f))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_c1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--scores", default="/workspace/three_c1.jsonl")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--thr", type=float, default=0.10)
    a = ap.parse_args()

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (int(f[2]), int(f[3]))

    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in jobs:
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] != "1":
                continue
            rows[rec][int(r["frame"])].append(
                {"box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                 "p_owner": float(r["p"]), "conf": float(r["conf"])})

    P = {json.loads(l)["stem"]: float(json.loads(l)["p"])
         for l in open(a.scores)}

    hands, parts = [], []
    for rec in sorted(jobs):
        s, n = jobs[rec]
        for f in range(s, s + n):
            its = rows[rec].get(f)
            if not its:
                continue
            keep = disjoint(its, a.iou)
            if len(keep) < 3:
                continue
            for k, it in enumerate(keep):
                ph = P.get("%s_f%06d_b%d" % (rec, f, k))
                if ph is None:
                    continue
                (parts if ph <= a.thr else hands).append(it)

    print("按 hand-ness 分开的 %d 个框：真手 %d、非手 %d"
          % (len(hands) + len(parts), len(hands), len(parts)))
    print()
    print("%-26s %8s %8s %8s %8s" % ("", "中位", "p25", "p75", "最小"))
    for name, xs in (("p_owner  真手", [x["p_owner"] for x in hands]),
                     ("p_owner  非手", [x["p_owner"] for x in parts]),
                     ("检测分数 真手", [x["conf"] for x in hands]),
                     ("检测分数 非手", [x["conf"] for x in parts])):
        print("%-26s %8.3f %8.3f %8.3f %8.3f"
              % (name, q(xs, .5), q(xs, .25), q(xs, .75), min(xs)))
    print()
    print("把 p_owner 当「是不是手」的分类器：AUC %.3f"
          % auc([x["p_owner"] for x in hands], [x["p_owner"] for x in parts]))
    print("把检测分数当同一个分类器：      AUC %.3f"
          % auc([x["conf"] for x in hands], [x["conf"] for x in parts]))
    print("（0.5 = 完全无信息；<0.5 = 非手的分数反而更高）")
    print()
    # the practical question: what does the pipeline actually do with a part
    hi = sum(1 for x in parts if x["p_owner"] >= 0.9)
    print("被 hand-ness 判为非手的 %d 个框里，分类器给出 p_owner>=0.9 的有 %d 个（%.0f%%）"
          % (len(parts), hi, 100.0 * hi / max(1, len(parts))))
    best, bt = 0.0, None
    for t in [i / 100.0 for i in range(1, 100)]:
        tp = sum(1 for x in hands if x["p_owner"] >= t)
        tn = sum(1 for x in parts if x["p_owner"] < t)
        b = 0.5 * (tp / max(1, len(hands)) + tn / max(1, len(parts)))
        if b > best:
            best, bt = b, t
    print("p_owner 最好的阈值 %.2f，平衡准确率 %.1f%%（随机 50%%）" % (bt, 100 * best))


if __name__ == "__main__":
    main()
