"""A floor on own-hand over-claiming that needs no labels at all.

THE WEARER HAS TWO HANDS. So a frame carrying three own-hand boxes that do
not overlap each other contains at least one box that is not the wearer's
hand, whatever anyone thinks about any individual box. No human judgement
enters, and -- this is the point -- the system's own ownership verdict is not
used to decide the question, only to define the population it is wrong about.
Every other cut at this has been circular: a colleague's hand already
labelled `own` is invisible to any test that filters on `own`.

DISJOINT, BECAUSE TWO BOXES ON ONE HAND ARE NOT TWO HANDS. Duplicate
detections on a single hand would otherwise be counted as a third hand and
the whole number would be detector noise. Boxes are taken greedily largest
first and a box overlapping one already taken is dropped.

IT IS A FLOOR AND NOT A RATE. Two own boxes where one is a stranger's is the
common case and is invisible here. What this bounds from below is how often
the delivered stream claims more hands than the wearer owns.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def disjoint(boxes, thr):
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in boxes:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--min_run", type=int, default=5)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    N = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            N[f[0]] = (int(f[2]), int(f[3]))

    dump = []
    for d in a.arm:
        tot = three = four = 0
        per = {}
        runs_all = []
        for rec, (s, n) in sorted(N.items()):
            p = os.path.join(d, rec + ".csv")
            if not os.path.exists(p):
                continue
            own = collections.defaultdict(list)
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if r["own"] == "1":
                    own[int(r["frame"])].append(
                        [float(r[c]) for c in ("x0", "y0", "x1", "y1")])
            bad = []
            t = t3 = 0
            for f in range(s, s + n):
                b = own.get(f)
                if not b:
                    continue
                t += 1
                k = len(disjoint(b, a.iou))
                if k >= 3:
                    t3 += 1
                    bad.append((f, k))
                    if k >= 4:
                        four += 1
            per[rec] = (t, t3)
            tot += t
            three += t3
            # consecutive stretches
            run = []
            for f, k in bad:
                if run and f == run[-1] + 1:
                    run.append(f)
                else:
                    if len(run) >= a.min_run:
                        runs_all.append((rec, run[0], run[-1]))
                    run = [f]
            if len(run) >= a.min_run:
                runs_all.append((rec, run[0], run[-1]))
        print("== %s" % os.path.basename(d))
        print("   %-16s %8s %8s %7s" % ("rec", "有手帧", ">=3只", "占比"))
        for rec in sorted(per):
            t, t3 = per[rec]
            print("   %-16s %8d %8d %6.1f%%" % (rec, t, t3, 100.0 * t3 / t if t else 0))
        print("   %-16s %8d %8d %6.1f%%   其中 >=4 只 %d 帧" % (
            "合计", tot, three, 100.0 * three / tot if tot else 0, four))
        print("   连续 >=%d 帧的段 %d 条：" % (a.min_run, len(runs_all)))
        for rec, lo, hi in runs_all:
            print("     %-16s f%d..f%d  (%d 帧)" % (rec, lo, hi, hi - lo + 1))
            dump.append((os.path.basename(d), rec, lo, hi))
        print()
    if a.dump and dump:
        with open(a.dump, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "rec", "lo", "hi"])
            w.writerows(dump)
        print("-> %s" % a.dump)


if __name__ == "__main__":
    main()
