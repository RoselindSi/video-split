"""How much of an arm's coverage gain rests on boxes only that arm has.

TWO DIFFERENT QUANTITIES, AND ONLY ONE NEEDS LABELS.

    exposure   frames the arm covers and base does not, where the ONLY own
               hand in the frame is a box base never had. If every such box
               were junk, the whole gain would be junk. Structural, computed
               from the runs.

    rate       what fraction of those boxes are not a hand. Needs a person.
               For C1 it is 38.9% of 90 judged; for the Rolan arms nobody has
               looked, and the 31.7% measured on Rolan's extra OTHER boxes
               cannot be borrowed -- that sample was drawn with `if own:
               continue`, so it says nothing about its own-hand boxes.

FALSE-BLUR NEEDS NO LABELS EITHER. Recomputing it with the arm-unique boxes
removed says whether the improvement depends on them, whatever they are.

AND `exposure` ONLY DISCRIMINATES BETWEEN ARMS THAT SHARE A DETECTOR. A box
is "unique" here when nothing in base overlaps it, so swapping the detector
makes every box unique by construction and the column reads 100% without
saying anything. It measures contamination risk for a threshold change and
nothing at all for a model change; for those the labels are the only route.

THE RATE APPLIES ONLY TO THE POPULATION IT WAS MEASURED ON. C1's 38.9% comes
from boxes that are unique AND under 0.25, so it may not be applied to every
unique box: 45 gain frames rest on a unique box alone, but 34 of those rest
on a unique box under 0.25, and only those 34 carry the rate.
"""
import collections
import csv
import glob
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def own(p):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["own"] == "1":
            out[int(r["frame"])].append(
                ([float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                 float(r["conf"] or 0), float(r["covered"])))
    return out


gone = set()
for r in csv.DictReader(open("/workspace/visiblepkg/hands.csv")):
    if r["label"] == "other":
        rec, rest = r["stem"].rsplit("_f", 1)
        gone.add((rec, int(rest.split("_h")[0])))

N = {}
for line in open("/workspace/cam3_jobs.txt"):
    f = line.strip().split("|")
    if len(f) >= 4 and not line.startswith("#"):
        N[f[0]] = (int(f[2]), int(f[3]))

# the not-a-hand rate among an arm's own unique own-hand boxes, where known
RATE = {"C1 cont.10": 0.389}

ARMS = [("C1 cont.10", "/workspace/cam3_c1"),
        ("Rolan", "/workspace/cam3_rolan"),
        ("C1+Rolan", "/workspace/cam3_c1r")]

print("  %-11s %7s %8s %9s %11s %11s %11s" % (
    "配置", "可见帧", "覆盖率", "新增覆盖", "只靠独有框", "false-blur", "剔掉独有框"))
for name, d in ARMS:
    vis = hav = gain = solo = rated = 0
    num = den = numx = denx = 0.0
    for rec, (s, n) in sorted(N.items()):
        pb, pa = "/workspace/cam3_base/%s.csv" % rec, os.path.join(d, rec + ".csv")
        if not (os.path.exists(pb) and os.path.exists(pa)):
            continue
        B, A = own(pb), own(pa)
        frames = [f for f in range(s, s + n) if (rec, f) not in gone]
        vis += len(frames)
        for f in frames:
            ab = A.get(f, [])
            bb = [x for x, _, _ in B.get(f, [])]
            if ab:
                hav += 1
            uniq = [t for t in ab if not any(iou(t[0], g) >= 0.3 for g in bb)]
            shared = [t for t in ab if t not in uniq]
            weak = [t for t in uniq if t[1] < 0.25]
            if ab and f not in B:
                gain += 1
                if uniq and not shared:
                    solo += 1
                    if weak:
                        rated += 1
            for x, c, cv in ab:
                ar = (x[2] - x[0]) * (x[3] - x[1])
                num += cv * ar
                den += ar
            for x, c, cv in shared:
                ar = (x[2] - x[0]) * (x[3] - x[1])
                numx += cv * ar
                denx += ar
    r = RATE.get(name)
    solo_txt = "%d/%d" % (solo, gain)
    est = ("需要标注" if r is None
           else "%d 帧带率, ~%.0f 假 -> 真增益 ~%d" % (rated, r * rated, gain - r * rated))
    print("  %-11s %7d %7.1f%% %9d %9s %10.3f%% %10.3f%%  %s" % (
        name, vis, 100.0 * hav / vis, gain, solo_txt,
        100 * num / max(1.0, den), 100 * numx / max(1.0, denx), est))
