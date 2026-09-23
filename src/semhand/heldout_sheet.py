"""The held-out sheet, and the half of the question it can answer now.

WHAT PRECISION NEEDS AND WHY IT CANNOT BE SAMPLED YET. `N` in the criteria is
class-2 precision: among boxes the STUDENT calls not-a-hand, how many are not
hands. Precision is a property of the flagged set, so the sample has to be
drawn from the flagged set -- and the student does not exist until training
ends. Drawing at random now and waiting would put about 7% of the sheet in
the flagged set, five boxes per band, which answers nothing.

WHAT IT CAN ANSWER NOW, AND IT IS NOT FILLER. How common non-hands are per
width band is unknown, and the band under 150 px is unknown in a way that
matters: every class-2 training example is 150 px or wider, so the student
meets small boxes having never been taught about them. If small non-hands are
vanishingly rare, the blind spot costs nothing and the class can simply be
disabled down there. If they are one box in five, the student will be asked
that question constantly and will answer it from nothing. Nobody knows which,
and the probe cannot be asked -- below 30 px its not-a-hand verdict was right
3 times in 55.

SO: EQUAL COUNTS PER BAND, NOT PROPORTIONAL. A proportional draw would spend
half the sheet on the two large bands, which the probe already measured at
178 of 180, and leave the two unmeasured bands thin. Equal counts measure
each band equally well; the population shares are recorded here so the bands
can be reweighted to the whole when a whole-pool number is wanted.

AND NOT ENRICHED BY ANY MODEL. The probe's opinion is what the student is
trained on, so a sheet enriched by it would measure agreement between a model
and its own teacher and report it as a rate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import shutil

BANDS = [(0, 100), (100, 150), (150, 200), (200, 340), (340, 10 ** 9)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", default="/workspace/heldout_views")
    ap.add_argument("--per_band", type=int, default=45)
    ap.add_argument("--max_per_rec", type=int, default=3)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    print("留出池 %d 个框 / %d 条录像" % (len(rows), len({r["rec"] for r in rows})))
    rnd = random.Random(a.seed)
    pick, key = [], []
    for lo, hi in BANDS:
        sub = [r for r in rows if lo <= int(r["w_px"]) < hi]
        share = len(sub) / float(len(rows))
        rnd.shuffle(sub)
        seen = collections.Counter()
        take = []
        for r in sub:
            if seen[r["rec"]] >= a.max_per_rec:
                continue
            seen[r["rec"]] += 1
            take.append(r)
            if len(take) >= a.per_band:
                break
        name = "%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo
        print("  %-10s 总体 %5d（%4.1f%%）  抽 %d" % (name, len(sub), 100 * share,
                                                   len(take)))
        pick += [(name, share, r) for r in take]
    rnd.shuffle(pick)

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out = []
    for band, share, r in pick:
        f = os.path.join(a.views, r["stem"] + "_full.jpg")
        c = os.path.join(a.views, r["stem"] + "_crop.jpg")
        if not (os.path.exists(f) and os.path.exists(c)):
            continue
        shutil.copy(f, os.path.join(a.out, "context", r["stem"] + ".jpg"))
        shutil.copy(c, os.path.join(a.out, "crops", r["stem"] + ".jpg"))
        out.append({"stem": r["stem"], "frame": int(r["frame"]), "hand": 0,
                    "conf": float(r["conf"]), "w_frac": 0.0, "h_frac": 0.0,
                    "cx_frac": 0.0, "cy_frac": 0.0, "model": "heldout",
                    "label": "", "label_mode": ""})
        key.append({"stem": r["stem"], "band": band,
                    "band_share": round(share, 4), "w_px": r["w_px"],
                    "rec": r["rec"], "cam": r["cam"], "det_conf": r["conf"]})
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
                 "每个宽度带等量抽样，不是按总体比例；带的总体占比在 "
                 "%s_key.csv 里，要还原到全体时用它加权。\n"
                 "没有任何模型参与抽样。\n" % a.out.rstrip("/"))
    print("-> %s (%d)  分带和权重另存 %s_key.csv"
          % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
