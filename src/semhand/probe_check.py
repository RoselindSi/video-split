"""Does the hand-ness probe still work where it is now being used.

WHY NOT TAKE THE 0.950 AND MOVE ON. It was measured on 88 boxes drawn from
C1's weak admissions -- a population picked for low detector confidence, not
for crowding. It is now deciding 852 boxes in frames that contain three or
more claimed hands, which is a different distribution in the one way that
matters: these frames are full of other people's hands, and a model that is
excellent at telling a hand from a machine part has never been asked to do it
with four arms in the picture.

BALANCED, AND THE IMBALANCE PUT BACK AFTERWARDS. Forty boxes the model called
hands and forty it called non-hands, rather than eighty at random -- at random
the rejected class would contribute about twenty-five items and its error bar
would be the whole result. The population is 586 to 266, so the rates are
reweighted to it when they are read back.

THE VIEWS ARE THE ONES THE MODEL SAW, unchanged, so a disagreement is about
the judgement and not about what was on the screen -- and the model's own
answer is nowhere on them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="/workspace/three_c1.jsonl")
    ap.add_argument("--views", default="/workspace/three_views_c1")
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--n_each", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.scores)]
    lo = [r for r in rows if r["p"] <= a.thr]
    hi = [r for r in rows if r["p"] > a.thr]
    print("总体 %d：模型判非手 %d、判手 %d" % (len(rows), len(lo), len(hi)))
    rnd = random.Random(a.seed)
    pick = rnd.sample(lo, min(a.n_each, len(lo))) + \
        rnd.sample(hi, min(a.n_each, len(hi)))
    rnd.shuffle(pick)

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out = []
    for r in pick:
        f = os.path.join(a.views, r["stem"] + "_full.jpg")
        c = os.path.join(a.views, r["stem"] + "_crop.jpg")
        if not (os.path.exists(f) and os.path.exists(c)):
            continue
        # `label_tool` reads the recording and the hand index back out of the
        # stem, so the probe's own `_b0` naming has to become the `_h0` the
        # rest of the packages use or the sheet comes out empty.
        stem = r["stem"].replace("_b", "_h")
        r["pkg_stem"] = stem
        shutil.copy(f, os.path.join(a.out, "context", stem + ".jpg"))
        shutil.copy(c, os.path.join(a.out, "crops", stem + ".jpg"))
        out.append({"stem": stem, "frame": r["frame"], "hand": 0,
                    "conf": round(float(r["conf"]), 3),
                    "w_frac": 0.0, "h_frac": 0.0, "cx_frac": 0.0, "cy_frac": 0.0,
                    "model": "three_hand_box", "label": "", "label_mode": ""})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    # the key is kept OUT of the package that gets labelled
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "p_hand", "model_says"])
        for r in pick:
            w.writerow([r.get("pkg_stem", r["stem"]), round(r["p"], 4),
                        "nonhand" if r["p"] <= a.thr else "hand"])
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：绿框里的东西是不是一只手（戴手套也算）？\n"
                 "是手   -> owner\n不是手 -> nothand\n看不清 -> unsure\n")
    print("-> %s (%d)  答案另存 %s_key.csv"
          % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
