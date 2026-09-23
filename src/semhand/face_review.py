"""Put the face question to a person, on the two populations it answers differently.

TWO QUESTIONS IN ONE SHEET, AND THEY NEED DIFFERENT BOXES.

The first is whether the verdict on the boxes the size cap would refuse can be
trusted. There are thirteen of those in eighty seconds of cam3 and all
thirteen go in, because thirteen is the whole population and sampling it would
be a choice nobody needs to make.

The second is what to make of the small ones. Of 271 mosaicked boxes at or
under the cap, the model calls 78% not-a-face. That is either the face
detector firing on bench and wall at small sizes -- which would be the first
false-positive rate this pipeline has ever had, its own code saying that until
those are labelled "every face number here is an ordering, not a rate" -- or
it is the probe losing its footing below 100 px, which is precisely what
hand-ness does. The two readings imply opposite actions and nothing in the
scores separates them.

SO THE SMALL ONES ARE DRAWN FROM BOTH SIDES OF THE MODEL'S OWN VERDICT, not at
random. A random draw of small boxes would be 78% rejections and would measure
the rejections well and the acceptances not at all, when it is the pair that
decides the question: if the rejections are mostly right the detector is
noisy, and if they are mostly wrong the probe is.

THE MODEL'S ANSWER IS IN A SEPARATE FILE. It is what the sheet is testing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="/workspace/faceness.jsonl")
    ap.add_argument("--views", default="/workspace/face_views")
    ap.add_argument("--cap", type=float, default=0.18)
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--n_small_yes", type=int, default=25)
    ap.add_argument("--n_small_no", type=int, default=35)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = {r["stem"]: r
           for r in csv.DictReader(open(os.path.join(a.views, "index.csv")))}
    rows = []
    for line in open(a.scores):
        d = json.loads(line)
        if d["stem"] in idx:
            rows.append({**d, **{k: idx[d["stem"]][k] for k in ("n_frames", "conf")}})
    print("打过分的段 %d" % len(rows))

    big = [r for r in rows if r["w_frac"] > a.cap]
    small_yes = [r for r in rows if r["w_frac"] <= a.cap and r["p"] > a.thr]
    small_no = [r for r in rows if r["w_frac"] <= a.cap and r["p"] <= a.thr]
    print("  cap 会拒的 %d（全放）" % len(big))
    print("  小框·模型说是脸 %d  小框·模型说不是脸 %d" % (len(small_yes), len(small_no)))

    rnd = random.Random(a.seed)
    pick = [("big", r) for r in big]
    pick += [("small_yes", r) for r in rnd.sample(small_yes, min(a.n_small_yes, len(small_yes)))]
    pick += [("small_no", r) for r in rnd.sample(small_no, min(a.n_small_no, len(small_no)))]
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
        out.append({"stem": r["stem"], "frame": int(r["frame"]), "hand": 0,
                    "conf": 0.0, "w_frac": round(float(r["w_frac"]), 4),
                    "h_frac": 0.0, "cx_frac": 0.0, "cy_frac": 0.0,
                    "model": "faceness", "label": "", "label_mode": ""})
        key.append({"stem": r["stem"], "group": g,
                    "p_face": round(float(r["p"]), 4),
                    "w_frac": round(float(r["w_frac"]), 4),
                    "n_frames": r["n_frames"], "det_conf": r["conf"],
                    "rec": r["rec"]})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：红框里的东西是不是一张脸或一个头（侧脸、后脑勺、侧过去的脸都算）？\n"
                 "是 -> owner   不是 -> nothand   看不清 -> unsure\n"
                 "模型的答案在 %s_key.csv，不在这张表里。\n" % a.out.rstrip("/"))
    print("-> %s (%d)  答案另存 %s_key.csv"
          % (a.out, len(out), a.out.rstrip("/")))
    print("   分组：%s" % dict(collections.Counter(g for g, _ in pick)))


if __name__ == "__main__":
    main()
