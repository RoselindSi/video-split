"""Where DS and the probe disagree, and a person settles it.

NEITHER SIDE IS GOLD. DS is unreviewed weak supervision whose own manifest
declines to claim coverage; the probe is a 27B model validated on 88 boxes
from a different population. Their agreement is worth something and their
disagreement is worth nothing until someone looks, so the sheet is built
around the disagreement and carries a slice of the agreement as a control.

THE TWO DISAGREEMENTS POINT OPPOSITE WAYS AND ONLY ONE IS DANGEROUS. Boxes
DS calls a colleague's hand and the probe calls not-a-hand are the gate's
failure mode: put hand-ness in front as a veto and each of those walks out of
the pipeline uncovered. Boxes DS calls not-a-hand and the probe calls a hand
cost nothing in the pipeline -- but they decide whether DS's negative class
can be believed at all, and 57% of it is in that state.

THE CONTROLS ARE NOT PADDING. Without them the sheet is all hard cases and
the labeller's calibration drifts toward doubt; with them the same person's
accuracy on the easy cells is visible, and a cell where they disagree with
both machines is a signal about the sheet rather than about the models.
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
    ap.add_argument("--scores", default="/workspace/ds_probe.jsonl")
    ap.add_argument("--key", default="/workspace/ds_views_key.csv")
    ap.add_argument("--views", default="/workspace/ds_views")
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--n_agree", type=int, default=25)
    ap.add_argument("--n_nothand_disagree", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = {r["stem"]: r for r in csv.DictReader(open(a.key))}
    rows = []
    for line in open(a.scores):
        d = json.loads(line)
        k = key.get(d["stem"])
        if k:
            rows.append({**d, "ds_status": k["ds_status"],
                         "ds_ownership": k["ds_ownership"]})

    def cell(st, own, lo, hi):
        return [r for r in rows if r["ds_status"] == st
                and r["ds_ownership"] == own and lo < r["p"] <= hi]

    rnd = random.Random(a.seed)
    groups = {
        # DS: a colleague's hand.  Probe: not a hand.  The gate's failure.
        "A_other_rejected": cell("hand", "other", -1, a.thr),
        # DS: not a hand.  Probe: a hand.  Decides whether DS's negatives hold.
        "B_nothand_accepted": None,
        "C_other_agreed": None,
        "D_nothand_agreed": None,
    }
    b = cell("not_hand", "", a.thr, 2)
    groups["B_nothand_accepted"] = rnd.sample(b, min(a.n_nothand_disagree, len(b)))
    c = cell("hand", "other", a.thr, 2)
    groups["C_other_agreed"] = rnd.sample(c, min(a.n_agree, len(c)))
    d = cell("not_hand", "", -1, a.thr)
    groups["D_nothand_agreed"] = rnd.sample(d, min(a.n_agree, len(d)))

    pick = []
    for g, items in groups.items():
        print("%-22s 总体 %4d  抽 %3d" % (
            g, len(cell("hand", "other", -1, a.thr)) if g == "A_other_rejected"
            else (len(b) if g == "B_nothand_accepted"
                  else len(c) if g == "C_other_agreed" else len(d)), len(items)))
        pick += [(g, r) for r in items]
    rnd.shuffle(pick)

    for sub in ("context", "crops"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out, ans = [], []
    for g, r in pick:
        f = os.path.join(a.views, r["stem"] + "_full.jpg")
        c_ = os.path.join(a.views, r["stem"] + "_crop.jpg")
        if not (os.path.exists(f) and os.path.exists(c_)):
            continue
        shutil.copy(f, os.path.join(a.out, "context", r["stem"] + ".jpg"))
        shutil.copy(c_, os.path.join(a.out, "crops", r["stem"] + ".jpg"))
        out.append({"stem": r["stem"], "frame": 0, "hand": 0, "conf": 0.0,
                    "w_frac": r.get("w_frac", 0.0), "h_frac": r.get("h_frac", 0.0),
                    "cx_frac": 0.0, "cy_frac": 0.0, "model": "ds_arbitration",
                    "label": "", "label_mode": ""})
        ans.append({"stem": r["stem"], "group": g, "ds_status": r["ds_status"],
                    "ds_ownership": r["ds_ownership"], "p_hand": round(r["p"], 4)})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ans[0]))
        w.writeheader()
        w.writerows(ans)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：绿框里的东西是不是一只手（戴手套也算）？\n"
                 "是手   -> owner\n不是手 -> nothand\n看不清 -> unsure\n"
                 "两台机器的答案都在 %s_key.csv，不在这张表里。\n"
                 % a.out.rstrip("/"))
    print("-> %s (%d)  两边的答案另存 %s_key.csv"
          % (a.out, len(out), a.out.rstrip("/")))


if __name__ == "__main__":
    main()
