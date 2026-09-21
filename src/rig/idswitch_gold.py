"""Did the track that survived the gap survive it with the same hand in it.

WHY LOWERING THE CONTINUATION FLOOR NEEDS THIS AND COVERAGE DOES NOT ANSWER
IT. C1 lets a detection too weak to start a track extend one, which either
carries the wearer's hand through a bad patch -- what it is for -- or lets a
different physical hand inherit a track that has already accumulated an OWNER
verdict. Both raise coverage. Both shorten the longest missing run. One is
the fix and the other is the failure the whole ownership layer exists to
prevent, and no aggregate distinguishes them.

SO THE UNIT IS A PAIR, not a frame: the hand as it was last seen before the
weak detection, and the hand in the weak detection. The question is whether
they are the same physical hand, and it is put with both ends in one picture
because a single crop cannot answer it.

TWO GROUPS, AND THE CONTROL IS NOT OPTIONAL. Auditing only what C1 added
measures C1's additions against nothing: a 3% switch rate is alarming or
reassuring depending on what the associations both arms already agreed on do,
so an equal random sample of those is drawn too, from the same recordings and
shuffled into the same sheet. Which group an item came from is in the CSV and
not on the screen.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import random


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def own(path):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["own"] == "1":
            out[int(r["frame"])].append(
                ([int(float(r[c])) for c in ("x0", "y0", "x1", "y1")],
                 float(r["conf"] or 0)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="/workspace/cam3_c1")
    ap.add_argument("--old", default="/workspace/cam3_base")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--per_group", type=int, default=90)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--ctx_px", type=int, default=1200)
    ap.add_argument("--crop_px", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    import numpy as np
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    added, control = [], []
    for p in sorted(glob.glob(os.path.join(a.new, "*.csv"))):
        if p.endswith(".assoc.csv"):
            continue
        rec = os.path.basename(p)[:-4]
        q = os.path.join(a.old, rec + ".csv")
        if rec not in jobs or not os.path.exists(q):
            continue
        N, O = own(p), own(q)
        for f in sorted(N):
            have = [b for b, _ in O.get(f, [])]
            for box, conf in N[f]:
                fresh = not any(iou(box, g) >= 0.3 for g in have)
                # the hand as last seen BEFORE this frame, in the same arm
                prior = [g for g in N if g < f and N[g]]
                if not prior:
                    continue
                pf = max(prior)
                if f - pf > 12:
                    continue           # too far to ask about as one hand
                pbox = max(N[pf], key=lambda t: iou(t[0], box))[0]
                item = (rec, pf, pbox, f, box, conf, fresh)
                if fresh and conf < 0.25:
                    added.append(item)
                elif not fresh:
                    control.append(item)
    print("C1 新增的弱观测关联 %d 个；两臂都有的对照 %d 个"
          % (len(added), len(control)))
    rng = random.Random(a.seed)
    rng.shuffle(added)
    rng.shuffle(control)
    pick = added[:a.per_group] + control[:a.per_group]
    rng.shuffle(pick)                  # the sheet must not be grouped

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for it in pick:
        by_rec[it[0]].append(it)
    rows = []
    for rec in sorted(by_rec):
        bag, start, n = jobs[rec]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        need = sorted({f for it in by_rec[rec] for f in (it[1], it[3])})
        rd = RawCameraReader(vids, "cam3", need[0])
        cur = need[0] - 1
        img_of = {}
        for f in need:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            img_of[f] = img.copy()
        rd.close()
        for j, (r_, pf, pbox, f, box, conf, fresh) in enumerate(by_rec[rec]):
            if pf not in img_of or f not in img_of:
                continue
            panes = []
            for tag, fr, bx, col in (("断档前 f%d" % pf, pf, pbox, (60, 220, 60)),
                                     ("弱观测 f%d" % f, f, box, (0, 200, 255))):
                v = img_of[fr].copy()
                cv2.rectangle(v, (bx[0], bx[1]), (bx[2], bx[3]), col, 4)
                cv2.putText(v, tag, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            col, 3, cv2.LINE_AA)
                panes.append(v)
            W = a.ctx_px // 2
            sc = W / float(panes[0].shape[1])
            panes = [cv2.resize(v, (W, int(v.shape[0] * sc))) for v in panes]
            ctx = np.hstack(panes)
            stem = "%s_f%06d_h%d" % (rec, f, 900 + j)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"), ctx,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            pad = max(120, (box[2] - box[0]))
            src = img_of[f]
            cx0, cy0 = max(0, box[0] - pad), max(0, box[1] - pad)
            cx1 = min(src.shape[1], box[2] + pad)
            cy1 = min(src.shape[0], box[3] + pad)
            crop = src[cy0:cy1, cx0:cx1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(crop, (a.crop_px, a.crop_px)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({"stem": stem, "frame": f, "hand": 900 + j,
                         "conf": round(conf, 4),
                         "w_frac": round((box[2] - box[0]) / float(src.shape[1]), 5),
                         "h_frac": round((box[3] - box[1]) / float(src.shape[0]), 5),
                         "cx_frac": round((box[0] + box[2]) / 2.0 / src.shape[1], 5),
                         "cy_frac": round((box[1] + box[3]) / 2.0 / src.shape[0], 5),
                         "model": "C1_added" if fresh else "both_agree",
                         "label": "", "label_mode": ""})
        print("  %-16s %d 对" % (rec, len(by_rec[rec])), flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = collections.Counter(r["model"] for r in rows)
    print("\n%d 对 -> %s  （%s）" % (len(rows), a.out, dict(c)))


if __name__ == "__main__":
    main()
