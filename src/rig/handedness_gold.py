"""Which of the wearer's two hands is this -- gold, and what the detector said.

WHY IT MATTERS FOR CONTINUITY AND NOT ONLY FOR TIDINESS. The goal is the
physical identity of the wearer's hands over time, and the wearer has two.
Every continuity number so far treats them as one population: the own-hand
chains are built by overlap between frames, so when the two hands cross, a
chain can carry on down the wrong arm and count as continuous. The tracker
has been charging SIDE_MISMATCH on the detector's `left`/`right` classes the
whole time and nothing downstream ever recorded which it said, so there has
never been a way to see that happen.

THE DETECTOR'S ANSWER COMES FREE -- its two classes ARE left and right -- so
this asks a person the same question and keeps the prediction beside it.
An earlier probe on the stitched render put own-hand purity at 0.64 against
0.89 for a colleague's and was filed inconclusive, which is a reason to
measure it on cam3 rather than to assume either number.

LEFT AND RIGHT ARE IN THE WEARER'S OWN FRAME, judged from the thumb and the
palm. For a colleague's hand the same words mean their body's frame and the
question is a different one, so only the wearer's hands are sampled.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_base")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--crop_px", type=int, default=256)
    ap.add_argument("--ctx_px", type=int, default=1100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    cand = []
    for rec in sorted(jobs):
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] != "1":
                continue
            cand.append((rec, int(r["frame"]),
                         [int(float(r[c])) for c in ("x0", "y0", "x1", "y1")],
                         float(r["conf"] or 0), r.get("side", "")))
    sides = collections.Counter(x[4] for x in cand)
    print("自己的手 %d 个框；检测器给的 side 分布 %s" % (len(cand), dict(sides)))
    if not any(k in ("left", "right") for k in sides):
        print("  !! 这一臂的 CSV 还没有 side 列，需要带上新代码重跑才有预测可比")
    random.Random(a.seed).shuffle(cand)
    pick = cand[:a.limit]

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for it in pick:
        by_rec[it[0]].append(it)
    rows = []
    for rec in sorted(by_rec):
        bag, start, n = jobs[rec]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        want = sorted({it[1] for it in by_rec[rec]})
        rd = RawCameraReader(vids, "cam3", want[0])
        cur = want[0] - 1
        cache = {}
        for f in want:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            cache[f] = img.copy()
        rd.close()
        for j, (r_, f, box, conf, side) in enumerate(by_rec[rec]):
            img = cache.get(f)
            if img is None:
                continue
            H, W = img.shape[:2]
            vis = img.copy()
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 230, 0), 4)
            stem = "%s_f%06d_h%d" % (rec, f, 900 + j)
            sc = a.ctx_px / float(W)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"),
                        cv2.resize(vis, (a.ctx_px, int(H * sc))),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            # a GENEROUS crop: which hand it is reads off the thumb and the
            # wrist, and a tight box cuts both
            side_px = int(max(box[2] - box[0], box[3] - box[1]) * 2.2)
            mx, my = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            x0, y0 = max(0, mx - side_px // 2), max(0, my - side_px // 2)
            x1, y1 = min(W, mx + side_px // 2), min(H, my + side_px // 2)
            crop = vis[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(crop, (a.crop_px, a.crop_px)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rows.append({"stem": stem, "frame": f, "hand": 900 + j,
                         "conf": round(conf, 4),
                         "w_frac": round((box[2] - box[0]) / float(W), 5),
                         "h_frac": round((box[3] - box[1]) / float(H), 5),
                         "cx_frac": round((box[0] + box[2]) / 2.0 / W, 5),
                         "cy_frac": round((box[1] + box[3]) / 2.0 / H, 5),
                         "model": side or "no_side",
                         "label": "", "label_mode": ""})
        print("  %-16s %d 个" % (rec, len(by_rec[rec])), flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 个 -> %s   （model 列存的是检测器的预测）" % (len(rows), a.out))


if __name__ == "__main__":
    main()
