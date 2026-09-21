"""The hands one detector finds and the other does not, put to a person.

WHY THIS CANNOT BE READ OFF THE NUMBERS. Rolan's detector reports 9,749
foreign hand-frames on the six cam3 recordings where ours reports 4,019, and
the exposure counts that go with them -- 75 against 23 -- are therefore not
comparable: a larger absolute count over a 2.4x larger population says
nothing about whether either is worse. The extra 5,730 detections are either
colleagues' hands our detector is missing, which would make the comparison
favour Rolan, or false boxes, which would make it favour ours, and the two
readings point opposite ways.

So the extra detections are sampled at random -- not by score, since score is
the thing under suspicion -- and asked about one at a time: a colleague's
hand, the wearer's own, or not a hand.

MATCHING IS BY OVERLAP WITH ANY OF OUR DETECTIONS IN THE SAME FRAME, owner or
foreign. A box we found and called the wearer's is not an extra box; it is a
disagreement about ownership, which is a different question with a different
answer sheet.
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


def boxes_of(path):
    """-> {frame: [(box, own)]}"""
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[int(r["frame"])].append((
            [float(r[c]) for c in ("x0", "y0", "x1", "y1")], r["own"] == "1",
            float(r.get("conf") or 0.0)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="/workspace/cam3_base")
    ap.add_argument("--theirs", default="/workspace/cam3_rolan")
    ap.add_argument("--reverse", action="store_true",
                    help="the boxes OURS has and theirs does not. The same "
                         "question has to be asked in both directions or the "
                         "comparison is one detector audited and the other "
                         "taken on trust")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=300, help="boxes to sample")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--crop_px", type=int, default=192)
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

    extra = []
    for p in sorted(glob.glob(os.path.join(a.theirs, "*.csv"))):
        if p.endswith(".assoc.csv"):
            continue
        rec = os.path.basename(p)[:-4]
        ours = os.path.join(a.ours, rec + ".csv")
        if rec not in jobs or not os.path.exists(ours):
            continue
        mine, yours = boxes_of(ours), boxes_of(p)
        if a.reverse:
            mine, yours = yours, mine
        for f, dets in yours.items():
            have = [b for b, _, _ in mine.get(f, [])]
            for box, own, conf in dets:
                if own:
                    continue                 # ownership disagreement, not this
                if any(iou(box, g) >= a.match_iou for g in have):
                    continue
                extra.append((rec, f, box, conf))
    print("两臂对齐后，%s独有的『别人的手』框 %d 个"
          % ("我们" if a.reverse else "Rolan", len(extra)))
    if not extra:
        return
    # RANDOM, NOT BY SCORE. Score is what is under suspicion, and a sheet
    # ordered by it measures the ordering rather than the detector.
    random.Random(a.seed).shuffle(extra)
    pick = extra[:a.limit]
    by_rec = collections.defaultdict(list)
    for rec, f, box, conf in pick:
        by_rec[rec].append((f, box, conf))
    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows = []
    for rec in sorted(by_rec):
        bag, start, n = jobs[rec]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        want = collections.defaultdict(list)
        for f, box, conf in by_rec[rec]:
            want[f].append((box, conf))
        rd = RawCameraReader(vids, "cam3", min(want))
        f0 = min(want)
        cur = f0 - 1
        for f in sorted(want):
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            H, W = img.shape[:2]
            for j, (box, conf) in enumerate(want[f]):
                x0, y0, x1, y1 = (int(v) for v in box)
                stem = f"{rec}_f{f:06d}_h{900 + j}"
                side = max(x1 - x0, y1 - y0) * 2
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cx0, cy0 = max(0, cx - side // 2), max(0, cy - side // 2)
                cx1, cy1 = min(W, cx + side // 2), min(H, cy + side // 2)
                crop = img[cy0:cy1, cx0:cx1]
                if crop.size:
                    cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                                cv2.resize(crop, (a.crop_px, a.crop_px)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                vis = img.copy()
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
                sc = a.ctx_px / float(W)
                cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"),
                            cv2.resize(vis, (a.ctx_px, int(H * sc))),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                rows.append({"stem": stem, "frame": f, "hand": j,
                             "conf": round(conf, 4),
                             "w_frac": round((x1 - x0) / float(W), 5),
                             "h_frac": round((y1 - y0) / float(H), 5),
                             "cx_frac": round(cx / float(W), 5),
                             "cy_frac": round(cy / float(H), 5),
                             "model": "ours" if a.reverse else "enhanced_8m2",
                             "label": "", "label_mode": ""})
        rd.close()
        print(f"  {rec}: {len(by_rec[rec])} 框", flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 个框 -> {a.out}")
    c = sorted(r["conf"] for r in rows)
    print(f"  分数 中位 {c[len(c) // 2]:.2f}  最小 {c[0]:.2f}  最大 {c[-1]:.2f}")


if __name__ == "__main__":
    main()
