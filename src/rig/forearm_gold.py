"""Is the wearer's FOREARM getting blurred, asked of the delivered pixels.

WHY THERE IS NO METRIC FOR THIS YET. `covered` is the fraction of a HAND box
whose pixels changed, and the forearm is in no box: nothing in the pipeline
represents it, so nothing measures it. A mask that clips the wrist and runs
up the arm scores zero on every number the project has.

WHAT THIS DELIBERATELY DOES NOT DO is invent a forearm region and measure
against it. A band extrapolated from the hand keypoints along the arm
direction would be a new construct with its own error, and validating it
needs exactly the judgement it is meant to replace. So the judgement comes
first: the frame that LEAVES the pipeline, the delivered own hand boxed so
the labeller knows which arm is the wearer's, and one question about the
pixels outside that box.

THE SAMPLE IS RANDOM over frames where an own hand was delivered. Picking
frames near a large mask would measure the picking.

AND THE VISIBILITY SPLIT IS PART OF THE QUESTION, not a separate sheet. A
frame with no forearm in it answers "not blurred" and inflates the clean rate
with frames that had nothing to blur, which is the same mistake as counting a
hand that was out of shot as a hand the pipeline lost. So the three answers
are visible-and-clean, visible-and-blurred, and not in shot, and only the
first two form the denominator.

WHAT THIS SAMPLE CANNOT SAY. It is drawn from frames where the pipeline DID
deliver an own hand, so it estimates a conditional quantity: given that the
hand came through, did the arm. Frames where the hand itself was lost are
outside it, and a clean result here does not mean the arm is continuous
through the recording.
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
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--ctx_px", type=int, default=1100)
    ap.add_argument("--crop_px", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    cand = []
    boxes = collections.defaultdict(list)
    for rec in sorted(jobs):
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] != "1":
                continue
            f = int(r["frame"])
            boxes[(rec, f)].append([int(float(r[c])) for c in ("x0", "y0", "x1", "y1")])
        cand += [(rec, f) for (rc, f) in boxes if rc == rec]
    cand = sorted(set(boxes))
    print("交付了自己的手的帧 %d 个" % len(cand))
    random.Random(a.seed).shuffle(cand)
    pick = sorted(cand[:a.limit])

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows = []
    by_rec = collections.defaultdict(list)
    for rec, f in pick:
        by_rec[rec].append(f)
    for rec in sorted(by_rec):
        mp4 = os.path.join(a.arm, rec + ".mp4")
        if not os.path.exists(mp4):
            print("  !! 没有 %s" % mp4)
            continue
        start = jobs[rec][1]
        want = set(by_rec[rec])
        cap = cv2.VideoCapture(mp4)
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            f = start + i
            i += 1
            if f not in want:
                continue
            # THE DELIVERED PANEL, which is the bottom one: the stacked layout
            # is bar, decision, bar, output, so the output is the last block of
            # source height. Reading the top panel would ask about a picture
            # with boxes drawn on it that nobody downstream ever receives.
            H = im.shape[0]
            src_h = (H - 68) // 2
            out = im[H - src_h:]
            vis = out.copy()
            for b in boxes[(rec, f)]:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (60, 220, 60), 3)
            stem = "%s_f%06d_h0" % (rec, f)
            W = vis.shape[1]
            sc = a.ctx_px / float(W)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"),
                        cv2.resize(vis, (a.ctx_px, int(vis.shape[0] * sc))),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b = boxes[(rec, f)][0]
            pad = max(200, (b[2] - b[0]) * 2)
            cx0, cy0 = max(0, b[0] - pad), max(0, b[1] - pad)
            cx1, cy1 = min(W, b[2] + pad), min(vis.shape[0], b[3] + pad)
            crop = vis[cy0:cy1, cx0:cx1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(crop, (a.crop_px, a.crop_px)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({"stem": stem, "frame": f, "hand": 0, "conf": 0.0,
                         "w_frac": round((b[2] - b[0]) / float(W), 5),
                         "h_frac": round((b[3] - b[1]) / float(vis.shape[0]), 5),
                         "cx_frac": round((b[0] + b[2]) / 2.0 / W, 5),
                         "cy_frac": round((b[1] + b[3]) / 2.0 / vis.shape[0], 5),
                         "model": os.path.basename(a.arm),
                         "label": "", "label_mode": ""})
        cap.release()
        print("  %-16s %d 帧" % (rec, len(by_rec[rec])), flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 帧 -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
