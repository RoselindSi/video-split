"""Was the wearer's hand there at all, on the frames nobody delivered it.

THE DENOMINATOR OF EVERY OWNER METRIC. "The hand is missing" and "the hand is
out of shot" produce identical output and mean opposite things: one is a
failure to fix, the other is the recording. Coverage, longest missing run and
fragmentation are all undefined until the frames are separated, and no model
can separate them -- both systems agreeing that nothing is there is exactly
what a hand out of shot looks like.

So the frames where BOTH systems deliver no own hand are put to a person, one
at a time, whole frame, no crop: the question is about the picture, not about
a box, and a crop would answer a different one.

THE LAST KNOWN POSITION IS DRAWN, in grey, labelled as such. Over a 70-frame
gap the hand may be nowhere near it, so it locates rather than proposes; a
judgement of "visible" still has to find the hand.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def own_frames(path):
    """-> ({frames with an own hand}, {frame: last own box})"""
    seen, box = set(), {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["own"] == "1":
            f = int(r["frame"])
            seen.add(f)
            box[f] = [int(float(r[c])) for c in ("x0", "y0", "x1", "y1")]
    return seen, box


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="/workspace/cam3_base")
    ap.add_argument("--theirs", default="/workspace/cam3_rolan")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--every", type=int, default=1,
                    help="keep one frame in N inside a long run")
    ap.add_argument("--ctx_px", type=int, default=1100)
    ap.add_argument("--crop_px", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows = []
    for rec in sorted(jobs):
        bag, start, n = jobs[rec]
        po = os.path.join(a.ours, rec + ".csv")
        pt = os.path.join(a.theirs, rec + ".csv")
        if not (os.path.exists(po) and os.path.exists(pt)):
            continue
        A, boxes = own_frames(po)
        B, _ = own_frames(pt)
        gaps = [f for f in range(start, start + n) if f not in A and f not in B]
        if not gaps:
            continue
        # one frame in `every` inside a run, but always its first and last
        runs, cur = [], None
        for f in range(start, start + n):
            if f in A or f in B:
                if cur:
                    runs.append(cur)
                    cur = None
            else:
                cur = cur or []
                cur.append(f)
        if cur:
            runs.append(cur)
        want = []
        for r in runs:
            keep = {r[0], r[-1]} | set(r[::a.every])
            want += sorted(keep)
        want = sorted(set(want))
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        rd = RawCameraReader(vids, "cam3", want[0])
        cur_f = want[0] - 1
        last_box = None
        for f in want:
            img = None
            while cur_f < f:
                img = rd.next()
                cur_f += 1
                if img is None:
                    break
            if img is None:
                break
            prior = [g for g in boxes if g < f]
            if prior:
                last_box = boxes[max(prior)]
            H, W = img.shape[:2]
            vis = img.copy()
            if last_box is not None:
                cv2.rectangle(vis, tuple(last_box[:2]), tuple(last_box[2:]),
                              (170, 170, 170), 3)
                cv2.putText(vis, "last seen here", (last_box[0], max(24, last_box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (170, 170, 170), 2,
                            cv2.LINE_AA)
            stem = f"{rec}_f{f:06d}_h0"
            sc = a.ctx_px / float(W)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"),
                        cv2.resize(vis, (a.ctx_px, int(H * sc))),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                        cv2.resize(img, (a.crop_px, a.crop_px)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            rows.append({"stem": stem, "frame": f, "hand": 0, "conf": 0.0,
                         "w_frac": 1.0, "h_frac": 1.0,
                         "cx_frac": 0.5, "cy_frac": 0.5,
                         "model": "both_miss", "label": "", "label_mode": ""})
        rd.close()
        print("  %-16s 两边都丢 %d 帧，取 %d 帧" % (rec, len(gaps), len(want)), flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n%d 帧 -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
