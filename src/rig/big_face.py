"""The boxes the size cap would refuse, on the SOURCE camera image.

WHAT THIS IS FOR. `MAX_FACE_FRAC = 0.18` is a measured number that has been
sitting switched off for one reason, written down in `face_mask`: the ten
recordings it was fitted on contain no colleague standing close to the
camera, so they can say the 25 huge boxes inspected there were all false but
not that a huge TRUE face is rare. The note says what would settle it -- "the
same detector on the SOURCE camera image, where the panorama's warp is not
there to invent one" -- and cam3 raw frames are exactly that.

EPISODES, NOT FRAME INSTANCES. A box that survives twelve frames of hold is
one thing seen twelve times, and counting the instances would turn a single
false detection into a dozen pieces of evidence for the same conclusion.
Consecutive frames whose refused boxes overlap are collapsed into one, and
the middle frame of each is what gets looked at.

THE WHOLE FRAME IS ON THE SHEET. The question is whether the box is a face,
and a crop of a stove or a tray of parts at 60% of the frame's width is a
wall of texture either way; what settles it is where it sits in the room.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", default="/workspace/cam3_faces")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--cap", type=float, default=0.18)
    ap.add_argument("--cell_w", type=int, default=620)
    ap.add_argument("--cols", type=int, default=4)
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

    eps = []
    for rec in sorted(jobs):
        p = os.path.join(a.faces, rec + ".faces.csv")
        if not os.path.exists(p):
            continue
        big = collections.defaultdict(list)
        n_all = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n_all += 1
            if float(r["w_frac"]) > a.cap:
                big[int(r["frame"])].append(
                    ([int(r[c]) for c in ("x0", "y0", "x1", "y1")],
                     float(r["w_frac"]), r["from_det"] == "1", r["conf"]))
        live = []          # [(last_frame, box, [frames], max_w, any_det, confs)]
        for f in sorted(big):
            for box, w, fd, cf in big[f]:
                for L in live:
                    if L[0] >= f - 2 and iou(L[1], box) > 0.3:
                        L[0], L[1] = f, box
                        L[2].append(f)
                        L[3] = max(L[3], w)
                        L[4] = L[4] or fd
                        if cf:
                            L[5].append(float(cf))
                        break
                else:
                    live.append([f, box, [f], w, fd, [float(cf)] if cf else []])
        for L in live:
            eps.append({"rec": rec, "frames": L[2], "box": L[1], "w": L[3],
                        "from_det": L[4], "confs": L[5]})
        print("  %-16s 打码框 %d，宽 >%.2f 的 %d 个实例 -> %d 段"
              % (rec, n_all, a.cap, sum(len(v) for v in big.values()), len(live)))
    print("\n共 %d 段" % len(eps))
    if not eps:
        return

    by_rec = collections.defaultdict(list)
    for e in eps:
        by_rec[e["rec"]].append(e)
    tiles, rows = [], []
    for rec in sorted(by_rec):
        bag = jobs[rec][0]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        want = sorted({e["frames"][len(e["frames"]) // 2] for e in by_rec[rec]})
        rd = RawCameraReader(vids, "cam3", want[0])
        cur, cache = want[0] - 1, {}
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
        for e in by_rec[rec]:
            f = e["frames"][len(e["frames"]) // 2]
            img = cache.get(f)
            if img is None:
                continue
            v = img.copy()
            b = e["box"]
            cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 6)
            cv2.putText(v, "%s f%d  w=%.2f  %dfr" % (rec[-6:], f, e["w"],
                                                     len(e["frames"])),
                        (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 255), 3, cv2.LINE_AA)
            sc = a.cell_w / float(v.shape[1])
            tiles.append(cv2.resize(v, (a.cell_w, int(v.shape[0] * sc))))
            rows.append({"rec": rec, "frame": f, "n_frames": len(e["frames"]),
                         "w_frac": round(e["w"], 4), "from_det": int(e["from_det"]),
                         "conf": max(e["confs"]) if e["confs"] else "",
                         "box": " ".join(str(x) for x in b)})
    h, w = tiles[0].shape[:2]
    nr = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.zeros((nr * h, a.cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        r_, c_ = divmod(k, a.cols)
        sheet[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("-> %s  (%dx%d)" % (a.out, sheet.shape[1], sheet.shape[0]))


if __name__ == "__main__":
    main()
