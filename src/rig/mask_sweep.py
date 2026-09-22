"""What the mosaic would have covered under a different pad, or a size cap.

WHY THIS IS NOT SIX MORE RENDERS. The mask is a pure function of the box list:
`cover` grows each box by the pad and paints it, and nothing downstream feeds
back into which boxes were found. So the boxes are written down once by the
shipped run and every candidate setting is evaluated on them, which turns an
hour of GPU per setting into a second of numpy and -- more to the point --
guarantees the arms differ only in the thing being swept.

THE CAP IS APPLIED AS AN APPROXIMATION AND HERE IS THE APPROXIMATION. In the
pipeline the cap sits inside `Hold.update`, so refusing a wide box also
refuses the twelve frames of hold behind it. Here every box wider than the cap
is dropped on every frame it appears. A held box has the width of the
detection that created it, so the two agree wherever the hold is the only
thing keeping the box alive -- which is the case that matters -- and the
answer is confirmed by a real run before anything ships.

THE 10% LINE IS BORROWED, NOT DERIVED. 120 judged frames put the wearer's
forearm under the mosaic on 9, every one of them above 10% of the frame and
none of the 111 clean ones anywhere near it. That makes mask area a usable
surrogate for arm damage at THIS pad; it does not make it one at another pad,
where the same area could be spread over more boxes. So the sweep reports the
area it is confident about and the arm question goes back to the pictures.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

SCALE = 4  # raster the union at a quarter of each side; the answer is a ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", default="/workspace/cam3_faces")
    ap.add_argument("--arm", default="/workspace/cam3_faces",
                    help="the per-hand CSVs, to count only frames that delivered a hand")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1520)
    ap.add_argument("--pad", type=float, action="append", default=[])
    ap.add_argument("--cap", type=float, action="append", default=[])
    ap.add_argument("--hot", type=float, default=0.10,
                    help="the mask fraction above which the forearm gold found damage")
    a = ap.parse_args()
    pads = a.pad or [0.35, 0.25, 0.15, 0.10]
    caps = a.cap or [0.0, 0.18]

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])

    # frames that delivered an own hand -- the population every rate so far uses
    delivered = collections.defaultdict(set)
    for rec in recs:
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] == "1":
                delivered[rec].add(int(r["frame"]))

    boxes = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in recs:
        p = os.path.join(a.faces, rec + ".faces.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            boxes[rec][int(r["frame"])].append(
                (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]),
                 float(r["w_frac"]), r["from_det"] == "1"))

    W, H = a.W // SCALE, a.H // SCALE

    def frac(bs, pad):
        if not bs:
            return 0.0
        m = np.zeros((H, W), bool)
        for x0, y0, x1, y1, _, _ in bs:
            dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
            gx0 = max(0, (x0 - dx)) // SCALE
            gy0 = max(0, (y0 - dy)) // SCALE
            gx1 = min(a.W, (x1 + dx)) // SCALE
            gy1 = min(a.H, (y1 + dy)) // SCALE
            if gx1 > gx0 and gy1 > gy0:
                m[gy0:gy1, gx0:gx1] = True
        return float(m.mean())

    print("交付了自己的手的帧 %d；有人脸框的帧 %d"
          % (sum(len(v) for v in delivered.values()),
             sum(len(v) for v in boxes.values())))
    print()
    print("%-18s %9s %9s %9s %9s %9s" % (
        "设置", "中位", "p90", "最大", ">=%.0f%% 的帧" % (100 * a.hot), "占比"))
    base = None
    for cap in caps:
        for pad in pads:
            vals, hot = [], 0
            for rec in recs:
                for f in sorted(delivered[rec]):
                    bs = boxes[rec].get(f, [])
                    if cap:
                        bs = [b for b in bs if b[4] <= cap]
                    v = frac(bs, pad)
                    vals.append(v)
                    hot += v >= a.hot
            vals.sort()
            n = len(vals)
            name = "pad %.2f%s" % (pad, "  cap %.2f" % cap if cap else "")
            if base is None:
                base = (vals[n // 2], hot)
            print("%-18s %9.4f %9.4f %9.4f %9d %8.1f%%" % (
                name, vals[n // 2], vals[int(n * .9)], vals[-1], hot,
                100.0 * hot / n))
    print()
    # what the cap would refuse, so the privacy side of the trade is visible
    for cap in caps:
        if not cap:
            continue
        kept = dropped = 0
        widths = []
        for rec in recs:
            for f, bs in boxes[rec].items():
                for b in bs:
                    if b[4] > cap:
                        dropped += 1
                        widths.append(b[4])
                    else:
                        kept += 1
        widths.sort()
        print("cap %.2f 会拒掉 %d 个打码框、保留 %d 个（%.1f%% 被拒）"
              % (cap, dropped, kept, 100.0 * dropped / max(1, dropped + kept)))
        if widths:
            print("   被拒的宽度：中位 %.3f  最小 %.3f  最大 %.3f"
                  % (widths[len(widths) // 2], widths[0], widths[-1]))


if __name__ == "__main__":
    main()
