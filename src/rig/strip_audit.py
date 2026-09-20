"""A window of consecutive frames around a track, every box in each frame.

One frame cannot say whose hand it is. The question is answered by where the
arm goes over time and whether a body stays attached to it, so this renders a
run of frames as a contact sheet and draws EVERY track in each, not only the
one under audit -- the neighbouring boxes are what reveal an association swap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

import cv2
import numpy as np

from src.rig.calibration import RigCalibration
from src.rig.geometry import VirtualWideCamera
from src.rig.render_wide import render
from src.rig.seam_fix import ClipReader
from src.rig.track_audit import clip_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--tid", required=True, help="the track under audit, drawn in amber")
    ap.add_argument("--f0", type=int, required=True)
    ap.add_argument("--f1", type=int, required=True)
    ap.add_argument("--box", nargs=4, type=int, required=True,
                    help="crop x0 y0 x1 y1 in render pixels")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    by_frame = collections.defaultdict(list)
    for p in sorted(glob.glob("/workspace/own_dump_b*.csv")):
        if "_s" in os.path.basename(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["rec"] == a.rec:
                by_frame[int(r["frame"])].append(r)

    bag, _ = clip_index()[a.rec]
    rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
    vcam = VirtualWideCamera.from_rig(rig)
    vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    mc = {}
    cx0, cy0, cx1, cy1 = a.box
    tiles = []
    for f in range(a.f0, a.f1 + 1):
        rd = ClipReader(rig, vids, f)
        src = rd.next()
        rd.close()
        if not src:
            continue
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        vis = rgb.copy()
        for r in by_frame.get(f, []):
            x0, y0, x1, y1 = (int(float(r[c])) for c in ("x0", "y0", "x1", "y1"))
            mine = str(r["tid"]) == a.tid
            col = (0, 200, 255) if mine else (190, 190, 190)
            own = "self" if r["final_owner_post_cap"] == "1" else "OTHER"
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
            cv2.putText(vis, "t%s p=%.2f %s" % (r["tid"], float(r["p_owner_raw"]), own),
                        (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                        cv2.LINE_AA)
        t = vis[cy0:cy1, cx0:cx1].copy()
        cv2.putText(t, "f%d" % f, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)

    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.zeros((rows * h, a.cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r_, c_ = divmod(i, a.cols)
        sheet[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t
    cv2.imwrite(a.out, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(a.out, sheet.shape, len(tiles))


if __name__ == "__main__":
    main()
