"""How many people are actually in shot, counted rather than assumed.

Other-hand frames are the obvious proxy and they are the wrong one: two
people working fast produce more hand-frames than five people standing
still, and a fragmented track inflates the count again. So this counts HEADS,
with the head detector, on frames sampled evenly across each candidate, and
reports the median and the maximum per frame.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rec", action="append", required=True)
    ap.add_argument("--model", default="/shared/models/yolov5-crowdhuman/crowdhuman_yolov5m.pt")
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--n", type=int, default=12, help="frames sampled per recording")
    a = ap.parse_args()
    import cv2
    from src.rig import face_mask
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.track_audit import clip_index

    ext = collections.defaultdict(list)
    for p in sorted(glob.glob("/workspace/own_dump_b*.csv")):
        if "_s" in os.path.basename(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["rec"] in a.rec:
                ext[r["rec"]].append(int(r["frame"]))

    det = face_mask.load_detector(a.model, a.conf)
    clips = clip_index()
    out = []
    for rec in a.rec:
        if rec not in clips or not ext.get(rec):
            print(f"  !! {rec} 没有片段或没有 dump")
            continue
        bag, _ = clips[rec]
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        mc = {}
        f0, f1 = min(ext[rec]), max(ext[rec])
        frames = [int(v) for v in np.linspace(f0, f1, a.n)]
        counts = []
        for f in frames:
            rd = ClipReader(rig, vids, f)
            src = rd.next()
            rd.close()
            if not src:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            counts.append(len(face_mask.detect_faces(det, rgb)))
        if not counts:
            continue
        c = sorted(counts)
        out.append((c[len(c) // 2], max(c), sum(c) / len(c), rec, f0))
        print(f"  {rec}  每帧头数 中位 {c[len(c) // 2]}  最多 {max(c)}  "
              f"均值 {sum(c) / len(c):.1f}  (采样 {len(c)} 帧)", flush=True)
    print("\n按每帧头数中位排序：")
    for med, mx, mean, rec, f0 in sorted(out, reverse=True):
        print(f"  {rec}  中位 {med}  最多 {mx}  均值 {mean:.1f}  起始帧 {f0}")


if __name__ == "__main__":
    main()
