"""The track itself, not the hole in it.

WHY THE GAP LABELS CANNOT SETTLE THIS. What was labelled is a position where a
track has no detection, and a person saying "no hand there" is saying the hand
was absent or the interpolation missed -- not that the track is a false one. A
track whose every mined gap reads `not a hand` is a CANDIDATE for being a
false track, and 27 of 144 tracks hold 92 of the 116 such labels, but the
frames where the track does have a box have never been looked at.

So this renders those frames: for each track named, a strip of the frames it
was actually detected in, its box drawn, so the question "is this a hand at
all" can be answered about the thing itself. The same strip settles the other
open case, a track a person called a colleague's hand while the classifier
called it the wearer's at p = 1.00 and the pipeline left it uncovered.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os


def clip_index():
    """-> {recording: (databag, start frame)} from every batch list on disk."""
    out = {}
    for p in glob.glob("/workspace/zonestereo*/batch*.txt"):
        if not os.path.basename(p)[5:-4].isdigit():
            continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(":")
            rec = os.path.basename(f[0]).replace("databag-26_", "R")
            out.setdefault(rec, (f[0], int(f[1]) if len(f) > 1 else 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", action="append", required=True,
                    help="recording:tid, repeatable")
    ap.add_argument("--per_track", type=int, default=4, help="frames to render")
    ap.add_argument("--crop", action="store_true",
                    help="crop to the box. Off by default: an ownership question "
                         "cannot be answered inside the box")
    ap.add_argument("--out", default="/workspace/track_audit")
    a = ap.parse_args()
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader

    dump = collections.defaultdict(list)
    for p in sorted(glob.glob("/workspace/own_dump_b*.csv")):
        if "_s" in os.path.basename(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            dump[(r["rec"], str(r["tid"]))].append(r)
    clips = clip_index()
    os.makedirs(a.out, exist_ok=True)
    want = collections.defaultdict(list)
    for spec in a.track:
        rec, tid = spec.split(":")
        rows = sorted(dump.get((rec, tid), []), key=lambda r: int(r["frame"]))
        if not rows or rec not in clips:
            print(f"  !! {spec}: 没有 dump 或没有片段定义")
            continue
        step = max(1, len(rows) // a.per_track)
        for r in rows[::step][:a.per_track]:
            want[rec].append((tid, int(r["frame"]), r))
    crop = a.crop
    for rec, items in want.items():
        bag, _start = clips[rec]
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        mc = {}
        for tid, f, r in sorted(items, key=lambda x: x[1]):
            rd = ClipReader(rig, vids, f)
            src = rd.next()
            if src:
                try:
                    rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
                except TypeError:
                    rgb, _, _, _ = render(rig, vcam, src, 0.6)
                x0, y0, x1, y1 = (int(float(r[c])) for c in ("x0", "y0", "x1", "y1"))
                vis = rgb.copy()
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 3)
                own = "self" if r["final_owner_post_cap"] == "1" else "other"
                cv2.putText(vis, f"{rec} tid {tid} f{f}  p={float(r['p_owner_raw']):.2f} {own}",
                            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
                            cv2.LINE_AA)
                # THE WHOLE FRAME, NOT A CROP OF THE BOX. Whether a hand is
                # the wearer's is decided by where its forearm goes and
                # whether a body is attached to it, and both live outside the
                # box by construction -- `label_tool` says so and this cropped
                # anyway, then could not answer the question it was built for.
                if not crop:
                    cv2.imwrite(os.path.join(a.out, f"{rec}_t{tid}_f{f:06d}.jpg"),
                                vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                else:
                    pad = max(120, (x1 - x0))
                    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                    cx1, cy1 = min(rgb.shape[1], x1 + pad), min(rgb.shape[0], y1 + pad)
                    cv2.imwrite(os.path.join(a.out, f"{rec}_t{tid}_f{f:06d}.jpg"),
                                vis[cy0:cy1, cx0:cx1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rd.close()
        print(f"  {rec}: {len(items)} 帧", flush=True)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
