"""Is the track a hand at all -- asked of the frames the track actually has.

A gap label says nothing about the track: it says a person looked at a place
where the track had no detection and saw no hand there. Thirty tracks have
every one of their mined gaps labelled `not a hand`, which makes them
candidates for being false tracks and nothing more, because the frames where
they DO have a box have never been shown to anyone.

So each track gets a row: for every sampled frame, the whole render above and
a zoom on the box below. Both are needed and for different reasons -- the
zoom answers "is this skin, fingers, a knee, a floor tile", and the whole
frame answers "whose body is it attached to", which the zoom destroys by
construction.
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

TILE_W, TILE_H = 640, 360


def _tile(rgb, box, caption):
    x0, y0, x1, y1 = box
    wide = rgb.copy()
    cv2.rectangle(wide, (x0, y0), (x1, y1), (0, 200, 255), 4)
    wide = cv2.resize(wide, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

    pad = max(90, (x1 - x0), (y1 - y0))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(rgb.shape[1], x1 + pad), min(rgb.shape[0], y1 + pad)
    zoom = rgb[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(zoom, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 200, 255), 3)
    zoom = cv2.resize(zoom, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)

    t = np.zeros((TILE_H * 2, TILE_W, 3), np.uint8)
    t[:TILE_H] = wide
    t[TILE_H:] = zoom
    cv2.putText(t, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2,
                cv2.LINE_AA)
    cv2.line(t, (0, TILE_H), (TILE_W, TILE_H), (40, 40, 40), 2)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True,
                    help="a file of rec:tid lines, one track per line")
    ap.add_argument("--per_track", type=int, default=3)
    ap.add_argument("--rows", type=int, default=5, help="tracks per sheet")
    ap.add_argument("--out", default="/workspace/nothand_audit")
    a = ap.parse_args()

    dump = collections.defaultdict(list)
    for p in sorted(glob.glob("/workspace/own_dump_b*.csv")):
        if "_s" in os.path.basename(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            dump[(r["rec"], str(r["tid"]))].append(r)

    specs = [l.strip() for l in open(a.tracks) if l.strip() and not l.startswith("#")]
    clips = clip_index()
    os.makedirs(a.out, exist_ok=True)

    # Group by recording so each databag is opened once, but keep the track
    # order the caller asked for when the rows are laid out.
    per_rec = collections.defaultdict(list)
    for spec in specs:
        rec, tid = spec.split(":")
        per_rec[rec].append(tid)

    tiles_of = {}
    for rec, tids in per_rec.items():
        if rec not in clips:
            print("  !! %s: 没有片段定义" % rec, flush=True)
            continue
        bag, _ = clips[rec]
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, "%s.mp4" % k) for k in ("cam12", "cam34", "cam56")}
        mc = {}
        wanted = []
        for tid in tids:
            rows = sorted(dump.get((rec, tid), []), key=lambda r: int(r["frame"]))
            if not rows:
                print("  !! %s:%s 没有 dump" % (rec, tid), flush=True)
                continue
            step = max(1, len(rows) // a.per_track)
            for r in rows[::step][:a.per_track]:
                wanted.append((int(r["frame"]), tid, r, len(rows)))
        for f, tid, r, n in sorted(wanted):
            rd = ClipReader(rig, vids, f)
            src = rd.next()
            rd.close()
            if not src:
                continue
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            box = tuple(int(float(r[c])) for c in ("x0", "y0", "x1", "y1"))
            cap = "%s t%s f%d  %dfr  p=%.2f %s" % (
                rec, tid, f, n, float(r["p_owner_raw"]),
                "self" if r["final_owner_post_cap"] == "1" else "OTHER")
            tiles_of.setdefault((rec, tid), []).append(_tile(rgb, box, cap))
        print("  %s: %d 帧" % (rec, len(wanted)), flush=True)

    order = []
    for spec in specs:
        rec, tid = spec.split(":")
        if (rec, tid) in tiles_of:
            order.append((rec, tid))
    cols = a.per_track
    for s in range(0, len(order), a.rows):
        chunk = order[s:s + a.rows]
        sheet = np.zeros((len(chunk) * TILE_H * 2, cols * TILE_W, 3), np.uint8)
        for ri, k in enumerate(chunk):
            for ci, t in enumerate(tiles_of[k][:cols]):
                sheet[ri * TILE_H * 2:(ri + 1) * TILE_H * 2,
                      ci * TILE_W:(ci + 1) * TILE_W] = t
        p = os.path.join(a.out, "sheet_%02d.jpg" % (s // a.rows + 1))
        cv2.imwrite(p, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(p, sheet.shape, [":".join(k) for k in chunk], flush=True)


if __name__ == "__main__":
    main()
