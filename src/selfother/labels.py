"""Zone-derived self/other labels for the detections a dump already holds.

THE BOXES ARE BORROWED, NOT THE VERDICTS. Every arm sees the same hands V1
saw, so a difference in results can only come from how ownership is decided.
A dump row also carries V1's outputs; this file reads the frame, the track id
and the box, and `KEEP` is the whole of what crosses over.

THROUGH THE RENDER'S OWN TABLE. Zones were drawn on raw cam3; detections sit
on the 0.6 m wide render. `source_maps(rig, "cam3", vcam, 0.6)` is the table
that render sampled cam3 with, so a panorama point pushed through it lands on
the cam3 pixel its content came from, and no depth is guessed on the way.

OWNERSHIP IS DECIDED PER TRACK. A hand does not change owner mid-track and the
boundary is where box-level labels are noisiest, so each track takes the mean
of its boxes' on-side fraction and every box in it inherits the verdict.
Tracks whose mean sits on the boundary are dropped, not guessed. The mean uses
only boxes whose centre cam3 actually rendered, because elsewhere the lookup
carries the render's parallax -- but once a track is labelled, all its boxes
keep the label, since whose hand it is does not depend on which module drew
that frame.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

import numpy as np

from src.rig.zone_stability import Eye, S as ZONE_RASTER

GRID = 5
DEPTH_M = 0.6              # what own_dump rendered with; the table must match
AMBIGUOUS = (0.2, 0.8)     # track mean strictly inside this is dropped
KEEP = ("rec", "frame", "tid", "x0", "y0", "x1", "y1")
COLUMNS = KEEP + ("pano_w", "pano_h", "zone_frac", "zone_n", "center_cam3",
                  "track_zone_frac", "label", "drop_reason")


def load_clips(paths):
    """-> {recording tag: (databag path, start frame)}"""
    out = {}
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bag, start, _end = line.rsplit(":", 2)
            tag = os.path.basename(bag.rstrip("/")).replace("databag-26_", "R")
            out[tag] = (bag, int(start))
    return out


def page_meta(pages, rec):
    for d in pages:
        p = os.path.join(d, f"zone_{rec}.html")
        if os.path.exists(p):
            head = open(p, encoding="utf-8").read(9000)
            m = re.search(r"const M = (\{.*?\});", head, re.S)
            if m:
                return json.loads(m.group(1))
    return None


def latest_zone(ann, rec, who):
    """The newest final for this recording that has a curve in cam3."""
    pattern = os.path.join(ann, "*", f"{rec}__{who}__final_*.json")
    for p in sorted(glob.glob(pattern), reverse=True):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("eyes", {}).get("cam3", {}).get("keyframes"):
            return d
    return None


def cam3_table(databag):
    """-> (map_x, map_y, valid, cam3_owns_pixel, width, height)

    The ownership map comes from running the renderer itself on blank frames,
    not from re-deriving its selection rule, so it cannot drift from what
    own_dump's panorama actually was."""
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.render_wide import render
    rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
    vcam = VirtualWideCamera.from_rig(rig)
    mx, my, ok = source_maps(rig, "cam3", vcam, DEPTH_M)
    blank = {m.left.name: np.zeros((rig.cameras[m.left.name].height,
                                    rig.cameras[m.left.name].width, 3),
                                   np.uint8)
             for m in rig.modules}
    _, owner, _, _ = render(rig, vcam, blank, DEPTH_M, colour_match=False)
    mid = len(rig.modules) // 2
    return mx, my, ok, owner == mid, vcam.width, vcam.height


def on_side(box, mask, table):
    """Share of a 5x5 grid over the box that lands on the wearer's side.

    -> (fraction or None, number of grid points cam3 can see)"""
    mx, my, ok, _owns, W, H = table
    x0, y0, x1, y1 = box
    hit = n = 0
    for v in np.linspace(y0, y1, GRID):
        for u in np.linspace(x0, x1, GRID):
            iu = min(max(int(round(u)), 0), W - 1)
            iv = min(max(int(round(v)), 0), H - 1)
            if not ok[iv, iu]:
                continue
            r = min(max(int(my[iv, iu] / ZONE_RASTER), 0), mask.shape[0] - 1)
            c = min(max(int(mx[iv, iu] / ZONE_RASTER), 0), mask.shape[1] - 1)
            hit += int(mask[r, c])
            n += 1
    return (hit / n if n else None), n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="own_dump csv (boxes only)")
    ap.add_argument("--clips", action="append", required=True)
    ap.add_argument("--ann", required=True, help="zone annotations root")
    ap.add_argument("--pages", action="append", required=True,
                    help="zone page directories (for clip start/length)")
    ap.add_argument("--who", default="roselindsi")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clips = load_clips(a.clips)
    by_rec = collections.defaultdict(list)
    for r in csv.DictReader(open(a.dump, encoding="utf-8")):
        by_rec[r["rec"]].append({k: r[k] for k in KEEP})

    out = []
    for rec in sorted(by_rec):
        z, meta = latest_zone(a.ann, rec, a.who), page_meta(a.pages, rec)
        if z is None or meta is None or rec not in clips:
            reason = ("no_zone" if z is None else
                      "no_page" if meta is None else "no_databag")
            out.extend(dict(r, drop_reason=reason) for r in by_rec[rec])
            print(f"  {rec}: 跳过 ({reason})", flush=True)
            continue
        start = int(meta["start"])
        end = start + int(round(meta["dur"] * meta["fps"]))
        eye = Eye(z["eyes"]["cam3"]["keyframes"])
        table = cam3_table(clips[rec][0])
        W, H = table[4], table[5]
        at, mask = None, None
        inside = []
        for r in sorted(by_rec[rec], key=lambda r: int(r["frame"])):
            f = int(r["frame"])
            row = dict(r, pano_w=W, pano_h=H)
            if not start <= f < end:
                out.append(dict(row, drop_reason="outside_zone_clip"))
                continue
            if f != at:
                at, mask = f, eye.mask(f)
            box = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
            frac, n = on_side(box, mask, table)
            cu = min(max(int(round((box[0] + box[2]) / 2)), 0), W - 1)
            cv_ = min(max(int(round((box[1] + box[3]) / 2)), 0), H - 1)
            row.update(zone_frac="" if frac is None else round(frac, 4),
                       zone_n=n, center_cam3=int(table[3][cv_, cu]))
            inside.append(row)

        tracks = collections.defaultdict(list)
        for row in inside:
            tracks[row["tid"]].append(row)
        for rows in tracks.values():
            evidence = [float(x["zone_frac"]) for x in rows
                        if x["center_cam3"] and x["zone_frac"] != ""]
            if not evidence:
                mean, label, reason = "", "", "no_cam3_evidence"
            else:
                mean = float(np.mean(evidence))
                if AMBIGUOUS[0] < mean < AMBIGUOUS[1]:
                    label, reason = "", "ambiguous_track"
                else:
                    label, reason = int(mean >= 0.5), ""
            for x in rows:
                out.append(dict(x, label=label, drop_reason=reason,
                                track_zone_frac=("" if mean == ""
                                                 else round(mean, 4))))
        print(f"  {rec}: {len(inside)} 个检测在区域片段里, "
              f"{len(tracks)} 条轨迹", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(out)

    kept = [r for r in out if r.get("label") in (0, 1)]
    why = collections.Counter(r.get("drop_reason") or "kept" for r in out)
    trk = collections.Counter()
    for rows in _group(out).values():
        judged = [r for r in rows if r.get("drop_reason") != "outside_zone_clip"]
        r0 = (judged or rows)[0]
        trk[r0.get("drop_reason") or f"label_{r0.get('label')}"] += 1
    print(f"\n{len(out)} 个检测 -> {a.out}")
    print(f"  有标签 {len(kept)}，其中自己的手 "
          f"{np.mean([r['label'] for r in kept]):.1%}" if kept else "  无标签")
    print("  检测按原因:", dict(why))
    print("  轨迹按原因:", dict(trk))


def _group(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["rec"], r["tid"])].append(r)
    return g


if __name__ == "__main__":
    main()
