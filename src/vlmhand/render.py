"""Render the frames the model will judge, with every detected hand numbered.

TWO SOURCES OF FRAMES.
  --pkg   V1's labelled packages. One query per labelled frame; the boxes are
          every hand the extractor wrote for that frame, whatever its label,
          because the model has to choose among all of them. Frames come
          from the render the packages were cut from (checked to 1.00
          correlation at the labelled frame).
  --clip  a stretch of a detection dump, every frame, boxes from the dump --
          for the video.

THE BOX IS DRAWN, THE CLEAN FRAME IS KEPT. The numbered image is what the
model sees; the unmarked frame is written beside it so the video can show
only the wearer's boxes without the numbers the model was asked about.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re

import numpy as np

COLOURS = [(0, 220, 255), (255, 80, 80), (80, 255, 80), (255, 0, 255),
           (0, 140, 255), (255, 255, 0), (180, 120, 255), (120, 255, 200)]
STEM_RE = re.compile(r"^(.*?_)f(\d+)_h(\d+)$")


def draw_numbered(img, boxes):
    import cv2
    out = img.copy()
    for b in boxes:
        c = COLOURS[(b["n"] - 1) % len(COLOURS)]
        x0, y0, x1, y1 = (int(round(v)) for v in (b["x0"], b["y0"], b["x1"], b["y1"]))
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 3)
        label = str(b["n"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ty = y0 - 6 if y0 - th - 10 > 0 else y1 + th + 6
        cv2.rectangle(out, (x0, ty - th - 6), (x0 + tw + 8, ty + 4), c, -1)
        cv2.putText(out, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


def pkg_queries(pkgs):
    """-> {(databag, frame): [box]} with every hand the package wrote there."""
    q = collections.defaultdict(list)
    for pkg in pkgs:
        name = os.path.basename(pkg.rstrip("/"))
        bags = {r["tag"]: r["databag"]
                for r in csv.DictReader(open(os.path.join(pkg, "sources.csv"), encoding="utf-8-sig"))}
        for r in csv.DictReader(open(os.path.join(pkg, "hands.csv"), encoding="utf-8-sig")):
            m = STEM_RE.match(r.get("stem", ""))
            if not m or not r.get("frame") or not r.get("box_cx"):
                continue
            bag = bags.get(m.group(1))
            if not bag:
                continue
            q[(bag, int(r["frame"]))].append(
                {"stem": r["stem"], "pkg": name, "label": r.get("label", ""),
                 "cx": float(r["box_cx"]), "cy": float(r["box_cy"]),
                 "w": float(r["box_w"]), "h": float(r["box_h"])})
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", default=[])
    ap.add_argument("--clip", help="rec:start:n from --dump")
    ap.add_argument("--dump")
    ap.add_argument("--clips", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    import cv2
    cv2.setNumThreads(2)
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    os.makedirs(os.path.join(a.out, "numbered"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "clean"), exist_ok=True)
    manifest = open(os.path.join(a.out, f"manifest_{a.shard}.jsonl"), "w")

    if a.pkg:
        q = pkg_queries(a.pkg)
        by_bag = collections.defaultdict(list)
        for (bag, f), boxes in q.items():
            by_bag[bag].append((f, boxes))
        for bag in sorted(by_bag)[a.shard::a.nshards]:
            cal = os.path.join(bag, "calibration.yaml")
            if not os.path.exists(cal):
                print(f"  跳过（无标定）{bag}", flush=True)
                continue
            rig = RigCalibration(cal)
            vcam = VirtualWideCamera.from_rig(rig)
            vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
            cache = {}
            # READ FORWARD, DON'T REOPEN. Package frames sit 15-30 frames
            # apart; a fresh seek per frame decodes from the previous keyframe
            # across three 1520p streams every time (about 30 s a frame here),
            # while grabbing forward from the last frame decodes only the gap.
            # A far jump still reopens, since grabbing thousands of frames
            # costs more than one seek.
            rd, at = None, None
            for f, boxes in sorted(by_bag[bag]):
                if rd is None or at is None or not (0 <= f - at - 1 <= 600):
                    if rd is not None:
                        rd.close()
                    rd = ClipReader(rig, vids, f)
                    src = rd.next()
                else:
                    src = rd.next(skip=f - at - 1)
                at = f
                if not src:
                    rd.close()
                    raise SystemExit(f"{bag}: 第 {f} 帧读不到")
                rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                H, W = rgb.shape[:2]
                items = []
                for n, b in enumerate(sorted(boxes, key=lambda b: b["stem"]), 1):
                    items.append(dict(b, n=n, x0=(b["cx"] - b["w"] / 2) * W,
                                      y0=(b["cy"] - b["h"] / 2) * H,
                                      x1=(b["cx"] + b["w"] / 2) * W,
                                      y1=(b["cy"] + b["h"] / 2) * H))
                qid = f"{os.path.basename(bag)}_f{f:06d}"
                img = os.path.join(a.out, "numbered", qid + ".jpg")
                cv2.imwrite(img, draw_numbered(rgb, items), [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.write(json.dumps({"id": qid, "image": img, "frame": f,
                                           "databag": bag, "boxes": items}) + "\n")
            if rd is not None:
                rd.close()
            manifest.flush()
            print(f"  {os.path.basename(bag)}: {len(by_bag[bag])} 帧", flush=True)

    if a.clip:
        from src.selfother.labels import load_clips
        rec, start, n = a.clip.split(":")
        start, n = int(start), int(n)
        bag, clip_start = load_clips(a.clips)[rec]
        dets = collections.defaultdict(list)
        for r in csv.DictReader(open(a.dump, encoding="utf-8")):
            if r["rec"] == rec and start <= int(r["frame"]) < start + n:
                dets[int(r["frame"])].append(r)
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        rd = ClipReader(rig, vids, start)
        cache = {}
        try:
            for k in range(n):
                src = rd.next()
                if not src:
                    break
                f = start + k
                rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                items = [{"n": i, "tid": r["tid"], "x0": float(r["x0"]), "y0": float(r["y0"]),
                          "x1": float(r["x1"]), "y1": float(r["y1"])}
                         for i, r in enumerate(sorted(dets.get(f, []), key=lambda r: int(r["tid"])), 1)]
                qid = f"{rec}_f{f:06d}"
                clean = os.path.join(a.out, "clean", qid + ".jpg")
                cv2.imwrite(clean, rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                img = os.path.join(a.out, "numbered", qid + ".jpg")
                cv2.imwrite(img, draw_numbered(rgb, items), [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.write(json.dumps({"id": qid, "image": img, "clean": clean, "frame": f,
                                           "rec": rec, "boxes": items}) + "\n")
        finally:
            rd.close()
        print(f"  {rec}: {k + 1} 帧", flush=True)
    manifest.close()


if __name__ == "__main__":
    main()
