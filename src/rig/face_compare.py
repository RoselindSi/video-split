"""Two face detectors on the same labelled frames, at each one's own scale.

WHY NOT ONE THRESHOLD FOR BOTH. SCRFD scores the same face 0.28 where
YOLOv8-face scores it 0.55; the boxes agree to an IoU of 0.78, so that is a
difference in calibration, not in what was found. Comparing both at 0.35 would
report the newer model as blind. Each is therefore swept over its own range
and the two are read off at matched operating points.

WHAT THE LABELS CAN AND CANNOT SAY. 1,464 crops have been labelled face or not
a face, and those boxes came from the SHIPPED detector, so:

    recall      answerable now. A labelled face is a box a human confirmed;
                whether a candidate finds it is a fact about the candidate.
    precision   NOT answerable for the candidate. Boxes it proposes that the
                shipped detector never did are unlabelled, and counting them
                as false would punish it for looking harder. They are counted
                and set aside for labelling.

So the output is: recall against known faces, and how many extra proposals
come with it. A model that finds more faces AND proposes fewer extras is
better without further work; anything else needs the extras labelled first.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def labelled(pkgs):
    """-> {context image: [items]}, ONE ENTRY PER FRAME, not per crop.

    EVERY LABELLED BOX HAS ITS OWN CONTEXT IMAGE, the same frame with that box
    drawn on it, so keying on the image path splits a frame into as many
    entries as it has labels: 247 of 1,100 frames here carry two or more. A
    candidate detector then gets run on that frame once per label, its
    proposals counted that many times, and a proposal matching a DIFFERENT
    label of the same frame is called unknown because that label is not in
    this entry. Measured on the first pass: 781 of 2,124 "unknown" boxes were
    the same box collected again.

    So the key is the frame -- recording and frame number off the stem -- one
    context image stands for it, and every label on it comes along.

    Keyed on (stem, box, score) rather than on the stem alone: the packages
    were harvested at different times and `h0` in one is not `h0` in another,
    so 267 stems name two different boxes. Merging them by name silently
    merges two detections into one."""
    import re
    stem_re = re.compile(r"^(.*)_f(\d{6})_h\d+$")
    out, seen, frame_ctx = collections.defaultdict(list), set(), {}
    for d in pkgs:
        for r in csv.DictReader(open(os.path.join(d, "hands.csv"))):
            if r["label"] not in ("owner", "other"):
                continue
            ctx = os.path.join(d, "context", r["stem"] + ".jpg")
            if not os.path.exists(ctx):
                continue
            m = stem_re.match(r["stem"])
            # THE PACKAGE IS NOT PART OF THE FRAME'S IDENTITY. Keying on it
            # too makes the same recording and frame, harvested into two
            # packages, two frames: the detector runs twice on it, its
            # proposals are collected twice, and a downstream package built
            # from them writes both under one name and loses 162 of 901.
            frame = (m.group(1), m.group(2)) if m else (r["stem"], "")
            ctx = frame_ctx.setdefault(frame, ctx)      # one image per frame
            key = (r["stem"], round(float(r["w_frac"]), 4), round(float(r["conf"]), 3))
            if key in seen:
                continue
            seen.add(key)
            out[ctx].append({"stem": r["stem"], "is_face": r["label"] == "owner",
                             "cx": float(r["cx_frac"]), "cy": float(r["cy_frac"]),
                             "w": float(r["w_frac"]), "h": float(r["h_frac"])})
    return out


def box_of(it, W, H):
    """The labelled box back in pixels. The package stores it as fractions of
    the frame, so nothing has to be guessed from width alone."""
    cx, cy = it["cx"] * W, it["cy"] * H
    w, h = it["w"] * W, it["h"] * H
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", default=None)
    ap.add_argument("--model", action="append", required=True,
                    help="path to a detector; repeatable")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="scan floor, well under any working point")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="frames, 0 = all")
    ap.add_argument("--out", default="/workspace/face_compare.json")
    a = ap.parse_args()
    import cv2
    from src.rig import face_mask

    pkgs = a.pkg or sorted(glob.glob("/workspace/facepkg_*"))
    frames = labelled(pkgs)
    keys = sorted(frames)
    if a.limit:
        keys = keys[:a.limit]
    n_face = sum(1 for k in keys for it in frames[k] if it["is_face"])
    print(f"{len(keys)} 张帧，已标注候选 {sum(len(frames[k]) for k in keys)}（其中真脸 {n_face}）")

    report = {}
    for path in a.model:
        det = face_mask.load_detector(path, a.floor)
        if det is None:
            print(f"  !! 打不开 {path}")
            continue
        name = os.path.basename(path)
        hits, extra_scores, miss = [], [], 0
        for k, ctx in enumerate(keys):
            img = cv2.imread(ctx)
            if img is None:
                continue
            props = face_mask.detect_faces(det, img)
            used = set()
            H, W = img.shape[:2]
            for it in frames[ctx]:
                if not it["is_face"]:
                    continue                      # recall is about real faces
                gb = box_of(it, W, H)
                best, bi = 0.0, None
                for i, p in enumerate(props):
                    if i in used:
                        continue
                    v = iou(gb, list(p[:4]))
                    if v > best:
                        best, bi = v, i
                if bi is not None and best >= a.match_iou:
                    used.add(bi)
                    hits.append(props[bi][4])
                else:
                    miss += 1
            extra_scores += [p[4] for i, p in enumerate(props) if i not in used]
            if (k + 1) % 200 == 0:
                print(f"    [{k + 1}/{len(keys)}] {name}", flush=True)
        report[name] = {"recalled": len(hits), "missed": miss,
                        "extra": len(extra_scores),
                        "hit_scores": sorted(hits), "extra_scores": sorted(extra_scores)}
        print(f"  {name}: 找到已知真脸 {len(hits)}/{len(hits) + miss}"
              f"（{len(hits) / max(1, len(hits) + miss):.1%}），额外候选 {len(extra_scores)}")
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    # matched operating points: for each model, the threshold that keeps the
    # same number of known faces, and what it costs in extras
    for name, r in report.items():
        h = r["hit_scores"]
        if not h:
            continue
        print(f"\n{name}  已知真脸 {len(h) + r['missed']} 张")
        print(f"  {'阈值':>6}{'找到':>7}{'召回':>8}{'额外候选':>9}")
        for t in (0.10, 0.20, 0.30, 0.35, 0.45, 0.60):
            kept = sum(1 for s in h if s >= t)
            ex = sum(1 for s in r["extra_scores"] if s >= t)
            print(f"  {t:>6.2f}{kept:>7}{kept / (len(h) + r['missed']):>8.1%}{ex:>9}")


if __name__ == "__main__":
    main()
