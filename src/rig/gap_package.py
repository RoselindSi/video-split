"""The mined gaps, cleaned of the miner's own artefacts, as a labelling package.

WHAT IS ASKED. One question per frame: is the green box on the wearer's hand?
The frames were chosen because a track the pipeline called the wearer's has a
hole in it, and the green box is where the hand has to be -- interpolated from
the frames on either side. A yes turns into a training box for the detector;
a no says the interpolation, or the track, was wrong.

THE BOX THAT BECOMES THE LABEL IS NOT ALWAYS THE GREEN ONE. Where the detector
did propose something at 0.05-0.25 (the `proposed` rows, two thirds of these),
its box is the better one to train on and the green box only says where to
look. Where it proposed nothing, the interpolation is all there is, and it is
approximate: good for a one-frame gap between two firm boxes, weaker as the
gap grows. The gap length is carried into the package so a training run can
weight or drop the long ones.

ARTEFACTS ARE DROPPED HERE RATHER THAN LABELLED. A gap that another track has
taken over is a rename, not a miss, and a "proposal" that belongs to another
track is a colleague's hand. `mine_gaps` no longer produces either, but the
frames already mined were produced before that, so the same two tests run
again over the index and the dump.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import shutil


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mine", default="/workspace/gapmine")
    ap.add_argument("--dump", action="append", default=None)
    ap.add_argument("--out", default="/workspace/handpkg_gaps")
    ap.add_argument("--crop_px", type=int, default=192)
    a = ap.parse_args()
    import cv2

    dumps = a.dump or sorted(glob.glob("/workspace/own_dump_b*.csv"))
    byframe = collections.defaultdict(list)
    for p in dumps:
        if "_s" in os.path.basename(p):
            continue                       # shard files, already merged
        for r in csv.DictReader(open(p, encoding="utf-8")):
            byframe[(r["rec"], int(r["frame"]))].append(
                (str(r["tid"]), [float(r[c]) for c in ("x0", "y0", "x1", "y1")]))

    rows, drop = [], collections.Counter()
    for p in sorted(glob.glob(os.path.join(a.mine, "index_*.csv"))):
        for r in csv.DictReader(open(p)):
            box = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
            here = byframe.get((r["rec"], int(r["frame"])), ())
            if any(t != r["tid"] and iou(box, b) >= 0.3 for t, b in here):
                drop["同一只手换了 track id"] += 1
                continue
            if r["kind"] == "proposed" and float(r["conf"]) >= 0.60:
                # The miner matched against every box in the frame, so a
                # confident "proposal" for a hand that is missing is almost
                # always the neighbouring hand. Measured: 16 of 16.
                drop["匹配到了别的手"] += 1
                continue
            rows.append(r)
    print(f"索引里 {sum(drop.values()) + len(rows)} 行，剔除 {dict(drop)}，留下 {len(rows)}")

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    out = []
    for r in rows:
        src = os.path.join(a.mine, r["image"])
        if not os.path.exists(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        # the mined image is 1200 wide; the boxes are in render pixels
        sx = W / 1600.0
        sy = H / 900.0
        x0, y0 = int(float(r["x0"]) * sx), int(float(r["y0"]) * sy)
        x1, y1 = int(float(r["x1"]) * sx), int(float(r["y1"]) * sy)
        stem = f"{r['rec']}_f{int(r['frame']):06d}_h{int(r['tid']):d}"
        side = max(x1 - x0, y1 - y0) * 2
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        crop = img[max(0, cy - side // 2):min(H, cy + side // 2),
                   max(0, cx - side // 2):min(W, cx + side // 2)]
        if crop.size:
            cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                        cv2.resize(crop, (a.crop_px, a.crop_px)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        shutil.copyfile(src, os.path.join(a.out, "context", stem + ".jpg"))
        out.append({"stem": stem, "frame": int(r["frame"]), "hand": int(r["tid"]),
                    "conf": r["conf"], "kind": r["kind"],
                    "w_frac": round((x1 - x0) / float(W), 5),
                    "h_frac": round((y1 - y0) / float(H), 5),
                    "cx_frac": round(cx / float(W), 5),
                    "cy_frac": round(cy / float(H), 5),
                    "model": "interpolated" if r["kind"] == "missing" else "detector",
                    "label": "", "label_mode": ""})
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    c = collections.Counter(r["kind"] for r in out)
    print(f"-> {a.out}: {len(out)} 帧  "
          f"（检测器提了但分数不够 {c['proposed']}，完全没提出 {c['missing']}）")
    print(f"   录像 {len({r['stem'].split('_f')[0] for r in out})} 段")


if __name__ == "__main__":
    main()
