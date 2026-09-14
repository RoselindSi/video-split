"""Cut V1's own labelled hands again, as rectified cam3|cam4 pairs.

THE ROWS ARE V1'S, CHOSEN BY V1'S LOADER. `own_ctx.load` decides which
labelled hands V1 trained on (label owner/other, crop and context on disk, all
geometry finite); calling it rather than re-deriving the rule is what makes
"the same rows" literal. Each row's databag comes from its package's
`sources.csv` and its frame from `hands.csv`.

THE BOX IS THE STORED ONE, IN TODAY'S PANORAMA. Boxes were saved normalised
by the 1600x900 render; multiplied back out they land on the same pixels --
checked by re-rendering sampled rows and correlating against the stored crops
(1.00 at the labelled frame, falling away by two frames either side) -- and
from there they cross to the rectified pair exactly as `selfother.crops` does.

SCATTERED FRAMES ARE SOUGHT, NOT DECODED THROUGH. A package samples a few
dozen frames spread over minutes of video; reading straight through would
decode tens of thousands of frames to use sixty. Frame-index seeking on these
files was the method the correlation check used, and it landed exactly.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?_)f(\d+)_h\d+$")
COLUMNS = ("stem", "tag", "pkg", "frame", "label", "status", "a_path",
           "b_path")


def v1_rows(pkg):
    """V1's own row set for one package, with databag and frame attached."""
    from src.rig import own_ctx
    rows = own_ctx.load([pkg], verbose=False)
    frames = {r["stem"]: r.get("frame", "")
              for r in csv.DictReader(open(os.path.join(pkg, "hands.csv"),
                                           encoding="utf-8-sig"))}
    bags = {r["tag"]: r["databag"]
            for r in csv.DictReader(open(os.path.join(pkg, "sources.csv"),
                                         encoding="utf-8-sig"))}
    out = []
    for r in rows:
        f = frames.get(r["stem"], "")
        bag = bags.get(r["tag"])
        out.append(dict(r, frame=int(f) if f else None, databag=bag))
    return out


def cut_databag(bag, rows, out_dir, pkg_name, quality=95):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.render_wide import split_halves
    from src.selfother.crops import (PX, context_crop, hand_crop, rectified_box,
                                     rectifier, shrink)
    from src.selfother.labels import cam3_table

    rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
    rect = rectifier(rig)
    table = cam3_table(bag)
    W, H = table[4], table[5]
    cw, ch = rect["size"]
    index = []
    need = collections.defaultdict(list)
    for r in rows:
        base = {"stem": r["stem"], "tag": r["tag"], "pkg": pkg_name,
                "frame": r["frame"], "label": r["y"]}
        cx, cy, bw, bh = r["box"]
        box = ((cx - bw / 2) * W, (cy - bh / 2) * H,
               (cx + bw / 2) * W, (cy + bh / 2) * H)
        rb = rectified_box(box, table, rect)
        if rb is None:
            index.append(dict(base, status="outside_rectified"))
            continue
        need[r["frame"]].append((base, rb))

    cap = cv2.VideoCapture(os.path.join(bag, "cam34.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"{bag}: 打不开 cam34.mp4")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for sub in ("A", "B"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    try:
        for f in sorted(need):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                if total and f < total - 1:
                    raise SystemExit(f"{bag}: 第 {f} 帧解码失败")
                index.extend(dict(b, status="frame_not_decoded")
                             for b, _ in need[f])
                continue
            if not frame[::64, ::64].any():
                raise SystemExit(f"{bag}: 第 {f} 帧全黑，拒绝写图")
            eye3, eye4 = split_halves(frame)
            r3 = cv2.remap(eye3, *rect["maps3"], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
            r4 = cv2.remap(eye4, *rect["maps4"], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
            s3, s4 = shrink(r3), shrink(r4)
            for base, rb in need[f]:
                b3, b4 = hand_crop(r3, rb), hand_crop(r4, rb)
                if b3 is None or b4 is None:
                    index.append(dict(base, status="hand_crop_too_small"))
                    continue
                a3, a4 = context_crop(s3, rb, cw, ch), context_crop(s4, rb, cw, ch)
                if a3 is None or a4 is None:     # V1 feeds the hand crop instead
                    a3 = cv2.resize(b3, (PX, PX), interpolation=cv2.INTER_AREA)
                    a4 = cv2.resize(b4, (PX, PX), interpolation=cv2.INTER_AREA)
                name = base["stem"] + ".jpg"
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                cv2.imwrite(os.path.join(out_dir, "A", name),
                            np.concatenate([a3, a4], 1), params)
                cv2.imwrite(os.path.join(out_dir, "B", name),
                            np.concatenate([b3, b4], 1), params)
                index.append(dict(base, status="ok",
                                  a_path=os.path.join(pkg_name, "A", name),
                                  b_path=os.path.join(pkg_name, "B", name)))
    finally:
        cap.release()
    return index


def read_rows(root, pkg_name):
    p = os.path.join(root, pkg_name, "rows.csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import cv2
    cv2.setNumThreads(2)

    for pkg in a.pkg:
        name = os.path.basename(pkg.rstrip("/"))
        out_dir = os.path.join(a.root, name)
        rows = v1_rows(pkg)
        missing = sum(1 for r in rows if r["frame"] is None or not r["databag"])
        by_bag = collections.defaultdict(list)
        for r in rows:
            if r["frame"] is not None and r["databag"]:
                by_bag[r["databag"]].append(r)
        index = []
        for bag in sorted(by_bag)[a.shard::a.nshards]:
            index.extend(cut_databag(bag, by_bag[bag], out_dir, name))
        part = os.path.join(out_dir, f"rows_{a.shard}.csv")
        os.makedirs(out_dir, exist_ok=True)
        with open(part, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
            w.writeheader()
            w.writerows(index)
        n = collections.Counter(r["status"] for r in index)
        print(f"  {name} shard {a.shard}: V1 行 {len(rows)}（无帧号/无 databag "
              f"{missing}），本分片 {len(index)}: {dict(n)}", flush=True)


def merge(root, pkg_name, nshards):
    """Shard files -> rows.csv, refusing to merge an incomplete set."""
    parts = [os.path.join(root, pkg_name, f"rows_{i}.csv") for i in range(nshards)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"缺分片: {missing}")
    rows = []
    for p in parts:
        rows.extend(csv.DictReader(open(p, encoding="utf-8")))
    with open(os.path.join(root, pkg_name, "rows.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
