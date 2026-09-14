"""Cut every arm's input from the rectified cam3|cam4 stereo pair, once.

THE INPUT IS THE STEREO PAIR, NOT THE PANORAMA. Both eyes of the middle module
are stereo-rectified onto one perspective canvas by
`stabstitch_pair.stereo_rectify_maps` -- the same maps `rectified_pair.mp4` is
written with -- and each input is the cam3 crop and the cam4 crop side by
side, 128 x 256.

THE SAME WINDOW IN BOTH EYES, ON PURPOSE. Rectification puts a point on the
same row in both images and offsets it horizontally by its disparity, so a
window cut at identical coordinates shows a near hand displaced between the
halves and a far one barely moved. The wearer's hands are nearer than a
colleague's; that offset is a depth cue the network can read straight from
the pixels, and no matcher is inserted that could get it wrong.

THE BOX CROSSES OVER THROUGH TABLES, NOT GUESSES. A detection lives on the
0.6 m wide render. Its 5x5 grid goes through the render's own cam3 sampling
table to raw cam3 pixels, then through the fisheye undistortion with cam3's
rectifying rotation to the canvas; the bounding box of those points is the
hand's box in the pair. Every recording reports the round trip back through
the rectification map, so a wrong rotation shows up as pixels of error rather
than as a quietly misplaced crop.

TWO SCALES, AS V1 CUT THEM. `A` is the 2.5x window cut from the eye shrunk to
900 px wide; `B` is the box padded by 0.6 of its longer side. The arithmetic
is V1's, re-implemented rather than imported (a test in `tests/` checks the
pixels match). V1's trunks were fitted to panorama crops, not rectified
pairs, so for the V1-initialised arms this is their learned filters on a new
input, and the comparison reads that way.

WIDER THAN `rectified_pair.mp4`, AND MEASURED TO BE. StabStitch's defaults
(960x720, fov_scale 0.8) trim the fisheye's edges, and on three recordings
that trim removed 30% of FOREIGN hands against 5% of the wearer's --
colleagues stand to the sides. 1280x960 at fov_scale 1.2 keeps every wearer
hand and 81% of foreign ones at an unchanged box resolution. The rest sit in
the side modules' field, which a cam3|cam4 input cannot see at any setting.

WHAT FALLS OUTSIDE IS COUNTED, NOT HIDDEN. A hand whose box maps off the
canvas has no input and gets `outside_rectified`; evaluation drops it from
every row alike and reports, by class, what that cost and how V1 did on it.
The rectification settings are written to the split's manifest, and a second
run with different settings is refused rather than mixed in.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

CTX_SCALE = 2.5
HAND_PAD = 0.6
PX = 128
CONTEXT_STORE_W = 900
GRID = 5
PAIR = "cam34"
RECT_SIZE = (1280, 960)
RECT_FOV_SCALE = 1.2
RECT_BALANCE = 0.0
INDEX_COLUMNS = ("rec", "frame", "tid", "x0", "y0", "x1", "y1", "rx0", "ry0",
                 "rx1", "ry1", "label", "status", "a_fallback", "a_path",
                 "b_path")


def hand_crop(img, box, pad=HAND_PAD, px=PX):
    """Input B for one eye: the box padded by `pad` of its longer side."""
    import cv2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    p = int(max(x1 - x0, y1 - y0) * pad)
    cx0, cy0 = max(0, x0 - p), max(0, y0 - p)
    cx1, cy1 = min(W, x1 + p), min(H, y1 + p)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    return cv2.resize(img[cy0:cy1, cx0:cx1], (px, px))


def shrink(img, width=CONTEXT_STORE_W):
    """The eye at the width V1's context frames were stored at."""
    import cv2
    H, W = img.shape[:2]
    if W <= width:
        return img
    return cv2.resize(img, (width, int(width * H / W)),
                      interpolation=cv2.INTER_AREA)


def context_crop(small, box, full_w, full_h, scale=CTX_SCALE, px=PX):
    """Input A for one eye: `scale` times the box, cut from the shrunk eye.

    The box is normalised by the full-size image and the window cut from the
    shrunk one, clipped at the edge and stretched square -- as V1 does."""
    import cv2
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0 / full_w, (y0 + y1) / 2.0 / full_h
    bw, bh = (x1 - x0) / full_w, (y1 - y0) / full_h
    h, w = small.shape[:2]
    half = max(bw * w, bh * h) * scale / 2.0
    ax0, ay0 = int(max(0, cx * w - half)), int(max(0, cy * h - half))
    ax1, ay1 = int(min(w, cx * w + half)), int(min(h, cy * h + half))
    if ax1 <= ax0 or ay1 <= ay0:
        return None
    return cv2.resize(small[ay0:ay1, ax0:ax1], (px, px),
                      interpolation=cv2.INTER_AREA)


def subsample(rows, k):
    """Evenly spaced along the track; consecutive frames are near-duplicates."""
    if k <= 0 or len(rows) <= k:
        return rows
    idx = sorted(set(np.linspace(0, len(rows) - 1, k).round().astype(int)))
    return [rows[i] for i in idx]


def read_index(root, split):
    rows = []
    base = os.path.join(root, split)
    for rec in sorted(os.listdir(base)):
        if not os.path.isdir(os.path.join(base, rec)):
            continue
        p = os.path.join(base, rec, "index.csv")
        if os.path.exists(p):
            rows.extend(csv.DictReader(open(p, encoding="utf-8")))
    return rows


def rectifier(rig, size=RECT_SIZE, fov_scale=RECT_FOV_SCALE,
              balance=RECT_BALANCE):
    """`stabstitch_pair`'s cam34 rectification, plus cam3's rectifying pose.

    `stereo_rectify_maps` returns the remap grids and projections but not the
    rotation, which mapping a raw point INTO the canvas needs. It is
    recomputed with the same arguments and refused unless the projection it
    yields is the one the maps were built from."""
    import cv2
    from src.rig.stabstitch_pair import find_module, stereo_rectify_maps
    module = find_module(rig, PAIR)
    maps3, maps4, P1, _P2 = stereo_rectify_maps(
        rig, module, size=tuple(size), fov_scale=fov_scale, balance=balance)
    K1, D1, K2, D2, sensor, R, T = rig.stereo_pair(module)
    R1, _R2, P1b, _P2b, _Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, sensor, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=tuple(size), balance=float(balance),
        fov_scale=float(fov_scale))
    if not np.allclose(P1, P1b):
        raise SystemExit("重算的校正投影和 stereo_rectify_maps 的不一致")
    return {"maps3": maps3, "maps4": maps4, "K": K1, "D": D1, "R": R1,
            "P": P1, "size": tuple(size)}


def to_canvas(raw_pts, rect):
    """Raw cam3 pixels -> rectified canvas pixels."""
    import cv2
    pts = np.asarray(raw_pts, np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, rect["K"], rect["D"],
                                       R=rect["R"], P=rect["P"]).reshape(-1, 2)


def rectified_box(box, table, rect):
    """Panorama box -> its box on the rectified canvas, or None off-canvas."""
    mx, my, ok, _owns, W, H = table
    x0, y0, x1, y1 = box
    raw = []
    for v in np.linspace(y0, y1, GRID):
        for u in np.linspace(x0, x1, GRID):
            iu = min(max(int(round(u)), 0), W - 1)
            iv = min(max(int(round(v)), 0), H - 1)
            if ok[iv, iu]:
                raw.append((mx[iv, iu], my[iv, iu]))
    if len(raw) < 4:
        return None
    cw, ch = rect["size"]
    pts = to_canvas(raw, rect)
    # Points far past the canvas are where the undistortion stops being a
    # mapping at all; they would stretch the box across the whole image.
    keep = (np.isfinite(pts).all(1) & (pts[:, 0] > -cw) & (pts[:, 0] < 2 * cw)
            & (pts[:, 1] > -ch) & (pts[:, 1] < 2 * ch))
    pts = pts[keep]
    if len(pts) < 4:
        return None
    bx0, by0 = pts.min(0)
    bx1, by1 = pts.max(0)
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    if not (0 <= cx < cw and 0 <= cy < ch):
        return None
    return (int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1)))


def round_trip_px(table, rect, boxes):
    """Median error of raw -> canvas -> back through the remap grid."""
    mx, my, ok, _owns, W, H = table
    errs = []
    cw, ch = rect["size"]
    for x0, y0, x1, y1 in boxes[:200]:
        iu = min(max(int(round((x0 + x1) / 2)), 0), W - 1)
        iv = min(max(int(round((y0 + y1) / 2)), 0), H - 1)
        if not ok[iv, iu]:
            continue
        raw = np.array([mx[iv, iu], my[iv, iu]])
        c = to_canvas([raw], rect)[0]
        ci, cj = int(round(c[1])), int(round(c[0]))
        if not (0 <= ci < ch and 0 <= cj < cw):
            continue
        back = np.array([rect["maps3"][0][ci, cj], rect["maps3"][1][ci, cj]])
        errs.append(float(np.hypot(*(back - raw))))
    return float(np.median(errs)) if errs else float("nan")


def cut_recording(rec, rows, databag, start, out_dir, quality, rect_args):
    import cv2
    from src.rig.calibration import RigCalibration
    from src.rig.render_wide import split_halves
    from src.selfother.labels import cam3_table

    rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
    rect = rectifier(rig, **rect_args)
    table = cam3_table(databag)
    cw, ch = rect["size"]

    need = collections.defaultdict(list)
    index = []
    boxes = []
    for r in rows:
        box = tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1"))
        boxes.append(box)
        rb = rectified_box(box, table, rect)
        row = {c: r.get(c, "") for c in INDEX_COLUMNS}
        if rb is None:
            index.append(dict(row, status="outside_rectified"))
            continue
        need[int(r["frame"])].append(
            dict(row, rx0=rb[0], ry0=rb[1], rx1=rb[2], ry1=rb[3]))
    trip = round_trip_px(table, rect, boxes)

    if need:
        if min(need) < start:
            raise SystemExit(f"{rec}: 检测帧 {min(need)} 早于片段起点 {start}，"
                             f"clips 文件和 dump 不是同一次运行")
        for sub in ("A", "B"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(databag, f"{PAIR}.mp4"))
        if not cap.isOpened():
            raise SystemExit(f"{rec}: 打不开 {PAIR}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        try:
            for k in range(max(need) - start + 1):
                ok_read, frame = cap.read()
                if not ok_read:
                    break
                f = start + k
                if f not in need:
                    continue
                eye3, eye4 = split_halves(frame)
                r3 = cv2.remap(eye3, *rect["maps3"], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
                r4 = cv2.remap(eye4, *rect["maps4"], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
                s3, s4 = shrink(r3), shrink(r4)
                for row in need.pop(f):
                    rb = (row["rx0"], row["ry0"], row["rx1"], row["ry1"])
                    b3, b4 = hand_crop(r3, rb), hand_crop(r4, rb)
                    if b3 is None or b4 is None:
                        index.append(dict(row, status="hand_crop_too_small"))
                        continue
                    a3 = context_crop(s3, rb, cw, ch)
                    a4 = context_crop(s4, rb, cw, ch)
                    fallback = a3 is None or a4 is None
                    if fallback:              # V1 feeds the hand crop instead
                        a3 = cv2.resize(b3, (PX, PX), interpolation=cv2.INTER_AREA)
                        a4 = cv2.resize(b4, (PX, PX), interpolation=cv2.INTER_AREA)
                    stem = f"{f:06d}_t{int(row['tid'])}.jpg"
                    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    cv2.imwrite(os.path.join(out_dir, "A", stem),
                                np.concatenate([a3, a4], 1), params)
                    cv2.imwrite(os.path.join(out_dir, "B", stem),
                                np.concatenate([b3, b4], 1), params)
                    index.append(dict(row, status="ok",
                                      a_fallback=int(fallback),
                                      a_path=os.path.join(rec, "A", stem),
                                      b_path=os.path.join(rec, "B", stem)))
        finally:
            cap.release()
        for rs in need.values():              # the video ended early
            index.extend(dict(r, status="frame_not_decoded") for r in rs)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS, restval="")
        w.writeheader()
        w.writerows(index)
    return index, trip


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="labels.py output")
    ap.add_argument("--clips", action="append", required=True)
    ap.add_argument("--root", required=True,
                    help="output root on the big volume (df -h it first)")
    ap.add_argument("--split", required=True, help="e.g. train_e2e, test_fresh")
    ap.add_argument("--labelled_only", action="store_true",
                    help="training: only detections that carry a zone label")
    ap.add_argument("--per_track", type=int, default=0,
                    help="max frames per track, evenly spaced; 0 keeps all")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--rect_size", type=int, nargs=2, default=list(RECT_SIZE))
    ap.add_argument("--fov_scale", type=float, default=RECT_FOV_SCALE)
    ap.add_argument("--balance", type=float, default=RECT_BALANCE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    import json
    rect_args = {"size": tuple(a.rect_size), "fov_scale": a.fov_scale,
                 "balance": a.balance}
    manifest = os.path.join(a.root, a.split, "manifest.json")
    want = {"pair": PAIR, "rect_size": list(a.rect_size),
            "fov_scale": a.fov_scale, "balance": a.balance,
            "ctx_scale": CTX_SCALE, "hand_pad": HAND_PAD, "px": PX}
    if os.path.exists(manifest):
        have = json.load(open(manifest))
        if have != want:
            raise SystemExit(f"{manifest} 是用别的设置切的: {have}；"
                             f"不混用，换一个 --split")
    else:
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        json.dump(want, open(manifest, "w"), indent=1)

    from src.selfother.labels import load_clips
    clips = load_clips(a.clips)
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
    if a.labelled_only:
        rows = [r for r in rows if r["label"] in ("0", "1")]
    tracks = collections.defaultdict(list)
    for r in rows:
        tracks[(r["rec"], r["tid"])].append(r)
    by_rec = collections.defaultdict(list)
    for key, rs in tracks.items():
        rs.sort(key=lambda r: int(r["frame"]))
        by_rec[key[0]].extend(subsample(rs, a.per_track))

    for rec in sorted(by_rec)[a.shard::a.nshards]:
        out_dir = os.path.join(a.root, a.split, rec)
        if os.path.exists(os.path.join(out_dir, "index.csv")):
            print(f"  {rec}: 已有，跳过", flush=True)
            continue
        if rec not in clips:
            print(f"  {rec}: clips 里没有，跳过", flush=True)
            continue
        databag, start = clips[rec]
        idx, trip = cut_recording(rec, by_rec[rec], databag, start, out_dir,
                                  a.quality, rect_args)
        n = collections.Counter(r["status"] for r in idx)
        print(f"  {rec}: 切好 {n['ok']}/{len(idx)}  出了校正画布 "
              f"{n['outside_rectified']}  往返误差中位 {trip:.2f}px -> "
              f"{out_dir}", flush=True)


if __name__ == "__main__":
    main()
