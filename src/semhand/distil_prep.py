"""The unlabelled pool Qwen will label, and both views of every hand in it.

WHY A WIDE POOL AND NOT MORE FRAMES. The swap experiment showed the student
should read the whole frame, and that a whole-frame student trained on 1,013
hands from 84 recordings learns the workstation instead of the hand: on a day
it had never seen, 133 of its 141 foreign-hand errors came from one recording.
Whole-frame input therefore needs MANY SCENES, which is exactly what a teacher
that labels for free can buy. So the pool is wide and shallow: many
recordings, few frames each, spread over the recording rather than taken from
one window.

BLIND, AND DISJOINT FROM EVERY TEST. Recordings are drawn by seed from the
calibrated pool after removing V1's packages, testpkg_2, fresh29, e2e_main2
and every zone batch (4, 5, 6 -- the clean tests). No model score picks a
recording, a window or a hand.

WHAT IS WRITTEN, for each sampled frame and each detection in it:
    fresh/<rec>/frames/<frame>.jpg   the clean 0.6 m render, for Qwen's view
    fresh/<rec>/index.csv            one row per detection (Qwen reads this)
    cross/index_test_<shard>.csv     V1's own view of the same detection:
                                     hand 128, context 128, 14 geometry
                                     features (keypoints come from the same
                                     detection, so `hand_span` is real)
There is no tracker here: training needs a hand and a label, not identity, and
`tid` is the detection's index within its frame.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import re

import numpy as np

from src.semhand.crossprep import COLS as CROSS_COLS
from src.semhand.frames import FRESH_COLS, cap_decoder_threads
from src.semhand.pick_batch import bag_of, day_of, device_of, listed_bags

N_RECORDINGS = 300
WINDOWS = 4                  # windows spread over the recording
PER_WINDOW = 10              # frames per window
STRIDE = 10                  # frames between samples inside a window
MIN_FRAMES = 2000            # a recording must be long enough to spread over
SEED = 20260916


def pick_from_scans(a):
    """A second pool, enriched for other people's hands, from scans already done.

    `pick_rich --mode scan` counted hand detections in 12 spread frames of 780
    candidate recordings while choosing zone batches 7-11. The recordings it
    scanned but no batch used are free, and the frames where it counted three
    or more hands are where other people are. Windows are centred on those
    frames, so the pool's foreign-hand share rises without any ownership model
    choosing anything."""
    import collections
    used = listed_bags(["/workspace/otherpkg_1/sources.csv",
                        "/workspace/crops/trainpkg_T1/sources.csv",
                        "/workspace/testpkg_2/sources.csv",
                        "/workspace/fresh29.txt", "/workspace/e2e_main2.txt"])
    used |= {bag_of(l.split(":")[0]) for p in glob.glob("/workspace/zonestereo*/batch*.txt")
             if re.fullmatch(r"batch\d+\.txt", os.path.basename(p))
             for l in open(p) if l.strip() and not l.startswith("#")}
    used |= {bag_of(r["databag"]) for r in json.load(open(a.pool))} if os.path.exists(a.pool) else set()
    scanned = {}
    for d in a.scans.split(","):
        for p in glob.glob(os.path.join(d, "scan_*.jsonl")):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    scanned[r["databag"]] = r
    out = []
    for path, r in sorted(scanned.items()):
        if bag_of(path) in used or r.get("status") != "ok" or not r.get("counts"):
            continue
        hot = sorted(((int(f), c) for f, c in r["counts"].items()), key=lambda t: -t[1])
        if hot[0][1] < 3:
            continue
        frames = set()
        for f, c in hot[:WINDOWS]:
            if c < 2:
                break
            for k in range(-PER_WINDOW // 2, PER_WINDOW // 2):
                g = f + k * STRIDE
                if 0 <= g < r["n_frames"] - 1:
                    frames.add(g)
        out.append({"databag": path, "rec": bag_of(path).replace("databag-26_", "R"),
                    "frames": sorted(frames)})
    json.dump(out, open(a.out_pool, "w"), indent=1)
    days = collections.Counter(day_of(bag_of(r["databag"])) for r in out)
    print(f"-> {a.out_pool}  {len(out)} 段（扫描过 {len(scanned)}，排除已用 {len(used)}），"
          f"{len({device_of(r['databag']) for r in out})} 台设备，{len(days)} 天，"
          f"共 {sum(len(r['frames']) for r in out)} 帧")


def pick(a):
    used = listed_bags(["/workspace/otherpkg_1/sources.csv",
                        "/workspace/crops/trainpkg_T1/sources.csv",
                        "/workspace/testpkg_2/sources.csv",
                        "/workspace/fresh29.txt", "/workspace/e2e_main2.txt"])
    used |= {os.path.basename(h)[5:-5].replace("R", "databag-26_", 1)
             for h in glob.glob("/workspace/zonestereo*/zone_*.html")}
    used |= listed_bags(sorted(glob.glob("/workspace/zonestereo*/batch*.txt")))
    pool = {}
    for line in open(a.pool_list):
        line = line.strip()
        if line and bag_of(line) not in used:
            pool.setdefault(bag_of(line), line.rstrip("/"))
    # Spread the draw over devices and days rather than taking whatever the
    # shuffle gives: the point of the pool is scene variety.
    by = collections.defaultdict(list)
    for bag, path in sorted(pool.items()):
        by[(device_of(path), day_of(bag))].append(path)
    keys = sorted(by)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    picked, r = [], 0
    while len(picked) < N_RECORDINGS and any(by[k] for k in keys):
        for k in keys:
            if len(picked) >= N_RECORDINGS:
                break
            if len(by[k]) > r:
                picked.append(by[k][r])
        r += 1
    out = []
    import cv2
    cv2.setNumThreads(2)
    for path in picked:
        cap = cv2.VideoCapture(os.path.join(path, "cam34.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n < MIN_FRAMES:
            continue
        span = WINDOWS * PER_WINDOW * STRIDE
        rng2 = random.Random(bag_of(path))
        starts = [rng2.randrange(int(n * i / WINDOWS), max(int(n * i / WINDOWS) + 1,
                                                          int(n * (i + 1) / WINDOWS) - span // WINDOWS))
                  for i in range(WINDOWS)]
        frames = sorted({s + k * STRIDE for s in starts for k in range(PER_WINDOW) if s + k * STRIDE < n - 1})
        out.append({"databag": path, "rec": bag_of(path).replace("databag-26_", "R"), "frames": frames})
    os.makedirs(os.path.dirname(a.pool), exist_ok=True)
    json.dump(out, open(a.pool, "w"), indent=1)
    days = collections.Counter(day_of(bag_of(r["databag"])) for r in out)
    devs = {device_of(r["databag"]) for r in out}
    print(f"-> {a.pool}  {len(out)} 段录像，{len(devs)} 台设备，{len(days)} 天，"
          f"共 {sum(len(r['frames']) for r in out)} 帧待渲染")
    print("  每天段数", dict(sorted(days.items())))


def prep(a):
    import cv2
    import torch
    from ultralytics import YOLO
    from src.rig import own_ctx
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.hand_detect import detect
    from src.rig.own_cnn import crop_of
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.semhand.crossprep import v1_views
    cap_decoder_threads()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    pool = json.load(open(a.pool))[a.shard::a.nshards]
    yolo = YOLO(a.weights)
    cross = []
    for item in pool:
        rec, bag = item["rec"], item["databag"]
        out_csv = os.path.join(a.out, "fresh", rec, "index.csv")
        cross_done = os.path.join(a.out, "cross", f"done_{rec}")
        if os.path.exists(out_csv) and os.path.exists(cross_done):
            continue
        os.makedirs(os.path.join(a.out, "fresh", rec, "frames"), exist_ok=True)
        os.makedirs(os.path.join(a.out, "cross", rec), exist_ok=True)
        index, mine = [], []
        try:
            rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
            vcam = VirtualWideCamera.from_rig(rig)
            vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
            cache, rd, at = {}, None, None
            for f in item["frames"]:
                if rd is None or not (0 <= f - at - 1 <= 300):
                    if rd is not None:
                        rd.close()
                    rd = ClipReader(rig, vids, f)
                    src = rd.next()
                else:
                    src = rd.next(skip=f - at - 1)
                at = f
                if not src:
                    raise SystemExit(f"{rec}: 第 {f} 帧读不到")
                rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                if not rgb[::32, ::32].any():
                    continue
                H, W = rgb.shape[:2]
                img = os.path.join(a.out, "fresh", rec, "frames", f"{f:06d}.jpg")
                cv2.imwrite(img, rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                for i, d in enumerate(detect(yolo, rgb, min_conf=0.25)):
                    x0, y0, x1, y1 = (float(v) for v in d["box"])
                    ident = f"{rec}|{f}|{i}"
                    index.append({"rec": rec, "frame": f, "tid": i, "databag": bag, "image": img,
                                  "x0": x0, "y0": y0, "x1": x1, "y1": y1, "W": W, "H": H,
                                  "side": d.get("side", ""), "p_dump": "", "status": "ok"})
                    if d.get("kp") is None:
                        cross.append({"id": ident, "kind": "test", "rec": rec, "status": "no_keypoints"})
                        continue
                    hand, ctx = v1_views(rgb, (x0, y0, x1, y1), own_ctx, crop_of, cv2)
                    if hand is None:
                        cross.append({"id": ident, "kind": "test", "rec": rec, "status": "hand_crop_too_small"})
                        continue
                    stem = ident.replace("|", "_")
                    hp = os.path.join(a.out, "cross", rec, stem + "_hand.png")
                    cp = os.path.join(a.out, "cross", rec, stem + "_ctx.png")
                    cv2.imwrite(hp, hand)
                    cv2.imwrite(cp, ctx)
                    g = own_ctx._geom_vector(d, rgb.shape)
                    cross.append({"id": ident, "kind": "test", "rec": rec, "hand": hp, "ctx": cp,
                                  "geom": json.dumps([round(float(x), 6) for x in g]), "status": "ok"})
            if rd is not None:
                rd.close()
        except (Exception, SystemExit) as e:
            print(f"  !! {rec} 失败 -- {type(e).__name__}: {e}", flush=True)
            continue
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FRESH_COLS, restval="")
            w.writeheader()
            w.writerows(index)
        open(cross_done, "w").close()
        print(f"  {rec}: {len(item['frames'])} 帧 / {len(index)} 只手", flush=True)
    path = os.path.join(a.out, "cross", f"index_test_{a.shard}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CROSS_COLS, restval="")
        w.writeheader()
        w.writerows(cross)
    print(f"-> {path}  {dict(collections.Counter(r['status'] for r in cross))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pool", "prep", "pool_from_scans"), required=True)
    ap.add_argument("--scans", default="/workspace/richscan,/workspace/richscan10")
    ap.add_argument("--out_pool", default="/workspace/distil2/pool.json")
    ap.add_argument("--out", default="/workspace/distil")
    ap.add_argument("--pool", default="/workspace/distil/pool.json")
    ap.add_argument("--pool_list", default="/workspace/calibrated.txt")
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    {"pool": pick, "pool_from_scans": pick_from_scans, "prep": prep}[a.mode](a)


if __name__ == "__main__":
    main()
