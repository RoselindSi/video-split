"""Panorama inputs for the scored fresh29 detections, cut exactly as V1 cuts them.

THE CONTROL NEEDS ITS OWN INPUT ON THE SAME HANDS. The stereo pairs for these
detections already exist; V1p13 reads the panorama instead, so the same
detections are rendered through the same 0.6 m render, and the hand crop and
the context window are produced by V1's own `crop_of` and `context_crop` --
imported, not re-implemented, because here the point is to be V1.

ONLY THE DETECTIONS THE STEREO ARMS CAN SCORE. The source list is the
`selfother` test index with status `ok`, so every row that gets a panorama
crop also has a stereo pair, and the comparison never runs on two different
sets of hands.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

COLUMNS = ("rec", "frame", "tid", "status", "hand_path", "ctx_path",
           "pano_w", "pano_h")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stereo_root", required=True, help="selfother crops root")
    ap.add_argument("--split", required=True, help="selfother test split")
    ap.add_argument("--clips", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    import cv2
    cv2.setNumThreads(2)
    from src.rig import own_ctx
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.own_cnn import crop_of
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.selfother.crops import read_index
    from src.selfother.labels import load_clips

    clips = load_clips(a.clips)
    rows = [r for r in read_index(a.stereo_root, a.split) if r["status"] == "ok"]
    by_rec = collections.defaultdict(list)
    for r in rows:
        by_rec[r["rec"]].append(r)

    for rec in sorted(by_rec)[a.shard::a.nshards]:
        out_dir = os.path.join(a.out, rec)
        if os.path.exists(os.path.join(out_dir, "index.csv")):
            print(f"  {rec}: 已有，跳过", flush=True)
            continue
        bag, start = clips[rec]
        need = collections.defaultdict(list)
        for r in by_rec[rec]:
            need[int(r["frame"])].append(r)
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        for sub in ("hand", "ctx"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        rd = ClipReader(rig, vids, start)
        cache, index = {}, []
        try:
            for k in range(max(need) - start + 1):
                src = rd.next()
                if not src:
                    raise SystemExit(f"{rec}: 第 {start + k} 帧读不到")
                f = start + k
                if f not in need:
                    continue
                rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                if not rgb[::32, ::32].any():
                    raise SystemExit(f"{rec}: 第 {f} 帧渲染全黑")
                H, W = rgb.shape[:2]
                small = cv2.resize(rgb, (own_ctx.CONTEXT_STORE_W,
                                         int(own_ctx.CONTEXT_STORE_W * H / W)),
                                   interpolation=cv2.INTER_AREA)
                for r in need.pop(f):
                    x0, y0, x1, y1 = (int(float(r[c])) for c in ("x0", "y0", "x1", "y1"))
                    base = {"rec": rec, "frame": f, "tid": r["tid"],
                            "pano_w": W, "pano_h": H}
                    hand = crop_of(rgb, {"box": (x0, y0, x1, y1)})
                    if hand is None:
                        index.append(dict(base, status="hand_crop_too_small"))
                        continue
                    norm = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                            (x1 - x0) / W, (y1 - y0) / H)
                    ctx = own_ctx.context_crop(small, norm, own_ctx.CTX_SCALE)
                    if ctx is None or ctx.size == 0:
                        ctx = hand                       # as own_ctx.predict
                    ctx = cv2.resize(ctx, (own_ctx.CTX_PX, own_ctx.CTX_PX),
                                     interpolation=cv2.INTER_AREA)
                    hand = cv2.resize(hand, (own_ctx.HAND_PX, own_ctx.HAND_PX),
                                      interpolation=cv2.INTER_AREA)
                    name = f"{f:06d}_t{int(r['tid'])}.jpg"
                    cv2.imwrite(os.path.join(out_dir, "hand", name), hand,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(out_dir, "ctx", name), ctx,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    index.append(dict(base, status="ok",
                                      hand_path=os.path.join(rec, "hand", name),
                                      ctx_path=os.path.join(rec, "ctx", name)))
        finally:
            rd.close()
        with open(os.path.join(out_dir, "index.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, restval="")
            w.writeheader()
            w.writerows(index)
        n = collections.Counter(x["status"] for x in index)
        print(f"  {rec}: {dict(n)}", flush=True)


if __name__ == "__main__":
    main()
