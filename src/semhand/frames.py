"""Clean frames for both families, and V1's frozen features for the test hands.

--mode fresh   fresh29, every STRIDE-th frame from the clip start, only the
               detections the earlier table scored (dump + stereo ok + pano ok).
               Each sampled frame is rendered once and then:
                 * saved clean, for Qwen;
                 * run through the shipped detector again, to get back the 21
                   keypoints the dump never stored. A dump box is matched to a
                   new detection by IoU; `hand_span` and the geometric prior
                   come from the match;
                 * cut exactly as `own_ctx.predict` cuts, and passed through
                   V1 once, keeping the 1088-d vector before its head.
               V1's P from that vector is written beside the dumped P: if the
               render, the keypoints or the crops differ from what V1 saw, the
               two disagree and it shows.
--mode pkg     the bank: V1's own training rows, their frames re-rendered clean
               from the databag (the stored context frames carry the labelling
               tool's box and forearm line).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np

from src.semhand import STRIDE, TRAIN_PKGS

FRESH_COLS = ("rec", "frame", "tid", "databag", "image", "x0", "y0", "x1", "y1",
              "W", "H", "side", "iou", "p_dump", "p_v1", "prior", "status")
PKG_COLS = ("stem", "pkg", "tag", "databag", "frame", "image", "x0", "y0", "x1", "y1",
            "W", "H", "y", "status")


def cap_decoder_threads(n=4):
    """Make every VideoCapture in this process decode on `n` threads.

    `seam_fix.ClipReader` opens three captures with FFmpeg's default of one
    thread per core, about 440 threads a process here, and the container
    stops at 4096: ten processes of this hit it, and the failures came back as
    unreadable frames and swscaler errors rather than as a thread error. The
    reader is V1's and stays untouched; the cap is applied from outside."""
    import cv2
    if getattr(cv2.VideoCapture, "_capped", False) or not hasattr(cv2, "CAP_PROP_N_THREADS"):
        return
    orig = cv2.VideoCapture

    # AND A LONGER READ TIMEOUT. Under load the shared volume stalls past
    # OpenCV's 30 s default, and a timed-out read is indistinguishable from
    # the end of the file: the reader returns nothing and the frame is lost.
    params = [cv2.CAP_PROP_N_THREADS, n]
    for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, prop):
            params += [getattr(cv2, prop), 300000]

    def capped(path, *args):
        return orig(path, *args) if args else orig(path, cv2.CAP_FFMPEG, params)
    capped._capped = True
    cv2.VideoCapture = capped


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def v1_features(model, h, c, g):
    """-> (1088-d features, P(owner)); asserts the split forward equals V1's."""
    import torch
    f = torch.cat([model.hand(h), model.ctx(c), model.gmlp(g)], 1)
    p = torch.softmax(model.head(f), 1)[:, 1]
    return f, p


def fresh_queries(dump, stereo_root, split, pano_root, clips):
    from src.selfother.crops import read_index
    d = {(r["rec"], r["frame"], r["tid"]): r
         for r in csv.DictReader(open(dump, encoding="utf-8"))}
    pano = set()
    for rec in os.listdir(pano_root):
        p = os.path.join(pano_root, rec, "index.csv")
        if os.path.exists(p):
            pano |= {(r["rec"], r["frame"], r["tid"])
                     for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"}
    out = collections.defaultdict(list)
    for r in read_index(stereo_root, split):
        k = (r["rec"], r["frame"], r["tid"])
        if r["status"] != "ok" or k not in pano or k not in d:
            continue
        if (int(r["frame"]) - clips[r["rec"]][1]) % STRIDE:
            continue
        out[r["rec"]].append(d[k])
    return out


def run_fresh(a):
    import cv2
    cap_decoder_threads()
    import torch
    from ultralytics import YOLO
    from src.rig import geom_prior, own_ctx
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.hand_detect import detect
    from src.rig.own_cnn import crop_of
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.selfother.labels import load_clips
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    clips = load_clips(a.clips)
    queries = fresh_queries(a.dump, a.stereo_root, a.split, a.pano_root, clips)
    yolo = YOLO(a.weights)
    v1, device, _ = own_ctx.load_model(a.clf_ctx)
    geom = geom_prior.load_model(a.geom)
    ds = own_ctx.Pairs([])
    for rec in sorted(queries)[a.shard::a.nshards]:
        out_csv = os.path.join(a.out, "fresh", rec, "index.csv")
        if os.path.exists(out_csv):
            print(f"  {rec}: 已有，跳过", flush=True)
            continue
        os.makedirs(os.path.join(a.out, "fresh", rec, "frames"), exist_ok=True)
        bag, start = clips[rec]
        need = collections.defaultdict(list)
        for r in queries[rec]:
            need[int(r["frame"])].append(r)
        rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        try:
            fresh_one(a, rec, bag, start, need, rig, vcam, vids, out_csv, yolo, v1, device, geom, ds)
        except (Exception, SystemExit) as e:
            # ONE BAD RECORDING DOES NOT TAKE THE SHARD WITH IT. Nothing is
            # written for it, so a second pass retries exactly the gaps, and
            # the run script refuses to go on while any remain.
            print(f"  !! {rec} 失败，未写出 -- {type(e).__name__}: {e}", flush=True)


def fresh_one(a, rec, bag, start, need, rig, vcam, vids, out_csv, yolo, v1, device, geom, ds):
    import cv2
    import torch
    from src.rig import geom_prior, own_ctx
    from src.rig.hand_detect import detect
    from src.rig.own_cnn import crop_of
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    if True:
        rd = ClipReader(rig, vids, start)
        cache, index, feats, fkeys = {}, [], [], []
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
                img = os.path.join(a.out, "fresh", rec, "frames", f"{f:06d}.jpg")
                cv2.imwrite(img, rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                dets = detect(yolo, rgb, min_conf=0.25)
                small = cv2.resize(rgb, (own_ctx.CONTEXT_STORE_W,
                                         int(own_ctx.CONTEXT_STORE_W * H / W)),
                                   interpolation=cv2.INTER_AREA)
                hs, cs, gs, rows = [], [], [], []
                for r in need.pop(f):
                    box = tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1"))
                    base = {"rec": rec, "frame": f, "tid": r["tid"], "databag": bag,
                            "image": img, "x0": box[0], "y0": box[1], "x1": box[2],
                            "y1": box[3], "W": W, "H": H, "side": r["side_raw"],
                            "p_dump": r["p_owner_raw"]}
                    best = max(dets, key=lambda d: iou(box, d["box"]), default=None)
                    ov = iou(box, best["box"]) if best is not None else 0.0
                    base["iou"] = round(ov, 4)
                    if best is None or ov < 0.9 or best.get("kp") is None:
                        index.append(dict(base, status="no_keypoints"))
                        continue
                    det = dict(best, box=np.array([int(v) for v in box]))
                    hand = crop_of(rgb, det)
                    if hand is None:
                        index.append(dict(base, status="hand_crop_too_small"))
                        continue
                    nb = ((box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H,
                          (box[2] - box[0]) / W, (box[3] - box[1]) / H)
                    ctx = own_ctx.context_crop(small, nb, own_ctx.CTX_SCALE)
                    if ctx is None or ctx.size == 0:
                        ctx = hand
                    hs.append(ds._prep(hand, own_ctx.HAND_PX))
                    cs.append(ds._prep(ctx, own_ctx.CTX_PX))
                    gs.append(torch.from_numpy(own_ctx._geom_vector(det, rgb.shape)))
                    prior = geom_prior.score(det, [det], rgb.shape, geom)
                    rows.append(dict(base, prior="" if prior is None else round(prior, 6),
                                     status="ok"))
                if rows:
                    with torch.no_grad():
                        fv, pv = v1_features(v1, torch.stack(hs).to(device),
                                             torch.stack(cs).to(device),
                                             torch.stack(gs).to(device))
                        ref = torch.softmax(v1(torch.stack(hs).to(device),
                                               torch.stack(cs).to(device),
                                               torch.stack(gs).to(device)), 1)[:, 1]
                    assert torch.allclose(pv, ref, atol=1e-5), "split forward != V1"
                    for r, x, p in zip(rows, fv.cpu().numpy(), pv.cpu().numpy()):
                        r["p_v1"] = round(float(p), 5)
                        index.append(r)
                        feats.append(x.astype(np.float32))
                        fkeys.append(f"{rec}|{r['frame']}|{r['tid']}")
        finally:
            rd.close()
        np.savez(os.path.join(a.out, "fresh", rec, "v1feat.npz"),
                 keys=np.array(fkeys), feat=np.stack(feats) if feats else np.zeros((0, 1088)))
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FRESH_COLS, restval="")
            w.writeheader()
            w.writerows(index)
        ok = [r for r in index if r["status"] == "ok"]
        dp = [abs(float(r["p_v1"]) - float(r["p_dump"])) for r in ok]
        print(f"  {rec}: {dict(collections.Counter(r['status'] for r in index))}  "
              f"|P重算-P dump| 中位 {np.median(dp) if dp else float('nan'):.5f} "
              f"最大 {max(dp) if dp else float('nan'):.4f}", flush=True)


def run_plain(a):
    """Clean frames only, for every dumped hand on a gold track, every STRIDE-th
    frame. No stereo filter and no V1 recomputation: the dump already holds
    V1's deployed verdict for each of these hands, and the Qwen arms need only
    the frame and the box."""
    import cv2
    cap_decoder_threads()
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.selfother.labels import load_clips
    from src.selfother.train import read_gold
    cv2.setNumThreads(2)
    clips = load_clips(a.clips)
    gold = read_gold(a.gold)
    queries = collections.defaultdict(list)
    for r in csv.DictReader(open(a.dump, encoding="utf-8")):
        if (r["rec"], str(r["tid"])) in gold and (int(r["frame"]) - clips[r["rec"]][1]) % STRIDE == 0:
            queries[r["rec"]].append(r)
    for rec in sorted(queries)[a.shard::a.nshards]:
        out_csv = os.path.join(a.out, "fresh", rec, "index.csv")
        if os.path.exists(out_csv):
            print(f"  {rec}: 已有，跳过", flush=True)
            continue
        os.makedirs(os.path.join(a.out, "fresh", rec, "frames"), exist_ok=True)
        bag, start = clips[rec]
        need = collections.defaultdict(list)
        for r in queries[rec]:
            need[int(r["frame"])].append(r)
        rd, index = None, []
        try:
            rig = RigCalibration(os.path.join(bag, "calibration.yaml"))
            vcam = VirtualWideCamera.from_rig(rig)
            vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
            rd, cache = ClipReader(rig, vids, start), {}
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
                img = os.path.join(a.out, "fresh", rec, "frames", f"{f:06d}.jpg")
                cv2.imwrite(img, rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                for r in need.pop(f):
                    index.append({"rec": rec, "frame": f, "tid": r["tid"], "databag": bag,
                                  "image": img, "x0": r["x0"], "y0": r["y0"], "x1": r["x1"],
                                  "y1": r["y1"], "W": W, "H": H, "side": r["side_raw"],
                                  "p_dump": r["p_owner_raw"], "status": "ok"})
        except (Exception, SystemExit) as e:
            print(f"  !! {rec} 失败，未写出 -- {type(e).__name__}: {e}", flush=True)
            continue
        finally:
            if rd is not None:
                rd.close()
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FRESH_COLS, restval="")
            w.writeheader()
            w.writerows(index)
        print(f"  {rec}: {len(index)} 只手", flush=True)


def run_pkg(a):
    import cv2
    cap_decoder_threads()
    from src.rig import own_ctx
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.vlmhand.render import STEM_RE
    cv2.setNumThreads(2)
    rows = own_ctx.load(list(TRAIN_PKGS), verbose=True)
    by_bag = collections.defaultdict(list)
    for pkg in TRAIN_PKGS:
        bags = {r["tag"]: r["databag"]
                for r in csv.DictReader(open(os.path.join(pkg, "sources.csv"), encoding="utf-8-sig"))}
        frame_of = {r["stem"]: r.get("frame", "")
                    for r in csv.DictReader(open(os.path.join(pkg, "hands.csv"), encoding="utf-8-sig"))}
        for r in rows:
            if r["pkg"] != pkg:
                continue
            m = STEM_RE.match(r["stem"])
            bag = bags.get(m.group(1)) if m else None
            fr = frame_of.get(r["stem"], "")
            by_bag[bag or ""].append((int(fr) if fr else -1, r))
    os.makedirs(os.path.join(a.out, "pkg", "frames"), exist_ok=True)
    out_csv = os.path.join(a.out, "pkg", f"index_{a.shard}.csv")
    index = []
    for bag in sorted(by_bag)[a.shard::a.nshards]:
        items = sorted(by_bag[bag], key=lambda t: t[0])
        base = lambda r, f: {"stem": r["stem"], "pkg": os.path.basename(r["pkg"].rstrip("/")),
                             "tag": r["tag"], "databag": bag, "frame": f, "y": r["y"]}
        cal = os.path.join(bag, "calibration.yaml") if bag else ""
        if not bag or not os.path.exists(cal) or any(f < 0 for f, _ in items):
            index.extend(dict(base(r, f), status="no_databag_or_frame") for f, r in items)
            print(f"  跳过 {bag or '(无 databag)'}: {len(items)} 行", flush=True)
            continue
        rig = RigCalibration(cal)
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        cache, rd, at, rgb = {}, None, None, None
        n_before = len(index)
        try:
            for f, r in items:
                if f != at:
                    if rd is None or not (0 <= f - at - 1 <= 600):
                        if rd is not None:
                            rd.close()
                        rd = ClipReader(rig, vids, f)
                        src = rd.next()
                    else:
                        src = rd.next(skip=f - at - 1)
                    at = f
                    if not src:
                        rgb = None
                    else:
                        rgb = render(rig, vcam, src, 0.6, map_cache=cache)[0]
                        img = os.path.join(a.out, "pkg", "frames",
                                           f"{os.path.basename(bag)}_f{f:06d}.jpg")
                        cv2.imwrite(img, rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if rgb is None or not rgb[::32, ::32].any():
                    index.append(dict(base(r, f), status="unreadable_frame"))
                    continue
                H, W = rgb.shape[:2]
                cx, cy, bw, bh = r["box"]
                index.append(dict(base(r, f), image=img, W=W, H=H,
                                  x0=(cx - bw / 2) * W, y0=(cy - bh / 2) * H,
                                  x1=(cx + bw / 2) * W, y1=(cy + bh / 2) * H, status="ok"))
        except (Exception, SystemExit) as e:
            # The bag's rows are kept, marked, and counted -- a bank that
            # silently shrinks would change what every template arm retrieves.
            del index[n_before:]
            index.extend(dict(base(r, f), status="bag_failed") for f, r in items)
            print(f"  !! {os.path.basename(bag)} 失败 {len(items)} 行 -- {type(e).__name__}: {e}",
                  flush=True)
            continue
        finally:
            if rd is not None:
                rd.close()
        print(f"  {os.path.basename(bag)}: {len(items)} 行", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PKG_COLS, restval="")
        w.writeheader()
        w.writerows(index)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("fresh", "pkg", "plain"), required=True)
    ap.add_argument("--out", default="/workspace/semhand")
    ap.add_argument("--dump", default="/workspace/own_dump_fresh.csv")
    ap.add_argument("--clips", action="append", default=None, help="default fresh29.txt")
    ap.add_argument("--gold", action="append", default=None, help="plain mode: gold track csvs")
    ap.add_argument("--stereo_root", default="/workspace/selfother")
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--pano_root", default="/workspace/v1stereo/pano_fresh")
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--geom", default="/workspace/geom_inv2.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    a.clips = a.clips or ["/workspace/fresh29.txt"]
    {"fresh": run_fresh, "pkg": run_pkg, "plain": run_plain}[a.mode](a)


if __name__ == "__main__":
    main()
