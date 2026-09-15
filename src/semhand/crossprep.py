"""V1's own input for every hand, cut from the saved frames both models share.

Runs in `/workspace/venv_rig`. Feeds the swap experiments (`crosscheck.py`):
Qwen reading only what V1 reads, and V1's architecture reading what Qwen reads.

ONE SOURCE OF PIXELS FOR BOTH SIDES. Qwen's views are cut from the JPEG frames
in `<root>/fresh/*/frames` and `<root>/pkg/frames`; V1's are cut here from
the same JPEGs, so a difference between the arms is the view and not a
decode, a render or a compression step. V1's shipped P on these JPEG crops is
written beside the dumped P, so the cost of the JPEG itself is measured.

WHAT V1 READS, EXACTLY AS `own_ctx.predict` CUTS IT:
    hand  `own_cnn.crop_of` (box + 0.6 x its longer side), 128 px
    ctx   the frame resized to 900 wide, `context_crop` at 2.5 x the box, 128 px
    geom  the 14 features; `hand_span` needs keypoints, so test hands are
          re-detected on the frame and matched to their dumped box by IoU.
          Bank hands use the geometry V1 was trained on (the package csv).

--mode test   every ok row of `<root>/fresh/*/index.csv`
--mode bank   V1's training rows with a re-rendered frame (`<root>/pkg`)
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

COLS = ("id", "kind", "rec", "hand", "ctx", "geom", "p_v1_jpeg", "p_dump", "y", "tag", "status")


def v1_views(img, box, own_ctx, crop_of, cv2):
    """-> (hand 128, ctx 128) BGR arrays, or (None, None) if the hand crop fails."""
    H, W = img.shape[:2]
    hand = crop_of(img, {"box": tuple(int(round(float(v))) for v in box)})
    if hand is None:
        return None, None
    small = cv2.resize(img, (own_ctx.CONTEXT_STORE_W, int(own_ctx.CONTEXT_STORE_W * H / W)),
                       interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = (float(v) for v in box)
    nb = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
    ctx = own_ctx.context_crop(small, nb, own_ctx.CTX_SCALE)
    if ctx is None or ctx.size == 0:
        ctx = hand
    rs = lambda x, px: cv2.resize(x, (px, px), interpolation=cv2.INTER_AREA)
    return rs(hand, own_ctx.HAND_PX), rs(ctx, own_ctx.CTX_PX)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=("test", "bank"), required=True)
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import cv2
    import torch
    from src.rig import own_ctx
    from src.rig.own_cnn import crop_of
    from src.semhand import TRAIN_PKGS
    from src.semhand.frames import iou
    from src.selfother.model import to_tensor
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    v1, device, _ = own_ctx.load_model(a.clf_ctx)
    out_dir = os.path.join(a.root, "cross")
    os.makedirs(out_dir, exist_ok=True)

    if a.mode == "test":
        rows = []
        for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "index.csv"))):
            rows += [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"]
        for r in rows:
            r["id"] = f"{r['rec']}|{r['frame']}|{r['tid']}"
        from ultralytics import YOLO
        yolo = YOLO(a.weights)
    else:
        idx = {}
        for p in glob.glob(os.path.join(a.root, "pkg", "index_*.csv")):
            idx.update({r["stem"]: r for r in csv.DictReader(open(p, encoding="utf-8"))
                        if r["status"] == "ok"})
        rows = []
        for r in own_ctx.load(list(TRAIN_PKGS), verbose=False):
            if r["stem"] in idx:
                rows.append(dict(idx[r["stem"]], id=r["stem"], rec=r["tag"].rstrip("_"),
                                 g=r["g"], y=r["y"], tag=r["tag"]))

    by_image = collections.defaultdict(list)
    for r in rows:
        by_image[r["image"]].append(r)
    images = sorted(by_image)[a.shard::a.nshards]
    out, batch = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            h = torch.stack([to_tensor(b[1]) for b in batch]).to(device)
            c = torch.stack([to_tensor(b[2]) for b in batch]).to(device)
            g = torch.stack([torch.from_numpy(b[3]) for b in batch]).to(device)
            p = torch.softmax(v1(h, c, g), 1)[:, 1].cpu().numpy()
        for (rec, _, _, _), q in zip(batch, p):
            rec["p_v1_jpeg"] = round(float(q), 5)
        batch.clear()

    for n, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            out += [dict(r, status="unreadable_frame") for r in by_image[path]]
            continue
        dets = detect_on(yolo, img) if a.mode == "test" else None
        for r in by_image[path]:
            box = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
            rec = {"id": r["id"], "kind": a.mode, "rec": r["rec"], "p_dump": r.get("p_dump", ""),
                   "y": r.get("y", ""), "tag": r.get("tag", "")}
            if a.mode == "test":
                best = max(dets, key=lambda d: iou(box, d["box"]), default=None)
                if best is None or iou(box, best["box"]) < 0.9 or best.get("kp") is None:
                    out.append(dict(rec, status="no_keypoints"))
                    continue
                det = dict(best, box=np.array([int(round(v)) for v in box]))
                g = own_ctx._geom_vector(det, img.shape)
            else:
                g = np.asarray(r["g"], np.float32)
            hand, ctx = v1_views(img, box, own_ctx, crop_of, cv2)
            if hand is None:
                out.append(dict(rec, status="hand_crop_too_small"))
                continue
            sub = os.path.join(out_dir, r["rec"])
            os.makedirs(sub, exist_ok=True)
            stem = r["id"].replace("|", "_")
            hp, cp = os.path.join(sub, stem + "_hand.png"), os.path.join(sub, stem + "_ctx.png")
            cv2.imwrite(hp, hand)
            cv2.imwrite(cp, ctx)
            rec.update(hand=hp, ctx=cp, geom=json.dumps([round(float(x), 6) for x in g]), status="ok")
            out.append(rec)
            batch.append((rec, hand, ctx, g.astype(np.float32)))
            if len(batch) >= 128:
                flush()
        if n % 500 == 0:
            print(f"  {n + 1}/{len(images)} 帧", flush=True)
    flush()
    path = os.path.join(out_dir, f"index_{a.mode}_{a.shard}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, restval="")
        w.writeheader()
        w.writerows(out)
    ok = [r for r in out if r["status"] == "ok"]
    dp = [abs(float(r["p_v1_jpeg"]) - float(r["p_dump"])) for r in ok if r.get("p_dump")]
    print(f"-> {path}  {dict(collections.Counter(r['status'] for r in out))}"
          + (f"  |P(JPEG)-P(dump)| 中位 {np.median(dp):.4f} p90 {np.percentile(dp, 90):.4f}" if dp else ""),
          flush=True)


def detect_on(yolo, img):
    from src.rig.hand_detect import detect
    return detect(yolo, img, min_conf=0.25)


if __name__ == "__main__":
    main()
