"""Which hands does the detector never propose, and which does the floor throw away?

WHY THE GOLD CANNOT ANSWER THIS. Every label in this project hangs off a track,
and a track begins with a detection, so a hand the detector never found is in
nobody's denominator. Ownership accuracy, M1, M2, the zone batches: all of them
are conditional on the detector having seen the hand. The blind audit found
GPT-6 proposing foreign hands ours had not (69 of 70 of its own boxes were real
hands), which is the first evidence of the size of that blind spot and is not a
measurement of it.

THE REFERENCE IS INDEPENDENT. GPT-6 drew its own boxes on monocular cam3/cam4
frames, without seeing our detector's output, so running our detector on those
same images compares two independent opinions about where the hands are. This
is not the deployed input -- the pipeline detects on the rendered panorama --
so the rate here is indicative of the detector, not of the pipeline; what makes
it worth having anyway is that it separates the two failures that need opposite
fixes:

    missed     no proposal at any score, even at the scan floor
               -> training data is the only thing that helps
    filtered   proposed, but under new_track_conf / continue_conf
               -> a threshold and tracking question, and new labels of the
                  same hands would change nothing

Recall is reported at the shipped floors and at the scan floor, split by the
label GPT-6 gave (a foreign hand missed is a privacy failure; the wearer's own
hand missed only costs the cover a box it did not need) and by hand size.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random

G = "/shared/ownership_labels/boxes_gpt6_qwen50958_v1"
NEW_TRACK_CONF = 0.60
CONTINUE_CONF = 0.25


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="/workspace/g6teach/gpt6_labels_snapshot.jsonl")
    ap.add_argument("--images", type=int, default=600, help="images to sample")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="scan floor: below the shipped 0.25, to see what was proposed at all")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", default="/workspace/detector_gap.csv")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    from src.rig.hand_detect import detect

    items = []
    for line in open(a.snapshot):
        r = json.loads(line)
        L = r.get("labels") or {}
        if L.get("image_readability") != "adequate":
            continue
        hands = [h for h in L.get("hands") or [] if h["label"] in ("owner", "other")
                 and h.get("crop_status") == "usable"]
        if hands:
            items.append((r, hands))
    rng = random.Random(a.seed)
    rng.shuffle(items)
    items = items[:a.images]
    print(f"{len(items)} 张图、{sum(len(h) for _, h in items)} 只手（GPT-6 画的框）")

    model = YOLO(a.weights)
    rows = []
    for k, (r, hands) in enumerate(items):
        path = os.path.join(G, r["image"])
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        props = detect(model, img, min_conf=a.floor)
        boxes = [[float(v) for v in d["box"]] for d in props]
        confs = [float(d["conf"]) for d in props]
        used = set()
        for h in hands:
            gb = [float(v) for v in h["box_xyxy"]]
            best, bi = 0.0, None
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                v = iou(gb, b)
                if v > best:
                    best, bi = v, i
            hit = bi is not None and best >= a.match_iou
            if hit:
                used.add(bi)
            rows.append({"recording": r["recording"], "camera": r["camera"],
                         "frame": r["cache_frame_idx"], "label": h["label"],
                         "w": round(gb[2] - gb[0], 1), "h": round(gb[3] - gb[1], 1),
                         "frac": round((gb[2] - gb[0]) * (gb[3] - gb[1]) / float(W * H), 5),
                         "found": int(hit), "conf": round(confs[bi], 4) if hit else 0.0,
                         "iou": round(best, 3)})
        # Boxes ours proposed that GPT-6 did not draw: the other direction, and
        # not evidence of a false positive -- GPT-6 misses hands too.
        rows.append({"recording": r["recording"], "camera": r["camera"],
                     "frame": r["cache_frame_idx"], "label": "__extra__",
                     "w": 0, "h": 0, "frac": 0,
                     "found": len(boxes) - len(used), "conf": 0.0, "iou": 0.0})
        if (k + 1) % 100 == 0:
            print(f"    [{k + 1}/{len(items)}]", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hands = [r for r in rows if r["label"] != "__extra__"]
    print(f"\n-> {a.out}   {len(hands)} 只手")
    print(f"  {'':<10}{'手数':>7}{'任何分数下找到':>15}{'>=0.25':>9}{'>=0.60':>9}")
    for lab in ("owner", "other"):
        g = [r for r in hands if r["label"] == lab]
        if not g:
            continue
        print(f"  {lab:<10}{len(g):>7}{sum(r['found'] for r in g) / len(g):>14.1%}"
              f"{sum(r['conf'] >= CONTINUE_CONF for r in g) / len(g):>9.1%}"
              f"{sum(r['conf'] >= NEW_TRACK_CONF for r in g) / len(g):>9.1%}")
    miss = [r for r in hands if not r["found"]]
    if miss:
        fr = sorted(r["frac"] for r in miss)
        ok = sorted(r["frac"] for r in hands if r["found"])
        print(f"  真漏掉的 {len(miss)} 只手，占图面积 中位 {fr[len(fr)//2]:.3%}；"
              f"找到的中位 {ok[len(ok)//2]:.3%}" if ok else "")
    extra = sum(r["found"] for r in rows if r["label"] == "__extra__")
    print(f"  我们提出、GPT-6 没画的框: {extra}（不等于误检——GPT-6 也会漏）")


if __name__ == "__main__":
    main()
