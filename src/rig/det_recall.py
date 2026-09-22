"""How many of other people's hands the detector never proposes.

THE HOLE THIS FILLS. Every ownership number in this project is conditioned on
a box the detector produced -- the student's acceptance tests, the exposure
rate, the continuity table. A colleague's hand the detector never found is in
nobody's denominator, and the one time it was looked at, 69 of 70 boxes GPT-6
drew and our detector did not were real hands. That said "a lot" and could not
say how many.

WHY THIS SET AND WHY IT IS NOT CIRCULAR. The boxes come from GPT-6, which
draws its own rather than scoring ours, so our detector has never seen them
and cannot be graded against its own output. Their ownership labels were
audited blind at 300 boxes: owner 99.8%, other 97.9%. The labels are NOT
exhaustive, which would wreck a precision measurement and does nothing to a
recall one -- a hand nobody labelled simply is not asked about.

AND IT IS DONE AT FULL RESOLUTION, which is the whole reason this file is
more than ten lines. The labelled images are 640x512 and the pipeline reads
1920x1520; measuring on the small copy would mix "the detector misses small
hands" with "the picture was thrown away", and today's hand-ness result makes
exactly that confusion the thing to avoid. Every row carries device,
recording, camera and a frame index, the raw databags are on disk, and the
index is recovered as

    frame = window_start_ms / 1000 * 30 + source_frame_idx

checked against the stored image on six recordings at correlation 0.93-0.98.
The 640x512 copy is that frame scaled by exactly 1/3 with two rows of
letterbox, so a label maps back as (x*3, (y-2)*3).

PROPOSED-BUT-LOW IS NOT MISSING. The detector runs at a floor far under the
shipped one, so a hand it saw and scored 0.12 is separated from a hand it
never proposed at all. Those need different fixes and the shipped thresholds
turn the first into the second.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random

PAIR = {"cam1": "cam12", "cam2": "cam12", "cam3": "cam34",
        "cam4": "cam34", "cam5": "cam56", "cam6": "cam56"}
LEFT = {"cam1", "cam3", "cam5"}
SCALE = 3.0
PAD_Y = 2


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot",
                    default="/shared/ownership_labels/direct_hand_set_20260922_v1"
                            "/snapshot_v1/train.jsonl")
    ap.add_argument("--labels",
                    default="/shared/ownership_labels/boxes_gpt6_qwen50958_v1/labels.jsonl")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    from src.rig.hand_detect import detect

    # where each image came from
    src = {}
    for line in open(a.labels):
        d = json.loads(line)
        if d.get("status") != "complete":
            continue
        src[os.path.basename(d["image"])] = d
    print("labels.jsonl 里有来源的图 %d 张" % len(src))

    # the audited boxes, from the frozen snapshot
    items = []
    for line in open(a.snapshot):
        d = json.loads(line)
        if d.get("label_basis") != "DS weak supervision":
            continue
        s = src.get(os.path.basename(d["image"]))
        if not s:
            continue
        gt = [h for h in d.get("hands", [])
              if h.get("status") == "hand"
              and h.get("ownership") in ("owner", "other")]
        if gt:
            items.append((s, gt))
    print("有框可评的图 %d 张" % len(items))
    random.Random(a.seed).shuffle(items)
    items = items[:a.n]

    roots = glob.glob("/shared/datasets/.incoming/video_1520p_raw/*/raw_databag")

    def bagpath(dev, rec):
        for r in roots:
            p = os.path.join(r, dev, rec)
            if os.path.isdir(p):
                return p
        return None

    # group by video so the seeks run forward
    by_vid = collections.defaultdict(list)
    for s, gt in items:
        p = bagpath(s["device"], s["recording"])
        if not p:
            continue
        v = os.path.join(p, PAIR[s["camera"]] + ".mp4")
        if not os.path.exists(v):
            continue
        start_ms = int(s["window_id"].rsplit("__", 1)[1].split("_")[0])
        f = int(round(start_ms / 1000.0 * 30)) + int(s["source_frame_idx"])
        by_vid[v].append((f, s, gt))
    print("涉及 %d 个视频文件" % len(by_vid))

    model = YOLO(a.weights)
    rows = []
    done = 0
    for v in sorted(by_vid):
        cap = cv2.VideoCapture(v)
        # SORT ON THE FRAME ONLY. Two labelled images can share a frame, and
        # then the tuple comparison falls through to the dicts behind it and
        # raises -- which is how the first run of this died at frame 200 with
        # nothing written.
        at = -1
        for f, s, gt in sorted(by_vid[v], key=lambda x: x[0]):
            # Seeking into a 28,000-frame file costs seconds and decoding
            # forward costs milliseconds, so a short hop is walked, not sought.
            if 0 <= f - at <= 240:
                fr = None
                while at < f:
                    ok, fr = cap.read()
                    at += 1
                    if not ok:
                        break
                if fr is None:
                    continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                at = f
                if not ok:
                    continue
            W = fr.shape[1] // 2
            half = fr[:, :W] if s["camera"] in LEFT else fr[:, W:]
            rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
            dets = detect(model, rgb, min_conf=a.floor)
            db = [([float(x) for x in d["box"]], float(d.get("conf", 0.0)))
                  for d in dets]
            for h in gt:
                x0, y0, x1, y1 = h["box_xyxy"]
                g = [x0 * SCALE, (y0 - PAD_Y) * SCALE,
                     x1 * SCALE, (y1 - PAD_Y) * SCALE]
                best, bc = 0.0, 0.0
                for b, c in db:
                    u = iou(g, b)
                    if u > best:
                        best, bc = u, c
                rows.append({"rec": s["recording"], "cam": s["camera"],
                             "frame": f, "ownership": h["ownership"],
                             "w_px": round(g[2] - g[0], 1),
                             "h_px": round(g[3] - g[1], 1),
                             "iou": round(best, 3),
                             "conf": round(bc, 3) if best >= a.iou else "",
                             "n_det": len(db)})
            done += 1
            if done % 50 == 0:
                print("  %d/%d 帧" % (done, len(items)), flush=True)
        cap.release()
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> %s (%d 个标注框，%d 帧)" % (a.out, len(rows), done))
    report(rows, a.iou)


def report(rows, thr):
    print()
    for own in ("owner", "other"):
        rs = [r for r in rows if r["ownership"] == own]
        if not rs:
            continue
        print("== %s  n=%d" % (own, len(rs)))
        print("   %-16s %6s %10s %10s %10s" % (
            "框宽(原生 px)", "n", "检到", "并且>=0.25", "并且>=0.60"))
        bins = [(0, 40), (40, 60), (60, 100), (100, 160), (160, 10000)]
        for lo, hi in bins:
            sub = [r for r in rs if lo <= r["w_px"] < hi]
            if not sub:
                continue
            hit = [r for r in sub if r["iou"] >= thr]
            c25 = [r for r in hit if r["conf"] != "" and float(r["conf"]) >= 0.25]
            c60 = [r for r in hit if r["conf"] != "" and float(r["conf"]) >= 0.60]
            print("   %-16s %6d %9.1f%% %9.1f%% %9.1f%%" % (
                "%d-%d" % (lo, hi) if hi < 10000 else "%d+" % lo, len(sub),
                100.0 * len(hit) / len(sub), 100.0 * len(c25) / len(sub),
                100.0 * len(c60) / len(sub)))
        hit = [r for r in rs if r["iou"] >= thr]
        c25 = [r for r in hit if r["conf"] != "" and float(r["conf"]) >= 0.25]
        c60 = [r for r in hit if r["conf"] != "" and float(r["conf"]) >= 0.60]
        print("   %-16s %6d %9.1f%% %9.1f%% %9.1f%%" % (
            "合计", len(rs), 100.0 * len(hit) / len(rs),
            100.0 * len(c25) / len(rs), 100.0 * len(c60) / len(rs)))
        print()


if __name__ == "__main__":
    main()
