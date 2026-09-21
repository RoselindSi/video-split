"""Where the wearer's own hand goes missing, and which stage lost it.

THE ONLY QUESTION IS OWNER CONTINUITY. A frame where the wearer's hand is
visible and the pipeline does not deliver it is a failure; a frame where the
hand is genuinely out of shot is not, and the two are indistinguishable from
the output. So this does the half that can be automated -- given two systems'
OWNER output on the same frames, cut the intervals where they disagree and
attribute ours to a stage -- and renders the intervals so the other half, was
the hand there at all, can be answered by eye.

ATTRIBUTION RUNS THE DETECTOR AGAIN AT A FLOOR BELOW THE SHIPPED ONE. The
per-hand CSV only holds detections that survived to carry a track id, so it
cannot distinguish "the detector never saw it" from "the detector saw it and
the tracker dropped it" -- which are opposite repairs. Re-detecting at 0.05
separates them:

    no box within reach          the detector missed it
    box below the 0.25 floor     a threshold refused it
    box above it, not in ours    association dropped it
    in ours but called foreign   ownership called it wrong
    in ours, own, but covered    the render covered it anyway
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load(path):
    """-> ({frame: [own boxes]}, {frame: [all boxes with own flag]})"""
    own, allb = collections.defaultdict(list), collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        b = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
        f = int(r["frame"])
        allb[f].append((b, r["own"] == "1", float(r["covered"])))
        if r["own"] == "1":
            own[f].append(b)
    return own, allb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--ours", default="/workspace/cam3_base")
    ap.add_argument("--theirs", default="/workspace/cam3_rolan")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--ship_floor", type=float, default=0.25)
    ap.add_argument("--match", type=float, default=0.3)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--owner_detector", action="store_true",
                    help="the weights name ownership themselves. Attributing a "
                         "loss means asking THAT detector whether it saw the "
                         "hand, so the re-detection has to use the same model "
                         "the arm under audit used")
    ap.add_argument("--out", help="write a per-frame CSV of the attribution")
    a = ap.parse_args()
    from ultralytics import YOLO
    from src.rig.hand_detect import detect, owner_detect
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))
    bag, start, n = jobs[a.rec]
    A_own, A_all = load(os.path.join(a.ours, a.rec + ".csv"))
    B_own, _ = load(os.path.join(a.theirs, a.rec + ".csv"))
    frames = list(range(start, start + n))
    want = [f for f in frames if f not in A_own and f in B_own]
    lost, kept = os.path.basename(a.ours.rstrip("/")), os.path.basename(a.theirs.rstrip("/"))
    print("%s：%s 有 OWNER 而 %s 没有的帧 %d 个（归因对象 = %s）"
          % (a.rec, kept, lost, len(want), lost))
    if not want:
        return

    model = YOLO(a.weights)
    vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    rd = RawCameraReader(vids, "cam3", min(want))
    cur = min(want) - 1
    tally = collections.Counter()
    rows = []
    for f in sorted(want):
        img = None
        while cur < f:
            img = rd.next()
            cur += 1
            if img is None:
                break
        if img is None:
            break
        props = (owner_detect(model, img, min_conf=a.floor) if a.owner_detector
                 else detect(model, img, min_conf=a.floor))
        for target in B_own[f]:
            best, bc = 0.0, 0.0
            for d in props:
                v = iou([float(x) for x in d["box"]], target)
                if v > best:
                    best, bc = v, float(d["conf"])
            mine = [(b, own, cov) for b, own, cov in A_all.get(f, [])
                    if iou(b, target) >= a.match]
            if best < a.match:
                why = "检测器没看到"
            elif not mine and bc < a.ship_floor:
                why = "分数低于续接门槛被扔掉"
            elif not mine:
                why = "检测到了但关联丢掉"
            elif any(own for _, own, _ in mine):
                why = "判为自己却仍算缺失(渲染?)"
            else:
                why = "归属判成别人的手"
            tally[why] += 1
            rows.append({"frame": f, "why": why, "best_iou": round(best, 3),
                         "conf": round(bc, 4),
                         "x0": int(target[0]), "y0": int(target[1]),
                         "x1": int(target[2]), "y1": int(target[3])})
    rd.close()
    print("\n  %-28s %6s" % ("%s 为什么丢了它" % lost, "帧次"))
    for k, v in tally.most_common():
        print("  %-28s %6d  (%.0f%%)" % (k, v, 100.0 * v / sum(tally.values())))
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
