"""Is the detector's left/right label steady enough to feed ownership?

THE OWNERSHIP HEAD HAS NEVER SEEN HANDEDNESS AND THE DETECTOR HAS ALWAYS
EMITTED IT. `detector.pt` is a two-class pose model -- `{0: left, 1: right}`,
21 keypoints -- and `hand_detect.detect` stores the class as `side` on every
detection, where nothing reads it. The fourteen geometry features the
ownership classifier does see carry position, forearm direction and exit
height, so `RIGHT hand entering low from the right` and `RIGHT hand reaching
in from across the bench` differ in a way the model cannot currently express:
the interaction needs the variable, and the variable is absent. That is why
`the geometry is saturated` does not settle this -- it was measured in a space
without handedness in it.

BEFORE ANY OF THAT IS WORTH TRAINING, THE FIELD HAS TO BE STABLE. A label that
flips within a track is noise wearing the shape of a feature, and the flip
risk is concentrated exactly where the feature would be useful: the checkpoint
was trained on InterHand -- clean, complete, third-person hands -- at 512, and
the cases that matter here are a colleague's hand, close, truncated and half
behind a workpiece. So consistency is reported PER STRATUM and the overall
number is deliberately not the headline; a 95% average made of 99% on the
wearer's own uncluttered hands and 55% on near foreign ones would be worse
than useless.

WHAT SPLITS THE STRATA. Truncation (does the box touch the frame edge),
crowding (does another detection overlap it) and side (own or foreign by the
FROZEN EXIT-HEIGHT RULE, not by the classifier -- using the classifier to
stratify a feature meant to improve that classifier would be circular).

NO OWNERSHIP MODEL, NO GRABCUT, NO FACES. This runs the detector and the
tracker and nothing else, because nothing else is being measured and the
decode is already the expensive part.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

# A box within this many pixels of any frame edge counts as truncated.
EDGE_PX = 6
# Another detection overlapping by more than this makes the hand `crowded`.
CROWD_IOU = 0.10


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load_clips(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(":", 2)
        d, s = parts[0], int(parts[1])
        n = int(parts[2]) if len(parts) == 3 else 1000
        tag = os.path.basename(d.rstrip("/")).replace("databag-26_", "R")
        out[tag] = (d, s, n)
    return out


def harvest(a):
    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect, is_owner
    from src.rig.hand_track import Tracker, MAX_LOST
    from src.rig import demo_video

    clips = load_clips(a.clips)
    want = [t for t in a.rec] if a.rec else sorted(clips)
    model = YOLO(a.weights)
    rows = []
    for tag in want:
        if tag not in clips:
            print(f"  !! {tag} 不在 clips 里")
            continue
        databag, start, _ = clips[tag]
        start = start + a.offset
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        # The same tracker the pipeline deploys, so a `track` here is the same
        # object a flip would be counted on there.
        tracker = Tracker(max_lost=max(MAX_LOST,
                                       demo_video.MAX_PREDICTION_AGE))
        rd = Prefetch(ClipReader(rig, vids, start), skip=0)
        mc = {}
        print(f"  {tag}: {start}-{start + a.n - 1}", flush=True)
        for k in range(a.n):
            src = rd.next()
            if not src:
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            H, W = rgb.shape[:2]
            dets = detect(model, rgb, min_conf=a.conf)
            ids = tracker.update(dets, rgb.shape,
                                 new_track_conf=a.new_track_conf,
                                 continue_conf=a.conf)
            for i, d in enumerate(dets):
                x0, y0, x1, y1 = [int(v) for v in d["box"]]
                crowd = max([iou(d["box"], o["box"])
                             for j, o in enumerate(dets) if j != i] or [0.0])
                kp = d.get("kp")
                rows.append({
                    "rec": tag, "frame": start + k,
                    "tid": "" if ids[i] is None else ids[i],
                    "side": d.get("side", ""), "conf": round(d["conf"], 4),
                    "area_frac": round((x1 - x0) * (y1 - y0) / float(W * H), 6),
                    "edge": int(x0 <= EDGE_PX or y0 <= EDGE_PX
                                or x1 >= W - EDGE_PX or y1 >= H - EDGE_PX),
                    "crowd": round(float(crowd), 4),
                    "kp_ok": int(kp is not None),
                    # The FROZEN RULE, not the classifier: stratifying a
                    # feature by the model it is meant to improve is circular.
                    "rule_owner": int(bool(d.get("rule_owner"))),
                    "exit_y": ("" if d.get("exit") is None
                               else round(float(d["exit"][1]) / H, 4))})
        rd.close()
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} 个检测 -> {a.out}")
    return rows


def report(rows):
    """Per-track majority fraction, split by the strata that matter."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["tid"] in ("", None):
            continue
        by[(r["rec"], r["tid"])].append(r)

    def stratum(det_rows):
        """A track is described by what it mostly was, not by any one frame."""
        n = len(det_rows)
        own = sum(int(r["rule_owner"]) for r in det_rows) > n / 2
        edge = sum(int(r["edge"]) for r in det_rows) > n / 2
        crowd = sum(float(r["crowd"]) >= CROWD_IOU for r in det_rows) > n / 2
        return ("自己的手" if own else "别人的手",
                "截断" if edge else "完整",
                "重叠" if crowd else "独立")

    agg = collections.defaultdict(lambda: {"tracks": 0, "det": 0, "flips": 0,
                                           "maj": [], "pure": 0})
    for _, det_rows in by.items():
        det_rows.sort(key=lambda r: int(r["frame"]))
        sides = [r["side"] for r in det_rows if r["side"]]
        if len(sides) < 3:
            continue                      # a two-frame track says nothing
        c = collections.Counter(sides)
        maj = c.most_common(1)[0][1] / len(sides)
        flips = sum(1 for x, y in zip(sides, sides[1:]) if x != y)
        for key in (stratum(det_rows), ("全部", "", "")):
            d = agg[key]
            d["tracks"] += 1
            d["det"] += len(sides)
            d["flips"] += flips
            d["maj"].append(maj)
            d["pure"] += (maj == 1.0)

    print(f"\n  {'分层':<26} {'轨迹':>5} {'检测':>6} {'纯净轨迹':>9} "
          f"{'多数占比中位':>12} {'翻转/百检测':>12}")
    for key in sorted(agg, key=lambda k: (k[0] != "全部", k)):
        d = agg[key]
        name = " ".join(x for x in key if x)
        med = sorted(d["maj"])[len(d["maj"]) // 2]
        lo, hi = wilson(d["pure"], d["tracks"])
        print(f"  {name:<26} {d['tracks']:>5} {d['det']:>6} "
              f"{d['pure']:>4}/{d['tracks']:<4} [{lo:.2f},{hi:.2f}] "
              f"{med:>10.3f} {100.0 * d['flips'] / d['det']:>12.1f}")
    print("\n  纯净轨迹 = 整条轨迹里 left/right 从没变过。这才是能直接当特征喂的那种。")
    print("  只有『别人的手 × 截断』和『别人的手 × 重叠』两格算数 —— 那是这个")
    print("  特征唯一会被用到的地方，也是 InterHand 训练分布离得最远的地方。")
    print("  分层用的是冻结的出口高度规则，不是归属分类器：用一个模型去分层一个")
    print("  打算改进它的特征，是循环论证。")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--rec", action="append",
                    help="recording tags; default every clip in the file")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="the deployed continue-track floor")
    ap.add_argument("--new_track_conf", type=float, default=0.60)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", default="/workspace/side_probe.csv")
    ap.add_argument("--report", help="read this csv and only report")
    a = ap.parse_args()

    if a.report:
        rows = list(csv.DictReader(open(a.report, encoding="utf-8-sig")))
        report(rows)
        return
    report(harvest(a))


if __name__ == "__main__":
    main()
