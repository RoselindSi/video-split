"""Does the hand V1 got wrong connect to somebody else's visible arm?

THE QUESTION V1 DOES NOT ASK. The ownership model answers "does this hand look
like the wearer's". Every input it has -- the crop, the surrounding window, the
frame-normalised geometry -- is a property of the hand and its neighbourhood.
The cases it still gets wrong are close-range: a colleague's hand beside the
wearer's, called the wearer's with high confidence. For those the useful
question is a different one, "is this hand attached to a body that is in the
frame", and no amount of tuning a hand classifier reaches it because the answer
lives outside the window.

THIS PROBE DOES NOT BUILD THAT. It asks whether the relation is present in the
data at all, which is the only thing worth knowing before anything is built.
Run a pose detector, take V1's FALSE OWNERS -- hands a person called foreign
that V1 called the wearer's -- and matched hands V1 called the wearer's
correctly, and measure how often each associates to a visible person's
wrist-elbow-shoulder chain.

WHAT THE TWO OUTCOMES MEAN. If the false owners associate far more often than
the true owners, an asymmetric veto has something to work with and its cost is
bounded by the true-owner rate. If both associate at similar rates the relation
is not discriminative here, and a veto built on it would trade one error for
another -- which is worth knowing for the price of a pose pass rather than for
the price of a trained model.

IT IS ASYMMETRIC ON PURPOSE, AND SO IS THE MEASUREMENT. A veto would only ever
turn `owner` into `other`. So the numbers that matter are the hit rate on
hands that should have been `other` and the false-alarm rate on hands that
really are the wearer's; nothing here needs to say anything about hands V1
already calls foreign.

NOTHING IN THE PIPELINE IMPORTS THIS. It reads labelled packages and writes a
table.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

# COCO keypoint indices, the layout every pose model in this family emits.
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
ARMS = ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR))

# A keypoint below this is not evidence of anything.
KP_CONF = 0.35

# How close a wrist has to be to the hand, as a fraction of the hand box's
# longer side. A wrist keypoint sits inside the hand, so this is generous
# rather than tight -- the question is association, not localisation.
WRIST_NEAR = 1.5


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _seg_dist(p, a, b):
    """Distance from point `p` to segment `ab`."""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    n = float(ab @ ab)
    if n < 1e-9:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / n))
    return float(np.linalg.norm(p - (a + t * ab)))


def associate(box, people, shape, kp_conf=KP_CONF, wrist_near=WRIST_NEAR):
    """Does this hand attach to a visible arm? -> dict

    `box` is (cx, cy, w, h) in frame fractions; `people` is a list of (17, 3)
    keypoint arrays in pixels. Returns the best chain found and how complete
    it was, so a partial chain -- a wrist with no shoulder -- is visible as
    partial rather than counted as an association."""
    H, W = shape[:2]
    cx, cy, bw, bh = box
    c = np.array([cx * W, cy * H])
    reach = max(bw * W, bh * H) * wrist_near
    best = {"wrist": None, "elbow": False, "shoulder": False,
            "person": None, "d": float("inf")}
    for pi, kp in enumerate(people):
        for sh, el, wr in ARMS:
            if kp[wr, 2] < kp_conf:
                continue
            d = float(np.linalg.norm(c - kp[wr, :2]))
            if d > reach or d >= best["d"]:
                continue
            has_el = kp[el, 2] >= kp_conf
            has_sh = kp[sh, 2] >= kp_conf
            # The forearm should point at the hand, not merely be nearby.
            if has_el:
                d_seg = _seg_dist(c, kp[wr, :2], kp[el, :2])
                if d_seg > reach:
                    has_el = False
            best = {"wrist": d, "elbow": has_el,
                    "shoulder": has_sh and has_el, "person": pi, "d": d}
    best["full"] = bool(best["wrist"] is not None and best["elbow"]
                        and best["shoulder"])
    best["any"] = best["wrist"] is not None
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--clf_ctx", required=True)
    ap.add_argument("--pose",
                    default="/workspace/models/yolo11n-pose.pt")
    ap.add_argument("--kp_conf", type=float, default=KP_CONF)
    ap.add_argument("--wrist_near", type=float, default=WRIST_NEAR)
    ap.add_argument("--dump")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from src.rig.track_pool import load_pkg, frame_scores

    rows = []
    for p in a.pkg:
        rows += load_pkg(p)[1]
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    for r in rows:
        r["box"] = tuple(float(r["raw"][c]) for c in
                         ("box_cx", "box_cy", "box_w", "box_h"))
    p_owner = frame_scores(rows, clf_ctx=a.clf_ctx)
    for r, p in zip(rows, p_owner):
        r["p"] = float(p)
        r["pred_owner"] = p >= 0.5

    # The two populations a veto would act on.
    false_owner = [r for r in rows if r["pred_owner"] and r["y"] == 0]
    true_owner = [r for r in rows if r["pred_owner"] and r["y"] == 1]
    print(f"  {len(rows)} 帧样本   V1 判成 owner 的 "
          f"{len(false_owner) + len(true_owner)}   其中错的 "
          f"{len(false_owner)}（真值是别人的手）")
    if not false_owner:
        raise SystemExit("  V1 在这批上没有 false owner，没有可测的对象")

    pose = YOLO(a.pose)
    cache = {}
    got = {"false": [], "true": []}
    for group, v in (("false", false_owner), ("true", true_owner)):
        for r in v:
            key = r["_ctx"]
            if key not in cache:
                img = cv2.imread(key)
                if img is None:
                    cache[key] = (None, None)
                else:
                    res = pose(img, verbose=False)[0]
                    k = (res.keypoints.data.cpu().numpy()
                         if res.keypoints is not None else
                         np.zeros((0, 17, 3)))
                    cache[key] = (k, img.shape)
            kps, shape = cache[key]
            if kps is None:
                continue
            got[group].append(associate(r["box"], kps, shape,
                                        a.kp_conf, a.wrist_near))

    print(f"\n  {'群体':<28} {'n':>5} {'有腕':>10} {'腕+肘':>10} "
          f"{'完整臂链':>11}")
    out = {}
    for group, name in (("false", "V1 误判成 owner（真 other）"),
                        ("true", "V1 正确判成 owner")):
        v = got[group]
        if not v:
            print(f"  {name:<28} {0:>5}")
            continue
        anyk = sum(1 for x in v if x["any"])
        el = sum(1 for x in v if x["any"] and x["elbow"])
        full = sum(1 for x in v if x["full"])
        out[group] = (full, len(v))
        print(f"  {name:<28} {len(v):>5} {anyk / len(v):>9.1%} "
              f"{el / len(v):>10.1%} {full / len(v):>11.1%}")

    if "false" in out and "true" in out:
        (fk, fn), (tk, tn) = out["false"], out["true"]
        flo, fhi = wilson(fk, fn)
        tlo, thi = wilson(tk, tn)
        print(f"\n  完整臂链命中率")
        print(f"    误判成 owner   {fk}/{fn} = {fk / fn:.3f} "
              f"[{flo:.3f}, {fhi:.3f}]   <- veto 能救回来的比例")
        print(f"    正确的 owner   {tk}/{tn} = {tk / tn:.3f} "
              f"[{tlo:.3f}, {thi:.3f}]   <- veto 会误伤的比例")
        print("\n  两个率差得远，非对称 veto 才有东西可用，而且它的代价就是第二"
              "行。\n  两个率接近，说明这个关系在这里没有判别力，veto 只是把一"
              "种错换成另一种。")
    print("\n  臂链是用 context 帧（900px 宽）跑的姿态，不是全分辨率全景；"
          "远处的人\n  关键点会更弱，所以命中率是下界。这决定了结论的方向："
          "低命中率可能是\n  分辨率造成的，高命中率不会是。")

    if a.dump:
        import csv as _csv
        with open(a.dump, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["group", "recording", "tid", "frame", "p_owner",
                        "wrist_d", "elbow", "shoulder", "full"])
            for group, v in (("false", false_owner), ("true", true_owner)):
                for r, x in zip(v, got[group]):
                    w.writerow([group, r["tag"], r["tid"], r["raw"]["frame"],
                                round(r["p"], 4),
                                "" if x["wrist"] is None
                                else round(x["wrist"], 1),
                                int(x["elbow"]), int(x["shoulder"]),
                                int(x["full"])])
        print(f"  per-row -> {a.dump}")


if __name__ == "__main__":
    main()
