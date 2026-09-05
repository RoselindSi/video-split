"""Four ways to reject a face that is not one, on faces a person labelled.

WHAT IS BROKEN AND WHY THE OBVIOUS FIX IS NOT ONE. A large patch of
skin-coloured background scores about 0.45 with the YOLO face detector and is
mosaicked. The two guards that would have caught it are both off, and both
were turned off for reasons that still hold: the size cap was refusing the
close colleague, who is the most identifiable face in any recording, and the
threshold was lowered because 0.60 missed 66 of 140 labelled faces. Turning
either back on trades this failure for a worse one.

The literature on this is consistent: face detector false positives concentrate
in a few visual classes -- hands, ears, torso -- and are better removed by
VERIFYING a proposal than by tightening the proposer. RetinaFace reports that
five-point landmark supervision is specifically what suppresses high-scoring
false positives; `It Takes Two to Tango` confirms one detector's proposals with
a second without retraining either.

FOUR ARMS ON ONE LABELLED SET.

    F0  what ships          YOLO at 0.35, no size cap
    F1  landmark geometry   YuNet re-run on the proposal, five points checked
                            for arrangement rather than presence
    F2  second detector     YuNet asked only whether a face is there
    F3  size cap            the rejected fix, kept as the control it has to
                            beat -- otherwise a more complicated method that
                            merely matches it looks like progress

YUNET IS THE VERIFIER HERE AND WAS REJECTED AS THE DETECTOR, WHICH IS NOT A
CONTRADICTION AND IS NOT A FREE PASS EITHER. As a detector it produced 1280
proposals to YOLO's 84 on 48 frames and called an engine cover a face at 0.57;
that is a failure to separate, and a verifier is asked a narrower question --
is there a face in THIS box. But the engine cover is exactly the kind of thing
F1 and F2 are wanted for, so if YuNet fires on skin-coloured background too,
the cascade adds nothing. That is what this measures rather than assumes.
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

# How far YuNet's box may sit from the proposal and still be called the same
# thing. Two detectors rarely agree on a box to the pixel.
MATCH_IOU = 0.30

# Landmark plausibility, all in units of the proposal box.
#   the eyes must be apart, but not further apart than the face is wide
#   the mouth must be below the eyes
#   the nose must sit between them
EYE_SEP = (0.18, 0.85)
MOUTH_BELOW = 0.10
INSIDE_PAD = 0.35


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def landmarks_plausible(box, pts):
    """Do five points sit like a face inside this box? -> bool

    ARRANGEMENT, NOT PRESENCE. A detector that emits five points always emits
    five points; on a patch of bench they land wherever the regression drifts.
    The checks are the ones that are true of every face and of few other
    things: the eyes are separated by a fair fraction of the width, the mouth
    is below them, the nose is horizontally between them, and none of it
    escapes the box by much."""
    if pts is None or len(pts) < 5:
        return False
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    re, le, no, rm, lm = [np.asarray(p, float) for p in pts[:5]]
    pad_x, pad_y = INSIDE_PAD * w, INSIDE_PAD * h
    for p in (re, le, no, rm, lm):
        if not (x0 - pad_x <= p[0] <= x1 + pad_x
                and y0 - pad_y <= p[1] <= y1 + pad_y):
            return False
    sep = abs(le[0] - re[0]) / w
    if not (EYE_SEP[0] <= sep <= EYE_SEP[1]):
        return False
    eye_y = (le[1] + re[1]) / 2.0
    mouth_y = (lm[1] + rm[1]) / 2.0
    if (mouth_y - eye_y) / h < MOUTH_BELOW:
        return False
    lo, hi = sorted((re[0], le[0]))
    if not (lo - 0.15 * w <= no[0] <= hi + 0.15 * w):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True,
                    help="a harvested face package with labels merged in")
    ap.add_argument("--yunet",
                    default="/workspace/models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--yunet_conf", type=float, default=0.30,
                    help="the verifier's own floor. Low on purpose: it is "
                         "answering about a box that already exists, not "
                         "searching.")
    ap.add_argument("--size_cap", type=float, default=0.055,
                    help="F3's cap, the value the joint sweep picked before "
                         "it was withdrawn")
    ap.add_argument("--dump")
    a = ap.parse_args()

    import cv2
    from src.rig import face_mask

    rows = []
    for r in csv.DictReader(open(os.path.join(a.pkg, "hands.csv"),
                                 encoding="utf-8-sig")):
        if r.get("label") not in ("owner", "other"):
            continue
        ctx = os.path.join(a.pkg, "context", r["stem"] + ".jpg")
        if not os.path.exists(ctx):
            continue
        rows.append({"stem": r["stem"], "_ctx": ctx, "raw": r,
                     # `owner` is the sheet's key 1; here it means A FACE.
                     "face": r["label"] == "owner",
                     "conf": float(r["conf"]),
                     "w_frac": float(r["w_frac"]),
                     "box_frac": (float(r["cx_frac"]), float(r["cy_frac"]),
                                  float(r["w_frac"]), float(r["h_frac"]))})
    if not rows:
        raise SystemExit(f"no labelled rows in {a.pkg}")
    n_face = sum(1 for r in rows if r["face"])
    print(f"  {len(rows)} 个候选   真脸 {n_face}   不是脸 "
          f"{len(rows) - n_face}   录像 "
          f"{len({r['stem'].rsplit('_f', 1)[0] for r in rows})} 段")

    det = face_mask.load_detector(a.yunet, a.yunet_conf)
    if det is None or getattr(det, "kind", "") != "yunet":
        raise SystemExit(f"no YuNet at {a.yunet}")

    cache = {}
    for r in rows:
        img = cv2.imread(r["_ctx"])
        if img is None:
            r["yunet"] = None
            continue
        H, W = img.shape[:2]
        cx, cy, bw, bh = r["box_frac"]
        box = (cx * W - bw * W / 2, cy * H - bh * H / 2,
               cx * W + bw * W / 2, cy * H + bh * H / 2)
        key = r["_ctx"]
        if key not in cache:
            found = det.detect(img)
            cache[key] = (found, list(getattr(det, "last_landmarks", [])))
        found, lms = cache[key]
        best, best_v, best_lm = None, MATCH_IOU, None
        for i, f in enumerate(found):
            v = iou(box, f[:4])
            if v >= best_v:
                best, best_v, best_lm = f, v, (lms[i] if i < len(lms) else None)
        r["yunet"] = (None if best is None else
                      {"score": float(best[4]), "iou": best_v,
                       "plaus": landmarks_plausible(box, best_lm)})

    def keeps(r, arm):
        if arm == "F0":
            return True
        if arm == "F1":
            return bool(r["yunet"] and r["yunet"]["plaus"])
        if arm == "F2":
            return r["yunet"] is not None
        if arm == "F3":
            return r["w_frac"] <= a.size_cap
        raise ValueError(arm)

    print(f"\n  {'arm':<26} {'覆盖真脸':>12} {'误糊非脸':>12} "
          f"{'精确':>8} {'召回':>8} {'f1':>8}")
    out = {}
    for arm, name in (("F0", "F0 现状 0.35 无上限"),
                      ("F1", "F1 + 五点几何"),
                      ("F2", "F2 + 二级检测器"),
                      ("F3", f"F3 + 尺寸上限 {a.size_cap:g}")):
        tp = sum(1 for r in rows if r["face"] and keeps(r, arm))
        fp = sum(1 for r in rows if not r["face"] and keeps(r, arm))
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / n_face if n_face else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and prec + rec else float("nan"))
        out[arm] = (tp, fp, prec, rec)
        print(f"  {name:<26} {tp:>4}/{n_face:<7} {fp:>4}/"
              f"{len(rows) - n_face:<7} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

    print()
    for arm in ("F0", "F1", "F2", "F3"):
        tp, fp, prec, rec = out[arm]
        lo, hi = wilson(tp, tp + fp) if tp + fp else (float("nan"),) * 2
        rlo, rhi = wilson(tp, n_face)
        print(f"  {arm}  精确 [{lo:.3f}, {hi:.3f}]   召回 [{rlo:.3f}, "
              f"{rhi:.3f}]")

    print("\n  这是一个非对称问题：漏一张脸是画面里留了一个能认出来的人，"
          "误糊一块工位是\n  一块马赛克。所以召回是要买的东西，精确是预算，"
          "F3 是必须被打败的对照——\n  一个更复杂的方法只是追平尺寸上限，"
          "那不是进步。")

    miss = [r for r in rows if r["face"] and not keeps(r, "F1")]
    if miss:
        print(f"\n  F1 拒掉的真脸 {len(miss)} 张，宽度中位 "
              f"{np.median([r['w_frac'] for r in miss]):.3f}，"
              f"分数中位 {np.median([r['conf'] for r in miss]):.2f}")
        no_y = sum(1 for r in miss if r["yunet"] is None)
        print(f"    其中 {no_y} 张 YuNet 根本没在那里检出 —— 那是二级检测器"
              f"的召回问题，不是几何判据的")

    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stem", "is_face", "yolo_conf", "w_frac",
                        "yunet_found", "yunet_score", "yunet_iou",
                        "landmarks_ok", "F0", "F1", "F2", "F3"])
            for r in rows:
                y = r["yunet"]
                w.writerow([r["stem"], int(r["face"]), round(r["conf"], 3),
                            round(r["w_frac"], 4), int(y is not None),
                            "" if not y else round(y["score"], 3),
                            "" if not y else round(y["iou"], 3),
                            "" if not y else int(y["plaus"])]
                           + [int(keeps(r, k)) for k in
                              ("F0", "F1", "F2", "F3")])
        print(f"\n  per-row -> {a.dump}")


if __name__ == "__main__":
    main()
