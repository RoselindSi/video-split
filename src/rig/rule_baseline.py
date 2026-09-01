"""Score the geometric rule on the same frames, upright and turned.

The segmentation head has no reference point. `owner_arm IoU 0.133` could mean
the model is weak, or that the task as posed is not learnable from 124 frames,
and nothing in the number says which. This supplies the two anchors that make
it readable:

    upright        the rule's own pipeline against the human labels
    turned 180     the same pipeline, same detections, same labels

Upright the rule is near-perfect by construction -- the ground-truth masks
were made by running GrabCut inside its detections and colouring them by a
human's ownership call, so the only disagreement is where the rule's verdict
differs from the person's, and that was measured at zero over 1028 hands.
That is not cheating, it is the point: the rule IS the incumbent, and a
segmenter has to be compared against the thing it would replace.

Turned 180 the rule inverts exactly. `is_owner` asks whether the forearm ray
leaves above 0.55 of the height; a half turn maps every exit height h to H-h,
so every call flips. The collapse is algebra, not an experiment.

DETECTIONS ARE ROTATED, NOT RECOMPUTED, AND THIS IS THE WHOLE VALIDITY OF THE
COMPARISON. Re-running a hand detector on an upside-down frame measures the
detector's own orientation tolerance, which is not what is being asked. The
boxes and keypoints are found once on the upright frame and carried into the
rotated one by coordinate transform, so detection is held fixed and the only
thing that varies is the ownership decision. A rotated-frame collapse then
belongs to the rule and to nothing else.
"""
from __future__ import annotations

import csv
import os

import numpy as np

CLASSES = ("background", "owner_arm", "other_arm")


def rot_points(pts, k, H, W):
    """Carry image points through `np.rot90(img, k)`. -> (N, 2) float

    `np.rot90` turns counter-clockwise, and for shape (H, W) it sends pixel
    (row, col) to (W-1-col, row). Written in (x, y) that is x' = y,
    y' = W-1-x -- the width, not the height, because the axes swap. Getting
    that backwards produces points that land inside the frame and look
    plausible, which is why it is checked against `np.rot90` itself below
    rather than reasoned about twice."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    for _ in range(k % 4):
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0], p[:, 1] = y, W - 1 - x
        H, W = W, H
    return p


def rot_shape(shape, k):
    H, W = shape[:2]
    return (W, H) if k % 2 else (H, W)


def predict(rgb, dets, k, y_frac=None):
    """The rule's 3-class mask for `np.rot90(rgb, k)`. -> uint8 mask

    Both the image and the detections are rotated; the ownership call is then
    made in the rotated frame, which is exactly what a rule that reads a
    height does when the camera it was tuned for is not the camera it gets."""
    from src.rig.hand_detect import (forearm_exit, is_owner, masks_from,
                                     OWNER_EXIT_Y_FRAC)
    if y_frac is None:
        y_frac = OWNER_EXIT_Y_FRAC
    H, W = rgb.shape[:2]
    img = np.ascontiguousarray(np.rot90(rgb, k))
    shape = rot_shape((H, W), k)

    own, oth = [], []
    for d in dets:
        kp = rot_points(d["kp"], k, H, W)
        box = rot_points([(d["box"][0], d["box"][1]),
                          (d["box"][2], d["box"][3])], k, H, W)
        x0, x1 = sorted(box[:, 0])
        y0, y1 = sorted(box[:, 1])
        r = dict(d)
        r["kp"] = kp
        r["box"] = (int(x0), int(y0), int(x1), int(y1))
        _edge, ex = forearm_exit(kp, shape)
        (own if is_owner(ex, shape, y_frac) else oth).append(r)

    mask = np.zeros(shape, np.uint8)
    if oth:
        mask[masks_from(img, oth)] = 2
    if own:
        mask[masks_from(img, own)] = 1
    return mask


def confusion(pred, true, n=len(CLASSES)):
    ok = (true >= 0) & (true < n) & (pred >= 0) & (pred < n)
    return np.bincount(true[ok].astype(np.int64) * n + pred[ok].astype(np.int64),
                       minlength=n * n).reshape(n, n)


def iou(cm):
    out = np.full(len(CLASSES), np.nan)
    for c in range(len(CLASSES)):
        denom = cm[c].sum() + cm[:, c].sum() - cm[c, c]
        if cm[c].sum() > 0 or cm[:, c].sum() > 0:
            out[c] = cm[c, c] / denom if denom else np.nan
    return out


def fp_share(cm):
    tot = cm.sum()
    return np.array([(cm[:, c].sum() - cm[c, c]) / tot if tot else np.nan
                     for c in range(len(CLASSES))])


def run(root, k, weights, limit=None, verbose=True):
    """-> (confusion, n_frames). Scores the rule over an eval dataset."""
    import cv2
    from ultralytics import YOLO
    from src.rig.hand_detect import detect

    rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"),
                                    encoding="utf-8-sig")))
    if limit:
        rows = rows[:limit]
    model = YOLO(weights)
    cm = np.zeros((len(CLASSES),) * 2, np.int64)
    n = 0
    for i, r in enumerate(rows, 1):
        stem = f"{r['recording']}_f{int(r['frame']):06d}.png"
        rgb = cv2.imread(os.path.join(root, "images", stem))
        gt = cv2.imread(os.path.join(root, "masks", stem), cv2.IMREAD_GRAYSCALE)
        if rgb is None or gt is None:
            continue
        # Detected ONCE, on the upright frame, then carried into the rotated
        # one. See the module docstring: re-detecting would measure the
        # detector instead of the rule.
        dets = detect(model, rgb)
        pred = predict(rgb, dets, k)
        cm += confusion(pred.ravel(), np.ascontiguousarray(np.rot90(gt, k)).ravel())
        n += 1
        if verbose and (i % 20 == 0 or i == len(rows)):
            print(f"    [{i}/{len(rows)}]", flush=True)
    return cm, n


def _cells(v):
    return "".join(f"{x:>13.3f}" if np.isfinite(x) else f"{'-':>13}" for x in v)


def report(results):
    """results: {label: (cm, n)}"""
    print(f"\n    {'':<16}{'n':>4}" + "".join(f"{c:>13}" for c in CLASSES))
    print(f"    IoU")
    for lab, (cm, n) in results.items():
        print(f"    {lab:<16}{n:>4}" + _cells(iou(cm)))
    print(f"\n    false positives, share of all pixels")
    for lab, (cm, n) in results.items():
        print(f"    {lab:<16}{n:>4}" + _cells(fp_share(cm)))


def _self_test():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok += bool(cond)

    # The transform is checked against np.rot90 itself: mark one pixel, rotate
    # the image, and see whether the mapped coordinate finds the mark.
    H, W = 7, 11
    for k in range(4):
        img = np.zeros((H, W), np.uint8)
        img[2, 3] = 255
        rot = np.rot90(img, k)
        p = rot_points([(3, 2)], k, H, W)[0]
        chk(f"rot_points agrees with np.rot90 at k={k}",
            rot.shape == rot_shape((H, W), k)
            and rot[int(round(p[1])), int(round(p[0]))] == 255)

    chk("k=4 is the identity",
        np.allclose(rot_points([(3, 2)], 4, H, W)[0], [3, 2]))

    cm = np.array([[8, 1, 0], [2, 4, 0], [0, 0, 0]])
    i = iou(cm)
    chk("IoU of an absent class is nan, not zero", np.isnan(i[2]))
    chk("IoU is computed per class",
        abs(i[0] - 8 / (9 + 10 - 8)) < 1e-9)
    f = fp_share(cm)
    chk("false-positive share counts columns, not rows",
        abs(f[1] - 1 / 15) < 1e-9)

    # A half turn must invert the rule's verdict. Built as a bare geometry
    # check so it holds without a detector or a video.
    from src.rig.hand_detect import forearm_exit, is_owner
    shape = (480, 640)
    kp = np.zeros((21, 2), np.float64)
    kp[0] = (320, 300)            # wrist, low
    kp[1:] = (320, 200)           # fingers, above it -> forearm points DOWN
    up = is_owner(forearm_exit(kp, shape)[1], shape)
    kp2 = rot_points(kp, 2, *shape)
    s2 = rot_shape(shape, 2)
    dn = is_owner(forearm_exit(kp2, s2)[1], s2)
    chk("a half turn inverts the rule", bool(up) != bool(dn))
    print(f"\n  {ok}/9")
    return ok == 9


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--rot", type=int, action="append", default=[],
                    help="quarter turns to score; repeatable. Default 0 and 2.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    ks = a.rot or [0, 2]
    results = {}
    for k in ks:
        print(f"  rot {k * 90} degrees")
        results[f"rule rot{k*90}"] = run(a.eval_root, k, a.weights, a.limit)
    report(results)
    print("\n  Upright is near-perfect by construction: the ground truth was "
          "made from\n  these detections and a person's ownership call, and "
          "the rule agreed with the\n  person on 1028 of 1028 hands. The "
          "number worth reading is the drop, and it\n  is the incumbent's "
          "cost of a coordinate system it was not tuned for.")
    print("  Detections were computed once on the upright frame and rotated, "
          "so this is\n  the rule's failure and not the detector's.")


if __name__ == "__main__":
    main()
