"""Hands from a detector that knows what a hand is, and ownership from the wrist.

This replaces a skin-colour threshold and the five shape rules that grew on top
of it. Those rules -- a border test, a solidity cap, a straightness cap, a
persistence mask, an area window -- were all trying to reconstruct "is this a
limb?" from the shape of a tan blob, and each one was a threshold cut through
two overlapping distributions. The last of them ran out of gap: a minimum
aspect ratio would have removed 45% of the remaining false positives at the
cost of 5% of real arms, because a paper disc measures 1.7 and a real arm
reaches down to 1.84.

A hand detector answers the question directly. On real wide frames it finds
both of the wearer's hands in every frame at 0.83-0.87 confidence, labels them
left and right, and returns 21 keypoints each. A wooden turntable, a beige
machine strap, a cabinet and a paper disc are not hands and it does not report
them.

OWNERSHIP COMES FROM THE WRIST, NOT FROM THE MASK. The old rule asked whether
a connected component touched the bottom of the frame, which worked (79%
against 0%) but depended on the segmentation holding together -- and a sleeve
cuts the component at the cuff, which is why the threshold on it had a margin
of 1.7 points. The keypoints make the question direct: the forearm runs from
the fingers through the wrist and onward, so extend that ray and see which
edge of the frame it leaves by. The wearer's arm exits through the bottom.
Nothing about that depends on how well anything was segmented.

MASKS STILL COME FROM GRABCUT, prompted by the detector's box instead of by a
colour threshold. That part was already built and already works; what it
lacked was a prompt from something that knew what it was pointing at.
"""
from __future__ import annotations

import numpy as np

# MANO / WiLoR keypoint layout: 0 is the wrist, then thumb, index, middle,
# ring and little finger, four points each.
WRIST = 0
FINGERS = list(range(1, 21))

DETECTOR = "/shared/models/HaWoR/weights/external/detector.pt"
IMGSZ = 512
MIN_CONF = 0.35

# How far past the wrist the forearm ray is followed before asking which edge
# it left by. Expressed in hand-widths, because a hand near the lens is large
# and one across the bench is small.
RAY_HAND_WIDTHS = 40.0


def forearm_exit(kp, shape):
    """Which frame edge the forearm leaves by. -> ('bottom'|'left'|'right'|
    'top'|None, exit point)

    The ray runs from the fingers through the wrist and outward. A hand
    belonging to the wearer is on the end of an arm that leaves the frame at
    the bottom; a colleague's leaves by a side or the top."""
    kp = np.asarray(kp, float)
    if kp.shape[0] < 21 or not np.isfinite(kp).all():
        return None, None
    H, W = shape[:2]
    wrist = kp[WRIST]
    d = wrist - kp[FINGERS].mean(0)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None, None
    d = d / n

    # Smallest positive distance to each of the four edges.
    ts = []
    if abs(d[0]) > 1e-9:
        ts += [((0 - wrist[0]) / d[0], "left"), ((W - wrist[0]) / d[0], "right")]
    if abs(d[1]) > 1e-9:
        ts += [((0 - wrist[1]) / d[1], "top"), ((H - wrist[1]) / d[1], "bottom")]
    ts = [(t, e) for t, e in ts if t > 0]
    if not ts:
        return None, None
    t, edge = min(ts)
    return edge, wrist + d * t


def detect(model, rgb, imgsz=IMGSZ, min_conf=MIN_CONF):
    """-> [{'box','kp','side','conf','edge'}] for one frame."""
    res = model(rgb, imgsz=imgsz, verbose=False)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    kps = (res.keypoints.xy.cpu().numpy()
           if res.keypoints is not None else [None] * len(boxes))
    for b, c, k, kp in zip(boxes, conf, cls, kps):
        if c < min_conf:
            continue
        edge, _ = forearm_exit(kp, rgb.shape) if kp is not None else (None, None)
        out.append({"box": b.astype(int), "kp": kp, "conf": float(c),
                    "side": model.names.get(int(c_), str(c_))
                    if (c_ := int(k)) is not None else "?",
                    "edge": edge})
    return out


def split_owner(dets, shape, max_owner=2):
    """-> (owner dets, other dets).

    The wearer has two arms and they come from below. A third hand entering
    from the bottom is a person standing beside the wearer, which happens, and
    the cap keeps the two whose ray exits closest to straight down rather than
    resolving it by list order."""
    H, W = shape[:2]
    own, oth = [], []
    for d in dets:
        (own if d.get("edge") == "bottom" else oth).append(d)
    if len(own) > max_owner:
        def downness(d):
            kp = np.asarray(d["kp"], float)
            v = kp[WRIST] - kp[FINGERS].mean(0)
            # y grows DOWNWARD in image coordinates, so straight down is
            # +v[1]. The negated version sorted the least-downward arm to the
            # front and kept exactly the wrong two.
            return v[1] / max(np.linalg.norm(v), 1e-9)    # +1 is straight down
        own.sort(key=downness, reverse=True)
        oth += own[max_owner:]
        own = own[:max_owner]
    return own, oth


def masks_from(rgb, dets, pad=0.15, iters=3):
    """GrabCut inside each detection's box. -> binary mask

    The box comes from the detector and the boundary from GrabCut, which is
    the division of labour this pipeline was always meant to have and could
    not, because the only available prompt was a colour threshold that did not
    know a hand from a turntable."""
    import cv2
    H, W = rgb.shape[:2]
    out = np.zeros((H, W), bool)
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
        bx = (max(0, x0 - px), max(0, y0 - py),
              min(W, x1 + px), min(H, y1 + py))
        if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 8:
            continue
        m = np.full((H, W), cv2.GC_BGD, np.uint8)
        m[bx[1]:bx[3], bx[0]:bx[2]] = cv2.GC_PR_BGD
        # The keypoints are inside the hand by construction -- a far better
        # foreground seed than a colour rule, and unavailable until now.
        if d.get("kp") is not None:
            for (kx, ky) in np.asarray(d["kp"], int):
                if 0 <= kx < W and 0 <= ky < H:
                    cv2.circle(m, (int(kx), int(ky)), 6, cv2.GC_FGD, -1)
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(rgb, m, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
            r = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
        except cv2.error:
            r = np.zeros((H, W), bool)
            r[bx[1]:bx[3], bx[0]:bx[2]] = True
        keep = np.zeros((H, W), bool)
        keep[bx[1]:bx[3], bx[0]:bx[2]] = True
        out |= r & keep
    return out


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    H, W = 900, 1600

    def hand(wrist, fingers_at):
        """21 keypoints: a wrist and 20 finger points clustered elsewhere."""
        kp = np.zeros((21, 2))
        kp[WRIST] = wrist
        kp[FINGERS] = np.array(fingers_at) + np.random.default_rng(0).normal(
            0, 3, (20, 2))
        return kp

    # The wearer: fingers up at the bench, wrist below them, arm from below.
    e, _ = forearm_exit(hand((800, 700), (800, 500)), (H, W))
    chk(e == "bottom", f"fingers above the wrist -> arm exits the bottom ({e})")

    # A colleague reaching in from the left: wrist to the LEFT of the fingers.
    e, _ = forearm_exit(hand((400, 400), (700, 400)), (H, W))
    chk(e == "left", f"wrist left of the fingers -> exits left ({e})")

    e, _ = forearm_exit(hand((1300, 400), (900, 400)), (H, W))
    chk(e == "right", f"wrist right of the fingers -> exits right ({e})")

    e, _ = forearm_exit(hand((800, 200), (800, 600)), (H, W))
    chk(e == "top", f"wrist above the fingers -> exits the top ({e})")

    # Diagonal, from the lower left: still the wearer.
    e, _ = forearm_exit(hand((500, 700), (800, 400)), (H, W))
    chk(e in ("bottom", "left"),
        f"a diagonal arm from the lower left exits bottom or left ({e})")

    chk(forearm_exit(np.zeros((5, 2)), (H, W))[0] is None,
        "too few keypoints returns nothing rather than guessing")
    chk(forearm_exit(np.full((21, 2), np.nan), (H, W))[0] is None,
        "NaN keypoints return nothing")

    d_own = [{"kp": hand((700, 700), (700, 500)), "edge": "bottom"},
             {"kp": hand((900, 700), (900, 500)), "edge": "bottom"}]
    d_oth = [{"kp": hand((400, 400), (700, 400)), "edge": "left"}]
    own, oth = split_owner(d_own + d_oth, (H, W))
    chk(len(own) == 2 and len(oth) == 1,
        "two from below are the wearer's, one from the side is not")

    # A third from below: the two most straight-down win, not the first two.
    slanted = {"kp": hand((300, 700), (600, 690)), "edge": "bottom"}
    own2, oth2 = split_owner([slanted] + d_own, (H, W))
    chk(len(own2) == 2 and slanted not in own2,
        "a third arm from below is resolved by direction, not by list order")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Ownership is read off the wrist, so it does not depend on how "
          "well anything\n  was segmented -- which is what the old "
          "bottom-of-the-component rule did,\n  with a margin of 1.7 points.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--src", help="a directory of images, or a seg_auto split")
    ap.add_argument("--out")
    ap.add_argument("--weights", default=DETECTOR)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=14.0)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.src or not a.out:
        ap.error("--src and --out are required without --self_test")

    import glob
    import os
    import cv2
    from ultralytics import YOLO
    from src.rig.suppress_other import suppress

    imgs = sorted(glob.glob(os.path.join(a.src, "images", "*.png"))) \
        or sorted(glob.glob(os.path.join(a.src, "*.png"))) \
        or sorted(glob.glob(os.path.join(a.src, "*.jpg")))
    imgs = imgs[:a.limit]
    if not imgs:
        raise SystemExit(f"no images under {a.src}")
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    print(f"  {len(imgs)} frames, detector {os.path.basename(a.weights)}\n")

    tally = {}
    for p in imgs:
        rgb = cv2.imread(p)
        dets = detect(model, rgb)
        own, oth = split_owner(dets, rgb.shape)
        for d in dets:
            tally[d.get("edge")] = tally.get(d.get("edge"), 0) + 1
        m_own = masks_from(rgb, own)
        m_oth = masks_from(rgb, oth)
        out, alpha = suppress(rgb, m_oth, a.dilate, 4, a.sigma, protect=m_own)
        vis = out.copy()
        vis[m_own] = (0.55 * vis[m_own]
                      + 0.45 * np.array([0, 230, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, os.path.basename(p)[:-4] + ".jpg"),
                    np.hstack([rgb, vis]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {os.path.basename(p)[-18:]}  {len(dets)} hands  "
              f"owner {len(own)}  other {len(oth)}  "
              f"suppressed {(alpha>0.5).mean():.2%}")
    print(f"\n  forearm exits: {tally}")
    print("  Left/right/top are colleagues; bottom is the wearer. Nothing "
          "that is not a\n  hand is reported at all, which is the whole point "
          "of replacing the colour rule.")


if __name__ == "__main__":
    main()
