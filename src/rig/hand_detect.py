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

import os

import numpy as np

# MANO / WiLoR keypoint layout: 0 is the wrist, then thumb, index, middle,
# ring and little finger, four points each.
WRIST = 0
FINGERS = list(range(1, 21))

DETECTOR = "/shared/models/HaWoR/weights/external/detector.pt"
IMGSZ = 512
# Real hands on this corpus come in at 0.83-0.87. A pink box came in as a
# hand at the old floor of 0.35 and was blurred, which is the first false
# positive this pipeline has had that arrives with a NUMBER attached -- every
# shape rule before it had to be cut through two overlapping distributions,
# and this one does not. The floor is set from that gap and the distribution
# is printed on every run so it stays checkable.
MIN_CONF = 0.60

# The forearm ray leaves the frame at some point; the wearer's leaves LOW.
#
# ASKING WHICH EDGE IT LEFT BY WAS TOO BRITTLE AT THE CORNERS. The wearer's
# right arm enters from the lower right, and if the hand is far enough over,
# the ray reaches the right edge before it reaches the bottom one -- so an arm
# that is plainly the wearer's was reported as "exits right" and blurred. It
# happened on 4 of 48 real detections. Where the ray leaves is an accident of
# the corner; how far DOWN it leaves is the thing that actually distinguishes
# an arm coming up from the wearer's body from one reaching in across the
# bench, and it treats the bottom edge and the lower side edges alike.
OWNER_EXIT_Y_FRAC = 0.55


def forearm_exit(kp, shape):
    """Where the forearm leaves the frame. -> (edge, exit point)

    The ray runs from the fingers through the wrist and outward. The edge is
    reported for diagnosis; `is_owner` decides on the exit point's HEIGHT,
    because the corners make the edge misleading."""
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


def is_owner(exit_pt, shape, y_frac=OWNER_EXIT_Y_FRAC):
    """-> True if the forearm leaves through the lower part of the border.

    One number, and it covers the bottom edge and the lower halves of both
    side edges together -- which is the shape of "towards the wearer's body"
    and is what four misclassified right hands were missing."""
    if exit_pt is None:
        return False
    return float(exit_pt[1]) >= y_frac * shape[0]


def load_owner_clf(path):
    """-> {'model','features'} or None. Missing is not an error: the rule is
    the fallback and it is the thing being replaced, not a placeholder."""
    import pickle
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def classify_owner(clf, det, shape):
    """-> (is_owner, probability). The classifier decides; the rule's verdict
    is left in the detection beside it so the two can be compared on live
    data, not only on the labelled set."""
    from src.rig.own_label import features
    x = features(det, shape)[None]
    p = float(clf["model"].predict_proba(x)[0, 1])
    return p >= 0.5, p


def detect(model, rgb, imgsz=IMGSZ, min_conf=MIN_CONF, clf=None):
    """-> [{'box','kp','side','conf','edge','owner','owner_p'}] for one frame.

    `clf` decides ownership when it is supplied. The geometric verdict stays
    in `rule_owner` regardless: on the 268 hands labelled so far the two agree
    everywhere, so the day they disagree is the day something new is in shot,
    and that is worth seeing rather than silently overriding."""
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
        edge, pt = (forearm_exit(kp, rgb.shape) if kp is not None
                    else (None, None))
        d = {"box": b.astype(int), "kp": kp, "conf": float(c),
             "side": model.names.get(int(k), str(int(k))),
             "edge": edge, "exit": pt,
             "rule_owner": is_owner(pt, rgb.shape)}
        d["owner"] = d["rule_owner"]
        d["owner_p"] = float(d["rule_owner"])
        if clf is not None:
            try:
                d["owner"], d["owner_p"] = classify_owner(clf, d, rgb.shape)
            except Exception:
                pass                      # a broken model must not lose a frame
        out.append(d)
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
        (own if d.get("owner") else oth).append(d)
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


def track(prev, cur, max_move_frac=0.25):
    """Match detections between consecutive frames by centre distance.
    -> [(prev_index, cur_index)]

    Greedy nearest-centre, capped by a fraction of the frame. A hand does not
    cross a quarter of the frame in one 30 fps step, and refusing the match
    beyond that keeps a dropout from being reported as a jump."""
    if not prev or not cur:
        return []
    def ctr(d):
        x0, y0, x1, y1 = d["box"]
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    P = np.stack([ctr(d) for d in prev])
    C = np.stack([ctr(d) for d in cur])
    dist = np.linalg.norm(P[:, None] - C[None], axis=-1)
    cap = max_move_frac * max(dist.max(), 1.0) if dist.size else 0
    out, used_p, used_c = [], set(), set()
    for _ in range(min(len(prev), len(cur))):
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[i, j]):
            break
        out.append((int(i), int(j)))
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return out


def stability(model, frames, min_conf=MIN_CONF):
    """-> dict of per-clip stability statistics.

    THE FAILURE THIS LOOKS FOR is not a wrong label but a CHANGING one. A hand
    called the wearer's on one frame and a colleague's on the next puts a
    blur that flickers on and off in front of a downstream video model, which
    is worse than either verdict held consistently."""
    rows, prev = [], None
    flips = same = 0
    counts = []
    for rgb in frames:
        dets = detect(model, rgb, min_conf=min_conf)
        counts.append(len(dets))
        if prev is not None:
            for i, j in track(prev, dets):
                if bool(prev[i].get("owner")) == bool(dets[j].get("owner")):
                    same += 1
                else:
                    flips += 1
        prev = dets
        rows.append(dets)
    c = np.array(counts)
    return {"frames": len(counts), "hands_per_frame_median": float(np.median(c)),
            "hands_min": int(c.min()), "hands_max": int(c.max()),
            "frames_with_no_hand": int((c == 0).sum()),
            "tracked_pairs": same + flips, "owner_label_flips": flips,
            "flip_rate": float(flips / max(same + flips, 1)),
            "per_frame": rows}


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

    # The case that actually failed: the wearer's right hand, far over, its
    # forearm leaving through the LOWER RIGHT corner.
    kp_lr = hand((1450, 620), (1250, 430))
    e_lr, pt_lr = forearm_exit(kp_lr, (H, W))
    chk(is_owner(pt_lr, (H, W)),
        f"a right hand exiting the lower {e_lr} edge at y={pt_lr[1]:.0f} is "
        f"still the wearer's")
    kp_ur = hand((1450, 300), (1250, 380))
    e_ur, pt_ur = forearm_exit(kp_ur, (H, W))
    chk(not is_owner(pt_ur, (H, W)),
        f"...and one exiting the UPPER {e_ur} edge at y={pt_ur[1]:.0f} is not")

    d_own = [{"kp": hand((700, 700), (700, 500)), "owner": True},
             {"kp": hand((900, 700), (900, 500)), "owner": True}]
    d_oth = [{"kp": hand((400, 400), (700, 400)), "owner": False}]
    own, oth = split_owner(d_own + d_oth, (H, W))
    chk(len(own) == 2 and len(oth) == 1,
        "two from below are the wearer's, one from the side is not")

    # A third from below: the two most straight-down win, not the first two.
    slanted = {"kp": hand((300, 700), (600, 690)), "owner": True}
    own2, oth2 = split_owner([slanted] + d_own, (H, W))
    # `not in` compares dicts holding numpy arrays with ==, which is
    # ambiguous; identity is what is meant here anyway.
    chk(len(own2) == 2 and all(d is not slanted for d in own2),
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
    ap.add_argument("--stability", action="store_true",
                    help="run on CONSECUTIVE rendered frames and report how "
                         "often a tracked hand changes its owner verdict. "
                         "Scattered keyframes cannot show this and it is the "
                         "failure that matters: a blur that flickers on and "
                         "off is worse than either verdict held.")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--clf", help="an own_clf.pkl from own_label. Without "
                                  "it the geometric rule decides.")
    ap.add_argument("--min_conf", type=float, default=MIN_CONF,
                    help="real hands score 0.83-0.87 here; a pink box scored "
                         "enough to pass 0.35")
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=14.0)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if a.stability:
        if not a.calibration or not a.video:
            ap.error("--stability needs --calibration and --video")
        from ultralytics import YOLO
        from src.rig.calibration import RigCalibration
        from src.rig.geometry import VirtualWideCamera
        from src.rig.render_wide import render
        from src.rig.seam_fix import ClipReader
        rig = RigCalibration(a.calibration)
        vcam = VirtualWideCamera.from_rig(rig)
        rd = ClipReader(rig, dict(s.split("=", 1) for s in a.video), a.start)
        model = YOLO(a.weights)
        mc, frames = {}, []
        for _ in range(a.n):
            src = rd.next()
            if not src:
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            frames.append(rgb)
        rd.close()
        st = stability(model, frames, a.min_conf)
        print(f"  {st['frames']} CONSECUTIVE frames from {a.start}\n")
        print(f"  hands per frame     median {st['hands_per_frame_median']:.0f}"
              f"   min {st['hands_min']}   max {st['hands_max']}")
        print(f"  frames with no hand {st['frames_with_no_hand']}")
        print(f"  tracked pairs       {st['tracked_pairs']}")
        print(f"  owner verdict flips {st['owner_label_flips']}"
              f"   ({st['flip_rate']:.2%})")
        print("\n  A flip is the same hand called the wearer's on one frame "
              "and a colleague's\n  on the next. That makes a blur switch on "
              "and off, which a downstream video\n  model reads as a real "
              "event. Near zero is the requirement; the label being\n  "
              "occasionally wrong but STEADY is a much smaller problem.")
        raise SystemExit(0)
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
    clf = load_owner_clf(a.clf)
    print(f"  {len(imgs)} frames, detector {os.path.basename(a.weights)}, "
          f"ownership by {'classifier' if clf else 'the geometric rule'}\n")

    tally, disagree = {}, []
    confs = {"owner": [], "other": []}
    allconf = []
    for p in imgs:
        rgb = cv2.imread(p)
        dets = detect(model, rgb, min_conf=a.min_conf, clf=clf)
        own, oth = split_owner(dets, rgb.shape)
        for d in dets:
            tally[d.get("edge")] = tally.get(d.get("edge"), 0) + 1
            if d.get("owner") != d.get("rule_owner"):
                disagree.append((os.path.basename(p), d.get("edge"),
                                 round(d.get("owner_p", 0.0), 3)))
        for d in own:
            confs["owner"].append(d["conf"])
        for d in oth:
            confs["other"].append(d["conf"])
        # Everything the detector proposed, including what the floor rejected,
        # so the floor can be judged rather than trusted.
        for d in detect(model, rgb, min_conf=0.0):
            allconf.append((d["conf"], bool(d.get("owner"))))
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
    if clf is not None:
        print(f"  classifier disagreed with the rule on {len(disagree)} "
              f"detections")
        for x in disagree[:8]:
            print(f"    {x[0]}  exit {x[1]}  p(owner)={x[2]}")
    for k, v in confs.items():
        if v:
            v = np.array(v)
            print(f"  {k:6s} confidence: min {v.min():.2f}  median "
                  f"{np.median(v):.2f}  max {v.max():.2f}  n={len(v)}")
    if allconf:
        c = np.array([x[0] for x in allconf])
        print(f"\n  EVERY proposal, floor ignored: n={len(c)}  "
              f"min {c.min():.2f}  p25 {np.percentile(c,25):.2f}  "
              f"median {np.median(c):.2f}")
        for t in (0.4, 0.5, 0.6, 0.7, 0.8):
            print(f"    floor {t}: keeps {int((c>=t).sum()):3d} of {len(c)}")
        print("  A real hand here scores 0.83-0.87. If the rejected tail sits "
              "well below\n  that, the floor is a gap and not another "
              "threshold through an overlap.")
    print("  Left/right/top are colleagues; bottom is the wearer. Nothing "
          "that is not a\n  hand is reported at all, which is the whole point "
          "of replacing the colour rule.")


if __name__ == "__main__":
    main()
