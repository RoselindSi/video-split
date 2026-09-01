"""Remove other people's hands from the frame without touching the frame.

The requirement is to keep a colleague's hands out of what the downstream
model sees, while leaving the scene otherwise intact. Cropping fails that on
its own terms -- the field of view would change from frame to frame, which is
a stronger artificial signal than the hands were, and it takes the bench and
the tools with it. A mosaic keeps the shape, the position and the motion of
the hand and adds a block texture that a model can learn to read as "somebody
else was here". So: dilate the mask, feather it, and blur hard inside it.

DILATION IS NOT A SAFETY MARGIN, IT IS THE POINT. A segmentation IoU is never
1.0, and the pixels it misses are the ones at the boundary: half a fingertip,
a wrist edge, a cuff. Suppressing exactly the predicted region leaves exactly
those. The dilation is sized against the error the mask actually has, not
against how it looks.

THE WHOLE FRAME IS BLURRED ONCE AND COMPOSITED THROUGH A SOFT ALPHA. Blurring
only the masked pixels would draw its samples from inside the region, so the
hand's own colours would be smeared into a hand-shaped smudge; blurring the
frame lets the surrounding bench flow inward instead. Pixels outside the
feather are bit-identical to the input, which is checked rather than assumed.

WHAT FEEDS IT. A hand detector proposes the regions and a classifier decides
whose each one is; GrabCut finds the boundary inside the detector's box. So
`other` is a positive claim about something already established to be a hand.
`suppress` is the last step and takes that mask as given -- it does not decide
anything, and nothing downstream of it defines `other`.

THE ORDER MATTERS AND USED TO BE THE OTHER WAY ROUND. The first version had no
hand detector, so it thresholded skin-like colour over the whole frame, took
connected components, subtracted the owner mask, and called the remainder
`other`. That defines a colleague's arm as a residue -- not "I recognised
someone else's hand" but "I could not explain this as yours" -- so anything
skin-coloured fell in. Of the 83 components it called `other_arm` on this
corpus, none was a hand: solidity 0.96-0.99 was a wooden turntable, 0.49-0.56 a
beige machine strap. Worse, a turntable sits in the same place on every frame,
so the downstream model saw a permanent smudge, which is a worse input than no
suppression at all.

`other_components` and `static_mask` below are that older path. Nothing in the
pipeline calls them; they are kept because the border and solidity statistics
in their comments are the measurements that justified replacing them.

The suppressed FRACTION is still reported per frame, and is still worth
reading: a number that never moves is the signature of a fixed object being
covered, which is the failure that survived from the old design into any
future one.
"""
from __future__ import annotations

import numpy as np

# Pixels. Sized against the boundary error a real mask has, not against how
# the mask looks: at 1600x900 an IoU around 0.76 leaves a rim several pixels
# wide, and a fingertip is wider than the rim.
DILATE_PX = 10

# The alpha ramp. Wide enough that no edge is visible, narrow enough that the
# ramp itself does not become a halo the model can read.
FEATHER_PX = 4

# Gaussian sigma inside the region. Large enough to destroy a fingerprint, a
# cuff seam and a ring; the identity detail is what has to go, not the fact
# that something is there.
BLUR_SIGMA = 14.0


def dilate_feather(mask, dilate_px=DILATE_PX, feather_px=FEATHER_PX):
    """binary mask -> float alpha in [0,1], grown then softened.

    Grow first and soften second. Softening a mask and then growing it would
    push the ramp outward and leave the hard original edge inside it."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    # Grow by the requested margin PLUS TWICE the feather width, and the
    # factor of two is corner geometry rather than caution. An isotropic
    # feather leaves a straight edge at 0.5 but a right-angled corner at about
    # 0.25, because the corner sees blur from two sides -- and a mask's worst
    # coverage is always at its corners. Measured on a square region short by
    # 10 px: dilating by 10 gave 0.13 at the corner, by 10+feather gave 0.46,
    # by 10+2*feather clears 0.5 everywhere. The feather is paid for on top of
    # the margin, twice.
    grow = int(dilate_px) + 2 * int(feather_px)
    if grow > 0:
        k = 2 * grow + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    a = m.astype(np.float32) / 255.0
    if feather_px > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(feather_px))
    return np.clip(a, 0.0, 1.0)


def suppress(rgb, other_mask, dilate_px=DILATE_PX, feather_px=FEATHER_PX,
             blur_sigma=BLUR_SIGMA, protect=None):
    """-> (frame, alpha). Blur hard inside the grown mask, leave the rest.

    `protect` is the owner's mask: wherever the two overlap the owner wins,
    because a suppressor that erases the hands the pipeline exists to watch
    has failed in the one way that matters. The overlap is real -- two arms
    meeting at the same part is exactly when the masks touch -- and resolving
    it silently in the other direction would be invisible until the downstream
    model started missing grasps."""
    import cv2
    a = dilate_feather(other_mask, dilate_px, feather_px)
    if protect is not None:
        keep = (np.asarray(protect) > 0)
        a = np.where(keep, 0.0, a).astype(np.float32)
        if feather_px > 0:
            # Re-soften only where the owner carved a hole, so the protected
            # boundary is a ramp too rather than a fresh hard edge.
            a = cv2.GaussianBlur(a, (0, 0), float(feather_px))
            a = np.where(keep, 0.0, a).astype(np.float32)
    if a.max() <= 0:
        return np.asarray(rgb).copy(), a
    blurred = cv2.GaussianBlur(np.asarray(rgb), (0, 0), float(blur_sigma))
    out = (np.asarray(rgb).astype(np.float32) * (1 - a[..., None])
           + blurred.astype(np.float32) * a[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), a


# A candidate must reach a border, because a person's arm is attached to a
# person, and everything but the wearer is outside the frame. Measured over
# 218 components on this corpus:
#
#                touches a border   L/R    top   bottom   solidity
#     owner  121        80%          4%     0%     79%      0.77
#     other   97        35%         30%      5%      0%      0.95
#
# 65% of what the old subtraction called "other" touched no border at all --
# the wooden turntable, floating in the middle of the bench. Requiring a
# border, forbidding the bottom one (which is where the wearer's own arm
# enters, 79% against 0%), and capping solidity removes it.
MUST_TOUCH_BORDER = True
MAX_SOLIDITY = 0.90
BORDER_PX = 3

# An arm bends. A strap, a conveyor edge and a pillar do not, and those were
# what survived every other filter -- 42 blurred components whose median
# straightness was 0.992, against 121 real owner arms whose median was 0.932
# and whose MAXIMUM was 0.974. The cut sits in the gap between those two
# numbers rather than at a round figure:
#
#     straight < 0.97   removes 74% of the false positives, costs 1% of arms
#     straight < 0.98   removes 69%, costs 0%
#
# Aspect ratio was tested alongside and adds nothing on top of it (74% / 1%
# either way), so it is not applied -- a second threshold that changes no
# decision is a second thing to get wrong later.
#
# 26% survive: a curved conveyor edge and a round paper disc, both bent and
# both skin-coloured. Single-frame shape has nothing left to say about those.
# What separates them from an arm is that they do not move, and that needs
# consecutive frames rather than a keyframe.
MAX_STRAIGHTNESS = 0.97

# A skin component overlapping the owner's mask by this much is the OWNER'S,
# entire. Not the overlapping pixels -- the whole component.
OWNER_OVERLAP_FRAC = 0.20


# A pixel that is skin-coloured in this share of a recording's frames is not
# a person. Nobody holds an arm in one place for minutes; a paper disc, a
# cabinet and a conveyor edge do exactly that.
STATIC_FRAC = 0.7


def static_mask(frames, frac=STATIC_FRAC):
    """Pixels that are skin-coloured in most frames of a recording. -> bool

    THIS IS THE FILTER SINGLE-FRAME SHAPE COULD NOT PROVIDE, and the reason to
    reach for it is that the shape cuts ran out of gap. Straightness worked
    because real arms topped out at 0.974 and the straps sat at 0.992 -- a
    real separation. A minimum aspect ratio does not: arms reach down to 1.84
    at the 5th percentile and the paper disc sits at 1.7, so buying 45% of the
    remaining false positives would cost 5% of real arms.

    Persistence has no such overlap and needs no threshold fitted to one. A
    colleague's arm enters, does something and leaves. A cabinet is in every
    frame of the recording. Computed once per recording and reused, so it
    costs nothing per frame."""
    from src.rig.near_other_miner import skin_mask
    acc = None
    n = 0
    for f in frames:
        m = skin_mask(f) > 0
        acc = m.astype(np.float32) if acc is None else acc + m
        n += 1
    if not n:
        return None
    return (acc / n) >= frac


def other_components(rgb, owner_mask, min_area_frac=0.006,
                     max_area_frac=0.25, max_solidity=MAX_SOLIDITY,
                     require_border=MUST_TOUCH_BORDER,
                     max_straightness=MAX_STRAIGHTNESS,
                     static=None, static_overlap=0.6):
    """-> (other mask, owner-protected mask, per-component reasons)

    OWNERSHIP IS DECIDED PER COMPONENT, NOT PER PIXEL, AND THAT IS THE WHOLE
    FIX. The first version took `skin AND NOT owner_mask`, which turns every
    pixel the owner mask MISSED into a foreign region -- and the owner mask
    scores 0.717 against hand-drawn truth, so it misses a rim on every arm.
    The result was the wearer's own forearm blurred along its edges, a
    compounding failure where an imperfect mask manufactures the very thing
    the suppressor then destroys. Here a component that overlaps the owner's
    mask at all is the owner's in full, and is protected in full.

    What remains has to look like an arm belonging to someone outside the
    frame: reaching a border, not the bottom one, and not compact."""
    import cv2
    from src.rig.near_other_miner import skin_mask
    sk = skin_mask(rgb) > 0
    own = np.asarray(owner_mask) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
    H, W = sk.shape
    other = np.zeros((H, W), bool)
    protect = own.copy()
    why = []
    for i in range(1, n):
        comp = lab == i
        a = stats[i, cv2.CC_STAT_AREA] / sk.size
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        L, R = x <= BORDER_PX, x + w >= W - BORDER_PX
        T, B = y <= BORDER_PX, y + h >= H - BORDER_PX
        # ENTERING FROM THE BOTTOM IS THE OWNER, AND IT IS DECIDED FIRST AND
        # WITHOUT A MODEL. 79% of owner components reach the bottom border
        # against 0% of everything else -- the cleanest separation in this
        # corpus, and the rule the task was originally specified with. Making
        # the owner's protection depend on a segmentation that scores 0.717
        # against hand-drawn truth was the mistake: the geometry is more
        # reliable than the mask it was guarding.
        if B:
            protect |= comp
            why.append((a, "owner: enters from the bottom", None))
            continue
        ov = float((comp & own).sum()) / max(int(comp.sum()), 1)
        if ov >= OWNER_OVERLAP_FRAC:
            # The other 21%: a hand cut off from its arm by a sleeve, floating
            # clear of every border. The mask is the only thing that can claim
            # those, so it is used here and only here.
            protect |= comp
            why.append((a, "owner: claimed by the mask", None))
            continue
        if not (min_area_frac <= a <= max_area_frac):
            why.append((a, "size", None))
            continue
        if require_border and not (L or R or T):
            why.append((a, "touches no border -- attached to nobody", None))
            continue
        ys, xs = np.where(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        sol = comp.sum() / max(cv2.contourArea(cv2.convexHull(pts)), 1e-6)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        ar = max(rw, rh) / max(min(rw, rh), 1e-6)
        # Straightness: how well the component's pixels fit a single line. A
        # pillar or a strap is straight; an arm bends at the elbow and at the
        # wrist. Reported rather than thresholded -- the threshold gets picked
        # from what the failures actually measure, the way the turntable's
        # was, not from what seems reasonable now.
        c = pts - pts.mean(0)
        ev = np.linalg.eigvalsh(np.cov(c.T)) if len(c) > 2 else np.array([1, 1])
        straight = float(1.0 - ev.min() / max(ev.max(), 1e-9))
        stat = {"area": a, "sol": float(sol), "ar": float(ar),
                "straight": straight, "bbox": (int(x), int(y), int(w), int(h)),
                "border": "".join(t for t, f in
                                  (("L", L), ("R", R), ("T", T), ("B", B)) if f)}
        if sol > max_solidity:
            why.append((a, f"solidity {sol:.2f} -- compact, not a limb", stat))
            continue
        if static is not None:
            ov_s = float((comp & static).sum()) / max(int(comp.sum()), 1)
            if ov_s >= static_overlap:
                why.append((a, f"static in {ov_s:.0%} of the component -- "
                               f"furniture, not a person", stat))
                continue
        if straight > max_straightness:
            why.append((a, f"straightness {straight:.3f} -- a strap or an "
                           f"edge, not a limb", stat))
            continue
        other |= comp
        why.append((a, "OTHER", stat))
    return other, protect, why


def _self_test():
    import cv2
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    H = W = 300
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)   # all high frequency
    true = np.zeros((H, W), bool)
    true[100:200, 100:200] = True

    out, a = suppress(rgb, true, dilate_px=10, feather_px=4, blur_sigma=14)

    def hf(img, sel):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F)[sel].std())

    core = np.zeros((H, W), bool)
    core[120:180, 120:180] = True
    chk(hf(out, core) < 0.15 * hf(rgb, core),
        f"detail inside is destroyed ({hf(rgb,core):.0f} -> {hf(out,core):.0f})")

    far = np.zeros((H, W), bool)
    far[:60, :60] = True
    chk(np.array_equal(out[far], rgb[far]),
        "pixels away from the mask are bit-identical to the input")

    # The case dilation exists for: a mask that is short of the truth.
    short = np.zeros((H, W), bool)
    short[110:190, 110:190] = True          # 10 px inside the real region
    _, a2 = suppress(rgb, short, dilate_px=10, feather_px=4)
    chk(a2[true].min() > 0.5,
        f"a mask 10 px short still covers the whole true region "
        f"(min alpha {a2[true].min():.2f})")
    _, a3 = suppress(rgb, short, dilate_px=0, feather_px=0)
    chk(a3[true].min() == 0.0,
        "...and without dilation it does not -- the rim survives")

    # The owner must never be blurred.
    own = np.zeros((H, W), bool)
    own[150:250, 150:250] = True
    out4, a4 = suppress(rgb, true, protect=own)
    chk(a4[own].max() == 0.0, "the owner's pixels are never suppressed")
    chk(np.array_equal(out4[own], rgb[own]),
        "...and come through untouched")

    # Component-level ownership: an owner mask that misses a rim must not
    # manufacture a foreign region out of the arm it failed to claim.
    from src.rig.suppress_other import other_components
    scene = np.full((H, W, 3), (60, 90, 45), np.uint8)
    scene[200:300, 120:190] = (110, 150, 200)     # owner arm, up from bottom
    # A real arm is bent -- measured solidity 0.77 median, 0.90 at p90 -- so
    # a rectangle is not a fair stand-in for one and would be rejected by the
    # solidity cap that exists to catch the turntable.
    scene[40:100, 0:60] = (110, 150, 200)         # other arm, in from the left
    scene[100:150, 0:25] = (110, 150, 200)        # ...bent at the elbow
    scene[120:180, 200:260] = (110, 150, 200)     # a compact object, mid-frame
    partial = np.zeros((H, W), bool)
    partial[215:290, 132:178] = True              # owner mask, a rim short
    oth, prot, why = other_components(scene, partial, min_area_frac=0.004)
    chk(not oth[200:300, 120:190].any(),
        "an owner mask short by a rim does not turn its own arm into `other`")
    chk(prot[200:300, 120:190].all(),
        "...the whole component is protected, not just the claimed pixels")
    chk(oth[40:150, 0:60].any(),
        "an arm entering from the left IS other")
    straightbar = np.full((H, W, 3), (60, 90, 45), np.uint8)
    straightbar[0:200, 40:70] = (110, 150, 200)   # a strap: skin-toned, T, bent-free
    o2, _, _ = other_components(straightbar, np.zeros((H, W), bool),
                                min_area_frac=0.004)
    chk(not o2.any(),
        "a straight skin-coloured bar from the top is a strap, not an arm")
    chk(not oth[120:180, 200:260].any(),
        "a compact object touching no border is left alone entirely")

    # Persistence: the same tan shape in every frame is furniture.
    from src.rig.suppress_other import static_mask
    # The arm MOVES between frames and the disc does not. A first version of
    # this test held both still, so the arm was static too and was correctly
    # excluded -- the test was wrong, not the filter.
    seq = []
    for k in range(6):
        f_ = np.full((H, W, 3), (60, 90, 45), np.uint8)
        dy = k * 18
        f_[40 + dy:100 + dy, 0:60] = (110, 150, 200)     # arm, sweeping down
        f_[100 + dy:150 + dy, 0:25] = (110, 150, 200)    # ...bent at the elbow
        f_[10:70, 200:255] = (110, 150, 200)             # a disc, never moves
        seq.append(f_)
    stat = static_mask(seq)
    chk(stat[20:60, 210:250].all(),
        "the disc is in the static mask")
    chk(not stat[40:100, 0:60].all(),
        "...and a moving arm is not")
    o6, _, _ = other_components(seq[0], np.zeros((H, W), bool),
                                min_area_frac=0.004, static=stat)
    chk(not o6[10:70, 200:255].any(),
        "a shape present in every frame is furniture and is left alone")
    chk(o6[40:150, 0:60].any(),
        "...while an arm in the same frames is still suppressed")

    o5, a5 = suppress(rgb, np.zeros((H, W), bool))
    chk(np.array_equal(o5, rgb) and a5.max() == 0,
        "an empty mask leaves the frame exactly alone")

    chk(0.0 <= dilate_feather(true).min() and dilate_feather(true).max() <= 1.0,
        "alpha stays inside [0,1]")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Dilation is not a margin: without it a mask short by 10 px "
          "leaves the rim,\n  which is where a fingertip and a cuff edge "
          "live.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--src", help="a seg_auto split with images/ and masks/")
    ap.add_argument("--out")
    ap.add_argument("--head", help="checkpoint; without it the owner mask "
                                   "comes from masks/ (the pseudo-labels)")
    ap.add_argument("--dilate", type=int, default=DILATE_PX)
    ap.add_argument("--feather", type=int, default=FEATHER_PX)
    ap.add_argument("--sigma", type=float, default=BLUR_SIGMA)
    ap.add_argument("--static_from", type=int, default=0,
                    help="build a per-recording static mask from this many of "
                         "its frames and exclude anything that sits inside "
                         "it. Persistence is what separates a paper disc from "
                         "an arm; shape has run out of gap.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.src or not a.out:
        ap.error("--src and --out are required without --self_test")

    import csv
    import os
    import cv2
    rows = list(csv.DictReader(open(os.path.join(a.src, "manifest.csv"),
                                    encoding="utf-8-sig")))
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    fracs, reasons, crops, blurred_stats = [], {}, [], []
    print(f"  dilate {a.dilate}px  feather {a.feather}px  sigma {a.sigma}\n")
    statics = {}
    if a.static_from:
        by_rec = {}
        for r in rows:
            by_rec.setdefault(r["recording"], []).append(r)
        for rid, rs in by_rec.items():
            imgs = []
            for r in rs[:a.static_from]:
                st = f"{r['recording']}_f{int(r['frame']):06d}.png"
                im = cv2.imread(os.path.join(a.src, "images", st))
                if im is not None:
                    imgs.append(im)
            if len(imgs) >= 3:
                statics[rid] = static_mask(imgs)
                print(f"  {rid}: static over {len(imgs)} frames, "
                      f"{statics[rid].mean():.1%} of the frame")
        print()
    for r in rows:
        stem = f"{r['recording']}_f{int(r['frame']):06d}.png"
        rgb = cv2.imread(os.path.join(a.src, "images", stem))
        m = cv2.imread(os.path.join(a.src, "masks", stem),
                       cv2.IMREAD_GRAYSCALE)
        if rgb is None or m is None:
            continue
        owner = m == 1
        other, protect, why = other_components(
            rgb, owner, static=statics.get(r["recording"]))
        out, alpha = suppress(rgb, other, a.dilate, a.feather, a.sigma,
                              protect=protect)
        for item in why:
            ar, w = item[0], item[1]
            reasons[w] = reasons.get(w, 0) + 1
            st = item[2] if len(item) > 2 else None
            if w == "OTHER" and st is not None:
                bx, by, bw, bh = st["bbox"]
                pad = int(max(bw, bh) * 0.25)
                y0, y1 = max(0, by-pad), min(rgb.shape[0], by+bh+pad)
                x0, x1 = max(0, bx-pad), min(rgb.shape[1], bx+bw+pad)
                crops.append((cv2.resize(rgb[y0:y1, x0:x1], (240, 240)), st))
                blurred_stats.append(dict(stem=stem, **{k: v for k, v in
                                     st.items() if k != "bbox"}))
        fracs.append(float((alpha > 0.5).mean()))
        cv2.imwrite(os.path.join(a.out, stem.replace(".png", ".jpg")), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not fracs:
        raise SystemExit("nothing processed")
    f = np.array(fracs)
    print(f"  {len(f)} frames, suppressed fraction: median {np.median(f):.2%}"
          f"  p10 {np.percentile(f,10):.2%}  p90 {np.percentile(f,90):.2%}"
          f"  std {f.std():.3%}")
    if crops:
        import math
        cols = 6
        rowsn = math.ceil(len(crops) / cols)
        sheet = np.zeros((rowsn*240, cols*240, 3), np.uint8)
        for j, (c, st) in enumerate(crops):
            cv2.rectangle(c, (0, 0), (240, 40), (0, 0, 0), -1)
            cv2.putText(c, f"sol{st['sol']:.2f} ar{st['ar']:.1f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(c, f"str{st['straight']:.2f} {st['border']} "
                        f"{st['area']:.1%}", (4, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            r_, c_ = divmod(j, cols)
            sheet[r_*240:(r_+1)*240, c_*240:(c_+1)*240] = c
        cv2.imwrite(os.path.join(a.out, "_blurred_components.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(os.path.join(a.out, "_blurred_components.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=list(blurred_stats[0].keys()))
            w_.writeheader(); w_.writerows(blurred_stats)
        print(f"\n  {len(crops)} components were blurred -- every one of them "
              f"is cropped into\n  _blurred_components.jpg with its "
              f"statistics. That is the picture to look at:\n  the filter for "
              f"whatever is still wrong gets chosen from what they measure.")
    print("\n  components by verdict: " + ", ".join(
        f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    print("\n  READ THE SPREAD, NOT THE MEDIAN. A fraction that barely moves "
          "between frames\n  means the same object is being blurred every "
          "time, and on this corpus that\n  object is a wooden turntable, not "
          "a colleague. A real hand comes and goes.")


if __name__ == "__main__":
    main()
