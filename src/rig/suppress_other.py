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

WHAT FEEDS IT, TODAY, AND THE PROBLEM WITH THAT. There is no trained
`other_arm` mask: the class was removed after all 83 of its components on this
corpus were inspected and none was a hand -- solidity 0.96-0.99 was a wooden
turntable and 0.49-0.56 a beige machine strap, because a colleague across the
aisle is smaller than the detector's area floor. So the honest default here is
`skin-like components minus the owner mask`, which is "hands in view that are
not yours" and which, on this corpus, is mostly furniture.

Blurring a turntable in a fixed position on every frame is worse than blurring
nothing: the downstream model sees a permanent smudge. That is why the
suppressed FRACTION is reported per frame -- a number that never moves is the
signature of furniture, and it is meant to be looked at before this is turned
on for a whole recording.
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

# A skin component overlapping the owner's mask by this much is the OWNER'S,
# entire. Not the overlapping pixels -- the whole component.
OWNER_OVERLAP_FRAC = 0.20


def other_components(rgb, owner_mask, min_area_frac=0.006,
                     max_area_frac=0.25, max_solidity=MAX_SOLIDITY,
                     require_border=MUST_TOUCH_BORDER):
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
            why.append((a, "owner: enters from the bottom"))
            continue
        ov = float((comp & own).sum()) / max(int(comp.sum()), 1)
        if ov >= OWNER_OVERLAP_FRAC:
            # The other 21%: a hand cut off from its arm by a sleeve, floating
            # clear of every border. The mask is the only thing that can claim
            # those, so it is used here and only here.
            protect |= comp
            why.append((a, "owner: claimed by the mask"))
            continue
        if not (min_area_frac <= a <= max_area_frac):
            why.append((a, "size"))
            continue
        if require_border and not (L or R or T):
            why.append((a, "touches no border -- attached to nobody"))
            continue
        ys, xs = np.where(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        sol = comp.sum() / max(cv2.contourArea(cv2.convexHull(pts)), 1e-6)
        if sol > max_solidity:
            why.append((a, f"solidity {sol:.2f} -- compact, not a limb"))
            continue
        other |= comp
        why.append((a, "OTHER"))
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
    chk(not oth[120:180, 200:260].any(),
        "a compact object touching no border is left alone entirely")

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
    fracs, reasons = [], {}
    print(f"  dilate {a.dilate}px  feather {a.feather}px  sigma {a.sigma}\n")
    for r in rows:
        stem = f"{r['recording']}_f{int(r['frame']):06d}.png"
        rgb = cv2.imread(os.path.join(a.src, "images", stem))
        m = cv2.imread(os.path.join(a.src, "masks", stem),
                       cv2.IMREAD_GRAYSCALE)
        if rgb is None or m is None:
            continue
        owner = m == 1
        other, protect, why = other_components(rgb, owner)
        out, alpha = suppress(rgb, other, a.dilate, a.feather, a.sigma,
                              protect=protect)
        for ar, w in why:
            reasons[w] = reasons.get(w, 0) + 1
        fracs.append(float((alpha > 0.5).mean()))
        cv2.imwrite(os.path.join(a.out, stem.replace(".png", ".jpg")), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not fracs:
        raise SystemExit("nothing processed")
    f = np.array(fracs)
    print(f"  {len(f)} frames, suppressed fraction: median {np.median(f):.2%}"
          f"  p10 {np.percentile(f,10):.2%}  p90 {np.percentile(f,90):.2%}"
          f"  std {f.std():.3%}")
    print("\n  components by verdict: " + ", ".join(
        f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    print("\n  READ THE SPREAD, NOT THE MEDIAN. A fraction that barely moves "
          "between frames\n  means the same object is being blurred every "
          "time, and on this corpus that\n  object is a wooden turntable, not "
          "a colleague. A real hand comes and goes.")


if __name__ == "__main__":
    main()
