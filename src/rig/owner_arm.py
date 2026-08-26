"""OWNER_ARM_MASK -- the wearer's own forearms and hands, from depth alone.

Three conditions together, and none of them alone:

    near          inside a hard physical range, and NEARER THAN ITS OWN
                  SURROUNDINGS by a margin
    connected     the component touches the bottom of the frame, where the
                  wearer's own body is
    persistent    it was there last frame, or it has to earn the mask afresh

The near test alone would take a bin sitting at 0.6 m. The bottom-connected
test alone would take the wearer's apron. Together they take an arm.

WHY A HARD DISTANCE IS NOT THE RULE. On this data the forearm sits at a median
0.39 m against a 3.14 m background, and it would be easy to freeze 0.20-0.85 m
and move on. But mounting height, the wearer's build and their posture all
shift that, and a hand at 0.9 m would fall off a hard 0.85 m cliff with no
warning. The hard range here is only a sanity bound; the discriminating test
is RELATIVE -- a component must be substantially nearer than the workspace
immediately around it. Separation of that kind is what the geometry actually
measured, and it survives a change of rig position.

WHAT THIS DOES NOT DO, stated because the name invites the mistake. It finds
the wearer's own arms. It does NOT find other people's hands. Everything
outside OWNER_ARM_MASK is bench, parts, bins, machines, other people's bodies
AND other people's hands, undifferentiated. Masking a colleague's hands in the
RGB while keeping the rest of the scene needs a hand detector -- and when that
arrives it should be a CANDIDATE GENERATOR only:

    detector says "this looks like a hand"
    this mask says whose it is

not the other way round. A detector guessing ownership from appearance is the
thing this module exists to avoid.

STATUS 2026-08-26: V0 DOES NOT YET WORK ON REAL FRAMES. It is committed with
the failure recorded rather than hidden, because the mechanism is worth
keeping and the next attempt should start from it.

Three formulations were tried against real frames and all three leaked the
bench into the mask:

    a range over the near band        the bench's front edge is genuinely
                                     near, genuinely reaches the bottom, and
                                     is adjacent to the sleeve. Merged
                                     component: 34% of the frame, passing the
                                     relative test on the arm's median.
    a narrower range around a seed    the entry band's depths pile up at the
                                     disparity limit, so a low percentile
                                     pinned the seed to 0.19 m and the window
                                     then cut off the hand.
    neighbour-to-neighbour growth     23% of the frame. The forearm RESTS ON
                                     the bench, so at the contact there is no
                                     depth step to stop at, and SGBM's own
                                     noise is already several centimetres.

The three are ONE failure wearing three costumes, and measuring the mask says
which. Across frames 6000 / 1000 / 12000 it spans 90.0% / 90.0% / 87.5% of
image COLUMNS at a mean column fill of 26% / 34% / 19%. A forearm is a compact
blob; nothing shaped like an arm reaches nine columns in ten. The mask is a
strip running the full width, which is the bench's front edge.

And the edge is not separable by distance. The bottom band's far RIGHT corner
-- bench, never arm -- measures 0.25 / 0.23 / 0.21 m, against a mask median of
0.26 / 0.30 / 0.23 m. THE BENCH EDGE IS AT THE SAME DEPTH AS THE FOREARM. No
threshold, no relative margin and no seed can separate two things the sensor
places at the same distance; every formulation above was asking depth for a
distinction depth does not carry.

What does carry it is depth's STRUCTURE rather than its value. The bench is a
plane and an arm is not, at any distance. A plane fitted to the workspace
removes every pixel consistent with it regardless of what touches it or how
near it is -- which is a scene assumption, not a local rule, and is the next
thing to try. Its own risk is stated in advance: a flat part lying on the
bench goes with the plane, and a forearm laid flat ALONG the bench may too.

Everything below the segmentation is sound and tested: the three conditions,
the hysteresis, the two-component cap, the refusal to own a component with no
measurable workspace behind it. Only the growth step is unresolved.

SGBM LEAVES HOLES and that is survivable here. Coverage runs 49-58% with heavy
speckle on the textureless bench, but a forearm is a large, high-contrast,
close object -- the one thing the matcher does find. Morphological closing
repairs the speckle inside a component; the component's median depth is
immune to what is left.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Sanity bounds, NOT the discriminating test. A human forearm cannot be at
# 5 cm or 3 m from a body-worn camera; between those the relative test decides.
HARD_MIN_M = 0.15
HARD_MAX_M = 1.00

# A component must be this much nearer than the workspace around it. Measured
# separation on this rig is 0.39 m against 3.14 m, a ratio of 8, so 1.6 is a
# wide margin that still rejects a bin resting on the same bench as the hand.
MIN_SURROUND_RATIO = 1.6
SURROUND_DILATE_PX = 45

# The band at the bottom of the frame the wearer's own limbs come through.
# The frame's bottom is the VIRTUAL camera's bottom -- cam1 is mounted rolled
# and its raw frame has the arm entering at 60 degrees from the lower left.
# geometry.FAN_UP_SIGN fixes that convention and inverting it inverts this.
ENTRY_BAND_FRAC = 0.12

# The largest step allowed BETWEEN NEIGHBOURING PIXELS while growing. This is
# a surface-continuity statement, not a distance rule: two adjacent samples on
# one forearm differ by centimetres, and the jump from a sleeve to the bench
# edge behind it is far larger. It does not care where the wearer's arm
# happens to be, which a range window did.
MAX_STEP_M = 0.06

MIN_AREA_FRAC = 0.004          # smaller than this is speckle, not a limb
MAX_COMPONENTS = 2             # a person has two arms
CLOSE_PX = 9                   # repairs SGBM speckle inside a limb

# Hysteresis. A component overlapping last frame's mask keeps it on weaker
# evidence, because an arm that briefly fails a test has not stopped being the
# wearer's, and a mask that flickers is worse downstream than one that lags.
HYSTERESIS_IOU = 0.20
HYSTERESIS_RATIO = 1.25
MAX_MISSING = 8


@dataclass
class OwnerArmResult:
    mask: np.ndarray                  # [H,W] bool
    components: list = field(default_factory=list)
    surround_depth_m: float = float("nan")
    seed_depth_m: float = float("nan")
    rejected: list = field(default_factory=list)

    def coverage(self):
        return float(self.mask.mean())


def _components(near, close_px=CLOSE_PX):
    import cv2
    m = near.astype(np.uint8)
    if close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    return lab, stats, cents, n


class OwnerArmSegmenter:
    """Per-frame OWNER_ARM_MASK with temporal hysteresis.

    Call `update` with each frame's depth map. Keeps the previous mask so a
    component that momentarily fails the relative test -- because the arm
    crossed in front of a near bin, or the matcher dropped half of it -- stays
    owned rather than blinking out."""

    def __init__(self, hard_min_m=HARD_MIN_M, hard_max_m=HARD_MAX_M,
                 min_surround_ratio=MIN_SURROUND_RATIO,
                 entry_band_frac=ENTRY_BAND_FRAC,
                 max_components=MAX_COMPONENTS, min_area_frac=MIN_AREA_FRAC):
        self.hard = (hard_min_m, hard_max_m)
        self.ratio = min_surround_ratio
        self.entry_band_frac = entry_band_frac
        self.max_components = max_components
        self.min_area_frac = min_area_frac
        self.prev_mask = None
        self.missing = 0

    def update(self, depth_m):
        import cv2
        H, W = depth_m.shape
        finite = np.isfinite(depth_m)
        lo, hi = self.hard
        near = finite & (depth_m >= lo) & (depth_m <= hi)

        res = OwnerArmResult(mask=np.zeros((H, W), bool))
        if not near.any():
            return self._settle(res)

        entry_row = int(H * (1 - self.entry_band_frac))

        # SEED FROM THE ENTRY BAND, GROW BY SURFACE CONTINUITY.
        #
        # A range mask over the whole near band merged the arm with the bench's
        # front edge, which is genuinely near, genuinely reaches the bottom and
        # is genuinely adjacent to the sleeve. The merged component then passed
        # the relative test on the arm's own median while dragging 34% of the
        # frame with it, because a component median cannot see that the
        # component is heterogeneous.
        #
        # Replacing the range with a NARROWER range only moved the failure: the
        # entry band's depths pile up at the disparity limit, so a low
        # percentile pins the seed to 0.19 m and the window then cuts the hand
        # off. Both are the same mistake -- a range, chosen globally.
        #
        # Growth is now by neighbour-to-neighbour continuity. Two samples on
        # one forearm differ by centimetres; the sleeve-to-bench jump does not.
        band = near[entry_row:, :]
        band_z = depth_m[entry_row:, :][band]
        if band_z.size < 64:
            return self._settle(res)
        res.seed_depth_m = float(np.median(band_z))

        fill_src = np.where(finite, depth_m, 1e6).astype(np.float32)
        fmask = np.zeros((H + 2, W + 2), np.uint8)
        fmask[1:-1, 1:-1] = (~near).astype(np.uint8)     # never leave `near`
        ys, xs = np.where(near[entry_row:, :])
        if ys.size == 0:
            return self._settle(res)
        # a spread of seeds along the band, so one noisy pixel cannot decide
        for j in np.linspace(0, ys.size - 1, min(48, ys.size)).astype(int):
            y, x = int(ys[j]) + entry_row, int(xs[j])
            if fmask[y + 1, x + 1]:
                continue
            cv2.floodFill(fill_src, fmask, (x, y), 0,
                          MAX_STEP_M, MAX_STEP_M,
                          4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        grown = (fmask[1:-1, 1:-1] == 255) & near
        if not grown.any():
            return self._settle(res)

        work = depth_m[finite & ~near]
        workspace_z = float(np.median(work)) if work.size > 256 \
            else float("nan")
        res.surround_depth_m = workspace_z

        lab, stats, cents, n = _components(grown)
        min_area = self.min_area_frac * H * W
        keep = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            comp = lab == i
            # 1. connected to the band the wearer's own limbs come through
            if not comp[entry_row:, :].any():
                res.rejected.append({"area": int(area),
                                     "why": "does not reach the bottom band"})
                continue
            zs = depth_m[comp & finite]
            if zs.size < 64:
                continue
            med = float(np.median(zs))

            # 2. nearer than the WORKSPACE, not than a fixed number.
            #
            # The comparison is the workspace at large -- everything measured
            # that is not in the near band -- and not a ring hugging the
            # component. A 45 px ring around a forearm is mostly the bench
            # edge a few centimetres behind it, which put the ratio at 1.34
            # and rejected the arm; the workspace behind it is at 1.92 m,
            # against the arm's 0.28, and that separation is unambiguous.
            #
            # A near bin resting on that bench would also clear this test.
            # It does not need to fail here: it fails the bottom-band test and
            # the continuity growth. The three conditions carry different
            # weight, and loading them all onto one is how a threshold ends up
            # tuned against the wrong failure.
            sur = workspace_z

            # No surroundings means the separation could not be MEASURED, and
            # absence of evidence is not evidence: a near mask that fills the
            # frame -- the wearer's own apron across the whole lower half, a
            # matcher failure that calls everything 0.5 m -- has nothing to be
            # nearer than. Treating that as infinite separation let the apron
            # case claim the entire frame.
            if not np.isfinite(sur) or med <= 0:
                res.rejected.append({
                    "area": int(area), "median_m": med,
                    "surround_m": float("nan"),
                    "why": "no measurable surroundings to be nearer than"})
                continue
            ratio = sur / med

            need = self.ratio
            was_owned = (self.prev_mask is not None
                         and (comp & self.prev_mask).sum()
                         / max(comp.sum(), 1) >= HYSTERESIS_IOU)
            if was_owned:
                need = HYSTERESIS_RATIO
            if ratio < need:
                res.rejected.append({
                    "area": int(area), "median_m": med, "surround_m": sur,
                    "why": f"only {ratio:.2f}x nearer than its surroundings, "
                           f"needs {need:.2f}x"})
                continue
            keep.append({"label": i, "area": int(area), "median_m": med,
                         "surround_m": sur, "ratio": float(ratio),
                         "held": bool(was_owned), "mask": comp})

        keep.sort(key=lambda c: -c["area"])
        if len(keep) > self.max_components:
            for c in keep[self.max_components:]:
                res.rejected.append({
                    "area": c["area"], "median_m": c["median_m"],
                    "why": f"a person has {self.max_components} arms and this "
                           f"was the smallest of {len(keep)} candidates"})
            keep = keep[:self.max_components]

        for c in keep:
            res.mask |= c["mask"]
            c.pop("mask")
        res.components = keep
        return self._settle(res)

    def _settle(self, res):
        if res.mask.any():
            self.prev_mask = res.mask
            self.missing = 0
        else:
            self.missing += 1
            if self.missing > MAX_MISSING:
                self.prev_mask = None
        return res


def overlay(rgb, res, alpha=0.45):
    """Debug view: the mask tinted, each component labelled with its depth."""
    import cv2
    out = rgb.copy()
    tint = np.zeros_like(out)
    tint[res.mask] = (60, 220, 90)
    out = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)
    out[res.mask] = cv2.addWeighted(rgb, 0.75, tint, 0.25, 0)[res.mask]
    h = out.shape[0]
    y = 34
    cv2.putText(out, f"OWNER_ARM_MASK  {res.coverage():.1%} of frame"
                     f"   seed {res.seed_depth_m:.2f} m",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 240, 100), 2)
    for c in res.components:
        y += 30
        cv2.putText(out, f"  {c['median_m']:.2f} m vs {c['surround_m']:.2f} m "
                         f"surround ({c['ratio']:.1f}x)"
                         + ("  [held]" if c["held"] else ""),
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 240, 100), 2)
    for r in res.rejected[:3]:
        y += 26
        cv2.putText(out, f"  rejected: {r['why']}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 140, 250), 2)
    return out


def _self_test():
    H, W = 300, 400
    ok = 0

    def depth(bg=3.0):
        return np.full((H, W), bg, np.float32)

    def arm(d, z=0.4, x0=170, x1=230, y0=150):
        d[y0:H, x0:x1] = z
        return d

    def case(name, d, expect_cov, **kw):
        nonlocal ok
        s = OwnerArmSegmenter(**kw)
        r = s.update(d)
        got = r.coverage() > 0.001
        assert got == expect_cov, \
            f"{name}: expected {'mask' if expect_cov else 'no mask'}, " \
            f"got {r.coverage():.3%}\n  rejected: {r.rejected}"
        ok += 1
        print(f"  ok  {name}")
        return r

    case("a near limb reaching the bottom is owned", arm(depth()), True)

    # a bin at the same distance, not touching the bottom, must not be an arm
    d = depth()
    d[60:130, 60:140] = 0.45
    case("a near object away from the bottom is not an arm", d, False)

    # the wearer's apron: touches the bottom, but is not nearer than the
    # things around it
    d = depth(0.5)
    d[250:H, :] = 0.45
    case("bottom-connected but no separation from its surroundings is not an "
         "arm", d, False)

    # the hard range is a sanity bound, not the discriminator: an arm at
    # 0.9 m -- a taller wearer, a different mount -- is still an arm
    r = case("an arm at 0.9 m against a 4 m background is still owned",
             arm(depth(4.0), z=0.9), True)
    assert r.components[0]["median_m"] > 0.85, r.components
    ok += 1
    print("  ok  ...and a hard 0.85 m cliff would have dropped it")

    # two arms in, a third candidate out
    d = depth()
    d = arm(d, x0=80, x1=140)
    d = arm(d, x0=180, x1=240)
    d = arm(d, x0=280, x1=330)
    r = case("three bottom-connected limbs keep the two largest", d, True)
    assert len(r.components) == 2, r.components

    # Hysteresis: a frame where the separation weakens keeps the mask.
    # The weak frame has to stay PHYSICAL -- an earlier version set the whole
    # scene to 0.62 m, which leaves the component nothing to be nearer than
    # and tests the wrong thing. Here the workspace crowds in to 1.05 m, just
    # outside the hard range, against an arm at 0.75 m: a ratio of 1.4, under
    # the 1.6 needed fresh and over the 1.25 needed to hold.
    s = OwnerArmSegmenter()
    s.update(arm(depth()))
    weak = depth(1.05)
    weak = arm(weak, z=0.75)
    r = s.update(weak)
    assert r.coverage() > 0.001 and r.components[0]["held"], \
        f"lost the arm when the surroundings closed in: {r.rejected}"
    ok += 1
    print("  ok  a held component survives a frame where separation weakens")

    s = OwnerArmSegmenter()
    r = s.update(weak)
    assert r.coverage() < 0.001, \
        "the same weak frame must NOT start a new mask from nothing"
    ok += 1
    print("  ok  ...but that leniency is only for something already owned")

    print(f"\n  {ok} cases pass. The near test alone takes a bin; the "
          f"bottom test alone\n  takes an apron; only together do they take "
          f"an arm.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
