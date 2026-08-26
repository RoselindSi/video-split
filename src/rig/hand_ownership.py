"""Which hands belong to the person wearing the camera.

A multi-hand detector says where hands are. It does not say whose they are,
and in a factory the frame regularly contains a colleague working across the
bench -- the raw cam1 view at frame 6000 has the operator's own arm at the
bottom and another worker's yellow glove in the top left. Everything
downstream that reasons about hand-object interaction needs those separated.

This module takes detections from ANY detector and decides ownership. The
detector is a parameter; the rule is the thing worth getting right.

TWO SIGNALS, AND THE DEPTH ONE IS THE STRONGER.

    depth      the wearer's own hands are 0.2-0.6 m away; a colleague across
               the bench is past 1.5 m. Measured on this rig: the working
               surface sits at a median 0.43 m with the 95th percentile at
               4.06 m, so the two populations barely overlap.
    entry      a track that first appears at the bottom of the frame came up
               from the wearer's own body.

Depth is available because each module is a calibrated 60 mm stereo pair, so
it is used when supplied and the geometry alone can carry the decision. Entry
position works without it and is what the first version runs on.

WHY ENTRY POSITION IS ABOUT THE TRACK, NOT THE FRAME. The obvious rule --
"a hand low in the frame is the wearer's" -- fails the moment the operator
reaches out, which is precisely when the hand matters most. Ownership is
therefore decided ONCE, when a track is born, and then held: a track that
entered from the bottom stays owned wherever it goes, and one that entered
from the top stays foreign however close it comes.

THE FRAME'S BOTTOM IS DEFINED BY THE RENDER, NOT BY THE SENSOR. cam1 is
mounted rolled; in its raw frame the wearer's arm enters from the lower left
at about 60 degrees. It is only in the virtual wide camera, whose up is the
fan's rotation axis, that the arm comes from the bottom. `FAN_UP_SIGN` in
geometry.py fixes that convention, and flipping it inverts this rule too --
which is why both modules name it.

MASK, DO NOT CROP. A colleague may be holding the very part the operator is
about to take. Removing their hands from the frame keeps that context; cutting
away their side of the image does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A track must start with its box bottom inside this fraction of the frame
# height, measured from the bottom, to be a candidate owner.
ENTRY_ZONE_FRAC = 0.28

# ...and it must not have entered through the top or the sides.
FOREIGN_MARGIN_FRAC = 0.06

# Metres. Below NEAR the hand is the wearer's on depth alone; above FAR it is
# someone else's. Between them depth abstains and entry decides.
OWNER_DEPTH_MAX_M = 0.85
FOREIGN_DEPTH_MIN_M = 1.40

# Frames a track survives without a detection before it is retired. Hands
# disappear behind the workpiece constantly; dropping the track would let it
# be reborn mid-frame and lose its entry evidence.
MAX_MISSING = 12
MAX_MATCH_DIST_FRAC = 0.09          # centroid distance, fraction of diagonal

OWNER, FOREIGN, UNDECIDED = "owner", "foreign", "undecided"


@dataclass
class Track:
    tid: int
    boxes: dict = field(default_factory=dict)      # frame -> (x0,y0,x1,y1)
    depths: dict = field(default_factory=dict)     # frame -> metres or None
    born_frame: int = 0
    born_box: tuple = (0, 0, 0, 0)
    missing: int = 0
    verdict: str = UNDECIDED
    reason: str = ""

    def centroid(self, f):
        x0, y0, x1, y1 = self.boxes[f]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def last_frame(self):
        return max(self.boxes)

    def predict(self, frame):
        """Where this track should be at `frame`, at constant velocity.

        Matching against the last SEEN position loses a track exactly when it
        moves fastest -- which is when the operator reaches out, and which is
        the moment ownership matters most, because a track reborn mid-frame
        has no entry evidence and falls back to undecided. Extrapolating from
        the last two observations keeps it."""
        fs = sorted(self.boxes)
        last = fs[-1]
        c = self.centroid(last)
        if len(fs) < 2:
            return c
        prev = fs[-2]
        dt = last - prev
        if dt <= 0:
            return c
        p = self.centroid(prev)
        v = ((c[0] - p[0]) / dt, (c[1] - p[1]) / dt)
        gap = frame - last
        return (c[0] + v[0] * gap, c[1] + v[1] * gap)


def _entry_verdict(box, w, h):
    """Where a track was born, as evidence rather than as a decision."""
    x0, y0, x1, y1 = box
    if y1 >= h * (1 - ENTRY_ZONE_FRAC):
        return OWNER, f"born with its lower edge in the bottom " \
                      f"{ENTRY_ZONE_FRAC:.0%} of the frame"
    m = FOREIGN_MARGIN_FRAC
    if y0 <= h * m:
        return FOREIGN, "entered through the top edge"
    if x0 <= w * m or x1 >= w * (1 - m):
        return FOREIGN, "entered through a side edge"
    return UNDECIDED, "appeared mid-frame, away from every edge"


def _depth_verdict(z):
    if z is None or not np.isfinite(z):
        return UNDECIDED, ""
    if z <= OWNER_DEPTH_MAX_M:
        return OWNER, f"{z:.2f} m away, inside the wearer's own reach"
    if z >= FOREIGN_DEPTH_MIN_M:
        return FOREIGN, f"{z:.2f} m away, beyond the wearer's reach"
    return UNDECIDED, f"{z:.2f} m away, between the two thresholds"


class HandOwnership:
    """Tracks hands across frames and decides whose they are.

    `update` takes one frame's detections and returns a verdict per detection.
    Verdicts are stable: a track's ownership is settled at birth from whatever
    evidence exists then, and revised only to RESOLVE an undecided track, never
    to flip a decided one. A hand that flickers between owner and foreign as it
    moves would be worse than a wrong constant answer, because everything
    downstream is temporal."""

    def __init__(self, width, height, max_owners=2,
                 entry_zone_frac=ENTRY_ZONE_FRAC, max_missing=MAX_MISSING):
        self.w, self.h = width, height
        self.diag = float(np.hypot(width, height))
        self.max_owners = max_owners
        self.entry_zone_frac = entry_zone_frac
        self.max_missing = max_missing
        self.tracks = {}
        self._next = 0

    def update(self, frame, boxes, depths=None):
        """boxes: [(x0,y0,x1,y1), ...]; depths: metres or None per box.

        -> [(track_id, verdict, reason), ...] aligned with `boxes`."""
        depths = list(depths) if depths is not None else [None] * len(boxes)
        live = [t for t in self.tracks.values()
                if frame - t.last_frame() <= self.max_missing]
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

        # greedy nearest-centroid association, closest pair first
        pairs = []
        for i, c in enumerate(cents):
            for t in live:
                lc = t.predict(frame)
                d = np.hypot(c[0] - lc[0], c[1] - lc[1]) / self.diag
                if d <= MAX_MATCH_DIST_FRAC:
                    pairs.append((d, i, t.tid))
        pairs.sort()
        taken_det, taken_trk = set(), set()
        assign = {}
        for d, i, tid in pairs:
            if i in taken_det or tid in taken_trk:
                continue
            assign[i] = tid
            taken_det.add(i)
            taken_trk.add(tid)

        out = []
        for i, b in enumerate(boxes):
            if i in assign:
                t = self.tracks[assign[i]]
            else:
                t = Track(tid=self._next, born_frame=frame, born_box=b)
                self.tracks[self._next] = t
                self._next += 1
                v, why = _entry_verdict(b, self.w, self.h)
                t.verdict, t.reason = v, why
            t.boxes[frame] = b
            t.depths[frame] = depths[i]

            # depth may settle a track that entry could not, and may confirm
            # one it did. It never overturns a settled verdict: a decision
            # that flips mid-track is worse than a wrong constant one.
            if t.verdict == UNDECIDED:
                dv, why = _depth_verdict(depths[i])
                if dv != UNDECIDED:
                    t.verdict, t.reason = dv, why
            out.append((t.tid, t.verdict, t.reason))

        return self._cap_owners(frame, out)

    def _cap_owners(self, frame, out):
        """At most `max_owners` owned tracks at once.

        A person has two hands. When a third claims ownership -- a colleague
        reaching in from below, a reflection, a detector error -- the ones kept
        are those with the most evidence: first a longer history, because a
        track owned for a hundred frames has been confirmed a hundred times,
        then a birth deeper into the entry zone.

        WHEN THE EVIDENCE IS EXACTLY TIED, NOBODY IS KEPT. Three tracks born
        in the same frame at the same height are indistinguishable, and
        picking two of them by list order would be an arbitrary decision made
        silently -- the caller would see two confident owners and no sign that
        the choice was a coin flip. Undecided is the honest answer, and it
        propagates: mask_foreign leaves undecided hands alone."""
        owned = [k for k, (_, v, _) in enumerate(out) if v == OWNER]
        if len(owned) <= self.max_owners:
            return out
        def evidence(k):
            t = self.tracks[out[k][0]]
            return (len(t.boxes), t.born_box[3])      # history, then depth in
        ranked = sorted(owned, key=evidence, reverse=True)
        cutoff = evidence(ranked[self.max_owners - 1])
        tied = [k for k in ranked if evidence(k) == cutoff]
        if len(tied) > 1 and len(ranked) > self.max_owners \
                and evidence(ranked[self.max_owners]) == cutoff:
            drop = set(tied)          # the tie spans the cut: keep none of it
        else:
            drop = set(ranked[self.max_owners:])
        for k in drop:
            tid, _, why = out[k]
            out[k] = (tid, UNDECIDED,
                      f"{why}, but {len(owned)} tracks claimed ownership and "
                      f"this one is not among the {self.max_owners} best "
                      f"supported")
        return out


def mask_foreign(image, boxes, verdicts, pad=8, blur=41):
    """Blur every hand that is not the wearer's, in place on a copy.

    Blur rather than fill: a filled rectangle is an object in its own right and
    invents an occluder that was never there. And blur rather than crop --
    a colleague may be holding the part the operator is reaching for, and that
    context is worth keeping even when their hands are not."""
    import cv2
    out = image.copy()
    h, w = out.shape[:2]
    for b, v in zip(boxes, verdicts):
        if v == OWNER:
            continue
        x0 = max(0, int(b[0]) - pad)
        y0 = max(0, int(b[1]) - pad)
        x1 = min(w, int(b[2]) + pad)
        y1 = min(h, int(b[3]) + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        k = blur | 1
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (k, k), 0)
    return out


def _self_test():
    """Every case here is a way the obvious rule gets it wrong."""
    W, H = 1600, 900
    ok = 0

    def case(name, frames, expect, **kw):
        nonlocal ok
        t = HandOwnership(W, H, **kw)
        last = None
        for f, (boxes, depths) in enumerate(frames):
            last = t.update(f, boxes, depths)
        got = [v for _, v, _ in last]
        assert got == expect, f"{name}: expected {expect}, got {got}\n  " \
                              f"{[r for _, _, r in last]}"
        ok += 1
        print(f"  ok  {name}")

    # reaching out must not lose ownership -- the naive "low in the frame"
    # rule fails exactly when the hand becomes interesting
    # ~120 px/frame at 30 fps is roughly 2.4 m/s at this working distance:
    # fast, and physical. Tests should reproduce the motion the matcher will
    # actually meet, not stress it past what a hand can do.
    case("own hand enters from the bottom, then reaches to the centre",
         [([(700, 830, 820, 899)], [None]),
          ([(700, 710, 820, 830)], [None]),
          ([(700, 590, 820, 710)], [None]),
          ([(700, 470, 820, 590)], [None]),
          ([(700, 380, 820, 500)], [None])],
         [OWNER])

    case("colleague's hand from the top stays foreign at the centre",
         [([(600, 0, 720, 100)], [None]),
          ([(600, 110, 720, 210)], [None]),
          ([(600, 220, 720, 320)], [None]),
          ([(600, 330, 720, 430)], [None])],
         [FOREIGN])

    case("appearing mid-frame with no depth is undecided, not guessed",
         [([(700, 400, 820, 520)], [None])], [UNDECIDED])

    case("depth settles a mid-frame appearance",
         [([(700, 400, 820, 520)], [0.42])], [OWNER])

    case("far depth settles it the other way",
         [([(700, 400, 820, 520)], [2.6])], [FOREIGN])

    case("depth between the thresholds still abstains",
         [([(700, 400, 820, 520)], [1.1])], [UNDECIDED])

    # a track survives a gap; without that it is reborn mid-frame and loses
    # the entry evidence that made it an owner
    t = HandOwnership(W, H)
    t.update(0, [(700, 830, 820, 899)], [None])
    for f in range(1, 6):
        t.update(f, [], [])
    r = t.update(6, [(706, 760, 826, 880)], [None])
    assert r[0][1] == OWNER, f"track lost across a gap: {r}"
    assert r[0][0] == 0, f"reborn as a new track: {r}"
    ok += 1
    print("  ok  a track survives a five-frame occlusion and keeps ownership")

    # three born together at the same height: nothing separates them, so no
    # two of them get to be confident owners
    case("three simultaneous, identical claims give nobody ownership",
         [([(200, 830, 320, 899), (700, 830, 820, 899),
            (1200, 830, 1320, 899)], [None, None, None])],
         [UNDECIDED, UNDECIDED, UNDECIDED])

    # ...but when the evidence differs, it decides
    case("the two born deeper into the entry zone win the tie-break",
         [([(200, 700, 320, 780), (700, 830, 820, 899),
            (1200, 820, 1320, 890)], [None, None, None])],
         [UNDECIDED, OWNER, OWNER])

    t = HandOwnership(W, H)
    for f in range(30):
        t.update(f, [(200, 830, 320, 899), (700, 830, 820, 899)],
                 [None, None])
    r = t.update(30, [(200, 830, 320, 899), (700, 830, 820, 899),
                      (1200, 830, 1320, 899)], [None, None, None])
    got = [v for _, v, _ in r]
    assert got == [OWNER, OWNER, UNDECIDED], got
    ok += 1
    print("  ok  two long-lived owners outrank a newcomer claiming the third "
          "slot")

    print(f"\n  {ok} cases pass. Each is a way the obvious rule gets it wrong.")


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
