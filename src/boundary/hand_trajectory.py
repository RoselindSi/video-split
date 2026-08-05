"""Pure trajectory logic for the hand-trajectory probe: track association,
gap handling, boxes.

Kept separate from the extraction script so every part that can be checked
without a GPU or a working mediapipe install IS checked. Cross-frame hand
identity in particular is not a detail -- if the two hands swap identity
between frames, the derived velocity flips sign, and a direction-reversal
feature built on that measures the matcher rather than the hand.
"""
from __future__ import annotations

import numpy as np


def box_from_landmarks(lm, w, h):
    """(x0, y0, x1, y1) in pixels from normalised landmarks. MediaPipe
    extrapolates landmarks outside the image when a hand is partly out of
    frame, so the box is NOT clipped here -- whether it exceeds the frame is
    signal (see edge_touch), not something to hide."""
    xs = [p[0] * w for p in lm]
    ys = [p[1] * h for p in lm]
    return (min(xs), min(ys), max(xs), max(ys))


def expand_box(b, margin, w=None, h=None):
    x0, y0, x1, y1 = b
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def edge_touch(b, w, h, tol=1.0):
    """Does the RAW box reach or cross a frame edge? A hand leaving the frame
    is the mechanism behind the offscreen subtype, so this is recorded per
    frame rather than inferred later from a clipped box."""
    x0, y0, x1, y1 = b
    return bool(x0 <= tol or y0 <= tol or x1 >= w - tol or y1 >= h - tol)


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _diag(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def match_cost(prev, cur, diag=None, handed_penalty=0.5):
    """Cost of calling `cur` the same hand as `prev`.

    Centre displacement normalised by the mean BOX diagonal (not the frame
    diagonal), minus IoU, plus a penalty when handedness disagrees.

    The normalisation choice decides whether handedness or geometry wins, and
    the first version got it wrong. Against the frame diagonal a 100 px move
    costs 0.125, so a 0.5 handedness penalty dominates and the "penalty"
    becomes a hard constraint -- on a test where two hands approach each other
    with the labels deliberately swapped, the matcher followed the labels and
    swapped the tracks. Against the box diagonal that same move costs about
    0.9, so geometry decides when it is clear and handedness only breaks ties.
    That is what was intended: mediapipe's Left/Right label is unreliable on
    egocentric footage, and an identity swap flips the sign of every derived
    velocity, so a direction-reversal feature built on it would be measuring
    the matcher."""
    pc, cc = _centre(prev["box"]), _centre(cur["box"])
    scale = (_diag(prev["box"]) + _diag(cur["box"])) / 2.0
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = diag if (diag and diag > 0) else 1.0
    d = np.hypot(cc[0] - pc[0], cc[1] - pc[1]) / scale
    cost = d - box_iou(prev["box"], cur["box"])
    if prev.get("handedness") and cur.get("handedness") \
            and prev["handedness"] != cur["handedness"]:
        cost += handed_penalty
    return cost


def associate(prev_tracks, dets, diag=None, max_cost=3.0):
    """Assign each detection in `dets` a track id, given the previous frame's
    tracks {track_id: det}. Exhaustive minimum-cost assignment -- with at most
    two hands there are at most two permutations, so no solver is needed and
    the greedy failure mode (locking in a bad first pair) cannot occur.

    Detections that match nothing under max_cost start a new track."""
    if not dets:
        return {}
    if not prev_tracks:
        return {i: d for i, d in enumerate(dets)}
    pids = sorted(prev_tracks)
    best, best_cost = None, float("inf")
    import itertools
    k = min(len(pids), len(dets))
    for pperm in itertools.permutations(range(len(dets)), k):
        for psub in itertools.combinations(range(len(pids)), k):
            c = sum(match_cost(prev_tracks[pids[p]], dets[d], diag)
                    for p, d in zip(psub, pperm))
            if c < best_cost:
                best_cost, best = c, list(zip(psub, pperm))
    out = {}
    used = set()
    nxt = max(pids) + 1
    for p, d in (best or []):
        if match_cost(prev_tracks[pids[p]], dets[d], diag) <= max_cost:
            out[pids[p]] = dets[d]
            used.add(d)
    for i, d in enumerate(dets):
        if i not in used:
            out[nxt] = d
            nxt += 1
    return out


def interpolate_gaps(values, valid, max_gap):
    """Linearly interpolate runs of at most `max_gap` invalid samples between
    valid ones. Longer runs stay missing.

    Never extrapolates past the first or last valid sample, and never bridges
    a long gap: a two-second hole filled by a straight line is not an
    observation, and the earlier local-crop work showed exactly that failure --
    a 64-second interpolated stretch scored as the most STABLE trajectory in
    the set because a straight line has no jitter."""
    v = np.asarray(values, dtype=float)
    ok = np.asarray(valid, dtype=bool).copy()
    out = v.copy()
    idx = np.nonzero(ok)[0]
    filled = np.zeros(len(v), dtype=bool)
    if len(idx) < 2:
        return out, ok, filled
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                out[k] = v[a] * (1 - w) + v[b] * w
                ok[k] = True
                filled[k] = True
    return out, ok, filled


def savgol(y, valid, window=5, poly=2):
    """Savitzky-Golay applied ONLY inside runs of consecutive valid samples.
    Smoothing across a missing stretch would blend observations from opposite
    sides of a hole into a value at neither."""
    y = np.asarray(y, dtype=float)
    ok = np.asarray(valid, dtype=bool)
    out = y.copy()
    i = 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        seg = y[i:j + 1]
        if len(seg) >= window:
            try:
                from scipy.signal import savgol_filter
                out[i:j + 1] = savgol_filter(seg, window, poly)
            except ImportError:
                k = window // 2
                sm = np.convolve(seg, np.ones(window) / window, mode="same")
                sm[:k], sm[-k:] = seg[:k], seg[-k:]
                out[i:j + 1] = sm
        i = j + 1
    return out
