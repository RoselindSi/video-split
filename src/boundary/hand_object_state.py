"""Per-frame hand-object relation, and the reset events derived from it.

Layers 1-3 of the C3.2 design, as pure logic over an abstract detection
record, so every part that can be checked without a detector IS checked. The
detector is deliberately not named here: the input is a list of boxes with
optional labels and scores, and swapping an open-vocabulary detector for a
hand-object interaction model changes the extractor, not this file.

CONTACT IS A PROXY AND MUST BE READ AS ONE. Nothing in this stack outputs a
contact flag. What is computed here is fingertip-to-object proximity,
normalised by hand size, with box overlap as a fallback when landmarks are
absent. That is a geometric approximation of contact, and its failure mode is
specific: a hand passing in front of an object at a different depth reads as
contact, because a single camera cannot separate them. The whole
interaction-reset layer inherits that error, so `contact_state` returns the
evidence (`iou`, `gap`) alongside the verdict and every downstream feature
keeps a variant computed at a stricter threshold. If the two variants disagree
about a result, the result is a threshold artefact.

OBJECT IDENTITY ACROSS FRAMES is what layers 2 and 3 actually rest on -- "the
same object, released and re-grasped" and "the same object, held throughout"
are the same measurement with opposite signs, and both are wrong if the object
track breaks. Association is IoU-plus-label with an explicit gap tolerance,
and the number of track breaks is reported so a feature built on a shattered
track can be recognised rather than trusted. The hand tracker learned this the
expensive way: normalising displacement by the frame diagonal made a
handedness penalty dominate, tracks swapped, and every derived velocity
flipped sign.
"""
from __future__ import annotations

import numpy as np

# fingertip landmark indices in mediapipe's 21-point hand model
FINGERTIPS = (4, 8, 12, 16, 20)

CONTACT, NEAR, FREE = "contact", "near", "free"


def _diag(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def _point_box_gap(p, b):
    """Distance from a point to a box; 0 inside."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def contact_state(hand, obj, contact_gap=0.15, near_gap=0.50, contact_iou=0.05):
    """Geometric proxy for whether `hand` is touching `obj`.

    Uses fingertip-to-object distance when landmarks are present and the hand
    box otherwise. Fingertips matter: a hand box drawn around a spread hand
    overlaps a nearby object long before anything touches it, so box overlap
    alone reports contact for a hand hovering over a plate.

    Returns (state, evidence) rather than a bare flag, because the states are
    threshold decisions on continuous evidence and a feature that cannot be
    re-derived at another threshold cannot be checked for threshold
    sensitivity."""
    hb = hand["box"]
    scale = _diag(hb)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    iou = _iou(hb, obj["box"])
    lm = hand.get("landmarks")
    if lm and len(lm) > max(FINGERTIPS):
        w, h = hand.get("frame_wh", (1.0, 1.0))
        pts = [(lm[i][0] * w, lm[i][1] * h) for i in FINGERTIPS]
        gap = min(_point_box_gap(p, obj["box"]) for p in pts) / scale
    else:
        cx = (hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0
        gap = _point_box_gap(cx, obj["box"]) / scale
    if iou >= contact_iou or gap <= contact_gap:
        st = CONTACT
    elif gap <= near_gap:
        st = NEAR
    else:
        st = FREE
    return st, {"iou": float(iou), "gap": float(gap)}


def associate_objects(prev, dets, iou_min=0.20, label_bonus=0.15, max_cost=0.90):
    """Track ids for object detections, given the previous frame's tracks.

    Greedy by IoU, with a bonus for a matching label. Greedy is adequate here
    and it is not adequate for hands: two hands are few enough to enumerate and
    they cross, while a scene holds many objects that rarely swap places, so
    the cost of an exhaustive assignment buys nothing.

    A detection matching nothing starts a new track. `prev` may hold tracks
    that were not seen this frame -- the caller decides how long to keep them,
    since an object occluded for three frames is still the same object and an
    object gone for three seconds is not."""
    out, used = {}, set()
    pairs = []
    for pid, p in prev.items():
        for j, d in enumerate(dets):
            iou = _iou(p["box"], d["box"])
            if iou < iou_min:
                continue
            cost = 1.0 - iou
            if p.get("label") and d.get("label") and p["label"] == d["label"]:
                cost -= label_bonus
            pairs.append((cost, pid, j))
    for cost, pid, j in sorted(pairs):
        if pid in out or j in used or cost > max_cost:
            continue
        out[pid] = dets[j]
        used.add(j)
    nxt = (max(prev) + 1) if prev else 0
    for j, d in enumerate(dets):
        if j not in used:
            out[nxt] = d
            nxt += 1
    return out


def relation_frame(hands, objects, **kw):
    """LAYER 1. One frame's hand-object relation.

    For every hand: the object it is closest to, the contact state, and the
    relative position of the hand within that object's box. Relative position
    is normalised into the object's own frame so approaching a cup and
    approaching a pan are comparable, which the raw pixel offset is not."""
    rel = {}
    for hid, h in hands.items():
        best = None
        for oid, o in objects.items():
            st, ev = contact_state(h, o, **kw)
            rank = (0 if st == CONTACT else 1 if st == NEAR else 2, ev["gap"])
            if best is None or rank < best[0]:
                best = (rank, oid, st, ev)
        if best is None:
            rel[hid] = {"object": None, "state": FREE, "iou": 0.0,
                        "gap": float("nan"), "rel_xy": (np.nan, np.nan)}
            continue
        _, oid, st, ev = best
        ob = objects[oid]["box"]
        hc = ((h["box"][0] + h["box"][2]) / 2.0, (h["box"][1] + h["box"][3]) / 2.0)
        ow, oh = max(ob[2] - ob[0], 1e-6), max(ob[3] - ob[1], 1e-6)
        rel[hid] = {"object": oid, "state": st, "iou": ev["iou"], "gap": ev["gap"],
                    "label": objects[oid].get("label"),
                    "rel_xy": ((hc[0] - ob[0]) / ow, (hc[1] - ob[1]) / oh)}
    return rel


def contact_runs(states):
    """Consecutive runs of a per-frame state sequence as (state, start, end).

    `end` is exclusive. Frames with state None (no observation) break a run
    rather than extending it: an unobserved frame is not evidence that contact
    continued, and treating it as one is how a detector's blind spell becomes
    a "continuously held object"."""
    runs = []
    for i, s in enumerate(states):
        if runs and runs[-1][0] == s and s is not None:
            runs[-1][2] = i + 1
        else:
            runs.append([s, i, i + 1])
    return [tuple(r) for r in runs]


def reset_events(rel_series, dt, min_free_frames=2):
    """LAYER 2. Interaction resets over a window.

    A reset is contact -> free -> contact, not merely a frame in which contact
    was absent. `min_free_frames` exists because a one-frame dropout of a
    proximity proxy is a detector artefact at 10 fps, not a release: a hand
    cannot let go and re-grasp in 100 ms. Set it to 1 only to measure how much
    of a result rests on that assumption.

    Whether the SAME object returns is the discriminating quantity, and it is
    the one that decides same-action from boundary. Released the knife and
    picked the knife back up -- one action, interrupted. Released the knife and
    picked up the bowl -- a boundary. The counts are kept apart for exactly
    that reason and must never be summed into a single "reset count"."""
    st = [r["state"] if r else None for r in rel_series]
    ob = [r["object"] if r else None for r in rel_series]
    runs = contact_runs([None if s is None else (s == CONTACT) for s in st])

    out = {"n_release": 0, "n_recontact_same": 0, "n_recontact_other": 0,
           "longest_free_s": 0.0, "total_free_s": 0.0, "n_target_switch": 0,
           "unobserved_frac": float(np.mean([s is None for s in st]))}
    last_obj, i = None, 0
    for state, s0, s1 in runs:
        if state is True:
            objs = [o for o in ob[s0:s1] if o is not None]
            cur = max(set(objs), key=objs.count) if objs else None
            if last_obj is not None and i > 0:
                out["n_recontact_same" if cur == last_obj
                    else "n_recontact_other"] += 1
            last_obj = cur if cur is not None else last_obj
            i += 1
        elif state is False:
            n = s1 - s0
            if n >= min_free_frames:
                out["n_release"] += 1
                out["total_free_s"] += n * dt
                out["longest_free_s"] = max(out["longest_free_s"], n * dt)
    seq = [o for o in ob if o is not None]
    out["n_target_switch"] = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    return out


def continuity_evidence(rel_series, dt):
    """LAYER 3. Positive evidence that ONE action continued.

    Not the complement of layer 2. "No reset was detected" is also what an
    absent detector produces, so a same-action decision resting on it is
    indistinguishable from a decision resting on nothing. These are stated
    positively -- one object held across a majority of observed frames, the
    grip changing while the object does not -- so the two cases separate.

    `held_fraction` is over OBSERVED frames and `observed_fraction` is
    reported next to it. A 0.95 held fraction over 4 observed frames of 41 is
    not the same claim as 0.95 over 40, and a single number cannot say which
    it is."""
    ob = [r["object"] if r else None for r in rel_series]
    seen = [o for o in ob if o is not None]
    out = {"observed_fraction": len(seen) / max(len(ob), 1),
           "held_fraction": float("nan"), "held_object_runs": 0,
           "longest_held_s": 0.0, "pose_change_without_object_change": float("nan")}
    if not seen:
        return out
    dom = max(set(seen), key=seen.count)
    out["held_fraction"] = seen.count(dom) / len(seen)
    runs = [r for r in contact_runs([o == dom if o is not None else None
                                     for o in ob]) if r[0] is True]
    out["held_object_runs"] = len(runs)
    out["longest_held_s"] = max((r[2] - r[1]) * dt for r in runs) if runs else 0.0

    # the same object, held throughout, while the hand's position within it
    # moves -- a regrasp. High here with no reset in layer 2 is the clearest
    # same-action signature the design produces.
    xy = np.array([r["rel_xy"] for r in rel_series
                   if r and r.get("object") == dom], float)
    if len(xy) >= 4 and np.isfinite(xy).all():
        out["pose_change_without_object_change"] = float(
            np.nanmax(np.hypot(*(xy - xy[0]).T)))
    return out
