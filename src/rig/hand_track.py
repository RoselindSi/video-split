"""Identity before ownership: which hand is this, asked separately from whose.

WHY THIS IS ITS OWN STEP. The pipeline used to answer both questions at once.
`OwnHold` kept the previous frame's detections in a list, matched them by
nearest centre, and carried each one's smoothed ownership across. Two things
were wrong with that and both produced the flicker the demos showed.

THE MATCH HAD NO GATE. `track` computed a distance cap and never compared
anything against it -- the variables holding the matched sets were built and
never read either. So every previous detection was assigned to some current
one no matter how far away it had jumped. When a hand left the frame and
another appeared, the departing hand's smoothed verdict was applied to the
arriving one, and a hysteresis meant to prevent flips became the channel that
propagated a wrong state onto a different hand.

AND THE CAP WOULD HAVE BEEN WRONG IF IT HAD BEEN USED. It was
`max_move_frac * dist.max()`, a fraction of how far apart the detections
happened to be, not of the frame. Hands close together shrank it and hands
spread apart grew it, which is backwards: the physical constraint is that a
hand cannot cross much of the FRAME between two adjacent video frames, and
that has nothing to do with where the other hands are.

WHAT REPLACES IT is the standard association step from the tracking
literature: predict, build a cost matrix, gate out the impossible pairs, solve
the assignment globally, and then handle what did not match. Tracks that find
no detection are kept alive for a few frames rather than deleted, and
detections that match no track start their own. Nothing inherits another
track's state, ever.

GREEDY NEAREST-CENTRE IS NOT MERELY UNTIDY. With two hands crossing, greedy
takes the smallest distance first and can be forced into the wrong pair for
the second; a global assignment minimises the total and gets both right. The
matrices here are at most a handful of rows, so the exact solution is free.
"""
from __future__ import annotations

import numpy as np

# A hand cannot cross this fraction of the frame's diagonal between two
# adjacent frames. At 30 fps and a 150-degree field of view this is generous;
# it exists to refuse impossible matches, not to model hand speed.
GATE_FRAC = 0.15

# How the cost is made. Distance is measured against a motion prediction,
# overlap disambiguates nearby hands, and pose/entry geometry stop a newly
# arrived hand from inheriting a track merely because it appeared nearby.
W_DIST, W_IOU, W_SCALE = 1.0, 0.75, 0.35
W_POSE, EDGE_MISMATCH, SIDE_MISMATCH = 0.50, 0.40, 0.25
REACQUIRE_EDGE_MISMATCH = 1.50

# A finite pair is not automatically a match. The old rectangular Hungarian
# solve assigned every pair inside the centre-distance gate, even when leaving
# both sides unmatched was the more plausible explanation. Dummy columns make
# that option explicit; MAX_ASSOC_COST remains a hard refusal above it.
UNMATCHED_COST = 1.35
MAX_ASSOC_COST = 1.60

# Exponential update for the constant-velocity box state. A high value follows
# real motion quickly; the old estimate still damps one-frame detector jitter.
VELOCITY_ALPHA = 0.70

# Frames a track survives with no detection supporting it. Long enough to
# bridge the detector's short dropouts, short enough that a hand which really
# left does not keep a stale identity waiting to be handed to a new arrival.
MAX_LOST = 5


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _centre(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


def _diag(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return float(np.hypot(x1 - x0, y1 - y0))


def _box_state(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0,
                     max(2.0, x1 - x0), max(2.0, y1 - y0)], float)


def _state_box(state, shape):
    H, W = shape[:2]
    cx, cy, w, h = [float(v) for v in state]
    w, h = max(2.0, w), max(2.0, h)
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if x1 > W:
        x0 -= x1 - W
        x1 = float(W)
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if y1 > H:
        y0 -= y1 - H
        y1 = float(H)
    return np.array([max(0, x0), max(0, y0), min(W, x1), min(H, y1)],
                    dtype=int)


def _normalised_pose(det):
    kp = det.get("kp")
    if kp is None:
        return None
    kp = np.asarray(kp, float)
    if kp.ndim != 2 or kp.shape[1] != 2 or not np.isfinite(kp).all():
        return None
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    scale = np.array([max(x1 - x0, 1.0), max(y1 - y0, 1.0)])
    return (kp - np.array([x0, y0])) / scale


def _move_detection(det, new_box):
    """Move a stale detection with its predicted box and keypoints."""
    out = dict(det)
    old = np.asarray(det["box"], float)
    new = np.asarray(new_box, float)
    old_wh = np.maximum(old[2:] - old[:2], 1.0)
    new_wh = np.maximum(new[2:] - new[:2], 1.0)
    for key in ("kp", "exit"):
        value = det.get(key)
        if value is None:
            continue
        pts = np.asarray(value, float)
        out[key] = new[:2] + (pts - old[:2]) * (new_wh / old_wh)
    out["box"] = np.asarray(new_box, dtype=int)
    out["predicted"] = True
    return out


def hungarian(cost):
    """Minimum-cost one-to-one assignment. -> [(row, col)]

    The classic O(n^3) algorithm with potentials, on a square-padded matrix.
    Written out rather than imported because this container's numpy is certain
    and its scipy is not, and the matrices here have single-digit dimensions.

    Pairs whose cost is infinite are dropped from the result afterwards: the
    solver needs finite numbers to work with, so the gate is applied by
    substituting a large finite value and then discarding those pairs."""
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return []
    n_r, n_c = cost.shape
    n = max(n_r, n_c)
    big = float(np.nanmax(cost[np.isfinite(cost)]) + 1.0) \
        if np.isfinite(cost).any() else 1.0
    pad = np.full((n, n), big * 10.0)
    finite = np.where(np.isfinite(cost), cost, big * 10.0)
    pad[:n_r, :n_c] = finite

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, int)
    way = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = pad[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_r and j <= n_c and np.isfinite(cost[i - 1, j - 1]):
            out.append((int(i - 1), int(j - 1)))
    return out


DUP_IOU = 0.70


def duplicate_pairs(dets, iou_min=DUP_IOU):
    """-> [(i, j, iou)] for detections that plausibly cover the same hand.

    WHY THIS IS COUNTED BEFORE ANYTHING IS DONE ABOUT IT. The two-hand cap
    downstream counts DETECTIONS, so a hand found twice reads as two owners
    and can push a real one over the limit. That is a plausible story for the
    flicker and it is currently only a story: nothing in this pipeline has
    ever recorded how often one hand arrives as two boxes. A deduplication
    step written against an unmeasured rate would be a fix aimed at a number
    nobody has.

    A pair here is not proof of a duplicate -- two hands really can overlap,
    and a hand crossing in front of another is the ordinary case in this
    corpus. It is an upper bound on how much of the problem duplicates could
    possibly explain."""
    out = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            v = box_iou(dets[i]["box"], dets[j]["box"])
            if v >= iou_min:
                out.append((i, j, float(v)))
    return out


def _pct(v, q):
    """-> the q-th percentile of a sorted list, without importing numpy."""
    if not v:
        return float("nan")
    i = min(len(v) - 1, int(round(q * (len(v) - 1))))
    return v[i]


class LowConfRuns:
    """How long a hand is visible, found, and treated as absent.

    THE FIRST VERSION OF THIS MEASURED THE WRONG POPULATION. It timed the wait
    only for detections that eventually became tracks, and reported a median
    of one to three frames -- which is survivorship. A hand that is found at a
    low score for forty frames and never clears the bar to start a track
    contributes nothing to that average, and it is the entire failure being
    looked for. The uncovered hand in the render is by definition one that was
    never promoted.

    So this follows RUNS rather than tracks. Every detection the tracker
    refused an id is chained frame to frame by overlap; a chain ends when
    nothing continues it, or when a real track takes it over. Whether it was
    ever promoted is an OUTCOME of the run, not the condition for counting it.

    A run is not proof of a hand. The detector fires at 0.25 on things that
    are not hands, and a run of two frames on a wrench looks like a run of two
    frames on a fingertip. What separates them is length: junk does not
    persist coherently in the same place for a second."""

    def __init__(self, iou_min=0.30):
        self.iou_min = float(iou_min)
        self.open = []
        self.done = []

    def update(self, k, dets, ids, new_ids=()):
        dropped = [(d["box"], float(d.get("conf", 1.0)))
                   for d, t in zip(dets, ids) if t is None]
        started = [d["box"] for d, t in zip(dets, ids)
                   if t is not None and t in new_ids]

        live = [r for r in self.open if r["last_k"] == k - 1]
        # Promotion first: a run that a real track continues ENDED by being
        # promoted, and must not also be extended as an ignored detection.
        for box in started:
            best, best_v = None, self.iou_min
            for r in live:
                v = box_iou(r["box"], box)
                if v >= best_v:
                    best, best_v = r, v
            if best is not None:
                best["promoted"] = True
                live.remove(best)
                self.open.remove(best)
                self.done.append(best)

        used, fresh = set(), []
        for box, conf in dropped:
            best, best_v = None, self.iou_min
            for r in live:
                if id(r) in used:
                    continue
                v = box_iou(r["box"], box)
                if v >= best_v:
                    best, best_v = r, v
            if best is None:
                fresh.append({"last_k": k, "n": 1, "confs": [conf],
                              "box": list(box), "promoted": False})
            else:
                used.add(id(best))
                best.update(last_k=k, n=best["n"] + 1, box=list(box))
                best["confs"].append(conf)
        # Anything not extended and not promoted is over.
        for r in list(self.open):
            if r["last_k"] < k:
                self.open.remove(r)
                self.done.append(r)
        self.open.extend(fresh)

    def runs(self):
        return self.done + self.open

    def report(self, fps=12.0):
        """-> a block about the run-length distribution, or None."""
        rs = self.runs()
        if not rs:
            return None
        n = sorted(r["n"] for r in rs)
        never = [r for r in rs if not r["promoted"]]
        long_never = sorted((r["n"] for r in never), reverse=True)

        def over(t):
            return sum(1 for x in n if x >= t)

        def over_never(t):
            return sum(1 for x in long_never if x >= t)
        out = [
            "\n  LOW-CONFIDENCE RUNS: a hand found by the detector, refused "
            "an id, and so\n    neither classified nor covered. Chained by "
            "overlap; length in frames.",
            f"    {len(rs)} runs   median {_pct(n, 0.5)}   p90 "
            f"{_pct(n, 0.9)}   max {n[-1]}",
            f"    >= 6 frames: {over(6)}    >= 12: {over(12)}    "
            f">= 24: {over(24)}    >= 36: {over(36)}",
            f"    never promoted: {len(never)} of {len(rs)}   "
            f"of those >= 12 frames: {over_never(12)}   "
            f">= 24: {over_never(24)}"]
        if long_never[:1]:
            worst = max(never, key=lambda r: r["n"])
            out.append(
                f"    longest unpromoted run {worst['n']} frames "
                f"({worst['n'] / fps:.1f}s on screen at {fps:g} fps), "
                f"peak conf {max(worst['confs']):.2f}")
        out.append(
            "    A run is not proof of a hand -- the detector fires on other "
            "things at this\n    score. Length is what separates them: junk "
            "does not persist in one place.")
        return "\n".join(out)


class Fragmentation:
    """Whether one physical hand keeps one id.

    EVERYTHING BUILT ON TOP OF A TRACK ASSUMES THIS. A smoothed ownership
    score, a hysteresis, a slot: each accumulates evidence per id and is
    worth exactly as much as the id is stable. Twenty-seven tracks over five
    hundred hand-frames is twenty frames per track, and a hysteresis that
    resets every twenty frames is not a hysteresis.

    A HANDOFF IS THE SIGNATURE. A track that ends and a new one that starts
    a frame or two later, in the same place, is one hand that lost its name.
    Counted against tracks that start with nothing behind them, which is what
    a hand genuinely entering the frame looks like."""

    def __init__(self, iou_min=0.20, max_gap=3):
        self.iou_min = float(iou_min)
        self.max_gap = int(max_gap)
        self.tracks = {}
        self.handoffs = []
        self.gaps = []

    def update(self, k, dets, ids):
        for d, t in zip(dets, ids):
            if t is None:
                continue
            r = self.tracks.get(t)
            if r is None:
                best, best_v = None, self.iou_min
                for ot, o in self.tracks.items():
                    gap = k - o["last"]
                    if not 1 <= gap <= self.max_gap:
                        continue
                    v = box_iou(o["last_box"], d["box"])
                    if v >= best_v:
                        best, best_v = (ot, gap), v
                if best is not None:
                    self.handoffs.append((best[0], t, best[1], float(best_v)))
                self.tracks[t] = {"first": k, "last": k, "n": 1,
                                  "last_box": list(d["box"])}
            else:
                if k - r["last"] > 1:
                    self.gaps.append(k - r["last"] - 1)
                r.update(last=k, n=r["n"] + 1, last_box=list(d["box"]))

    def report(self):
        if not self.tracks:
            return None
        life = sorted(r["n"] for r in self.tracks.values())
        span = sorted(r["last"] - r["first"] + 1
                      for r in self.tracks.values())
        g = sorted(self.gaps)
        out = [
            "\n  IDENTITY FRAGMENTATION: whether one physical hand keeps one "
            "id. Everything\n    accumulated per track -- the smoothed score, "
            "the hysteresis -- is worth what\n    this is worth.",
            f"    {len(self.tracks)} tracks   frames seen: median "
            f"{_pct(life, 0.5)}  p90 {_pct(life, 0.9)}  max {life[-1]}",
            f"    lifespan first-to-last: median {_pct(span, 0.5)}  "
            f"max {span[-1]}   under 10 frames: "
            f"{sum(1 for x in life if x < 10)}",
            f"    HANDOFFS: {len(self.handoffs)} of {len(self.tracks)} tracks "
            f"started within {self.max_gap} frames\n      of another ending "
            f"in the same place. Those are one hand renamed, not a\n      "
            f"hand arriving."]
        if g:
            out.append(f"    within-track gaps: {len(g)}   median "
                       f"{_pct(g, 0.5)}   max {g[-1]}")
        if self.handoffs:
            w = sorted(self.handoffs, key=lambda h: -h[3])[:4]
            out.append("    worst by overlap: " + ", ".join(
                f"{a}->{b} gap {gp} IoU {v:.2f}" for a, b, gp, v in w))
        return "\n".join(out)


class Tracker:
    """Hand identities across frames. Knows nothing about ownership.

    Deliberately identity-neutral: an ownership guess must not be allowed to
    decide which hand is which, or a wrong verdict on one frame becomes a
    wrong identity on the next and the two errors reinforce. Every consumer
    that wants to accumulate something per hand -- a smoothed score, a size
    history, a verdict -- keys it on the id this returns."""

    def __init__(self, gate_frac=GATE_FRAC, max_lost=MAX_LOST,
                 predict_motion=True, rich_association=True,
                 max_assoc_cost=MAX_ASSOC_COST,
                 unmatched_cost=UNMATCHED_COST):
        self.gate_frac = float(gate_frac)
        self.max_lost = int(max_lost)
        self.predict_motion = bool(predict_motion)
        self.rich_association = bool(rich_association)
        self.max_assoc_cost = (None if max_assoc_cost is None
                               else float(max_assoc_cost))
        self.unmatched_cost = (None if unmatched_cost is None
                               else float(unmatched_cost))
        self.tracks = {}
        self._next = 0
        self.n_new = 0
        self.n_lost = 0
        self.reacquired = set()
        self.low_matches = set()
        self.new_ids = set()
        self.provenance = []
        self.unmatched = []

    def _predicted_boxes(self, shape):
        out = {}
        for tid, track in self.tracks.items():
            state = track["state"].copy()
            if self.predict_motion:
                state += track["velocity"]
            state[2:] = np.maximum(state[2:], 2.0)
            out[tid] = _state_box(state, shape)
        return out

    def _cost(self, tid, det, predicted_box, shape):
        H, W = shape[:2]
        track = self.tracks[tid]
        uncertainty = 1.0 + 0.25 * min(track["lost"], 3)
        gate = self.gate_frac * float(np.hypot(W, H)) * uncertainty
        dist = float(np.linalg.norm(_centre(predicted_box)
                                    - _centre(det["box"])))
        if dist > gate:
            return np.inf
        td = _diag(predicted_box)
        sc = abs(_diag(det["box"]) - td) / max(td, 1.0)
        cost = (W_DIST * dist / max(gate, 1.0)
                + W_IOU * (1.0 - box_iou(predicted_box, det["box"]))
                + W_SCALE * min(sc, 2.0))
        if not self.rich_association:
            return cost

        old_pose, new_pose = _normalised_pose(track["det"]), \
            _normalised_pose(det)
        if old_pose is not None and new_pose is not None \
                and old_pose.shape == new_pose.shape:
            pose = float(np.linalg.norm(old_pose - new_pose, axis=1).mean())
            cost += W_POSE * min(pose, 2.0)

        old_edge, new_edge = track["det"].get("edge"), det.get("edge")
        if old_edge and new_edge and old_edge != new_edge:
            cost += (REACQUIRE_EDGE_MISMATCH if track["lost"] > 0
                     else EDGE_MISMATCH)
        old_side, new_side = track["det"].get("side"), det.get("side")
        if old_side and new_side and old_side != new_side:
            cost += SIDE_MISMATCH
        return cost

    def _associate(self, tids, det_indices, dets, predicted, shape):
        if not tids or not det_indices:
            return []
        cost = np.full((len(tids), len(det_indices)), np.inf)
        for row, tid in enumerate(tids):
            for col, det_i in enumerate(det_indices):
                value = self._cost(tid, dets[det_i], predicted[tid], shape)
                if self.max_assoc_cost is None or value <= self.max_assoc_cost:
                    cost[row, col] = value

        if self.unmatched_cost is None:
            assigned = hungarian(cost)
        else:
            # One private dummy column per track. Choosing it means the track
            # remains lost and the detection remains free to start a new track.
            aug = np.full((len(tids), len(det_indices) + len(tids)), np.inf)
            aug[:, :len(det_indices)] = cost
            for row in range(len(tids)):
                aug[row, len(det_indices) + row] = self.unmatched_cost
            assigned = [(row, col) for row, col in hungarian(aug)
                        if col < len(det_indices)]
        return [(tids[row], det_indices[col]) for row, col in assigned]

    def _start(self, det):
        tid = self._next
        state = _box_state(det["box"])
        self.tracks[tid] = {
            "box": np.asarray(det["box"], dtype=int),
            "det": dict(det), "state": state,
            "velocity": np.zeros(4, float), "lost": 0, "age": 1,
        }
        self._next += 1
        self.n_new += 1
        self.new_ids.add(tid)
        return tid

    def _match(self, tid, det):
        track = self.tracks[tid]
        was_lost = track["lost"] > 0
        observed = _box_state(det["box"])
        measured = observed - track["state"]
        if track["age"] <= 1:
            velocity = measured
        else:
            velocity = (VELOCITY_ALPHA * measured
                        + (1.0 - VELOCITY_ALPHA) * track["velocity"])
        track.update({"box": np.asarray(det["box"], dtype=int),
                      "det": dict(det), "state": observed,
                      "velocity": velocity, "lost": 0,
                      "age": track["age"] + 1})
        if was_lost:
            self.reacquired.add(tid)

    def update(self, dets, shape, new_track_conf=None, continue_conf=None):
        """Associate detections and return track ids aligned with ``dets``.

        With thresholds omitted every unmatched detection starts a track, which
        preserves the original API. With thresholds supplied, high-confidence
        detections are associated first and may start tracks; lower-confidence
        detections get a second association pass and may only continue one.
        Unmatched low detections return ``None`` and must be dropped by callers.
        """
        ids = [None] * len(dets)
        self.reacquired, self.low_matches, self.new_ids = set(), set(), set()
        self.provenance = ["dropped" for _ in dets]
        live = sorted(self.tracks)
        predicted = self._predicted_boxes(shape)

        if new_track_conf is None:
            high = list(range(len(dets)))
            low = []
        else:
            new_track_conf = float(new_track_conf)
            continue_conf = (new_track_conf if continue_conf is None
                             else float(continue_conf))
            high = [i for i, d in enumerate(dets)
                    if float(d.get("conf", 1.0)) >= new_track_conf]
            low = [i for i, d in enumerate(dets)
                   if continue_conf <= float(d.get("conf", 1.0))
                   < new_track_conf]

        matched_tracks = set()
        matched_dets = set()
        for stage, candidates in (("high", high), ("low", low)):
            remaining_tracks = [tid for tid in live if tid not in matched_tracks]
            remaining_dets = [i for i in candidates if i not in matched_dets]
            for tid, det_i in self._associate(remaining_tracks, remaining_dets,
                                               dets, predicted, shape):
                self._match(tid, dets[det_i])
                ids[det_i] = tid
                self.provenance[det_i] = f"matched_{stage}"
                matched_tracks.add(tid)
                matched_dets.add(det_i)
                if stage == "low":
                    self.low_matches.add(tid)

        for det_i in high:
            if det_i in matched_dets:
                continue
            tid = self._start(dets[det_i])
            ids[det_i] = tid
            self.provenance[det_i] = "new_high"

        # WHY A LIVE TRACK WENT WITHOUT A BOX. 15 of the mined own-hand gaps
        # have a detection sitting on the interpolated position at 0.27-0.57,
        # above the 0.25 continuation floor, while the track was still within
        # its lost budget -- so the score was never the reason and the failure
        # is somewhere in the association. Which part cannot be read off the
        # output: an unmatched track and a refused pair look the same
        # afterwards. This records, per unmatched live track, the cheapest
        # candidate and the rule that refused it, computed BEFORE the lost and
        # age counters move so the cost is the one the assignment saw.
        self.unmatched = []
        for tid in live:
            if tid in matched_tracks:
                continue
            best = None
            for i, d in enumerate(dets):
                conf = float(d.get("conf", 1.0))
                if new_track_conf is not None and conf < continue_conf:
                    continue
                value = self._cost(tid, d, predicted[tid], shape)
                if best is not None and not value < best["cost"]:
                    continue
                best = {"det": i, "conf": round(conf, 4),
                        "cost": value,
                        "iou": round(box_iou(predicted[tid], d["box"]), 3),
                        "taken_by": ids[i],
                        "stage": ("high" if new_track_conf is None
                                  or conf >= new_track_conf else "low")}
            if best is None:
                why = "no_candidate"
            elif not np.isfinite(best["cost"]):
                why = "distance_gate"
            elif (self.max_assoc_cost is not None
                  and best["cost"] > self.max_assoc_cost):
                why = "cost_gate"
            elif best["taken_by"] is not None:
                why = "taken_by_other_track"
            elif (self.unmatched_cost is not None
                  and best["cost"] > self.unmatched_cost):
                why = "unmatched_cost_preferred"
            else:
                why = "unexplained"
            self.unmatched.append(dict(best or {}, tid=tid, why=why,
                                       lost=self.tracks[tid]["lost"]))

        for tid in live:
            if tid in matched_tracks:
                continue
            track = self.tracks[tid]
            new_box = predicted[tid]
            track["det"] = _move_detection(track["det"], new_box)
            track["box"] = new_box
            track["state"] = _box_state(new_box)
            track["lost"] += 1
            track["age"] += 1
            if track["lost"] > self.max_lost:
                del self.tracks[tid]
                self.n_lost += 1
        return ids

    def coasting(self, max_prediction_age=2):
        """Predicted tracks missing a detection but still within the horizon.
        -> [(track_id, detection)]

        THE TRACKER KNEW WHERE THE HAND WAS AND NOBODY ASKED IT. Masks are
        built from the CURRENT frame's detections, so a track the tracker was
        holding through a one-frame dropout contributed no box, no mask and no
        blur -- the identity was bridged and the cover was not. On the clip
        that prompted this, every drop was the detector losing one hand for
        one or two frames while the wearer's two stayed put: 3 detections
        became 2, the colleague's hand went uncovered, and the frame counted
        as a drop.

        ``max_prediction_age`` is deliberately bounded separately from track
        lifetime. Prediction answers where a missing hand may be; this horizon
        answers how long that estimate is allowed to affect output pixels."""
        return [(tid, t["det"]) for tid, t in sorted(self.tracks.items())
                if 0 < t["lost"] <= max_prediction_age
                and t.get("det") is not None]


class FlipCount:
    """Verdict changes per track, which is what a viewer sees as flicker.

    THE FRAME-LEVEL COUNT MISSED THIS ENTIRELY. The demo's trace called a
    frame a `drop` when the previous frame suppressed something and this one
    suppressed nothing. Two hands in view and one of them alternating leaves
    the frame's suppressed fraction above zero throughout, so the alternation
    was invisible and the trace reported zero label drops while the video
    plainly flickered. A flip is a property of a hand over time, so it has to
    be counted on the thing that persists over time."""

    def __init__(self):
        self.last = {}
        self.flips = 0
        self.frames = {}          # id -> frames seen
        self.per_track = {}

    def update(self, ids, owners):
        for tid, own in zip(ids, owners):
            self.frames[tid] = self.frames.get(tid, 0) + 1
            if tid in self.last and bool(self.last[tid]) != bool(own):
                self.flips += 1
                self.per_track[tid] = self.per_track.get(tid, 0) + 1
            self.last[tid] = bool(own)

    def report(self):
        seen = sum(self.frames.values())
        worst = sorted(self.per_track.items(), key=lambda kv: -kv[1])[:5]
        return {"flips": self.flips, "tracks": len(self.frames),
                "hand_frames": seen,
                "flips_per_100_hand_frames": (100.0 * self.flips / seen
                                              if seen else 0.0),
                "worst": worst}


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    c = np.array([[1.0, 5.0], [5.0, 1.0]])
    chk(sorted(hungarian(c)) == [(0, 0), (1, 1)], "the obvious assignment")
    # Greedy takes the smallest cell (0,1)=1 first and is then forced into
    # (1,0)=9 for a total of 10. The optimum is (0,0)+(1,1)=5, and getting it
    # is the whole reason for solving the assignment instead of sorting.
    c2 = np.array([[2.0, 1.0], [9.0, 3.0]])
    chk(sorted(hungarian(c2)) == [(0, 0), (1, 1)],
        "and it beats greedy when the smallest cell is a trap")
    c3 = np.array([[1.0, np.inf], [np.inf, np.inf]])
    got = hungarian(c3)
    chk(got == [(0, 0)], "gated pairs are dropped rather than assigned")

    shape = (900, 1600)

    def box(cx, cy, s=80):
        return {"box": (cx - s, cy - s, cx + s, cy + s)}

    t = Tracker()
    a = t.update([box(200, 400), box(900, 400)], shape)
    b = t.update([box(210, 405), box(915, 402)], shape)
    chk(a == b, "a hand that barely moves keeps its id")
    chk(len(set(a)) == 2, "and two hands get two ids")

    # The failure the old matcher had: one hand leaves, a different one
    # appears far away. It must NOT inherit the departed hand's identity.
    t2 = Tracker()
    x = t2.update([box(200, 400)], shape)
    y = t2.update([box(1400, 800)], shape)
    chk(x != y, "a detection beyond the gate starts a new track")

    # Crossing hands: greedy would swap, the assignment should not.
    t3 = Tracker()
    p = t3.update([box(700, 400), box(800, 400)], shape)
    q = t3.update([box(730, 400), box(830, 400)], shape)
    chk(p == q, "two hands crossing keep their own ids")

    t4 = Tracker(max_lost=2)
    i1 = t4.update([box(400, 400)], shape)
    t4.update([], shape)
    i2 = t4.update([box(405, 402)], shape)
    chk(i1 == i2, "a track survives a frame the detector missed")
    t5 = Tracker(max_lost=1)
    j1 = t5.update([box(400, 400)], shape)
    t5.update([], shape)
    t5.update([], shape)
    j2 = t5.update([box(405, 402)], shape)
    chk(j1 != j2, "...but not an outage longer than the limit")

    f = FlipCount()
    f.update([1, 2], [True, False])
    f.update([1, 2], [True, False])
    chk(f.flips == 0, "a steady verdict is not a flip")
    f.update([1, 2], [False, False])
    chk(f.flips == 1, "a changed verdict on the same track is")
    # The case the frame-level counter could not see: one hand flips while
    # the other keeps the frame's suppression non-empty throughout.
    chk(f.report()["flips_per_100_hand_frames"] > 0,
        "and it is reported per hand-frame rather than per frame")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Identity is decided without reference to ownership. A wrong "
          "verdict must not\n  be able to become a wrong identity, because "
          "the two would then reinforce.")
    return all(ok)


if __name__ == "__main__":
    import sys
    raise SystemExit(0 if _self_test() else 1)
