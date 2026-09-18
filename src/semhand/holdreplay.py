"""Replay V1's post-processing on any model's per-hand probability.

THE SAME CLASS, NOT A COPY. `OwnHold` from `hand_detect` is driven with the
blend already applied (geom None, rule weight 0), so the EMA, the Schmitt
thresholds, the reacquire rule, state expiry and the two-owner cap are the
shipped code.

THE PRIOR IS RECOVERED FROM THE DUMP, THEN CHECKED. The dump kept V1's raw P
and the smoothed score but not the geometric prior. OwnHold's update is
invertible per hand: a new track starts at the blend, and afterwards the
blend is solved from the EMA step whose direction the change reveals. Driving
the replay with V1's raw P and that prior has to reproduce the dump's final
verdict before the prior is used for anything else (`verify`).
"""
from __future__ import annotations

import collections
import csv

GEOM_W = 0.5
MAX_OWNER = 2


def _hold():
    from src.rig.demo_video import MAX_PREDICTION_AGE
    from src.rig.hand_detect import OwnHold
    from src.rig.hand_track import MAX_LOST
    return OwnHold(geom=None, rule_w=0.0, max_owner=MAX_OWNER,
                   state_ttl=max(MAX_LOST, MAX_PREDICTION_AGE))


def frames_of(rows):
    """-> [(rec, frame, [row])] in file order within a frame, frames ascending."""
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault((r["rec"], int(r["frame"])), []).append(r)
    return [(k[0], k[1], v) for k, v in sorted(by.items(), key=lambda kv: kv[0])]


def key(r):
    return (r["rec"], int(r["frame"]), str(r["tid"]))


def invert_prior(dump_rows):
    """-> {key: geometric prior score}, from p_owner_raw and ema_owner."""
    hold, rec, out = None, None, {}
    for rc, f, rows in frames_of(dump_rows):
        if rc != rec:
            hold, rec = _hold(), rc
        blends = []
        for r in rows:
            ema = float(r["ema_owner"])
            prev = hold.state.get(int(r["tid"]))
            if prev is None:
                pb = ema
            elif ema < prev[0]:
                pb = (ema - (1 - hold.fast) * prev[0]) / hold.fast
            else:
                pb = (ema - (1 - hold.slow) * prev[0]) / hold.slow
            blends.append(pb)
            out[key(r)] = (pb - (1 - GEOM_W) * float(r["p_owner_raw"])) / GEOM_W
        hold.update([None] * len(rows), [(None, pb) for pb in blends],
                    ids=[int(r["tid"]) for r in rows])
        for r in rows:                      # stop rounding error accumulating
            hold.state[int(r["tid"])][0] = float(r["ema_owner"])
    return out


def replay(rows, p, prior, geom_w=GEOM_W, cap=MAX_OWNER, reacquire=False):
    """-> {key: held verdict} for `rows` (any subset of frames), given
    {key: P(self)} and {key: prior}. `geom_w` and `cap` default to V1's
    shipped values; the ablation moves them.

    `reacquire` adds the one thing the dump cannot carry: the live pipeline
    tells OwnHold which tracks the tracker LOST and found again, and a
    reacquired track that was the wearer's is reset to `other` until two
    frames confirm it. A hand the detector drops for a frame therefore gets
    covered on the frame it comes back, with no label change on either side of
    the gap. Off by default, because the dumps every published number was
    measured on were written without it."""
    hold, rec, out = None, None, {}
    last = {}
    for rc, f, fr in frames_of(rows):
        if rc != rec:
            hold, rec, last = _hold(), rc, {}
            hold.max_owner = cap
        flags, rq = [], set()
        for r in fr:
            k = key(r)
            flags.append((None, min(1.0, max(0.0, (1 - geom_w) * p[k] + geom_w * prior[k]))))
            tid = int(r["tid"])
            if reacquire and last.get(tid, f) < f - 1:
                rq.add(tid)
        res = hold.update([None] * len(fr), flags, ids=[int(r["tid"]) for r in fr],
                          reacquired=rq)
        for r in fr:
            last[int(r["tid"])] = f
        for r, (lab, _) in zip(fr, res):
            out[key(r)] = bool(lab)
    return out


def verify(dump_path):
    rows = list(csv.DictReader(open(dump_path, encoding="utf-8")))
    prior = invert_prior(rows)
    p = {key(r): float(r["p_owner_raw"]) for r in rows}
    held = replay(rows, p, prior)
    same = sum(held[key(r)] == (r["final_owner_post_cap"] == "1") for r in rows)
    pre = sum(1 for v in prior.values() if not (-0.01 <= v <= 1.01))
    return {"rows": len(rows), "match": same, "rate": same / len(rows),
            "prior_out_of_range": pre}


if __name__ == "__main__":
    import sys
    print(verify(sys.argv[1]))
