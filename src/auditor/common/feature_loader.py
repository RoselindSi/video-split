"""Per-event frame SEQUENCES from the frozen caches. No new extraction.

Every model in this project so far collapsed the window before the model saw
it: `hal_features_at` averages the frames on each side of the candidate and
hands over a cosine distance. That is a reasonable summary of "how different
are the two sides", and it is exactly the wrong input for morphology, which
asks WHERE inside the window the change sits and HOW WIDE it is. A pre- and
post- mean cannot distinguish a step from a ramp; both give the same distance.

So this returns the frames, resampled onto a fixed grid around the candidate,
and the temporal encoder is what collapses them.

RESAMPLING IS NEAREST-FRAME WITH A VALIDITY MASK. The caches are ~2 fps but
the spacing is not exactly uniform, and near the start or end of a recording
part of the window does not exist. A frame is marked invalid when the nearest
cached frame is further than half the target spacing, or when the requested
time falls outside the recording; invalid positions are zero-filled AND
masked, never silently filled with the nearest available frame. Padding a
missing edge with a repeated frame would manufacture "no change is happening
here", which is a morphology answer.

THE DIFFERENCES ARE TAKEN AFTER PROJECTION, not here. PCA is linear, so the
projection of a difference equals the difference of projections up to the mean
term, which cancels -- computing them post-projection is equivalent and avoids
carrying a second full-dimensional sequence per event. The loader therefore
returns raw levels only, and the model builds the deltas.

NOTHING IS FITTED HERE. PCA, scaling and calibration all belong to the
training fold, and a loader that fitted anything would leak the held-out
recordings into every event it returned.
"""
from __future__ import annotations

import numpy as np
import torch


def load_caches(paths):
    """extract_features_recseg .pt caches, indexed by recording_id. Later
    paths win on collision, so a val cache may follow a train cache."""
    by_id = {}
    for p in paths:
        for rec in torch.load(p, weights_only=False):
            rid = rec.get("recording_id") or rec.get("video")
            by_id[rid] = rec
    return by_id


def window(rec, t0, half_s=6.0, n_frames=25):
    """(feats [T,D] float32, rel_t [T], valid [T] bool) on a uniform grid.

    rel_t is relative to the candidate, so t=0 is always the same index and
    the encoder never has to learn where the candidate is."""
    if rec is None:
        return None, None, None
    feats = rec["feats"]
    times = rec["times"]
    if not isinstance(times, torch.Tensor):
        times = torch.as_tensor(times)
    times = times.detach().cpu().numpy().astype(float)
    feats = feats.detach().cpu().numpy().astype(np.float32)
    if len(times) == 0:
        return None, None, None

    grid = np.linspace(-half_s, half_s, n_frames)
    step = grid[1] - grid[0] if n_frames > 1 else half_s
    out = np.zeros((n_frames, feats.shape[1]), np.float32)
    valid = np.zeros(n_frames, bool)
    lo, hi = times.min(), times.max()
    for i, dt in enumerate(grid):
        t = t0 + dt
        if t < lo or t > hi:
            continue
        j = int(np.argmin(np.abs(times - t)))
        # a cached frame further than half the target spacing is not a
        # measurement of this instant, and pretending otherwise would smear a
        # step across the grid
        if abs(times[j] - t) > step / 2:
            continue
        out[i] = feats[j]
        valid[i] = True
    return out, grid.astype(np.float32), valid


def build_events(events, gcache, lcache, half_s=6.0, n_frames=25,
                 verbose=True):
    """One entry per event with both streams on the same grid.

    An event is DROPPED only when a stream is missing entirely, and the reason
    is counted and reported. Events kept with partial coverage carry their
    mask, because "the evidence was not visible" is a label this model has --
    UNOBSERVABLE -- and discarding those events would remove the class from
    its own training set."""
    out, drop = [], {"no global cache": 0, "no local cache": 0,
                     "no frames in window": 0}
    for e in events:
        rid, t0 = e["recording_id"], e.get("candidate_time")
        if t0 is None:
            drop["no frames in window"] += 1
            continue
        g, rel, gv = window(gcache.get(rid), t0, half_s, n_frames)
        if g is None:
            drop["no global cache"] += 1
            continue
        l, _, lv = window(lcache.get(rid), t0, half_s, n_frames)
        if l is None:
            drop["no local cache"] += 1
            continue
        if not gv.any():
            drop["no frames in window"] += 1
            continue
        out.append({**e, "g": g, "l": l, "rel_t": rel,
                    "valid_g": gv, "valid_l": lv,
                    "coverage_g": float(gv.mean()),
                    "coverage_l": float(lv.mean())})
    if verbose:
        print(f"  {len(out)} events with sequences, "
              f"{sum(drop.values())} dropped {dict(drop)}")
        if out:
            cg = np.array([e["coverage_g"] for e in out])
            cl = np.array([e["coverage_l"] for e in out])
            print(f"  window coverage  global mean {cg.mean():.3f} "
                  f"(full on {int((cg == 1).sum())}/{len(out)})   "
                  f"local mean {cl.mean():.3f} "
                  f"(full on {int((cl == 1).sum())}/{len(out)})")
    return out


def stack(events, key):
    return np.stack([e[key] for e in events], 0)
