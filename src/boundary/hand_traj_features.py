"""The ~28 trajectory features, as pure functions over one event's frames.

Every one of these can be wrong without raising: a reversal counter with a bad
threshold returns a plausible integer, an autocorrelation with an off-by-one
lag returns a plausible period. So each is defined against a synthetic
trajectory whose answer is known by construction, and the tests live beside
the definitions rather than downstream of a classifier that would hide them.

Five groups, matching the failure modes they target:
  observability   the 24 offscreen events and the detection failures
  direction       direction_reversal
  stop-start      regrasp and reposition
  periodicity     periodic_repetition. Read autocorrelation_lag_s ONLY
                  together with velocity_autocorrelation_max -- the lag is an
                  argmax and always returns a value.
  two-hand        an interaction proxy, deliberately NOT called contact --
                  without object tracks nothing here observes contact state

Speeds are normalised by the hand-box diagonal, so a hand near the camera does
not read as faster than the same motion further away.
"""
from __future__ import annotations

import numpy as np

from src.boundary.hand_trajectory import box_iou, interpolate_gaps, savgol

NEAR = 0.5          # seconds counted as "near the candidate"
STOP_FRAC = 0.25    # speed below this fraction of the window median is a stop


def _series(frames, key="centre"):
    """Per-frame value for the LONGEST-LIVED track, plus its validity mask.

    The longest-lived track rather than a per-frame average: averaging two
    hands moving in opposite directions gives a stationary phantom, which is
    exactly the situation direction_reversal is about."""
    lens = {}
    for f in frames:
        for tid in f["tracks"]:
            lens[tid] = lens.get(tid, 0) + 1
    if not lens:
        return None, None, None
    tid = max(lens, key=lens.get)
    cen, box, ok = [], [], []
    for f in frames:
        d = f["tracks"].get(tid)
        if d is None:
            cen.append((np.nan, np.nan))
            box.append((np.nan,) * 4)
            ok.append(False)
        else:
            b = d["box"]
            cen.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            box.append(b)
            ok.append(True)
    return np.array(cen, float), np.array(box, float), np.array(ok, bool)


def _speed(cen, box, ok, dt, max_gap=2):
    """Box-diagonal-normalised speed, smoothed inside valid runs only."""
    cx, okx, _ = interpolate_gaps(cen[:, 0], ok, max_gap)
    cy, oky, _ = interpolate_gaps(cen[:, 1], ok, max_gap)
    good = okx & oky
    cx, cy = savgol(cx, good), savgol(cy, good)
    diag = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    scale = np.nanmedian(diag[ok]) if ok.any() else 1.0
    scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    vx = np.gradient(cx) / dt / scale
    vy = np.gradient(cy) / dt / scale
    vx[~good], vy[~good] = np.nan, np.nan
    return vx, vy, good, scale


def observability(frames, dt, rel_t):
    n = len(frames)
    nh = np.array([f["n_hands"] for f in frames], float)
    det = nh > 0
    out = {"hand_detect_coverage": float(det.mean()),
           "mean_n_hands": float(nh.mean()),
           "n_hands_change_count": float((np.diff(nh) != 0).sum())}
    runs, cur = [], 0
    for v in det:
        cur = 0 if v else cur + 1
        runs.append(cur)
    out["longest_missing_gap_s"] = float(max(runs) * dt)
    near = np.abs(rel_t) <= NEAR
    # A hole AT the candidate matters more than one at the window edge: the
    # decision is made at the candidate, and evidence missing there cannot be
    # replaced by evidence two seconds away.
    nr, cur = [], 0
    for v in det[near]:
        cur = 0 if v else cur + 1
        nr.append(cur)
    out["candidate_centered_missing_gap_s"] = float(max(nr) * dt) if nr else 0.0
    et = [any(d.get("edge_touch") for d in f["tracks"].values()) if f["tracks"] else False
          for f in frames]
    out["edge_touch_rate"] = float(np.mean(et))
    cen, box, ok = _series(frames)
    if cen is None:
        out.update({"median_consecutive_box_iou": np.nan, "box_center_jitter": np.nan,
                    "box_scale_jitter": np.nan, "interpolated_fraction": np.nan})
        return out
    ious = [box_iou(tuple(box[i]), tuple(box[i + 1]))
            for i in range(n - 1) if ok[i] and ok[i + 1]]
    out["median_consecutive_box_iou"] = float(np.median(ious)) if ious else np.nan
    diag = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    scale = np.nanmedian(diag[ok]) if ok.any() else 1.0
    d = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    pair = ok[:-1] & ok[1:]
    out["box_center_jitter"] = float(np.nanmedian(d[pair]) / max(scale, 1e-6)) \
        if pair.any() else np.nan
    out["box_scale_jitter"] = float(np.nanmedian(np.abs(np.diff(diag))[pair]) /
                                    max(scale, 1e-6)) if pair.any() else np.nan
    _, okx, filled = interpolate_gaps(cen[:, 0], ok, 2)
    out["interpolated_fraction"] = float(filled.mean())
    return out


def direction(frames, dt, rel_t):
    cen, box, ok = _series(frames)
    out = dict.fromkeys(["pre_motion_speed", "post_motion_speed",
                         "direction_cosine_pre_vs_post", "reversal_count",
                         "candidate_velocity_change", "candidate_acceleration_peak"],
                        np.nan)
    if cen is None or ok.sum() < 6:
        return out
    vx, vy, good, _ = _speed(cen, box, ok, dt)
    pre, post = (rel_t < 0) & good, (rel_t >= 0) & good
    sp = np.hypot(vx, vy)
    if pre.any():
        out["pre_motion_speed"] = float(np.nanmean(sp[pre]))
    if post.any():
        out["post_motion_speed"] = float(np.nanmean(sp[post]))
    if pre.any() and post.any():
        a = np.array([np.nanmean(vx[pre]), np.nanmean(vy[pre])])
        b = np.array([np.nanmean(vx[post]), np.nanmean(vy[post])])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out["direction_cosine_pre_vs_post"] = float(a @ b / (na * nb)) \
            if na > 1e-9 and nb > 1e-9 else np.nan
        out["candidate_velocity_change"] = float(np.linalg.norm(b - a))
    # a reversal is a sign flip of the velocity projected on its dominant axis,
    # counted only where the speed is above a floor -- otherwise jitter around
    # a stationary hand counts as dozens of reversals
    v = np.stack([vx, vy], 1)
    m = good & np.isfinite(sp)
    if m.sum() >= 4:
        # project velocity on its dominant axis before counting sign flips:
        # a hand oscillating along one direction reverses in that direction,
        # and counting flips in x and y separately would double-count a
        # diagonal motion and miss one aligned with neither axis
        pc = np.linalg.svd(v[m] - v[m].mean(0), full_matrices=False)[2][0]
        proj = v[m] @ pc
        floor = 0.25 * np.nanmedian(np.abs(proj))
        s = np.sign(np.where(np.abs(proj) > floor, proj, 0))
        s = s[s != 0]
        out["reversal_count"] = float((np.diff(s) != 0).sum()) if len(s) > 1 else 0.0
    acc = np.hypot(np.gradient(vx), np.gradient(vy)) / dt
    near = (np.abs(rel_t) <= NEAR) & good & np.isfinite(acc)
    if near.any():
        out["candidate_acceleration_peak"] = float(np.max(acc[near]))
    return out


def stop_start(frames, dt, rel_t):
    cen, box, ok = _series(frames)
    out = dict.fromkeys(["minimum_speed_near_candidate", "stop_duration_near_candidate",
                         "stop_then_start_score", "box_scale_change_across_candidate",
                         "hand_orientation_change"], np.nan)
    if cen is None or ok.sum() < 6:
        return out
    vx, vy, good, _ = _speed(cen, box, ok, dt)
    sp = np.hypot(vx, vy)
    med = np.nanmedian(sp[good]) if good.any() else np.nan
    near = (np.abs(rel_t) <= NEAR) & good
    if near.any():
        out["minimum_speed_near_candidate"] = float(np.nanmin(sp[near]))
        stopped = sp[near] < STOP_FRAC * med if np.isfinite(med) else np.zeros(near.sum(), bool)
        out["stop_duration_near_candidate"] = float(np.sum(stopped) * dt)
    pre = (rel_t < -NEAR / 2) & good
    post = (rel_t > NEAR / 2) & good
    if pre.any() and post.any() and near.any() and np.isfinite(med) and med > 0:
        # high when motion runs, pauses at the candidate, then runs again --
        # the regrasp signature, and NOT the same thing as a low mean speed
        lo = np.nanmin(sp[near])
        out["stop_then_start_score"] = float(
            (np.nanmean(sp[pre]) + np.nanmean(sp[post])) / 2.0 / max(lo, 1e-6) / max(med, 1e-6))
    diag = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    if pre.any() and post.any():
        a, b = np.nanmedian(diag[pre]), np.nanmedian(diag[post])
        out["box_scale_change_across_candidate"] = float(abs(b - a) / max(a, 1e-6))
    lm = [f["tracks"] for f in frames]
    ang = []
    for f in frames:
        for d in f["tracks"].values():
            p = d.get("landmarks")
            if p and len(p) > 9:
                ang.append(np.arctan2(p[9][1] - p[0][1], p[9][0] - p[0][0]))
                break
        else:
            ang.append(np.nan)
    ang = np.array(ang)
    va = np.isfinite(ang)
    if va.sum() >= 4:
        pa = ang[va & (rel_t < 0)]
        po = ang[va & (rel_t >= 0)]
        if len(pa) and len(po):
            d_ = abs(np.angle(np.exp(1j * (np.median(po) - np.median(pa)))))
            out["hand_orientation_change"] = float(d_)
    return out


def periodicity(frames, dt, rel_t, min_lag_s=0.3, max_lag_s=1.5, n_perm=200):
    """Autocorrelation of speed, over lags the 4 s window can actually resolve.
    A 4 s window cannot evidence a period longer than about 1.5 s, so longer
    lags are not searched and no claim is made about them."""
    cen, box, ok = _series(frames)
    out = dict.fromkeys(["velocity_autocorrelation_max", "autocorrelation_lag_s",
                         "autocorrelation_beats_shuffled",
                         "dominant_frequency_strength", "repeated_reversal_rate"], np.nan)
    if cen is None or ok.sum() < 12:
        return out
    vx, vy, good, _ = _speed(cen, box, ok, dt)
    sp = np.hypot(vx, vy)
    s = sp.copy()
    s[~good] = np.nan
    if np.isfinite(s).sum() < 12:
        return out
    s = np.nan_to_num(s - np.nanmean(s))
    denom = float(s @ s)
    if denom <= 0:
        return out
    lo, hi = int(round(min_lag_s / dt)), int(round(max_lag_s / dt))
    best, blag = -np.inf, np.nan
    for lag in range(lo, min(hi, len(s) - 4) + 1):
        c = float(s[:-lag] @ s[lag:]) / denom
        if c > best:
            best, blag = c, lag * dt
    out["velocity_autocorrelation_max"] = float(best)
    # The lag is an argmax and ALWAYS returns something: a straight-line
    # trajectory reported the same 0.50 s lag as a genuine 0.5 s oscillation,
    # separated only by a correlation of 0.05 against 0.81. Guarding on
    # "correlation > 0" does not help, since 0.05 is positive. The
    # non-arbitrary test is whether this series' own peak beats what SHUFFLING
    # THE SAME SERIES reaches -- same permutation logic used for the feature
    # baselines elsewhere in this project, and it needs no threshold chosen by
    # hand. Below that, there is no period to report.
    rng = np.random.RandomState(0)
    null = []
    for _ in range(n_perm):
        z = rng.permutation(s)
        dz = float(z @ z)
        null.append(max((float(z[:-l] @ z[l:]) / dz
                         for l in range(lo, min(hi, len(z) - 4) + 1)), default=0.0))
    thr = float(np.percentile(null, 95)) if null else 0.0
    out["autocorrelation_beats_shuffled"] = float(best > thr)
    out["autocorrelation_lag_s"] = float(blag) if best > thr else np.nan
    f = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), dt)
    band = (freqs >= 1 / max_lag_s) & (freqs <= 1 / min_lag_s)
    out["dominant_frequency_strength"] = float(f[band].max() / max(f[1:].sum(), 1e-9)) \
        if band.any() else np.nan
    d = direction(frames, dt, rel_t)
    if np.isfinite(d.get("reversal_count", np.nan)):
        out["repeated_reversal_rate"] = float(d["reversal_count"] / (len(frames) * dt))
    return out


def two_hand(frames, dt, rel_t):
    """An interaction PROXY. Without object tracks nothing here observes
    contact, so these are not named contact features."""
    out = dict.fromkeys(["inter_hand_distance_change", "two_hand_synchrony",
                         "two_hand_presence_change"], np.nan)
    d, both = [], []
    for f in frames:
        ts = list(f["tracks"].values())
        both.append(len(ts) >= 2)
        if len(ts) >= 2:
            c = [((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
                 for t in ts[:2]]
            sc = np.mean([np.hypot(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1])
                          for t in ts[:2]])
            d.append(np.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]) / max(sc, 1e-6))
        else:
            d.append(np.nan)
    d = np.array(d)
    both = np.array(both)
    out["two_hand_presence_change"] = float((np.diff(both.astype(int)) != 0).sum())
    pre, post = (rel_t < 0) & np.isfinite(d), (rel_t >= 0) & np.isfinite(d)
    if pre.any() and post.any():
        out["inter_hand_distance_change"] = float(abs(np.nanmedian(d[post]) -
                                                      np.nanmedian(d[pre])))
    if np.isfinite(d).sum() >= 6:
        dd = np.diff(np.nan_to_num(d, nan=np.nanmean(d)))
        out["two_hand_synchrony"] = float(1.0 - np.std(dd) / (np.nanmean(np.abs(dd)) + 1e-9)) \
            if np.nanmean(np.abs(dd)) > 0 else np.nan
    return out


def all_features(entry):
    frames = entry["frames"]
    dt = 1.0 / float(entry.get("fps", 10.0))
    rel_t = np.array([f["rel_t"] for f in frames], float)
    out = {}
    for fn in (observability, direction, stop_start, periodicity, two_hand):
        out.update(fn(frames, dt, rel_t))
    return out
