"""Score a generated timeline against HUMAN-AUDITED points, on both sides.

Every previous boundary number in this project was measured against stored
ground truth on the positive side and audited labels on the negative side. The
batch4 join showed what that costs: 52 of 117 injected stored-GT boundaries
were audited as NOT boundaries, so the positives carry roughly 44% noise while
the negatives carry almost none, and every accuracy computed across that
asymmetry is pulled toward chance by an unknown amount.

This module does not use stored ground truth at all. Both sides come from the
same human audit:

    POSITIVE probes   `true_boundary_start_s` from rows the auditor called
                      task_boundary -- the auditor's own corrected time, not
                      the candidate's time and not the stored label's.
    NEGATIVE probes   `candidate_time_s` from rows the auditor called
                      no_boundary -- a specific instant a person looked at
                      and said nothing ends here.

THE MEASUREMENT THAT MATTERS MOST IS THE THIRD ONE. Among the negative probes,
the ones whose candidate_type is `gt_boundary` are places the STORED LABEL
said boundary and the HUMAN said no. If the pipeline declines to cut there, it
is agreeing with the person against the stored annotation -- which is a claim
about the labels, not about the model, and it is the only cheap way to find
out whether a generated timeline is better than what we already have.

WHAT THE FALSE-CUT RATE IS NOT. The negative probes were chosen by the
detector and by stored GT, not sampled from time. So the rate below is "how
often does it cut at a place someone already suspected", not a false-positive
rate over the recording. A pipeline that cuts every 3 seconds would still
score well here, so the cut count per minute is printed beside it.

Tolerance is 1.0s throughout.

Usage:
    python -m src.auditor.boundary.pipeline_vs_audit \
        --timeline RUNS/*/global_timeline.json \
        --audit data/gold/batch4_joint_audit.csv \
        --manifest results/hal/batch4/batch4_manifest.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict

import numpy as np

TOL_S = 1.0
BOOT = 2000
SEED = 0


def _norm_rid(x):
    m = re.search(r"(\d+)$", str(x).strip())
    return f"recording_{int(m.group(1)):06d}" if m else str(x).strip()


def load_timelines(patterns):
    """-> {recording_id: {"cuts": [t...], "duration_s": float}}.

    Boundary times are every episode start and end EXCEPT the two at the ends
    of the recording. Those correspond to initial_action_start and
    terminal_action_end, which the audit treats as a separate class because
    they are not disengagements between episodes, and counting them would
    hand the pipeline two free hits per recording."""
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            d = json.loads(open(p, encoding="utf-8").read())
            eps = d.get("episodes") or []
            dur = float(d.get("recording_duration_s") or 0.0)
            rid = d.get("recording_id") or _rid_from_path(p, d)
            cuts = set()
            for e in eps:
                for k in ("start_s", "end_s"):
                    try:
                        t = round(float(e[k]), 3)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if t <= TOL_S or (dur and t >= dur - TOL_S):
                        continue
                    cuts.add(t)
            out[_norm_rid(rid)] = {"cuts": sorted(cuts), "duration_s": dur,
                                   "n_episodes": len(eps), "path": p}
    return out


def _rid_from_path(p, d):
    src = str(d.get("source_video") or p)
    m = re.search(r"(recording_\d+|\d{4,})", src)
    return m.group(1) if m else p


def probes(audit_rows, manifest):
    """The two probe sets, both from the human audit."""
    pos, neg = [], []
    skipped = defaultdict(int)
    for r in audit_rows:
        rid = _norm_rid(r.get("recording_id", ""))
        ev = r.get("temporal_event_type", "")
        ct = manifest.get((rid, _round(r.get("candidate_time_s"))), "UNKNOWN")
        if ev == "task_boundary":
            if r.get("within_1s_tolerance") != "yes":
                # the auditor located a boundary but more than a second from
                # the candidate. The corrected time is still a human-confirmed
                # boundary, so it is kept as a probe -- what is discarded is
                # only the claim that the CANDIDATE hit it, which this module
                # never uses.
                pass
            t = _round(r.get("true_boundary_start_s")) or \
                _round(r.get("candidate_time_s"))
            if t is None:
                skipped["task_boundary with no time"] += 1
                continue
            pos.append({"recording_id": rid, "t": t, "candidate_type": ct})
        elif ev == "no_boundary":
            t = _round(r.get("candidate_time_s"))
            if t is None:
                skipped["no_boundary with no time"] += 1
                continue
            neg.append({"recording_id": rid, "t": t, "candidate_type": ct})
        else:
            skipped[f"not a probe: {ev}"] += 1
    return pos, neg, dict(skipped)


def _round(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def hit(cuts, t, tol=TOL_S):
    if not cuts:
        return False
    a = np.asarray(cuts, float)
    return bool(np.abs(a - t).min() <= tol)


def _cluster_ci(per_rec, boot=BOOT, seed=SEED):
    """Resample RECORDINGS. Probes inside one are not independent."""
    keys = [k for k in per_rec if per_rec[k][1]]
    if not keys:
        return float("nan"), float("nan"), float("nan"), float("nan")
    h = np.array([per_rec[k][0] for k in keys], float)
    n = np.array([per_rec[k][1] for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(boot, len(keys)))
    micro = h.sum() / n.sum()
    macro = float((h / n).mean())
    mi = h[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    ma = (h[idx] / n[idx]).mean(1)
    return micro, macro, float(np.quantile(ma, .025)), \
        float(np.quantile(ma, .975))


def score(name, items, timelines, want_hit, tol=TOL_S):
    per = defaultdict(lambda: [0, 0])
    off = 0
    for it in items:
        tl = timelines.get(it["recording_id"])
        if tl is None:
            off += 1
            continue
        if tl["duration_s"] and not (0 <= it["t"] <= tl["duration_s"]):
            off += 1
            continue
        got = hit(tl["cuts"], it["t"], tol)
        per[it["recording_id"]][0] += int(got == want_hit)
        per[it["recording_id"]][1] += 1
    micro, macro, lo, hi = _cluster_ci(per)
    n = sum(v[1] for v in per.values())
    print(f"  {name:<44}{micro:>8.3f}{macro:>9.3f}  "
          f"[{lo:.3f}, {hi:.3f}]  n={n} over {len(per)} rec"
          + (f", {off} unscorable" if off else ""))
    return {"metric": name, "micro": micro, "macro": macro, "lo": lo,
            "hi": hi, "n": n, "n_recordings": len(per), "unscorable": off}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", action="append", required=True,
                    help="global_timeline.json path or glob; APPEND")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--manifest", help="for candidate_type, which the "
                                       "stored-GT comparison needs")
    ap.add_argument("--tol_s", type=float, default=TOL_S)
    ap.add_argument("--out")
    a = ap.parse_args()

    tls = load_timelines(a.timeline)
    if not tls:
        raise SystemExit("no timeline matched")
    rows = [{k.lstrip("﻿").strip(): (v or "").strip()
             for k, v in r.items()}
            for r in csv.DictReader(open(a.audit, encoding="utf-8-sig"))]
    mf = {}
    if a.manifest:
        for l in open(a.manifest, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                mf[(_norm_rid(d["recording_id"]), _round(d["t"]))] = \
                    d.get("candidate_type", "")

    print(f"{len(tls)} timelines, {len(rows)} audit rows, "
          f"tolerance {a.tol_s}s")
    tot_cuts = sum(len(v["cuts"]) for v in tls.values())
    tot_min = sum(v["duration_s"] for v in tls.values()) / 60 or float("nan")
    print(f"  {tot_cuts} internal cuts over {tot_min:.1f} minutes of video "
          f"= {tot_cuts / tot_min:.2f} cuts/min")
    print(f"  (a pipeline that cuts often scores well on the negative probes "
          f"by\n   accident, so this rate belongs next to every number below)")

    pos, neg, skipped = probes(rows, mf)
    print(f"\n  {len(pos)} positive probes, {len(neg)} negative probes")
    for k, v in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"    skipped {v:>4}  {k}")

    covered = {p["recording_id"] for p in pos + neg} & set(tls)
    print(f"  {len(covered)} audited recordings have a timeline")
    if not covered:
        raise SystemExit(
            "no audited recording has a timeline. The probe recording ids and "
            "the timeline recording ids do not meet -- check that "
            "`source_video` carries the recording id, since `4` and "
            "`recording_000004` normalise to the same thing but a path that "
            "carries neither does not.")

    print(f"\n{'=' * 78}\nAGAINST HUMAN AUDIT, BOTH SIDES\n{'=' * 78}")
    print(f"  {'':<44}{'micro':>8}{'macro':>9}  "
          f"{'95% CI (macro)':<18}")
    res = []
    res.append(score("cut at a human-confirmed boundary", pos, tls, True,
                     a.tol_s))
    res.append(score("no cut at a human-confirmed non-boundary", neg, tls,
                     False, a.tol_s))

    gt_rej = [x for x in neg if x["candidate_type"] == "gt_boundary"]
    gt_conf = [x for x in pos if x["candidate_type"] == "gt_boundary"]
    peak_neg = [x for x in neg if x["candidate_type"] == "raw_change_peak"]
    if gt_rej:
        print(f"\n  the labels, not the model:")
        res.append(score("agrees with human AGAINST stored GT", gt_rej, tls,
                         False, a.tol_s))
        print(f"    These are instants the stored annotation calls a boundary "
              f"and the\n    auditor calls not one. Declining to cut here is "
              f"the pipeline siding\n    with the person. It is the cheapest "
              f"evidence that a generated\n    timeline beats the labels we "
              f"already hold.")
    if gt_conf:
        res.append(score("cut where stored GT and human agree", gt_conf, tls,
                         True, a.tol_s))
    if peak_neg:
        res.append(score("no cut at a detector false peak", peak_neg, tls,
                         False, a.tol_s))

    print(f"\n  micro weights probes, macro weights recordings, and the CI is "
          f"a cluster\n  bootstrap over recordings because probes inside one "
          f"share whatever that\n  recording does. When the two columns "
          f"disagree the number belongs to a\n  few videos -- which is how "
          f"the detector's .5400 was misread once already.")

    if a.out:
        json.dump({"tolerance_s": a.tol_s, "results": res,
                   "cuts_per_min": tot_cuts / tot_min if tot_min else None,
                   "n_timelines": len(tls),
                   "recordings_scored": sorted(covered),
                   "skipped": skipped},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
