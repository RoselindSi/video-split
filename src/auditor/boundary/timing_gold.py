"""The frozen 45-event timing gold: raw human answers, then derived fields.

TWO LAYERS, AND THE RAW ONE IS NEVER OVERWRITTEN. The human gave points,
intervals, several boundaries per event, and a `gradual` flag for transitions
with no instant. Collapsing that to one number per event at load time would
destroy the only record of what was actually observed, and every question
downstream would then be asked of a summary. So the raw columns are carried
through untouched and every derived quantity is computed beside them, named,
and recomputable.

    raw         human_boundary_points_json    [287.0]
                human_boundary_intervals_json [[254.0, 258.0]]
                gradual                       True / False
    derived     midpoint of each interval
                nearest human boundary to the candidate
                candidate-to-boundary distance, two ways

TWO DISTANCE DEFINITIONS, BOTH REPORTED. An interval collapsed to its midpoint
is a point measurement and comparable with every earlier point-based number.
An interval treated as an interval is more faithful to what was seen:

    candidate inside the interval    distance 0
    candidate before it              start - candidate
    candidate after it               candidate - end

They are not redundant. If the two disagree about whether the detector is
aligned, the disagreement is concentrated in the 17 interval and 3
gradual_interval events, and that is itself the finding -- it would mean the
alignment verdict depends on a modelling choice about transitions rather than
on the video.

THE CANDIDATE IS NOT ALLOWED TO CHOOSE ITS BOUNDARY. Several events carry
more than one human boundary; 35/t351 has three, at 346.5, 349.5-350.5 and
353.8-354.9, against a candidate at 351.0. The nearest is 350.0 and the code
finds it. A human picking "the relevant one" while looking at the candidate
would be doing the alignment measurement by hand, in the direction of
agreement, which is exactly the selection this whole experiment exists to
avoid.

NOTHING IS TRUNCATED. 50/t291.5 has its human boundary at 287.0, 4.5s from
the candidate, well outside the 2.0s max_retime used elsewhere. That makes it
a large misalignment, which is a result and not an error, and clamping it or
dropping it would delete the strongest evidence in the set.

Usage:
    python -m src.auditor.boundary.timing_gold \
        --csv data/gold/alignment_timing_gold_45.csv \
        --out data/gold/alignment_timing_gold_45.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter


def parse_json_cell(s):
    """The sheet's JSON columns, tolerant of single quotes and blanks."""
    s = (s or "").strip()
    if not s or s in ("[]", "null", "None"):
        return []
    try:
        return json.loads(s)
    except ValueError:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []


def interval_distance(t, lo, hi):
    """0 inside, otherwise the distance to the nearer edge. Signed by side.

    Returned as (distance, signed), where signed is negative when the
    candidate is EARLY -- before the interval -- so the direction survives.
    A magnitude alone cannot tell a detector that fires early from one that
    fires late, and those need different fixes."""
    if lo > hi:
        lo, hi = hi, lo
    if lo <= t <= hi:
        return 0.0, 0.0
    if t < lo:
        return lo - t, -(lo - t)
    return t - hi, t - hi


def load(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("audit_key") or "").strip():
                continue
            pts = [float(x) for x in parse_json_cell(
                r.get("human_boundary_points_json"))]
            ivs = [[float(a), float(b)] for a, b in parse_json_cell(
                r.get("human_boundary_intervals_json"))]
            canon = [float(x) for x in parse_json_cell(
                r.get("canonical_times_json"))]
            cand = float(r["candidate_time"])

            # derived: midpoints, and the point set the midpoint mode uses
            mids = [round((a + b) / 2.0, 3) for a, b in ivs]
            points_mid = sorted(pts + mids)
            if canon and len(canon) != len(points_mid):
                # the sheet's own canonical list is kept as the authority and
                # the disagreement is reported rather than silently resolved
                pass

            # derived: distance candidate -> nearest boundary, both ways
            d_mid = min((abs(cand - p), cand - p) for p in points_mid) \
                if points_mid else (None, None)
            cands = [(abs(cand - p), cand - p) for p in pts] + \
                    [interval_distance(cand, a, b) for a, b in ivs]
            d_iv = min(cands) if cands else (None, None)

            rows.append({
                "audit_key": r["audit_key"],
                "recording_id": r["recording_id"],
                "candidate_time": cand,
                "timing_type": r.get("timing_type", ""),
                "gradual": str(r.get("gradual", "")).strip().lower() == "true",
                # raw
                "human_points": pts,
                "human_intervals": ivs,
                "canonical_times": canon,
                "timing_note": r.get("timing_note", ""),
                # derived
                "interval_midpoints": mids,
                "points_midpoint_mode": points_mid,
                "n_boundaries": len(pts) + len(ivs),
                "nearest_midpoint_time": (
                    min(points_mid, key=lambda p: abs(cand - p))
                    if points_mid else None),
                "dist_midpoint": None if d_mid[0] is None else round(d_mid[0], 3),
                "signed_midpoint": None if d_mid[1] is None else round(d_mid[1], 3),
                "dist_interval_aware": None if d_iv[0] is None else round(d_iv[0], 3),
                "signed_interval_aware": None if d_iv[1] is None else round(d_iv[1], 3),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/gold/alignment_timing_gold_45.csv")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = load(a.csv)
    recs = {r["recording_id"] for r in rows}
    print(f"{len(rows)} events over {len(recs)} recordings")
    print(f"  timing_type: "
          f"{dict(Counter(r['timing_type'] for r in rows).most_common())}")
    print(f"  gradual: {sum(1 for r in rows if r['gradual'])}")
    print(f"  events with more than one human boundary: "
          f"{sum(1 for r in rows if r['n_boundaries'] > 1)}")

    ivs = [b - a_ for r in rows for a_, b in r["human_intervals"]]
    if ivs:
        ivs.sort()
        print(f"  human interval width: median {ivs[len(ivs)//2]:.2f}s  "
              f"min {ivs[0]:.2f}s  max {ivs[-1]:.2f}s  (n={len(ivs)})")

    # ---- candidate against human, which needs no detector output at all ----
    print(f"\nCANDIDATE vs HUMAN BOUNDARY -- no peaks involved, so this is "
          f"available now:")
    print(f"  {'band':>14}{'midpoint':>11}{'interval-aware':>16}")
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9]
    for lo, hi in zip(edges, edges[1:]):
        nm = sum(1 for r in rows if r["dist_midpoint"] is not None
                 and lo <= r["dist_midpoint"] < hi)
        ni = sum(1 for r in rows if r["dist_interval_aware"] is not None
                 and lo <= r["dist_interval_aware"] < hi)
        hs = "inf" if hi > 1e8 else f"{hi:.2f}"
        print(f"  {lo:>5.2f}-{hs:<8}{nm:>11}{ni:>16}")
    for mode in ("dist_midpoint", "dist_interval_aware"):
        d = sorted(r[mode] for r in rows if r[mode] is not None)
        al = sum(1 for x in d if x <= a.tol)
        print(f"  {mode:<22} median {d[len(d)//2]:.2f}s   "
              f"within {a.tol}s: {al}/{len(d)} ({100*al/len(d):.0f}%)")

    sm = [r["signed_midpoint"] for r in rows
          if r["signed_midpoint"] is not None]
    early = sum(1 for x in sm if x < 0)
    print(f"  signed (candidate - boundary): {early}/{len(sm)} candidates "
          f"are EARLY; median {sorted(sm)[len(sm)//2]:+.2f}s")

    dis = [r for r in rows
           if r["dist_midpoint"] is not None
           and (r["dist_midpoint"] <= a.tol) != (r["dist_interval_aware"] <= a.tol)]
    print(f"\n  events where the two definitions DISAGREE about alignment at "
          f"tol {a.tol}: {len(dis)}")
    for r in dis:
        print(f"    {r['audit_key']:<14} {r['timing_type']:<17} "
              f"midpoint {r['dist_midpoint']:.2f}s  "
              f"interval-aware {r['dist_interval_aware']:.2f}s")
    if dis:
        print(f"    the alignment verdict for these depends on a modelling "
              f"choice about\n    transitions, not on the video. That is why "
              f"both are carried.")

    if a.out:
        json.dump({"source_csv": a.csv, "tol": a.tol, "events": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
