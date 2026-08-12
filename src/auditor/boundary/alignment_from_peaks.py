"""Build the alignment set from detector peaks. The offsets need no annotator.

THE PREVIOUS PLAN WAS HALF WRONG AND THIS FILE REPLACES THAT HALF. A targeted
audit cannot supply misaligned candidates, because the corpus does not contain
them and relabelling does not create them. Three measurements, in order:

1.  `cand_time(event_id)` MEANS TWO DIFFERENT THINGS. 94 of the 188 audited
    events are GT-centred -- exact / early / late / missed_* -- and for those
    the id's time is the ANNOTATION time, with the model's prediction
    somewhere else or absent entirely. The other 94 are prediction-centred.
    So the existing alignment derivation measures how far the ANNOTATION moved
    on half the corpus and how far the CANDIDATE was off on the other half,
    and reports them in one column. That is the same conflation the two-field
    split was made to remove, one level further down.

2.  ONLY 38 EVENTS ASK THE QUESTION AT ALL -- prediction-centred, boundary
    valid, corrected time known. Of those, 7 are misaligned at tol 0.5s and 13
    at tol 0.25s, over 6 and 10 recordings. That is the whole pool in the whole
    gold, so a targeted audit of it cannot reach 30.

3.  THE CANDIDATE GRID IS 0.5s AND THE TOLERANCE IS 0.5s. Every pred_time in
    the corpus falls on a half-second, so one grid step of error sits exactly
    at the tolerance and reads as EXACT. The audit's own `early` and `late`
    categories are all one step -- median AND max |pred - gt| is 0.50s across
    all 27 of them. The categories name an error the tolerance then absorbs.

WHAT ACTUALLY GENERATES THE DATA. Each audited event kept ONE candidate. The
detector produced many, and predictions.jsonl still has them. A corrected
boundary time plus the detector's peaks in a window around it gives one
labelled example per peak, with the offset known exactly and no annotator
involved -- the human already did the only part that needs a human, which is
saying where the boundary is.

    ALIGNED       nearest peak, |offset| <= tol
    EARLY / LATE  nearest peak, tol < |offset| <= max_assoc_s
    DUPLICATE     a further peak assigned to a boundary another peak owns
    (a boundary with no peak in the window is a MISS -- an existence failure,
     not an alignment case, and it is reported separately rather than dropped)

WHAT THIS SET IS NOT. The offsets are this detector's offsets. Retrain the
detector and the dataset changes, so a verifier fitted here is calibrated to
one upstream model and its threshold does not transfer to another. That is a
real limitation and it is the price of not waiting for an annotation round
that would take weeks and still land on the same 0.5s grid.

Usage:
    python -m src.auditor.boundary.alignment_from_peaks \
        --predictions /workspace/tr1/results/boundary/error_audit/predictions.jsonl \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --out /workspace/tr1/results/auditor/alignment_from_peaks.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from src.auditor.boundary.labels import TOL, cand_time, nearest_corrected

POSITIVE = ("new_action", "same_action_new_instance")


def corrected_boundaries(gold, mig, positives_only):
    """One entry per (recording, corrected boundary time), deduplicated.

    Several audited events can point at the same boundary, so the boundary --
    not the event -- is the unit here. That is also why DUPLICATE becomes
    computable: it is a property of the boundary's peak list."""
    by = defaultdict(set)
    prov = {}
    for eid, g in gold.items():
        if g.get("no_valid_boundary") or g.get("boundary_time_unresolved"):
            continue
        if g.get("temporal_truth") != "valid":
            continue
        rel = mig.get(eid)
        if positives_only and rel not in POSITIVE:
            continue
        times = [float(x) for x in (g.get("corrected_boundary_times_json")
                                    or [])]
        p = g.get("primary_corrected_boundary_time")
        if p is not None:
            times.append(float(p))
        for t in times:
            key = (g["recording_id"], round(t, 1))
            by[g["recording_id"]].add(round(t, 1))
            prov.setdefault(key, []).append((eid, rel))
    return by, prov


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="boundary_error_audit.py output; needs "
                         "predicted_peaks per recording")
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--migrated",
                    help="pair_schema_v2_migrated.csv. Without it the "
                         "ontology filter cannot be applied and the set will "
                         "include boundaries the ontology denies")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max_assoc_s", type=float, default=2.0,
                    help="beyond this a peak is not this boundary's candidate")
    ap.add_argument("--positives_only", action="store_true",
                    help="restrict to new_action / same_action_new_instance")
    ap.add_argument("--sweep", default="0.25,0.5,1.0")
    ap.add_argument("--out")
    a = ap.parse_args()

    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r
    mig = {}
    if a.migrated:
        with open(a.migrated, newline="", encoding="utf-8-sig") as f:
            mig = {r["event_id"]: r["instance_relation"]
                   for r in csv.DictReader(f)}
    elif a.positives_only:
        print("!! --positives_only needs --migrated; ignoring the filter")
        a.positives_only = False

    peaks = defaultdict(list)
    n_rec = 0
    with open(a.predictions, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rec += 1
            for p in r.get("predicted_peaks") or []:
                if p.get("pred_time") is not None:
                    peaks[r["recording_id"]].append(
                        (float(p["pred_time"]), p.get("score")))
    for v in peaks.values():
        v.sort()
    print(f"{n_rec} recordings in predictions; "
          f"{sum(len(v) for v in peaks.values())} peaks over "
          f"{len(peaks)} of them")
    print(f"  median peaks per recording: "
          f"{sorted(len(v) for v in peaks.values())[len(peaks)//2]}")

    bounds, prov = corrected_boundaries(gold, mig, a.positives_only)
    n_b = sum(len(v) for v in bounds.values())
    print(f"{n_b} distinct corrected boundaries over {len(bounds)} recordings"
          + ("  (ontology positives only)" if a.positives_only else ""))
    missing = [r for r in bounds if r not in peaks]
    if missing:
        print(f"  !! {len(missing)} of those recordings have no peaks in "
              f"predictions; their boundaries cannot be scored: "
              f"{missing[:4]}")

    def assign(tol):
        rows, miss = [], 0
        for rid, times in bounds.items():
            pk = peaks.get(rid, [])
            for bt in sorted(times):
                near = [(abs(t - bt), t, s) for t, s in pk
                        if abs(t - bt) <= a.max_assoc_s]
                if not near:
                    miss += 1
                    continue
                near.sort()
                for i, (d, t, s) in enumerate(near):
                    cls = ("DUPLICATE" if i else
                           "ALIGNED" if d <= tol else
                           ("EARLY" if t < bt else "LATE"))
                    rows.append({"recording_id": rid, "boundary_time": bt,
                                 "peak_time": t, "offset_s": round(bt - t, 3),
                                 "score": s, "label": cls,
                                 "rank_at_boundary": i,
                                 "relation": sorted(
                                     {r for _, r in prov.get((rid, bt), [])
                                      if r})})
        return rows, miss

    rows, miss = assign(a.tol)
    print(f"\nat tol {a.tol}s, association window {a.max_assoc_s}s:")
    c = Counter(r["label"] for r in rows)
    for k in ("ALIGNED", "EARLY", "LATE", "DUPLICATE"):
        sel = [r for r in rows if r["label"] == k]
        print(f"  {k:<12} {c.get(k, 0):>5} over "
              f"{len({r['recording_id'] for r in sel})} recordings")
    print(f"  boundaries with NO peak within {a.max_assoc_s}s: {miss}"
          f"  <- existence failures, not alignment cases")
    mis_n = sum(c.get(k, 0) for k in ("EARLY", "LATE", "DUPLICATE"))
    mis_r = len({r["recording_id"] for r in rows if r["label"] != "ALIGNED"})
    print(f"\n  ALIGNED    {c.get('ALIGNED', 0):>5} over "
          f"{len({r['recording_id'] for r in rows if r['label'] == 'ALIGNED'})}"
          f" recordings")
    print(f"  MISALIGNED {mis_n:>5} over {mis_r} recordings")

    print(f"\n|offset| distribution of the nearest peak per boundary:")
    nearest = [r for r in rows if r["rank_at_boundary"] == 0]
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        n = sum(1 for r in nearest if lo <= abs(r["offset_s"]) < hi)
        print(f"  {lo:>4.2f}-{hi:>4.2f}s  {n:>5}")
    print(f"  the grid is 0.5s, so these bins are not equally reachable -- "
          f"0.25-0.50 and\n  0.00-0.25 are one grid step and zero steps, and "
          f"nothing lands between them.")

    print(f"\nthe threshold is the whole labelling, so:")
    print(f"  {'tol':>6}{'ALIGNED':>10}{'MISALIGN':>10}{'mis recs':>10}")
    sweep = {}
    for t in [float(x) for x in a.sweep.split(",") if x.strip()]:
        rr, _ = assign(t)
        cc = Counter(r["label"] for r in rr)
        mr = len({r["recording_id"] for r in rr if r["label"] != "ALIGNED"})
        m = sum(cc.get(k, 0) for k in ("EARLY", "LATE", "DUPLICATE"))
        print(f"  {t:>6.2f}{cc.get('ALIGNED', 0):>10}{m:>10}{mr:>10}")
        sweep[str(t)] = {"aligned": cc.get("ALIGNED", 0), "misaligned": m,
                         "mis_recordings": mr}

    print(f"\nWHAT THIS SET IS NOT: these are THIS detector's offsets. Retrain "
          f"it and the\ndataset changes, so a verifier fitted here is "
          f"calibrated to one upstream model.\nIts threshold does not "
          f"transfer, and that has to be restated wherever it is used.")

    if a.out:
        json.dump({"tol": a.tol, "max_assoc_s": a.max_assoc_s,
                   "positives_only": a.positives_only, "sweep": sweep,
                   "no_peak_boundaries": miss, "rows": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out} ({len(rows)} labelled candidates)")


if __name__ == "__main__":
    main()
