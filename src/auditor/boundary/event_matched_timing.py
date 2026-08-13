"""Same event, same peaks: human timing against the stored timing it replaced.

WHY THE PAIRED RESULT IS NOT YET THE CLAIM. The significant paired difference
compared 58 human boundaries from 45 audited events against all 2713 segment
joins in those 36 recordings. Same recordings, same peaks, same null -- but not
the same boundary POPULATION. The 45 events were chosen for an ontology audit,
and an alternative reading survives that: those boundaries may simply be more
visually salient than the average stored join, so a detector would find them
more easily whoever timed them. That is not peak-based selection, and the
earlier checks do not exclude it. It is a composition difference and it needs
its own control.

THIS FILE REMOVES THE POPULATION DIFFERENCE. For one event there is one human
timing and one stored timing, and both are scored against the same peaks. What
is left is only the correction.

THE JOIN RULE IS VERIFIED, NOT TRUSTED. The first version of this file
decided which events had a stored boundary by matching category NAMES --
exact, early, late, missed_* -- and it matched none of them, because these 45
events came from batch3 and carry `batch3_gt_boundary` and
`batch3_raw_change_peak`. Reading the first of those as GT-centred because of
its name would be the same mistake with a longer list.

So the rule is an identity CHECK against the annotations: an event is usable
when its candidate time coincides, within --identity_eps, with an actual
segment edge in that recording. If it does, that edge IS the stored timing for
the event, exactly and by construction. If it does not, the event has no
stored boundary attached and pairing it would need a nearest-stored rule
invented after seeing the data -- one that could be pulled toward whichever
answer the peaks favour.

The check reads recseg and the event ids. It never reads a peak, so it cannot
select toward the result. Categories are reported beside it because a usable
set dominated by one category is a different population from a mixed one, and
that has to be visible.

INTERVALS KEEP THE INTERVAL-AWARE DEFINITION: a peak inside a human-marked
transition is at distance 0, outside it the distance to the nearer edge. The
stored side is a point and has no interval form, which is itself part of what
is being compared -- the stored schema cannot express a transition that takes
time, and 20 of the 45 events needed one.

TWO INTERVALS ON THE OUTPUT, because they answer different questions. The
recording-clustered bootstrap says how much the observed difference would move
on another sample of recordings. The circular-shift permutation says how large
a difference appears when the peaks are unrelated to both timings. A result
needs both: a tight bootstrap around an effect that the null also produces is
not evidence.

Usage:
    python -m src.auditor.boundary.event_matched_timing \
        --gold_json data/gold/alignment_timing_gold_45.json \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --predictions .../timing36_audit/predictions.jsonl \
        --n_boot 2000 --n_perm 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

EID = re.compile(r"^(recording_\d+)_(.+)_t(\d+(?:\.\d+)?)$")


def parse_eid(eid):
    m = EID.match(eid)
    return (m.group(1), m.group(2), float(m.group(3))) if m else (None, None,
                                                                  None)


def load_peaks(path):
    peaks = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                continue
            for p in r.get("predicted_peaks") or []:
                if p.get("pred_time") is not None:
                    peaks[r["recording_id"]].append(float(p["pred_time"]))
    for v in peaks.values():
        v.sort()
    return peaks


def d_human(t, ev, mode):
    """Distance from a peak to this event's human timing."""
    best = None
    for p in ev["human_points"]:
        best = min(best, abs(t - p)) if best is not None else abs(t - p)
    for lo, hi in ev["human_intervals"]:
        lo, hi = min(lo, hi), max(lo, hi)
        if mode == "midpoint":
            d = abs(t - (lo + hi) / 2.0)
        else:
            d = 0.0 if lo <= t <= hi else (lo - t if t < lo else t - hi)
        best = min(best, d) if best is not None else d
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold_json",
                    default="data/gold/alignment_timing_gold_45.json")
    ap.add_argument("--migrated", required=True,
                    help="pair_schema_v2_migrated.csv -- the only place the "
                         "audit_key can be turned back into an event_id, and "
                         "the category lives in the event_id")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--recseg", action="append", required=True,
                    help="recseg json(s). An event is usable only if its "
                         "candidate time coincides with a real segment edge, "
                         "which is what makes the stored timing an identity "
                         "rather than a nearest-neighbour guess")
    ap.add_argument("--identity_eps", type=float, default=0.06,
                    help="how close the candidate time must be to a segment "
                         "edge to count as being that edge. Larger than the "
                         "0.1s annotation grid step would start matching "
                         "DIFFERENT boundaries")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    evs = json.load(open(a.gold_json, encoding="utf-8"))["events"]
    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        mig = [r["event_id"] for r in csv.DictReader(f) if r.get("event_id")]
    by_key = {}
    for eid in mig:
        rid, cat, t = parse_eid(eid)
        if rid:
            by_key[(rid, round(t, 1))] = (eid, cat)

    from src.auditor.semantic.render_ontology_clips import get_segments
    edges = defaultdict(set)
    for p in a.recseg:
        if not os.path.exists(p):
            print(f"  !! {p} not found")
            continue
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid:
                continue
            for sg in get_segments(r)[0]:
                edges[rid].add(round(float(sg[1]), 2))
                edges[rid].add(round(float(sg[2]), 2))

    peaks = load_peaks(a.predictions)
    print(f"{len(evs)} timing-gold events; {len(mig)} migrated event ids; "
          f"{sum(len(v) for v in peaks.values())} peaks over {len(peaks)} "
          f"recordings")
    print(f"{sum(len(v) for v in edges.values())} segment edges over "
          f"{len(edges)} recordings")

    matched, unmatched, no_identity, no_peaks = [], [], [], []
    cat_of = {}
    for e in evs:
        key = (e["recording_id"], round(e["candidate_time"], 1))
        hit = by_key.get(key)
        cat = hit[1] if hit else None
        cat_of[e["audit_key"]] = cat
        if not hit:
            unmatched.append(e["audit_key"])
            continue
        # THE IDENTITY CHECK -- annotations only, no peaks
        near = [x for x in edges.get(e["recording_id"], ())
                if abs(x - e["candidate_time"]) <= a.identity_eps]
        if not near:
            no_identity.append((e["audit_key"], cat))
            continue
        if not peaks.get(e["recording_id"]):
            no_peaks.append(e["audit_key"])
            continue
        matched.append(dict(e, event_id=hit[0], source_category=cat,
                            stored_time=min(
                                near, key=lambda x: abs(x - e["candidate_time"]))))

    print(f"\nJOIN -- identity checked against the annotations, no peak "
          f"read:")
    print(f"  usable (candidate time IS a segment edge)  {len(matched):>3} "
          f"over {len({m['recording_id'] for m in matched})} recordings")
    if matched:
        print(f"    by category: "
              f"{dict(Counter(m['source_category'] for m in matched).most_common())}")
    print(f"  excluded, no segment edge within {a.identity_eps}s  "
          f"{len(no_identity):>3}")
    if no_identity:
        print(f"    by category: "
              f"{dict(Counter(c for _k, c in no_identity).most_common())}")
    print(f"  excluded, audit_key not in --migrated      {len(unmatched):>3}")
    print(f"  excluded, recording has no peaks           {len(no_peaks):>3}")
    print(f"  all 45 categories seen: "
          f"{dict(Counter(v for v in cat_of.values()).most_common())}")
    if not matched:
        raise SystemExit(
            "nothing to compare. If everything is prediction-centred, this "
            "test cannot be\n  run without a mapping rule from candidate to "
            "stored boundary that is defined\n  independently of the peaks -- "
            "and nearest-stored is not one.")
    if len(matched) < 15:
        print(f"  !! {len(matched)} events is a small paired sample. The "
              f"intervals below will be wide\n     and that width is the "
              f"result, not a detail to round off.")

    def stats(pk_by_rec, mode, rows):
        """Per event: is a peak within tol of the human timing, of the stored
        one, and which is closer."""
        out = []
        for m in rows:
            pk = pk_by_rec.get(m["recording_id"]) or []
            if not pk:
                continue
            dh = min(d_human(t, m, mode) for t in pk)
            ds = min(abs(t - m["stored_time"]) for t in pk)
            out.append((dh, ds, dh <= a.tol, ds <= a.tol))
        return out

    results = {}
    for mode in ("midpoint", "interval_aware"):
        obs = stats(peaks, mode, matched)
        n = len(obs)
        hh = sum(1 for _dh, _ds, h, _s in obs if h)
        ss = sum(1 for _dh, _ds, _h, s in obs if s)
        closer_h = sum(1 for dh, ds, _h, _s in obs if dh < ds - 1e-9)
        closer_s = sum(1 for dh, ds, _h, _s in obs if ds < dh - 1e-9)
        tie = n - closer_h - closer_s
        med_h = sorted(x[0] for x in obs)[n // 2]
        med_s = sorted(x[1] for x in obs)[n // 2]
        delta = (hh - ss) / n

        # recording-clustered bootstrap: resample RECORDINGS, not events
        rng = random.Random(a.seed)
        by_rec = defaultdict(list)
        for m, o in zip([m for m in matched
                         if peaks.get(m["recording_id"])], obs):
            by_rec[m["recording_id"]].append(o)
        rids = list(by_rec)
        boots = []
        for _ in range(a.n_boot):
            pick = [by_rec[rng.choice(rids)] for _ in rids]
            flat = [o for g in pick for o in g]
            if flat:
                boots.append((sum(1 for o in flat if o[2])
                              - sum(1 for o in flat if o[3])) / len(flat))
        boots.sort()
        blo = boots[int(0.025 * len(boots))]
        bhi = boots[min(int(0.975 * len(boots)), len(boots) - 1)]

        # circular-shift permutation, same peaks moved together
        span = {r: max(peaks[r] + [m["stored_time"] for m in matched
                                   if m["recording_id"] == r] + [1.0])
                for r in rids}
        rng2 = random.Random(a.seed)
        nulls = []
        for _ in range(a.n_perm):
            sh = {r: [(t + rng2.uniform(0, span[r])) % span[r]
                      for t in peaks[r]] for r in rids}
            o2 = stats(sh, mode, matched)
            if o2:
                nulls.append((sum(1 for x in o2 if x[2])
                              - sum(1 for x in o2 if x[3])) / len(o2))
        nulls.sort()
        nm = sum(nulls) / len(nulls)
        nlo = nulls[int(0.025 * len(nulls))]
        nhi = nulls[min(int(0.975 * len(nulls)), len(nulls) - 1)]
        pv = (1 + sum(1 for x in nulls if x >= delta)) / (len(nulls) + 1)

        print(f"\n{'=' * 74}\n{mode.upper()}   {n} paired events over "
              f"{len(rids)} recordings\n{'=' * 74}")
        print(f"  human closer   {closer_h:>4}")
        print(f"  stored closer  {closer_s:>4}")
        print(f"  tie            {tie:>4}")
        print(f"  median distance to nearest peak   human {med_h:.2f}s   "
              f"stored {med_s:.2f}s")
        print(f"  hit rate at tol {a.tol}s          human "
              f"{hh}/{n} = {hh/n:.3f}   stored {ss}/{n} = {ss/n:.3f}")
        print(f"  delta hit rate  {delta:+.3f}")
        print(f"    recording-clustered bootstrap 95%  "
              f"[{blo:+.3f}, {bhi:+.3f}]"
              + ("   spans zero" if blo <= 0 <= bhi else ""))
        print(f"    shift-null mean {nm:+.3f}, null 95% [{nlo:+.3f}, "
              f"{nhi:+.3f}], permutation p = {pv:.4f}")
        results[mode] = {
            "n": n, "recordings": len(rids), "human_closer": closer_h,
            "stored_closer": closer_s, "tie": tie,
            "median_human": round(med_h, 3), "median_stored": round(med_s, 3),
            "hit_human": hh, "hit_stored": ss, "delta": round(delta, 4),
            "bootstrap_95": [round(blo, 4), round(bhi, 4)],
            "null_mean": round(nm, 4), "null_95": [round(nlo, 4),
                                                   round(nhi, 4)],
            "p_value": round(pv, 5)}

    print(f"\n  The bootstrap says how far this would move on another sample "
          f"of recordings.\n  The permutation says how large a difference "
          f"appears when the peaks are unrelated\n  to both timings. An "
          f"effect needs to clear the second and have the first exclude\n"
          f"  zero; either alone is not the result.")
    print(f"\n  STILL NOT HELD OUT. These peaks come from a head fitted on "
          f"these recordings'\n  stored annotations, which biases this "
          f"particular comparison AGAINST the human\n  side. The clean "
          f"version needs recording-grouped out-of-fold peaks.")

    if a.out:
        json.dump({"tol": a.tol, "n_matched": len(matched),
                   "excluded_no_identity": len(no_identity),
                   "excluded_unmatched": len(unmatched),
                   "results": results},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
