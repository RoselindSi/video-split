"""New human task timing against the old corrected times. No detector anywhere.

WHY THIS ONE IS CLEAN. Every other timing comparison in this project has had a
detector somewhere in it, and every detector-based pool carries the selection
that built it. This has neither: both sides are human judgements about the
same events, so whatever disagreement shows up is between two annotation
passes and nothing else. Peak selection cannot reach it.

WHAT THE TWO SIDES ARE.

    old   `corrected_boundary_times_json` and `primary_corrected_boundary_time`
          from audit_188_gold_v2, written during the boundary error audit,
          under the schema in force then.
    new   task-level boundaries from the semantic/enrichment timing sheets,
          written under the FROZEN task-level ontology, which explicitly
          treats a continuous held sequence as one instance and has a value
          for "no task boundary here".

They are not two measurements of one quantity. The old pass was asked where
the boundary is; the new pass was asked where the TASK-LEVEL boundary is, and
was allowed to answer "nowhere". A distance between them is therefore a
mixture of imprecision and ontology change, and the categorical disagreements
below separate those two better than any distance does.

FOUR DISAGREEMENT KINDS, counted separately, because they mean different
things and averaging them hides all four:

    both_have      both passes give times -- a distance question
    new_says_none  old gave a time, the new pass says no task boundary. Under
                   the frozen ontology this is the EXPECTED outcome for
                   repeated-instance and motion-phase cases, not an error
    old_says_none  old recorded no_valid_boundary and the new pass found one
    neither        both say none -- agreement, and invisible in any distance

MOTION-PHASE TIMES ARE NOT BOUNDARIES and are excluded on the new side, as
they are everywhere else. Including them would manufacture agreement with
exactly the old times the frozen ontology was written to stop treating as
boundaries.

Usage:
    python -m src.auditor.boundary.timing_qa_compare \
        --new data/gold/task_timing_gold.json \
        --old data/gold/audit_188_gold_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

EID = re.compile(r"^recording_0*(\d+)_.*_t(\d+(?:\.\d+)?)$")
KEY = re.compile(r"^(\d+)/t(\d+(?:\.\d+)?)$")


def dist(t, b):
    """Point-to-boundary, interval-aware: 0 inside, else to the nearer edge."""
    _k, lo, hi = b
    if lo <= t <= hi:
        return 0.0
    return lo - t if t < lo else t - hi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="data/gold/task_timing_gold.json")
    ap.add_argument("--old", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--fuzzy_join_s", type=float, default=0.6)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    new = json.load(open(a.new, encoding="utf-8"))["events"]
    old_rows = [json.loads(l) for l in open(a.old, encoding="utf-8")
                if l.strip()]
    by_rec = defaultdict(list)
    for g in old_rows:
        m = EID.match(g["event_id"])
        if m:
            by_rec[int(m.group(1))].append((float(m.group(2)), g))

    pairs, unmatched = [], []
    for e in new:
        m = KEY.match(e["audit_key"])
        if not m:
            unmatched.append(e["audit_key"])
            continue
        rid, t = int(m.group(1)), float(m.group(2))
        c = sorted(((abs(x - t), g) for x, g in by_rec.get(rid, ())),
                   key=lambda z: z[0])
        if c and (c[0][0] <= 0.01 or
                  (c[0][0] <= a.fuzzy_join_s
                   and (len(c) == 1 or c[1][0] > 5 * c[0][0]))):
            pairs.append((e, c[0][1]))
        else:
            unmatched.append(e["audit_key"])
    print(f"{len(new)} new timing events; {len(pairs)} joined to the old "
          f"gold; {len(unmatched)} unmatched")
    if unmatched:
        print(f"  unmatched: {unmatched[:6]}")

    kinds = Counter()
    both = []
    for e, g in pairs:
        old_t = [float(x) for x in
                 (g.get("corrected_boundary_times_json") or [])]
        p = g.get("primary_corrected_boundary_time")
        if p is not None:
            old_t.append(float(p))
        old_t = sorted(set(round(x, 2) for x in old_t))
        old_none = bool(g.get("no_valid_boundary")) or not old_t
        new_none = e["asserts_no_boundary"] or not e["boundaries"]
        if not old_none and not new_none:
            kinds["both_have"] += 1
            both.append((e, g, old_t))
        elif not old_none and new_none:
            kinds["new_says_none"] += 1
        elif old_none and not new_none:
            kinds["old_says_none"] += 1
        else:
            kinds["neither"] += 1

    print(f"\nDISAGREEMENT KINDS (counted separately on purpose):")
    for k in ("both_have", "new_says_none", "old_says_none", "neither"):
        print(f"  {k:<16}{kinds.get(k, 0):>4}")
    print(f"  `new_says_none` is the EXPECTED outcome under the frozen "
          f"ontology for repeated\n  instances and motion-phase changes -- it "
          f"is an ontology difference, not an error.")

    if not both:
        raise SystemExit("no event has times on both sides; nothing to "
                         "measure")

    # old time -> nearest NEW boundary, and new boundary -> nearest OLD time
    d_old, d_new, per_ev = [], [], []
    for e, g, old_t in both:
        bs = [tuple(b) for b in e["boundaries"]]
        do = [min(abs(dist(t, b)) for b in bs) for t in old_t]
        dn = [min(abs(t - x) for x in old_t)
              for _k, lo, hi in bs for t in [(lo + hi) / 2.0]]
        d_old += do
        d_new += dn
        per_ev.append({"audit_key": e["audit_key"],
                       "recording_id": e["recording_id"],
                       "n_old": len(old_t), "n_new": len(bs),
                       "median_old_to_new": sorted(do)[len(do) // 2],
                       "matched_old": sum(1 for x in do if x <= a.tol),
                       "status": e["task_timing_status"]})

    def summ(name, d):
        d = sorted(d)
        within = sum(1 for x in d if x <= a.tol)
        print(f"  {name:<34}n={len(d):>4}  median {d[len(d)//2]:>6.2f}s  "
              f"within {a.tol}s {within}/{len(d)} "
              f"({100*within/len(d):.0f}%)")
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 1e9)):
            n = sum(1 for x in d if lo <= x < hi)
            print(f"      {lo:>4.1f}-{'inf' if hi > 1e8 else f'{hi:.1f}':<5} "
                  f"{n:>4}")

    print(f"\nDISTANCES on the {len(both)} events where both passes gave "
          f"times:")
    summ("old time -> nearest new boundary", d_old)
    summ("new boundary -> nearest old time", d_new)

    # recording-clustered bootstrap on the agreement rate
    rng = random.Random(a.seed)
    by_r = defaultdict(list)
    for e, g, old_t in both:
        bs = [tuple(b) for b in e["boundaries"]]
        for t in old_t:
            by_r[e["recording_id"]].append(
                1 if min(abs(dist(t, b)) for b in bs) <= a.tol else 0)
    keys = list(by_r)
    boots = []
    for _ in range(a.n_boot):
        flat = [x for _ in keys for x in by_r[rng.choice(keys)]]
        if flat:
            boots.append(sum(flat) / len(flat))
    boots.sort()
    rate = sum(1 for x in d_old if x <= a.tol) / len(d_old)
    print(f"\n  agreement rate (old time within {a.tol}s of a new boundary) "
          f"{rate:.3f}\n  recording-clustered 95% "
          f"[{boots[int(0.025*len(boots))]:.3f}, "
          f"{boots[min(int(0.975*len(boots)), len(boots)-1)]:.3f}]  over "
          f"{len(keys)} recordings")

    worst = sorted(per_ev, key=lambda r: -r["median_old_to_new"])[:8]
    print(f"\n  furthest apart:")
    for r in worst:
        print(f"    {r['audit_key']:<14}{r['status']:<18}"
              f"old {r['n_old']} new {r['n_new']}  "
              f"median {r['median_old_to_new']:.1f}s  "
              f"matched {r['matched_old']}/{r['n_old']}")

    print(f"\n  NO DETECTOR IS INVOLVED. Both sides are human. A disagreement "
          f"here is between two\n  annotation passes, and the peak-selection "
          f"that contaminates every other timing\n  comparison in this "
          f"project cannot reach it.")

    if a.out:
        json.dump({"tol": a.tol, "kinds": dict(kinds),
                   "agreement_rate": rate, "events": per_ev},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
