"""Is there enough gold to train a candidate verifier? Counts only, no model.

Two questions were entangled for months and the two-field migration finally
separates them:

    does a boundary belong here          instance_relation
    is THIS candidate that boundary      candidate alignment

Everything downstream of the second question was previously trained on a
population that included events where the answer to the first was no. A
spurious peak with no boundary near it is not an `early` candidate; it is not
a candidate for anything. So the alignment population is defined here as the
ontology positives and nothing else:

    new_action                  boundary exists
    same_action_new_instance    boundary exists
    same_instance               NO BOUNDARY -- excluded, not relabelled

`same_instance` events are not `NO_VALID` alignment cases. They end at stage
one. Feeding them to an alignment head is exactly the contamination the split
was made to remove, and it is a filter rather than a class here.

THE DERIVATION IS NOT REWRITTEN. relation() and find_duplicates() come from
labels.py, so this file cannot quietly disagree with the trained labels about
what EXACT means. What it adds is the join to instance_relation, the counts by
recording, and the conflicts.

WHAT DECIDES GO / NO-GO is the misaligned count and the recordings it spans,
not the total. A 58-event pool that is 50 EXACT and 3 EARLY cannot train a
verifier at any architecture, and the answer to that is a targeted audit of
the error pool -- sampled for learnability, not for prevalence -- rather than
a smaller model.

EVERY COUNT MOVES WITH --tol AND --max_retime_s, so a sweep is printed rather
than one table at one setting. EXACT against EARLY is a threshold on the same
continuous offset; reporting a single tolerance would make an arbitrary choice
look like a measurement.

THE CONFLICTS ARE THE OTHER HALF. Three of them say the old candidate gold
still carries ontology judgements:

    boundary exists (ontology) but the candidate was called NO_VALID
    same_instance (ontology) but the candidate was called EXACT
    several candidates resolving onto one corrected boundary

Usage:
    python -m src.auditor.boundary.candidate_alignment_v2 \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --out /workspace/tr1/results/auditor/candidate_alignment_v2.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from src.auditor.boundary.labels import (
    TOL, MAX_RETIME_S, cand_time, recording_of, nearest_corrected, relation,
    find_duplicates)

POSITIVE = ("new_action", "same_action_new_instance")
ALIGN = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID", "UNDECIDABLE"]
# the binary the product actually asks. NO_VALID is not misaligned -- on an
# ontology positive it is a contradiction, and it is counted as one below
BINARY = {"EXACT": "ALIGNED", "EARLY": "MISALIGNED", "LATE": "MISALIGNED",
          "DUPLICATE": "MISALIGNED"}


def derive(mig, gold, tol, max_retime):
    dup = find_duplicates(gold, {}, tol)  # subtypes is unused by that function
    out = {}
    for eid, rel in mig.items():
        g = gold.get(eid)
        cls, off, why = relation(eid, g, dup, tol, max_retime)
        out[eid] = {
            "instance_relation": rel, "alignment": cls, "offset_s": off,
            "why": why, "recording_id": recording_of(eid, g),
            "candidate_time": cand_time(eid),
            "corrected_time": nearest_corrected(eid, g) if g else None,
            "split": (g or {}).get("split"),
            "source_category": (g or {}).get("source_category"),
            "in_gold": g is not None}
    return out, dup


def table(rows, key, order=None, title=""):
    """n and distinct recordings per class -- 30 events in 3 recordings is not
    30 events for a grouped split."""
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    keys = order or sorted(by, key=lambda k: -len(by[k]))
    if title:
        print(f"\n{title}")
    print(f"  {'class':<26} {'n':>5} {'recordings':>12}")
    for k in keys:
        v = by.get(k, [])
        print(f"  {str(k):<26} {len(v):>5} "
              f"{len({x['recording_id'] for x in v}):>12}")
    return {str(k): {"n": len(by.get(k, [])),
                     "recordings": len({x["recording_id"]
                                        for x in by.get(k, [])})}
            for k in keys}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrated", required=True)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max_retime_s", type=float, default=MAX_RETIME_S)
    ap.add_argument("--sweep", default="0.25,0.5,1.0,1.5",
                    help="tolerances to report alongside the main table")
    ap.add_argument("--min_aligned", type=int, default=50)
    ap.add_argument("--min_misaligned", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()

    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        mig = {r["event_id"]: r["instance_relation"] for r in csv.DictReader(f)}
    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r

    print(f"{len(mig)} migrated events; {len(gold)} audited gold rows")
    rc = Counter(mig.values())
    for k, v in rc.most_common():
        print(f"  {k:<28} {v:>4}")

    derived, dup = derive(mig, gold, a.tol, a.max_retime_s)
    pos = [d for d in derived.values() if d["instance_relation"] in POSITIVE]
    neg = [d for d in derived.values()
           if d["instance_relation"] == "same_instance"]
    print(f"\nontology positives: {len(pos)} events over "
          f"{len({d['recording_id'] for d in pos})} recordings")
    print(f"  in the timing gold: {sum(1 for d in pos if d['in_gold'])}; "
          f"absent: {sum(1 for d in pos if not d['in_gold'])}")
    print(f"  `same_instance` excluded at stage one, not relabelled: "
          f"{len(neg)}")

    # ------------------------------------------------------------- the table
    res = {"tol": a.tol, "max_retime_s": a.max_retime_s,
           "n_positive": len(pos)}
    res["alignment"] = table(
        pos, "alignment", ALIGN,
        f"alignment of the {len(pos)} ontology positives "
        f"(tol {a.tol}s, max retime {a.max_retime_s}s)")

    print(f"\ncrosstab: instance_relation x alignment")
    print(f"  {'':<28}" + "".join(f"{c[:9]:>11}" for c in ALIGN))
    for rel in POSITIVE:
        row = [d for d in pos if d["instance_relation"] == rel]
        c = Counter(d["alignment"] for d in row)
        print(f"  {rel:<28}" + "".join(f"{c.get(k, 0):>11}" for k in ALIGN))

    if any(d["split"] for d in pos):
        table(pos, "split", None, "by split")
    und = [d for d in pos if d["alignment"] == "UNDECIDABLE"]
    if und:
        print(f"\nwhy the {len(und)} UNDECIDABLE positives are undecidable:")
        for k, v in Counter(d["why"] or "unstated" for d in und).most_common():
            print(f"  {v:>4}  {k}")

    # ------------------------------------------------------ the binary target
    b = Counter(BINARY.get(d["alignment"]) for d in pos)
    rec = {k: len({d["recording_id"] for d in pos
                   if BINARY.get(d["alignment"]) == k})
           for k in ("ALIGNED", "MISALIGNED")}
    print(f"\nthe binary the product asks -- can this candidate be kept?")
    print(f"  ALIGNED     {b.get('ALIGNED', 0):>4}  over "
          f"{rec['ALIGNED']:>3} recordings")
    print(f"  MISALIGNED  {b.get('MISALIGNED', 0):>4}  over "
          f"{rec['MISALIGNED']:>3} recordings")
    print(f"  neither     {b.get(None, 0):>4}  (NO_VALID or UNDECIDABLE -- "
          f"not trainable either way)")
    res["binary"] = {"aligned": b.get("ALIGNED", 0),
                     "misaligned": b.get("MISALIGNED", 0),
                     "unusable": b.get(None, 0), "recordings": rec}

    go = (b.get("ALIGNED", 0) >= a.min_aligned
          and b.get("MISALIGNED", 0) >= a.min_misaligned)
    res["go"] = go
    print(f"\n  stated before the numbers: >= {a.min_aligned} aligned and "
          f">= {a.min_misaligned} misaligned -> {'GO' if go else 'NO-GO'}")
    if not go:
        short = []
        if b.get("ALIGNED", 0) < a.min_aligned:
            short.append(f"{a.min_aligned - b.get('ALIGNED', 0)} aligned")
        if b.get("MISALIGNED", 0) < a.min_misaligned:
            short.append(
                f"{a.min_misaligned - b.get('MISALIGNED', 0)} misaligned")
        print(f"  short by {' and '.join(short)}. The fix is a targeted audit "
              f"of the error pool for the\n  missing side, sampled for "
              f"learnability rather than prevalence -- not a smaller model.")

    # ------------------------------------------------------- tolerance sweep
    print(f"\nevery count above moves with the tolerance. EXACT against "
          f"EARLY is a threshold on\none continuous offset, so:")
    print(f"  {'tol':>6}" + "".join(f"{c[:9]:>11}" for c in ALIGN)
          + f"{'ALIGNED':>10}{'MISALIGN':>10}")
    sweep = {}
    for t in [float(x) for x in a.sweep.split(",") if x.strip()]:
        d2, _ = derive(mig, gold, t, a.max_retime_s)
        p2 = [d for d in d2.values()
              if d["instance_relation"] in POSITIVE]
        c = Counter(d["alignment"] for d in p2)
        bb = Counter(BINARY.get(d["alignment"]) for d in p2)
        print(f"  {t:>6.2f}" + "".join(f"{c.get(k, 0):>11}" for k in ALIGN)
              + f"{bb.get('ALIGNED', 0):>10}{bb.get('MISALIGNED', 0):>10}")
        sweep[str(t)] = {"alignment": dict(c),
                         "aligned": bb.get("ALIGNED", 0),
                         "misaligned": bb.get("MISALIGNED", 0)}
    res["sweep"] = sweep

    # ------------------------------------------------------------- conflicts
    print(f"\n{'=' * 74}\nCONFLICTS -- how much ontology is still inside the "
          f"candidate gold\n{'=' * 74}")
    c1 = [d for d in pos if d["alignment"] == "NO_VALID"]
    print(f"\n1. ontology says a boundary exists, the audit called the "
          f"candidate invalid: {len(c1)}")
    print(f"   Both cannot hold. Either the relation label is wrong or the "
          f"candidate was judged\n   against a boundary the auditor did not "
          f"believe in.")
    for d in c1[:6]:
        print(f"     {d['recording_id']}  {d['instance_relation']:<26} "
              f"t={d['candidate_time']}  {d['source_category']}")

    c2 = [d for d in neg if d["alignment"] in ("EXACT", "EARLY", "LATE",
                                               "DUPLICATE")]
    print(f"\n2. ontology says same_instance, the old candidate gold gave it "
          f"a timing class: {len(c2)}")
    print(f"   These are the events that would have trained an alignment head "
          f"on a boundary the\n   ontology denies. They are excluded here; "
          f"the count is how much that exclusion costs.")
    for k, v in Counter(d["alignment"] for d in c2).most_common():
        print(f"     {k:<12} {v:>4}")

    groups = defaultdict(list)
    for eid, d in derived.items():
        if d["corrected_time"] is not None and d["in_gold"]:
            groups[(d["recording_id"], round(d["corrected_time"], 1))].append(
                (eid, d))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n3. several candidates resolving onto one corrected boundary: "
          f"{len(multi)} boundaries, "
          f"{sum(len(v) for v in multi.values())} candidates")
    print(f"   find_duplicates keeps the nearest and marks the rest "
          f"DUPLICATE. Where the members\n   carry DIFFERENT instance_"
          f"relations, the ontology disagrees with itself:")
    bad = {k: v for k, v in multi.items()
           if len({d["instance_relation"] for _, d in v}) > 1}
    print(f"   groups with mixed instance_relation: {len(bad)}")
    for (rid, ct), v in list(bad.items())[:5]:
        print(f"     {rid} @ {ct}: "
              + ", ".join(f"{d['instance_relation']}/{d['alignment']}"
                          for _, d in v))
    res["conflicts"] = {"exists_but_invalid": len(c1),
                        "same_instance_with_timing": len(c2),
                        "shared_boundary_groups": len(multi),
                        "shared_boundary_mixed_relation": len(bad)}

    if a.out:
        res["events"] = {e: d for e, d in derived.items()
                         if d["instance_relation"] in POSITIVE}
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
