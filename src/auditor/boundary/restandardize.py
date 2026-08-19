"""Re-derive every historical boundary judgement under the CURRENT standards.

Two things changed after most of this gold was collected, and neither was
retro-applied:

    instance_relation_policy_v2   boundary existence comes from
                                  instance_relation and from nothing else;
                                  transition_shape no longer gates it, and
                                  POINT / INTERVAL / NO_TRANSITION is retired
                                  as a training target
    tolerance = 1.0s              up from 0.5s, from 2026-08-19

So the historical numbers describe a standard the project no longer uses. This
re-derives them and prints what moves, rather than leaving two standards in
circulation.

THE POLICY IS READ FROM ITS YAML, never restated here. A re-derivation that
carried its own copy of the mapping would be a third standard.

WHAT THIS IS EXPECTED TO SHOW, stated before it runs so the result is not read
backwards: `instance_relation` was added late, so a large share of the
historical set never had it recorded, and v2 sends every one of those to
REVIEW. That is not a regression -- those events never carried an answer to the
question v2 asks. The old label answered a different question and the old
number counted it anyway.

PROVENANCE IS PART OF THE ANSWER. The migration recorded where each field came
from, and a value inherited from the very label under audit is not evidence.
The breakdown by `relation_source` / `shape_source` is printed because that is
the distinction v2 was written to enforce.

Usage:
    python -m src.auditor.boundary.restandardize \
        --pairs data/gold/pair_schema_v2_migrated.csv \
        --legacy data/gold/boundary_v1_labels.json \
        --audit data/gold/audit_72_gold_v2.jsonl \
        --audit data/gold/audit_188_gold_v2.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

POLICY = "configs/auditor/instance_relation_policy_v2.yaml"

# The OLD standard, stated once so the crosstab has something to compare with.
# `sharp_visible_transition` was the positive class and everything else the
# negative -- one head carrying perception, annotation convention and policy,
# which is the entanglement v2 exists to undo.
LEGACY_POSITIVE = "sharp_visible_transition"


def load_policy(path=POLICY):
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is required; the policy is not duplicated "
                         "in code on purpose.")
    with open(path, encoding="utf-8") as f:
        p = yaml.safe_load(f)
    if not p.get("frozen"):
        print(f"  !! {path} is not marked frozen; re-deriving against a "
              f"moving target")
    return {k: v["decision"] for k, v in p["instance_relation"].items()}, p


def read_rows(path):
    if path.endswith(".csv"):
        return list(csv.DictReader(open(path, newline="",
                                        encoding="utf-8-sig")))
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8-sig")
                if l.strip()]
    blob = json.load(open(path, encoding="utf-8-sig"))
    rows = blob.get("events", blob if isinstance(blob, list)
                    else list(blob.values()))
    return [r for r in rows if isinstance(r, dict)]


def pct(n, d):
    return f"{n / d:.1%}" if d else "—"


def section_v2(rows, dec):
    print(f"\n{'=' * 72}\n1. instance_relation_policy_v2 applied to "
          f"{len(rows)} migrated events\n{'=' * 72}")
    got = Counter(dec.get(r.get("instance_relation") or "UNKNOWN", "REVIEW")
                  for r in rows)
    for k in ("BOUNDARY", "NO_BOUNDARY", "REVIEW"):
        print(f"  {k:<14}{got[k]:>5}  {pct(got[k], len(rows))}")
    print(f"\n  trainable positive : negative = {got['BOUNDARY']} : "
          f"{got['NO_BOUNDARY']}")

    print(f"\n  why each REVIEW:")
    rev = Counter(r.get("instance_relation") or "UNKNOWN" for r in rows
                  if dec.get(r.get("instance_relation") or "UNKNOWN",
                             "REVIEW") == "REVIEW")
    for k, v in rev.most_common():
        print(f"    {k:<28}{v:>5}  {pct(v, len(rows))}")
    never = rev.get("UNKNOWN", 0)
    print(f"\n  {never} of {len(rows)} ({pct(never, len(rows))}) are REVIEW "
          f"only because instance_relation\n  was never recorded. Those events "
          f"never carried an answer to the question v2\n  asks; the old label "
          f"answered a different one and the old count included them.")

    print(f"\n  where the relation came from:")
    for k, v in Counter((r.get("relation_source") or "(never annotated)")
                        for r in rows).most_common():
        print(f"    {k:<48}{v:>5}  {pct(v, len(rows))}")

    # THE CHECK v2 WAS WRITTEN FOR. A shape inherited from the label under
    # audit is not evidence about that label.
    print(f"\n  same_action_new_instance, by shape and where the shape came "
          f"from:")
    t = Counter((r.get("transition_shape"), r.get("shape_source") or "(none)")
                for r in rows
                if r.get("instance_relation") == "same_action_new_instance")
    inherited = sum(n for (sh, src), n in t.items()
                    if str(src).startswith("legacy:"))
    for (sh, src), n in sorted(t.items(), key=lambda x: -x[1]):
        mark = "   <- inherited from the label under audit" \
            if str(src).startswith("legacy:") else ""
        print(f"    shape={sh:<14}from {src:<46}{n:>4}{mark}")
    tot = sum(t.values())
    print(f"    {inherited} of {tot} inherited. v1 gated "
          f"same_action_new_instance on the shape\n    being a gap or a point, "
          f"and that condition was being satisfied by a value\n    nobody "
          f"observed -- which is why v2 drops it.")
    return got


def section_legacy(rows, legacy, dec):
    """Old standard against new, on the events both can score."""
    print(f"\n{'=' * 72}\n2. old standard (sharp = boundary) against v2"
          f"\n{'=' * 72}")
    sub = {}
    for r in legacy:
        k = r.get("event_id") or r.get("pair_id")
        s = r.get("subtype") or r.get("temporal_pair_subtype")
        if k and s:
            sub[k] = s
    print(f"  legacy subtypes available for {len(sub)} events")
    if not sub:
        print("  !! no legacy subtype joined; nothing to compare")
        return

    tab = Counter()
    miss = 0
    for r in rows:
        k = r.get("event_id")
        s = sub.get(k)
        if s is None:
            miss += 1
            continue
        old = "BOUNDARY" if s == LEGACY_POSITIVE else "NO_BOUNDARY"
        new = dec.get(r.get("instance_relation") or "UNKNOWN", "REVIEW")
        tab[(old, new)] += 1
    n = sum(tab.values())
    print(f"  joined {n}; {miss} migrated events have no legacy subtype\n")
    print(f"  {'old':<14}{'v2':<14}{'n':>6}")
    for (o, nw), v in sorted(tab.items(), key=lambda x: -x[1]):
        flip = "   <- decision changes" if o != nw and nw != "REVIEW" else \
               "   <- now REVIEW" if nw == "REVIEW" else ""
        print(f"  {o:<14}{nw:<14}{v:>6}{flip}")
    changed = sum(v for (o, nw), v in tab.items() if o != nw)
    to_rev = sum(v for (o, nw), v in tab.items() if nw == "REVIEW")
    print(f"\n  {changed} of {n} ({pct(changed, n)}) get a different answer, "
          f"{to_rev} of them ({pct(to_rev, n)})\n  because v2 declines to "
          f"answer at all.")


def section_timing(paths, tolerances, old_tol, new_tol):
    print(f"\n{'=' * 72}\n3. timing at 1.0s against the 0.5s the older tables "
          f"used\n{'=' * 72}")
    summary = []
    for p in paths:
        rows = read_rows(p)
        errs, no_truth = [], 0
        for r in rows:
            pt = r.get("pred_time")
            tt = r.get("primary_corrected_boundary_time")
            if tt in (None, "") :
                tt = r.get("gt_time")
            try:
                errs.append(float(pt) - float(tt))
            except (TypeError, ValueError):
                no_truth += 1
        name = os.path.basename(p)
        if not errs:
            print(f"\n  {name}: no scorable pred/truth pair "
                  f"({no_truth} rows skipped)")
            continue
        print(f"\n  {name}: {len(errs)} scorable, {no_truth} without both "
              f"times")
        for tol in tolerances:
            k = sum(abs(e) <= tol for e in errs)
            print(f"    |err| <= {tol}s   {k:>4}/{len(errs)}  "
                  f"{pct(k, len(errs))}")
        early = sum(e < 0 for e in errs)
        late = sum(e > 0 for e in errs)
        print(f"    early {early} / late {late} / exact "
              f"{len(errs) - early - late}")
        # THE STANDARD CHANGE IS 0.5 -> 1.0, not first-to-last of whatever
        # sweep was passed. Reporting the widest pair in the sweep overstates
        # the change by counting tolerances nobody adopted.
        d = (sum(abs(e) <= new_tol for e in errs)
             - sum(abs(e) <= old_tol for e in errs))
        print(f"    THE STANDARD CHANGE {old_tol}s -> {new_tol}s: {d:+d} "
              f"events ({pct(d, len(errs))}) become within tolerance")
        summary.append((os.path.basename(p), len(errs),
                        sum(abs(e) <= old_tol for e in errs),
                        sum(abs(e) <= new_tol for e in errs),
                        early, late))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/gold/pair_schema_v2_migrated.csv")
    ap.add_argument("--legacy", default="data/gold/boundary_v1_labels.json")
    ap.add_argument("--audit", action="append", default=[])
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--tolerances", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--old_tolerance", type=float, default=0.5)
    ap.add_argument("--new_tolerance", type=float, default=1.0,
                    help="the project tolerance from 2026-08-19")
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, raw = load_policy(a.policy)
    print(f"policy {os.path.basename(a.policy)} v{raw['version']}  "
          f"frozen={raw.get('frozen')}  supersedes={raw.get('supersedes')}")
    print(f"  mapping: {dec}")

    rows = read_rows(a.pairs)
    got = section_v2(rows, dec)

    if a.legacy and os.path.exists(a.legacy):
        section_legacy(rows, read_rows(a.legacy), dec)

    tols = [float(x) for x in a.tolerances.split(",")]
    if a.audit:
        section_timing(a.audit, tols, a.old_tolerance, a.new_tolerance)

    print(f"\n{'=' * 72}\n4. what this does to the supervision counts"
          f"\n{'=' * 72}")
    print(f"  boundary_v1_heads.yaml gives morphology 383 trainable events. "
          f"That head answers\n  'what kind of change is this', which v2 "
          f"retired as the boundary-existence target.\n  Under v2 the "
          f"existence question has {got['BOUNDARY'] + got['NO_BOUNDARY']} "
          f"decided events "
          f"({got['BOUNDARY']} positive, {got['NO_BOUNDARY']} negative).")
    if 'summary' in dir():
        pass
    print(f"\n  The drop is not attrition. The two counts answer different "
          f"questions, and the\n  larger one was never counting answers to "
          f"this one.")

    if a.out:
        json.dump({"policy": os.path.basename(a.policy),
                   "decisions": dict(got), "n": len(rows)},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
