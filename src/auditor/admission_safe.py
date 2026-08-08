"""Is `subtype == sharp` the same thing as "this timestamp may enter the
dataset unreviewed"? Answer it from the audit gold before touching the gate.

Every teacher number so far scores against `arm`, which is defined as

    true_keep   temporal_pair_subtype == sharp_visible_transition
    false_keep  anything else

and that is a claim about the PAIR, not about the CANDIDATE. A recording can
contain a real sharp transition at 233.25 s while the candidate sits at 233.0
and still be labelled sharp -- the subtype describes what happens across the
boundary, not whether this particular timestamp is the boundary. If a teacher
routes that to a human it is behaving correctly and the current scoring calls
it a false negative.

So the target is rebuilt from the fields the human auditor actually recorded:

    temporal_truth              valid
    candidate_boundary_validity valid
    no_valid_boundary           False
    boundary_time_unresolved    False
    timing                      the candidate lies within TOL of
                                primary_corrected_boundary_time, or inside
                                [boundary_interval_start, boundary_interval_end]
    subtype                     sharp_visible_transition

WHAT THIS IS NOT. Not an independent check. Every one of those fields comes
from the same human pass that produced the subtype, so agreement between them
is not corroboration -- it is the same judgement read out along a different
axis. This changes WHICH question the score answers; it does not make the
answer more reliable.

UNKNOWN IS ITS OWN CATEGORY. Events outside the audit gold (batch3 carries
subtype and nothing else) have no timing record, so their safety cannot be
decided. Counting them either way would invent the very number this file
exists to measure. They are reported separately -- except where the subtype
alone settles it, since a gradual or annotation-convention event is unsafe to
admit whatever its timing.

`multiple_valid` is NOT treated as unsafe. It means the region admits more
than one defensible boundary, not that this one is wrong, and the duplicate
question -- would admitting this create a second boundary for one event -- is
about the candidate set, not about this candidate. It is printed in its own
column so it can be seen rather than assumed.

THE SECOND TABLE IS THE ONE THAT DECIDES A GATE. For every eligibility check
it counts failures among admission-safe candidates (the cost) against failures
among unsafe ones (the benefit), and then removes each check in turn. A check
that fails on both sides at the same rate is buying nothing, and no threshold
on it will help; a check that fails almost only on unsafe candidates is worth
keeping even if it is strict.

Usage:
    python -m src.auditor.admission_safe \
        --review /workspace/tr1/results/hal/c3/teacher_observe_only.json \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

SHARP = "sharp_visible_transition"
TOL = 0.5


def load_gold(paths):
    g = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    g[r["event_id"]] = r
    return g


def cand_time(eid, gold_row):
    """The candidate's own timestamp, from the event id.

    NOT gt_time. gt_time is the GT boundary the candidate was drawn near, and
    for the model-derived sources (false_mid_segment, false_near_edge,
    false_gap, duplicate) it deliberately differs from the candidate -- that
    offset is what makes them negatives. Comparing the candidate against
    gt_time would score every one of those as mistimed by construction.

    Across the 188 audited events the id's t matches pred_time for the
    model-derived sources and gt_time for the GT-derived ones (early, late,
    missed_*), and matches neither for none of them -- so a row that matches
    neither is a genuine inconsistency and is reported rather than absorbed."""
    t = None
    if "_t" in eid:
        try:
            t = float(eid.rsplit("_t", 1)[1])
        except ValueError:
            t = None
    if t is None:
        return None, "no timestamp in the event id"
    g, p = (gold_row or {}).get("gt_time"), (gold_row or {}).get("pred_time")
    hit = [v for v in (g, p) if v is not None and abs(float(v) - t) < 1e-6]
    if gold_row is not None and not hit:
        return t, (f"id t={t} matches neither gt_time={g} nor pred_time={p}")
    return t, None


def admission_safe(eid, subtype, gold_row, tol=TOL):
    """(verdict, reasons). verdict is True, False, or None for undecidable.

    None is not a failure mode to be tidied away: an event with no timing
    record has not been shown to be safe OR unsafe, and folding it into either
    count would fabricate the quantity being measured."""
    reasons = []
    if subtype and subtype != SHARP:
        # settled without timing: no amount of correct localisation makes a
        # gradual or annotation-convention pair safe to admit unreviewed
        return False, [f"subtype={subtype}"]
    if gold_row is None:
        return None, ["not in the audit gold, so timing is unrecorded"]
    if subtype is None:
        reasons.append("subtype unknown")

    if gold_row.get("temporal_truth") != "valid":
        reasons.append(f"temporal_truth={gold_row.get('temporal_truth')}")
    if gold_row.get("candidate_boundary_validity") != "valid":
        reasons.append(
            f"candidate_boundary_validity={gold_row.get('candidate_boundary_validity')}")
    if gold_row.get("no_valid_boundary"):
        reasons.append("no_valid_boundary")
    if gold_row.get("boundary_time_unresolved"):
        reasons.append("boundary_time_unresolved")

    t, warn = cand_time(eid, gold_row)
    if warn:
        reasons.append(warn)
    corr = gold_row.get("primary_corrected_boundary_time")
    s, e = (gold_row.get("boundary_interval_start"),
            gold_row.get("boundary_interval_end"))
    if t is None or corr is None:
        reasons.append("no corrected boundary time to compare against")
    else:
        d = abs(float(corr) - t)
        inside = (s is not None and e is not None
                  and float(s) - tol <= t <= float(e) + tol)
        if d > tol and not inside:
            reasons.append(f"candidate is {d:.2f}s from the corrected boundary "
                           f"{corr} (tolerance {tol})")
    return (not reasons), reasons


def is_v2(r):
    return "review" in r or "eligible" in r


def rev(r):
    return (r.get("review") if is_v2(r) else r.get("blind")) or {}


def checks(b, rule, observe_only):
    out = [("evidence_sufficient", bool(b.get("evidence_sufficient")))]
    if rule.get("require_decision") is not None and not observe_only:
        out.append((f"decision == {rule['require_decision']}",
                    b.get("decision") == rule["require_decision"]))
    for k in rule.get("require_yes", []):
        out.append((f"{k} == yes", b.get(k) == "yes"))
    for k in rule.get("forbid_no", []):
        out.append((f"{k} != no", b.get(k) != "no"))
    for k in rule.get("forbid_yes", []):
        out.append((f"{k} != yes", b.get(k) != "yes"))
    for k in rule.get("forbid_insufficient", []):
        out.append((f"{k} != insufficient", b.get(k) != "insufficient"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--review", help="a teacher review json; without it this "
                                     "only reports the gold itself")
    ap.add_argument("--config", help="eligibility rule; defaults to the path "
                                     "recorded in the review file")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.boundary.pair_taxonomy import load_pair_labels
    labels = {}
    for p in a.pair_labels:
        for e, v in load_pair_labels(p).items():
            labels[e] = v["temporal_pair_subtype"]
    gold = load_gold(a.gold)
    print(f"{len(gold)} audited events, {len(labels)} subtype labels")

    # ---------------------------------------------------- the gold itself
    rowsg = []
    for eid, sub in labels.items():
        ok, why = admission_safe(eid, sub, gold.get(eid), a.tol)
        rowsg.append((eid, sub, ok, why))
    c = Counter(str(r[2]) for r in rowsg)
    n_sharp = sum(1 for r in rowsg if r[1] == SHARP)
    sharp_unsafe = [r for r in rowsg if r[1] == SHARP and r[2] is False]
    sharp_unk = [r for r in rowsg if r[1] == SHARP and r[2] is None]
    print(f"\n{'=' * 74}\nADMISSION-SAFE OVER EVERY LABELLED EVENT\n{'=' * 74}")
    print(f"  safe {c.get('True', 0)}   unsafe {c.get('False', 0)}   "
          f"undecidable {c.get('None', 0)}")
    print(f"  of {n_sharp} sharp: {n_sharp - len(sharp_unsafe) - len(sharp_unk)} "
          f"safe, {len(sharp_unsafe)} unsafe, {len(sharp_unk)} undecidable")
    if sharp_unsafe:
        print("  sharp but NOT admission-safe -- these are the events the "
              "current arm scores as false negatives when a teacher blocks them:")
        for eid, _, _, why in sharp_unsafe[:20]:
            print(f"    {eid[-44:]:<45} {'; '.join(why)}")
    if sharp_unk:
        print(f"  sharp with no timing record: {len(sharp_unk)} "
              f"(e.g. {sharp_unk[0][0]})")

    if not a.review:
        return

    # ------------------------------------------------- against a review file
    blob = json.load(open(a.review, encoding="utf-8"))
    res = blob["results"]
    observe_only = bool(blob.get("observe_only")) or all(
        rev(r).get("decision") is None for r in res)
    cfgp = a.config or blob.get("config")
    rule = (json.load(open(cfgp, encoding="utf-8")).get("eligibility")
            if cfgp and os.path.exists(cfgp) else None)
    if rule is None:
        raise SystemExit("no eligibility config found; the per-check table "
                         "cannot be recomputed without it")
    print(f"\n{os.path.basename(a.review)}: {len(res)} events"
          + ("   (observe-only: the decision check is not applied, because the "
             "run never asked for one)" if observe_only else ""))

    for r in res:
        r["_safe"], r["_why"] = admission_safe(
            r["event_id"], r.get("subtype"), gold.get(r["event_id"]), a.tol)

    print(f"\n{'=' * 74}\nARM vs ADMISSION-SAFE\n{'=' * 74}")
    print(f"  {'arm':<12} {'safe':>6} {'unsafe':>7} {'undecidable':>12}")
    for arm in sorted({r.get("arm") for r in res}):
        g = [r for r in res if r.get("arm") == arm]
        print(f"  {str(arm):<12} {sum(1 for r in g if r['_safe'] is True):>6} "
              f"{sum(1 for r in g if r['_safe'] is False):>7} "
              f"{sum(1 for r in g if r['_safe'] is None):>12}")
    mis = [r for r in res if r.get("arm") == "true_keep" and r["_safe"] is not True]
    if mis:
        print("\n  scored as a true keep, not admission-safe:")
        for r in mis:
            print(f"    {r['event_id'][-44:]:<45} {'; '.join(r['_why'])}")
    else:
        print("\n  every true keep in this file is also admission-safe, so the "
              "retention figure is measuring what it claims to.")
    mv = [r for r in res
          if (gold.get(r["event_id"]) or {}).get("gt_boundary_relation")
          == "multiple_valid"]
    print(f"  gt_boundary_relation == multiple_valid: {len(mv)}"
          + (f"  ({', '.join(r['event_id'][-28:] for r in mv)})" if mv else ""))

    # ---------------------------------------------- per-check discrimination
    dec = [r for r in res if r["_safe"] is not None]
    safe = [r for r in dec if r["_safe"]]
    unsafe = [r for r in dec if not r["_safe"]]
    print(f"\n{'=' * 96}\nPER-CHECK DISCRIMINATION   "
          f"({len(safe)} admission-safe, {len(unsafe)} unsafe, "
          f"{len(res) - len(dec)} undecidable and excluded)\n{'=' * 96}")
    print(f"  {'check':<46} {'fails on safe':>14} {'fails on unsafe':>16} "
          f"{'ratio':>7}")
    names = [n for n, _ in checks(rev(res[0]), rule, observe_only)]
    per = {}
    for n in names:
        fs = sum(1 for r in safe
                 if not dict(checks(rev(r), rule, observe_only)).get(n, True))
        fu = sum(1 for r in unsafe
                 if not dict(checks(rev(r), rule, observe_only)).get(n, True))
        per[n] = (fs, fu)
        rs = fs / len(safe) if safe else float("nan")
        ru = fu / len(unsafe) if unsafe else float("nan")
        print(f"  {n:<46} {fs:>6} / {len(safe):<5} {fu:>7} / {len(unsafe):<6} "
              f"{(ru / rs if rs else float('inf')):>7.2f}")
    print("  ratio = (fail rate on unsafe) / (fail rate on safe). At 1.0 the "
          "check fails equally often on both and is buying nothing;\n  below "
          "1.0 it costs more real boundaries than it catches wrong ones.")

    # ------------------------------------------------- leave one check out
    def passes(r, drop=()):
        return all(ok for n, ok in checks(rev(r), rule, observe_only)
                   if n not in drop)

    print(f"\n{'=' * 96}\nLEAVE ONE CHECK OUT\n{'=' * 96}")
    print(f"  {'removed':<46} {'admitted safe':>14} {'admitted unsafe':>16} "
          f"{'precision':>10}")

    def line(tag, drop):
        ts = sum(1 for r in safe if passes(r, drop))
        tu = sum(1 for r in unsafe if passes(r, drop))
        p = ts / (ts + tu) if (ts + tu) else float("nan")
        print(f"  {tag:<46} {ts:>8} / {len(safe):<4} {tu:>9} / {len(unsafe):<5} "
              f"{p:>10.3f}")
        return ts, tu

    line("(nothing -- the rule as it stands)", ())
    for n in names:
        line(n, (n,))
    print("  A row that leaves 'admitted unsafe' unchanged while raising "
          "'admitted safe' is a check to delete: it was rejecting real\n"
          "  boundaries and catching nothing. A row that raises both is a "
          "genuine trade and needs the precision column to decide.")

    if a.out:
        json.dump({"tol": a.tol, "observe_only": observe_only,
                   "per_check": per,
                   "events": [{"event_id": r["event_id"], "arm": r.get("arm"),
                               "subtype": r.get("subtype"),
                               "admission_safe": r["_safe"], "why": r["_why"]}
                              for r in res]},
                  open(a.out, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
