"""How many DISTINCT RECORDINGS can supply a within-recording contrast.

The unit is the recording, not the pair, and that is the whole point of this
module. recording_000157 alone contributed 20829 of the 81416 pairs in the
evaluation and sits at chance; two recordings hold 48% of the pair mass. So a
Cartesian pair count says almost nothing about how much independent evidence
is available, while the number of recordings that can pose the question at all
says most of it.

WHAT COUNTS AS A CONTRAST. A recording supplies one when it holds both a
candidate that IS a boundary and a candidate that is NOT, because the failure
being attacked is a real boundary ranked below an internal motion in the SAME
video. Positives and negatives sitting in different recordings train the
separation that recognising the kitchen already solves.

TWO EXCLUSIONS THAT MATTER MORE THAN THE HEADLINE NUMBER.

`same_action_new_instance` is NOT a negative. It is the repeated-instance gap:
the same action performed twice in a row, where the stored ground truth cuts
and the subtype audit produced five different answers across annotators. 56%
of the disputed events sit on exactly this configuration. Training a
discriminator on it teaches a distinction the annotation itself has not
settled, and the model would learn the annotator's coin flip.

`within_1s_tolerance == no` is NOT a positive. A boundary exists near that
candidate but the candidate is more than a second away, and under the frozen
1.0s tolerance that is a miss, not a hit. It is also not internal motion. It
is genuinely neither, so it is counted and set aside rather than pushed into
whichever class needs the volume.

Both exclusions cost pairs. That is the intended direction: a contrast whose
label is contested is worse than no contrast, because it is indistinguishable
from a good one at training time and only shows up as a ceiling later.

Usage:
    python -m src.auditor.boundary.contrast_census \
        --audit data/gold/batch4_joint_audit.csv \
        --emit_pairs results/auditor/batch4_contrast_pairs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

# --- the class definitions, in one place so they can be argued with --------
POSITIVE_EVENT = {"task_boundary"}
NEGATIVE_EVENT = {"no_boundary"}
# reported on their own: recording-edge events are not inter-episode
# disengagements, and the ontology's boundary is defined between episodes.
EDGE_EVENT = {"initial_action_start", "terminal_action_end",
              "visibility_endpoint"}
CLEAN_NEGATIVE_RELATION = {"same_instance"}
CONTESTED_RELATION = {"same_action_new_instance"}
NEAR_GAPS = (60.0, 30.0, 10.0)


POLICIES = (
    ("strict (shipped)", {}),
    ("+ mislocalised as positive", {"loose_tolerance": True}),
    ("+ repeated instance as negative", {"contested_negative": True}),
    ("+ no-action as negative", {"no_action_negative": True}),
    ("everything admitted", {"loose_tolerance": True,
                             "contested_negative": True,
                             "no_action_negative": True}),
)


def _clean(row):
    return {k.lstrip("﻿").strip(): (v or "").strip()
            for k, v in row.items()}


def classify(r, loose_tolerance=False, contested_negative=False,
             no_action_negative=False):
    """-> ('pos' | 'neg' | 'edge' | 'excluded', reason).

    The keyword arguments exist only so `--sensitivity` can price each
    exclusion. They are not an interface for choosing a looser policy at
    training time: the shipped defaults are the strict ones, and a relaxation
    has to be argued for and written down, not passed as a flag."""
    ev = r.get("temporal_event_type", "")
    tol = r.get("within_1s_tolerance", "")
    rel = r.get("interaction_relation", "")
    if ev in POSITIVE_EVENT:
        if tol == "yes" or loose_tolerance:
            return "pos", "task_boundary within 1s"
        return "excluded", f"task_boundary but within_1s_tolerance={tol!r}"
    if ev in NEGATIVE_EVENT:
        if rel in CONTESTED_RELATION:
            if contested_negative:
                return "neg", "no_boundary, same_action_new_instance"
            return "excluded", "no_boundary but same_action_new_instance"
        if rel in CLEAN_NEGATIVE_RELATION:
            return "neg", "no_boundary, same_instance"
        if rel == "not_applicable_no_action":
            if no_action_negative:
                return "neg", "no_boundary, no action at all"
            return "excluded", "no_boundary, relation='not_applicable_no_action'"
        return "excluded", f"no_boundary, relation={rel!r}"
    if ev in EDGE_EVENT:
        return "edge", ev
    return "excluded", f"event={ev!r}"


def census(rows, label="", quiet=False, **policy):
    by = defaultdict(lambda: {"pos": [], "neg": [], "edge": 0, "excluded": 0})
    reasons = Counter()
    for r in rows:
        kind, why = classify(r, **policy)
        rid = r.get("recording_id", "")
        t = r.get("candidate_time_s", "")
        try:
            t = float(t)
        except ValueError:
            t = None
        reasons[f"{kind}: {why}"] += 1
        if kind in ("pos", "neg") and t is not None:
            by[rid][kind].append((t, r))
        elif kind == "edge":
            by[rid]["edge"] += 1
        else:
            by[rid]["excluded"] += 1

    if not quiet:
        print("=" * 78)
        print(f"HOW EVERY ROW WAS CLASSIFIED{'  -- ' + label if label else ''}")
        print("=" * 78)
        for k, n in reasons.most_common():
            print(f"  {n:>5}  {k}")

    rec = []
    for rid, d in sorted(by.items()):
        p = sorted(x[0] for x in d["pos"])
        n = sorted(x[0] for x in d["neg"])
        row = {"recording_id": rid, "n_pos": len(p), "n_neg": len(n),
               "has_both": bool(p and n), "n_edge": d["edge"],
               "n_excluded": d["excluded"],
               "pairs_all": len(p) * len(n)}
        for g in NEAR_GAPS:
            row[f"pairs_{int(g)}s"] = sum(1 for a in p for b in n
                                          if abs(a - b) <= g)
        rec.append(row)
    return rec, by


def report(rec, label):
    both = [r for r in rec if r["has_both"]]
    print(f"\n{'=' * 78}\nPER RECORDING -- {label}\n{'=' * 78}")
    print(f"  {'recording':<24}{'pos':>5}{'neg':>5}{'all':>7}{'<=60s':>7}"
          f"{'<=30s':>7}{'<=10s':>7}")
    for r in sorted(rec, key=lambda x: -x["pairs_all"]):
        if not r["has_both"]:
            continue
        print(f"  {r['recording_id'][:24]:<24}{r['n_pos']:>5}{r['n_neg']:>5}"
              f"{r['pairs_all']:>7}{r['pairs_60s']:>7}{r['pairs_30s']:>7}"
              f"{r['pairs_10s']:>7}")

    print(f"\n  THE NUMBER THAT DECIDES, and the ones that do not:")
    print(f"    recordings with BOTH classes          {len(both):>6}"
          f"   <- the gate is on this")
    for g in NEAR_GAPS:
        k = sum(1 for r in both if r[f"pairs_{int(g)}s"] > 0)
        print(f"    ... of which some pair is <={int(g):>3}s apart "
              f"{k:>6}")
    print(f"    recordings seen at all                {len(rec):>6}")
    print(f"    positives / negatives kept            "
          f"{sum(r['n_pos'] for r in rec):>6} / "
          f"{sum(r['n_neg'] for r in rec)}")
    print(f"    total pairs (NOT the sample size)     "
          f"{sum(r['pairs_all'] for r in rec):>6}")
    if both:
        top = max(both, key=lambda r: r["pairs_all"])
        tot = sum(r["pairs_all"] for r in both)
        print(f"    largest single recording's share      "
              f"{top['pairs_all'] / max(tot, 1):>6.1%}   ({top['recording_id']})")
    print(f"\n  A pair count is a Cartesian product and grows quadratically "
          f"inside one\n  recording; the evidence does not. Read the first "
          f"line.")
    return both


def emit(by, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rid, d in sorted(by.items()):
            if not d["pos"] or not d["neg"]:
                continue
            for tp, rp in d["pos"]:
                for tn, rn in d["neg"]:
                    f.write(json.dumps({
                        "recording_id": rid,
                        "pos_time_s": tp, "neg_time_s": tn,
                        "gap_s": abs(tp - tn),
                        "pos_key": rp.get("candidate_key", ""),
                        "neg_key": rn.get("candidate_key", ""),
                        "neg_relation": rn.get("interaction_relation", ""),
                        "pos_timing": rp.get(
                            "boundary_timing_judgment_at_1s", ""),
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"\nwrote {n} pairs to {path}")
    print(f"  These pairs are WITHIN recording by construction. Any training "
          f"that\n  batches them must also SPLIT by recording, or the "
          f"validation set will\n  share a kitchen with the training set and "
          f"report the shortcut back.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="append", required=True,
                    help="joint audit CSV; APPEND one flag per file")
    ap.add_argument("--eval_candidates",
                    help="the frozen evaluation pool. Every recording it "
                         "contains is REMOVED from the census, because a "
                         "development set that overlaps the only unfitted "
                         "test set stops being a test set quietly.")
    ap.add_argument("--emit_pairs")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for p in a.audit:
        with open(p, encoding="utf-8-sig") as f:
            got = [_clean(r) for r in csv.DictReader(f)]
        print(f"{len(got)} rows from {p}")
        rows += got

    if a.eval_candidates:
        ev = {json.loads(l)["recording_id"]
              for l in open(a.eval_candidates, encoding="utf-8") if l.strip()}
        before = len({r.get("recording_id", "") for r in rows})
        rows = [r for r in rows if r.get("recording_id") not in ev]
        after = len({r.get("recording_id", "") for r in rows})
        print(f"  {before - after} of {before} recordings dropped for "
              f"overlapping the frozen evaluation pool ({len(ev)} recordings)")

    rec, by = census(rows)
    both = report(rec, "candidate contrast availability")

    print(f"\n{'=' * 78}\nWHAT EACH EXCLUSION COSTS\n{'=' * 78}")
    print(f"  {'policy':<34}{'recs both':>11}{'<=60s':>8}{'pairs':>8}"
          f"{'pos':>6}{'neg':>6}")
    sens = {}
    for name, pol in POLICIES:
        r2, _ = census(rows, quiet=True, **pol)
        b2 = [x for x in r2 if x["has_both"]]
        sens[name] = {"recordings_with_both": len(b2),
                      "recordings_with_60s_pair":
                          sum(1 for x in b2 if x["pairs_60s"] > 0),
                      "pairs": sum(x["pairs_all"] for x in r2),
                      "n_pos": sum(x["n_pos"] for x in r2),
                      "n_neg": sum(x["n_neg"] for x in r2), "policy": pol}
        s = sens[name]
        print(f"  {name:<34}{s['recordings_with_both']:>11}"
              f"{s['recordings_with_60s_pair']:>8}{s['pairs']:>8}"
              f"{s['n_pos']:>6}{s['n_neg']:>6}")
    print(f"\n  This table prices the exclusions; it does not license them. "
          f"Admitting\n  `same_action_new_instance` buys recordings by "
          f"training on the one\n  configuration the annotators disagreed "
          f"about five ways, and admitting\n  mislocalised positives teaches "
          f"the model that a candidate more than a\n  second from the "
          f"boundary is the boundary -- which is the tolerance the\n  whole "
          f"evaluation is defined by. `no-action` negatives are honest but "
          f"EASY:\n  they are not the internal-motion confusion the failure "
          f"is made of, so\n  they raise a score without touching it.")

    if a.emit_pairs:
        emit(by, a.emit_pairs)
    if a.out:
        json.dump({"per_recording": rec,
                   "n_recordings_with_both": len(both),
                   "sensitivity": sens,
                   "definitions": {
                       "positive": sorted(POSITIVE_EVENT),
                       "positive_requires": "within_1s_tolerance == yes",
                       "negative": sorted(NEGATIVE_EVENT),
                       "negative_requires_relation":
                           sorted(CLEAN_NEGATIVE_RELATION),
                       "excluded_contested": sorted(CONTESTED_RELATION),
                       "edge_events": sorted(EDGE_EVENT)}},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
