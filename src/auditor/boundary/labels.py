"""Boundary v1 label layer: morphology, candidate relation, offset -- and masks.

The old target was `sharp_visible_transition` versus everything else, which
made one head carry three unrelated jobs: seeing what happened, knowing this
dataset's annotation conventions, and deciding what to do about it. Every
failure of the last months traces back to that. `annotation_convention` events
were supervised as negatives although nothing is visually absent there;
`gradual` was supervised as a negative although a real change occurs, just not
at an instant; and an event whose transition is real but sits 1.2 s from the
candidate was supervised as a positive, so the head was taught that a
mistimed candidate is correct.

This file produces the learned layer's targets only. What those targets MEAN
for a dataset is the ontology's job and lives in a rule file; what to DO about
them is the policy's job. Nothing here decides an action.

MORPHOLOGY -- does a transition exist, and is it a point or an interval?

    sharp_visible_transition      POINT_TRANSITION
    gradual_phase_transition      INTERVAL_TRANSITION
    same_action_internal_motion   NO_TRANSITION
    visibility_or_offscreen       UNOBSERVABLE
    camera_or_viewpoint_shift     MASKED, and fed to the nuisance head
    annotation_convention         MASKED
    ambiguous                     MASKED

The two masks are the point of the reformulation and not an omission.
`annotation_convention` means the split exists by labelling rule; the pixels
do not encode it, and training a perception head to reproduce it is asking the
model to guess a convention. `camera_or_viewpoint_shift` is masked because a
camera can move WHILE the interaction changes -- forcing NO_TRANSITION there
teaches that a moving camera means nothing happened. Neither is
"unknown, so negative": a masked event contributes no morphology gradient, and
the policy still refuses to act on it automatically.

CANDIDATE RELATION -- separate from morphology, because the two questions have
different answers on the same event. A candidate at 456.0 whose corrected
boundary is 455.0 is POINT_TRANSITION and LATE: the boundary exists and the
candidate is wrong. Collapsing them is what made 25 of the 108 timing-decidable
sharp events unsafe to admit while the label said positive.

    EXACT       within tolerance of the corrected boundary
    EARLY       the corrected boundary is later than the candidate
    LATE        the corrected boundary is earlier
    DUPLICATE   another audited candidate already sits on this same boundary
    NO_VALID    the auditor recorded no valid boundary here
    UNDECIDABLE no timing record, or the timing itself was unresolved

DUPLICATE IS DERIVED FROM THE CANDIDATE SET, NOT FROM THE EVENT ID. The
`_duplicate_` tag in an event id is the generator bucket the candidate was
drawn from, not a human verdict -- `recording_000102_duplicate_t164.5` was
audited as a valid boundary corrected to 164.25. Reading the tag as truth was
already caught once in this project and would be caught again here. It is
computed by grouping audited candidates onto their corrected boundaries, so it
UNDERSTATES: the audit set is a sample, and two candidates that both survive in
deployment may both be absent here.

OFFSET is corrected minus candidate, supervised only where a valid point
boundary and a corrected time both exist. Positive means the true boundary is
later than the candidate.

Usage:
    python -m src.auditor.boundary.labels \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --out data/gold/boundary_v1_labels.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

MORPHOLOGY = ["POINT_TRANSITION", "INTERVAL_TRANSITION", "NO_TRANSITION",
              "UNOBSERVABLE"]
RELATION = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID", "UNDECIDABLE"]

SUBTYPE_TO_MORPHOLOGY = {
    "sharp_visible_transition": "POINT_TRANSITION",
    "gradual_phase_transition": "INTERVAL_TRANSITION",
    "same_action_internal_motion": "NO_TRANSITION",
    "visibility_or_offscreen": "UNOBSERVABLE",
    # masked, with the reason, so a reader of the output never has to guess
    # whether an absent label is an oversight
    "camera_or_viewpoint_shift": None,
    "annotation_convention": None,
    "ambiguous": None,
}
MASK_REASON = {
    "camera_or_viewpoint_shift":
        "a camera can move while the interaction also changes; forcing "
        "NO_TRANSITION would teach that global motion means nothing happened",
    "annotation_convention":
        "the split exists by labelling rule and is not encoded in the pixels",
    "ambiguous":
        "unresolvable from the clip, so any target would be invented",
}
TOL = 0.5
# past this, the nearest recorded boundary is a different boundary and
# moving the candidate onto it is not a retime
MAX_RETIME_S = 2.0


RECORDING = re.compile(r"^(recording_\d+)")


def recording_of(eid, gold_row):
    """The recording, from the audit record when there is one and from the
    event id otherwise.

    NOT string surgery on the tail. The previous fallback took
    `eid.split("_t")[0].rsplit("_", 1)[0]`, which turns
    `recording_000001_batch3_gt_boundary_t473.0` into
    `recording_000001_batch3_gt` -- an id that exists in no cache, so all 240
    batch3 events looked like missing features when the features were there.
    Worse, it gave nearly every batch3 event its OWN pseudo-recording, so the
    recording-grouped folds were grouping nothing and two events from one
    video could sit on both sides of a split."""
    if gold_row and gold_row.get("recording_id"):
        return gold_row["recording_id"]
    m = RECORDING.match(eid)
    return m.group(1) if m else eid


def cand_time(eid):
    try:
        return float(eid.rsplit("_t", 1)[1])
    except (IndexError, ValueError):
        return None


def morphology(subtype):
    if subtype not in SUBTYPE_TO_MORPHOLOGY:
        return None, f"unknown subtype {subtype!r}"
    m = SUBTYPE_TO_MORPHOLOGY[subtype]
    return m, (None if m else MASK_REASON[subtype])


def nearest_corrected(eid, g):
    """The recorded boundary CLOSEST to the candidate, not the auditor's
    designated primary.

    16 audited events carry more than one corrected boundary, and on 7 of them
    the nearest is not the primary. `recording_000213_false_gap_t242.5` records
    [240.15, 242.5]: the candidate sits exactly on the second one, and reading
    the primary makes it LATE by 2.35 s. The relation head is asked whether
    THIS candidate is on a boundary, so it must be compared with the boundary
    it would be on."""
    t = cand_time(eid)
    times = [float(x) for x in (g.get("corrected_boundary_times_json") or [])]
    p = g.get("primary_corrected_boundary_time")
    if p is not None:
        times.append(float(p))
    if t is None or not times:
        return None
    return min(times, key=lambda x: abs(x - t))


def relation(eid, g, dup_ids, tol=TOL, max_retime_s=MAX_RETIME_S):
    """(class, offset_or_None, why). offset is corrected minus candidate.

    THE AUDITOR'S VERDICT ON THE CANDIDATE COMES FIRST. Timing is only asked
    about candidates the auditor called valid: an event marked spurious with a
    real boundary two seconds away is not an early candidate for that boundary,
    it is a spurious peak that happens to be near one, and two such events were
    coming out as NO_TRANSITION plus EARLY -- a combination that cannot be true.

    EARLY and LATE MEAN RETIMEABLE. Past max_retime_s the nearest recorded
    boundary is a different boundary and moving the candidate onto it is not a
    correction. `_000224_missed_signal_present_not_top_t479.0` records its
    boundary at 253.0 -- calling that LATE by 226 s would have put a 226-second
    target into the offset regression."""
    if g is None:
        return "UNDECIDABLE", None, "not in the audit gold"
    if g.get("boundary_time_unresolved"):
        return "UNDECIDABLE", None, "boundary_time_unresolved"
    if (g.get("no_valid_boundary")
            or g.get("temporal_truth") == "spurious"
            or g.get("candidate_boundary_validity") == "invalid"):
        return "NO_VALID", None, None
    if g.get("temporal_truth") != "valid":
        return "UNDECIDABLE", None, f"temporal_truth={g.get('temporal_truth')}"
    t, c = cand_time(eid), nearest_corrected(eid, g)
    if t is None or c is None:
        return "UNDECIDABLE", None, "no candidate or corrected time"
    off = c - t
    s, e = g.get("boundary_interval_start"), g.get("boundary_interval_end")
    inside = (s is not None and e is not None
              and float(s) - tol <= t <= float(e) + tol)
    if abs(off) <= tol or inside:
        return ("DUPLICATE" if eid in dup_ids else "EXACT"), off, None
    if abs(off) > max_retime_s:
        return "UNDECIDABLE", None, (
            f"the nearest recorded boundary is {abs(off):.1f}s away, beyond "
            f"{max_retime_s}s, so it is a different boundary rather than a "
            f"retime of this candidate")
    return ("EARLY" if off > 0 else "LATE"), off, None


def find_duplicates(gold, subtypes, tol=TOL):
    """Candidates that land on a boundary another candidate already occupies.

    Grouped by recording and corrected boundary; the candidate nearest the
    corrected time keeps EXACT and the rest are duplicates of it. Only
    candidates that are themselves near a valid boundary can be duplicates --
    a spurious peak is not a duplicate of anything."""
    by = defaultdict(list)
    for eid, g in gold.items():
        if g.get("no_valid_boundary") or g.get("boundary_time_unresolved"):
            continue
        c, t = nearest_corrected(eid, g), cand_time(eid)
        if c is None or t is None or abs(c - t) > tol:
            continue
        by[g["recording_id"]].append((c, t, eid))
    dup = set()
    for rid, items in by.items():
        items.sort()
        used = []
        for c, t, eid in items:
            hit = next((u for u in used if abs(u[0] - c) <= tol), None)
            if hit is None:
                used.append((c, t, eid))
            else:
                # whichever candidate sits closer to the corrected time is the
                # one the dataset should keep; the other is the duplicate
                if abs(t - c) < abs(hit[1] - hit[0]):
                    dup.add(hit[2])
                    used[used.index(hit)] = (c, t, eid)
                else:
                    dup.add(eid)
    return dup


def build(pair_label_paths, gold_paths, tol=TOL,
          max_retime_s=MAX_RETIME_S):
    from src.boundary.pair_taxonomy import load_pair_labels
    subtypes = {}
    for p in pair_label_paths:
        for e, v in load_pair_labels(p).items():
            subtypes[e] = v["temporal_pair_subtype"]
    gold = {}
    for p in gold_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r
    dup = find_duplicates(gold, subtypes, tol)

    rows = []
    for eid, sub in sorted(subtypes.items()):
        g = gold.get(eid)
        m, why_m = morphology(sub)
        rel, off, why_r = relation(eid, g, dup, tol, max_retime_s)
        rows.append({
            "event_id": eid,
            "recording_id": recording_of(eid, g),
            "subtype": sub,
            "candidate_time": cand_time(eid),
            "morphology": m,
            "morphology_masked": m is None,
            "morphology_mask_reason": why_m,
            "candidate_relation": rel,
            "relation_masked": rel == "UNDECIDABLE",
            "relation_mask_reason": why_r,
            # offset is supervised only where a point boundary really exists;
            # regressing toward a boundary that is not there is not a target
            "offset_s": off if (m == "POINT_TRANSITION"
                                and rel in ("EXACT", "EARLY", "LATE")) else None,
            "nuisance_camera": sub == "camera_or_viewpoint_shift",
            "audited": g is not None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max_retime_s", type=float, default=MAX_RETIME_S,
                    help="beyond this the nearest recorded boundary is treated "
                         "as a different boundary, not a retime target")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = build(a.pair_labels, a.gold, a.tol, a.max_retime_s)
    print(f"{len(rows)} labelled events, "
          f"{sum(1 for r in rows if r['audited'])} of them audited")

    print(f"\n{'=' * 78}\nMORPHOLOGY\n{'=' * 78}")
    c = Counter(r["morphology"] or "MASKED" for r in rows)
    for k in MORPHOLOGY + ["MASKED"]:
        print(f"  {k:<22} {c.get(k, 0):>4}")
    print("\n  masked, by reason:")
    for sub, n in Counter(r["subtype"] for r in rows
                          if r["morphology_masked"]).most_common():
        print(f"    {n:>4}  {sub:<28} {MASK_REASON.get(sub, '')[:60]}")
    trainable = [r for r in rows if not r["morphology_masked"]]
    print(f"\n  trainable for morphology: {len(trainable)} over "
          f"{len({r['recording_id'] for r in trainable})} recordings")

    print(f"\n{'=' * 78}\nCANDIDATE RELATION\n{'=' * 78}")
    c = Counter(r["candidate_relation"] for r in rows)
    for k in RELATION:
        print(f"  {k:<22} {c.get(k, 0):>4}")
    dec = [r for r in rows if not r["relation_masked"]]
    print(f"\n  trainable for relation: {len(dec)} over "
          f"{len({r['recording_id'] for r in dec})} recordings")
    print(f"  why the rest are undecidable: "
          f"{dict(Counter(r['relation_mask_reason'] for r in rows if r['relation_masked']))}")

    print(f"\n{'=' * 78}\nMORPHOLOGY x RELATION  (the pair the old target "
          f"collapsed)\n{'=' * 78}")
    print(f"  {'':<22}" + "".join(f"{k:>12}" for k in RELATION))
    for m in MORPHOLOGY + ["MASKED"]:
        g = [r for r in rows if (r["morphology"] or "MASKED") == m]
        cc = Counter(r["candidate_relation"] for r in g)
        print(f"  {m:<22}" + "".join(f"{cc.get(k, 0):>12}" for k in RELATION))
    pe = sum(1 for r in rows if r["morphology"] == "POINT_TRANSITION"
             and r["candidate_relation"] == "EXACT")
    pw = sum(1 for r in rows if r["morphology"] == "POINT_TRANSITION"
             and r["candidate_relation"] in ("EARLY", "LATE", "DUPLICATE"))
    print(f"\n  POINT_TRANSITION with a wrong candidate: {pw}, against {pe} "
          f"exact. Under the old target all {pe + pw} were positives,\n  which "
          f"is what taught the head that a mistimed candidate is correct.")

    off = [r["offset_s"] for r in rows if r["offset_s"] is not None]
    if off:
        import statistics
        print(f"\n{'=' * 78}\nOFFSET (supervised on {len(off)} events)"
              f"\n{'=' * 78}")
        print(f"  mean {statistics.fmean(off):+.3f}s   median "
              f"{statistics.median(off):+.3f}s   "
              f"|max| {max(abs(x) for x in off):.2f}s")
        print(f"  within +/-0.5s {sum(1 for x in off if abs(x) <= 0.5)}   "
              f"0.5-2s {sum(1 for x in off if 0.5 < abs(x) <= 2)}   "
              f">2s {sum(1 for x in off if abs(x) > 2)}")

    if a.out:
        json.dump({"tol": a.tol, "morphology_classes": MORPHOLOGY,
                   "relation_classes": RELATION, "events": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
