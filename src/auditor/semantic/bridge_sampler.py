"""Which recordings are one audit away from a usable within-recording contrast.

Two arms are currently unidentifiable, and neither is a modelling problem:

    semantic   765 YES/NO pairs, 6 of them within a recording (0.8%). 32
               recordings, 1 carrying both classes. An AUROC on that structure
               cannot separate "predicts the label" from "recognises the
               kitchen".
    span       reorder_span's 0.730 splits 0.882 / 0.704 by whether the
               internal boundary was human-confirmed, but only 11 recordings
               hold both kinds and the intervals overlap.

Both need the SAME thing: a recording that contains both sides of the
comparison. So the sampling unit is the RECORDING, and the quantity to
maximise is not events audited but **recordings that gain a contrast**.

This ranks recordings by how close they already are, and emits a packet of
concrete targets for each. It does not decide anything; it says where an hour
of auditing buys the most identifiability.

NO MODEL SCORE IS USED FOR SELECTION, and that is not an oversight. Choosing
the events a reranker thinks are wrong would enrich the NO class with exactly
the errors that model detects, and evaluating that model on the result would
be circular. The selection signals here are properties of the annotation
alone -- a rare verb, a label reused across many segments, a duration far from
the recording's own distribution, a multi-clause label -- and every candidate
carries the reason it was picked, so the enrichment is visible rather than
implicit.

SPANS ALREADY IN THE BENCHMARK COME FIRST. Confirming the internal join of a
span that reorder_span already scored upgrades an existing measurement with no
rescoring; confirming a new one requires a GPU pass before it is worth
anything.

Usage:
    python -m src.auditor.semantic.bridge_sampler \
        --gold data/gold/semantic_ontology_gold_48.json \
        --gold data/gold/semantic_enrichment_gold_41.csv \
        --boundary_gold data/gold/audit_72_gold_v2.jsonl \
        --boundary_gold data/gold/audit_188_gold_v2.jsonl \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --span_benchmark data/gold/reorder_arms_benchmark.jsonl \
        --claims data/gold/atomic_claims_frozen.json \
        --top 40 --out data/gold/bridge_audit_targets.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np

from src.auditor.semantic.claim_support_diagnostic import load_gold, norm_key
from src.auditor.semantic.compose_supervision import resolve
from src.auditor.semantic.paired_benchmark import SPLIT_CLAUSE
from src.auditor.semantic.render_ontology_clips import get_segments
from src.auditor.semantic.span_boundary_split import TIME_COLS


def rid_of(row):
    r = row.get("recording_id")
    if r:
        return r
    m = re.match(r"^(recording_\d+)", str(row.get("event_id")
                                          or row.get("audit_key") or ""))
    return m.group(1) if m else None


def read_rows(path):
    if path.lower().endswith(".csv"):
        return list(csv.DictReader(open(path, newline="",
                                        encoding="utf-8-sig")))
    if path.lower().endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8-sig")
                if l.strip()]
    blob = json.load(open(path, encoding="utf-8-sig"))
    return blob.get("events", blob if isinstance(blob, list)
                    else list(blob.values()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--boundary_gold", action="append", default=[])
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--span_benchmark")
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    ap.add_argument("--per_recording", type=int, default=4,
                    help="targets per packet. Three to five focused examples "
                         "in one recording beat one example in each of a "
                         "hundred")
    ap.add_argument("--tol_s", type=float, default=0.5,
                    help="how close a span's internal join must be to an "
                         "audited boundary to inherit its status")
    ap.add_argument("--time_col")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()

    # --- what each recording already has -------------------------------
    sem = defaultdict(set)
    for r in load_gold(a.gold):
        rid = rid_of(r)
        cs = (r.get("claim_support") or "").strip().lower()
        if rid and cs in ("yes", "no", "partial", "uncertain"):
            sem[rid].add(cs)

    # BOUNDARY STATE MUST BE READ AT THE SPAN JOIN, NOT AT THE RECORDING.
    # The boundary audits judged times the DETECTOR proposed; the span arm
    # needs the internal join of its own pairs, and 118 confirmed boundaries
    # matched only 25 of 385 spans. Counting at recording level said 32
    # recordings were complete while the within-recording analysis found 11,
    # and 11 is the number that can be used.
    btimes = defaultdict(lambda: {"confirmed": [], "not_confirmed": []})
    for p in a.boundary_gold:
        for r in read_rows(p):
            rid = rid_of(r)
            rel = str(r.get("gt_boundary_relation") or "").strip().lower()
            tc = a.time_col or next((c for c in TIME_COLS if r.get(c)), None)
            if not rid or not rel or not tc:
                continue
            try:
                t = float(r[tc])
            except (TypeError, ValueError):
                continue
            btimes[rid]["confirmed" if rel == "correctly_annotated"
                        else "not_confirmed"].append(t)
    print(f"       {sum(len(v['confirmed']) for v in btimes.values())} "
          f"confirmed and "
          f"{sum(len(v['not_confirmed']) for v in btimes.values())} "
          f"not-confirmed boundary judgements over {len(btimes)} recordings")

    both_sem = {r for r, v in sem.items() if {"yes", "no"} <= v}
    print(f"today: {len(sem)} recordings with semantic audits, "
          f"{len(both_sem)} carrying BOTH yes and no")

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    vcount = Counter(x["verb"] for d in claims.values()
                     for x in d["actions"] if x.get("verb"))
    rare = {v for v, c in vcount.items() if c <= 2}

    scored_spans = set()
    if a.span_benchmark:
        for l in open(a.span_benchmark, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("kind") == "reorder_span":
                    scored_spans.add((r["recording_id"],
                                      round(float(r["start"]), 3)))
        print(f"       {len(scored_spans)} spans already scored by the "
              f"reorder_span arm")

    # --- candidates per recording, from the annotation only ------------
    sem_cand, span_cand = defaultdict(list), defaultdict(list)
    label_uses = Counter()
    per_rec = {}
    for path in resolve(a.recseg):
        blob = json.load(open(path, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid or rid in per_rec:
                continue
            segs = sorted(([str(x[0]), float(x[1]), float(x[2])]
                           for x in get_segments(r)[0]), key=lambda x: x[1])
            per_rec[rid] = segs
            for lab, _s, _e in segs:
                label_uses[lab] += 1

    for rid, segs in per_rec.items():
        durs = np.array([e - s for _l, s, e in segs]) if segs else np.array([])
        med = float(np.median(durs)) if len(durs) else 0.0
        for lab, st, en in segs:
            why = []
            d = claims.get(lab)
            if d and any(x["verb"] in rare for x in d["actions"]):
                why.append("rare verb")
            if label_uses[lab] >= 5:
                why.append(f"label reused {label_uses[lab]}x")
            if med and (en - st) > 3 * med:
                why.append("duration >3x this recording's median")
            if len([q for q in SPLIT_CLAUSE.split(lab)
                    if q and q.strip()
                    and not SPLIT_CLAUSE.fullmatch(q)]) > 1:
                why.append("multi-clause")
            if why:
                sem_cand[rid].append({"start": st, "end": en, "label": lab,
                                      "why_selected": why})
        for i in range(len(segs) - 1):
            x, y = segs[i], segs[i + 1]
            if y[1] - x[2] > 2.0 or y[2] - x[1] > 30.0 or x[0] == y[0]:
                continue
            bt = btimes.get(rid, {"confirmed": [], "not_confirmed": []})
            st = ("confirmed"
                  if any(abs(x[2] - t) <= a.tol_s for t in bt["confirmed"])
                  else "not_confirmed"
                  if any(abs(x[2] - t) <= a.tol_s
                         for t in bt["not_confirmed"])
                  else "unaudited")
            span_cand[rid].append({
                "start": x[1], "end": y[2], "internal_join": x[2],
                "clause_a": x[0], "clause_b": y[0], "join_status": st,
                "already_scored": (rid, round(x[1], 3)) in scored_spans})

    # --- rank ------------------------------------------------------------
    # THE QUANTITY IS RECORDINGS THAT GAIN A CONTRAST, not events audited. A
    # recording already holding both classes gains nothing from more of them;
    # one holding neither needs two lucky outcomes; one holding exactly one
    # side needs a single audit to land the other way, which is why it ranks
    # above both.
    # The span arm's usable state: which join statuses appear among the spans
    # this recording ALREADY has scored. An unscored span's status is worth
    # nothing until a GPU pass exists for it.
    both_bnd = set()
    bnd_state = {}
    for rid, cands in span_cand.items():
        sc_ = [c for c in cands if c["already_scored"]] or cands
        st = {c["join_status"] for c in sc_} - {"unaudited"}
        bnd_state[rid] = st
        if len(st) > 1:
            both_bnd.add(rid)
    print(f"       {len(both_bnd)} recordings carry BOTH a confirmed and an "
          f"unconfirmed join AMONG THEIR SCORED SPANS")

    rows = []
    for rid in sorted(per_rec):
        s, b = sem.get(rid, set()), bnd_state.get(rid, set())
        sem_gap = ("complete" if {"yes", "no"} <= s else
                   "needs NO" if "yes" in s else
                   "needs YES" if "no" in s else "unaudited")
        bnd_gap = ("complete" if len(b) > 1 else
                   "needs not_confirmed" if "confirmed" in b else
                   "needs confirmed" if "not_confirmed" in b else "unaudited")
        n_sem = len(sem_cand.get(rid, ()))
        n_span = len(span_cand.get(rid, ()))
        n_scored = sum(1 for c in span_cand.get(rid, ())
                       if c["already_scored"])
        reach_sem = sem_gap in ("needs NO", "needs YES") and n_sem > 0
        reach_bnd = bnd_gap in ("needs not_confirmed",
                                "needs confirmed") and n_span > 0
        cold_sem = sem_gap == "unaudited" and n_sem >= 2
        cold_bnd = bnd_gap == "unaudited" and n_span >= 2
        if reach_sem and reach_bnd:
            tier, why = 3, "one audit from BOTH on both arms"
        elif reach_sem or reach_bnd:
            tier, why = 1, ("one audit from BOTH on "
                            + ("semantic" if reach_sem else "span"))
        elif cold_sem and cold_bnd:
            tier, why = 2, "unaudited but rich on both arms"
        elif cold_sem or cold_bnd:
            tier, why = 0, "unaudited, one arm only"
        else:
            continue
        rows.append({"recording_id": rid, "tier": tier, "why": why,
                     "sem_gap": sem_gap, "bnd_gap": bnd_gap,
                     "n_sem_candidates": n_sem, "n_span_candidates": n_span,
                     "n_spans_already_scored": n_scored})
    # PRIORITY IS 3 > 1 > 2, NOT NUMERIC. Tier 1 is one audit from a contrast
    # and Tier 2 is an unaudited recording that might become one after two;
    # sorting by -tier put Tier 2 above Tier 1 and --top then cut every Tier 1
    # row out of the table entirely, which read as "there are none".
    order = {3: 0, 1: 1, 2: 2, 0: 3}
    rows.sort(key=lambda r: (order[r["tier"]], -r["n_spans_already_scored"],
                             -min(r["n_sem_candidates"], 4),
                             r["recording_id"]))
    rows = rows[:a.top]

    print(f"\n  {'recording':<20}{'tier':>5}{'sem':>10}{'bnd':>20}"
          f"{'semC':>6}{'spanC':>7}{'scored':>8}")
    for r in rows:
        print(f"  {r['recording_id']:<20}{r['tier']:>5}{r['sem_gap']:>10}"
              f"{r['bnd_gap']:>20}{r['n_sem_candidates']:>6}"
              f"{r['n_span_candidates']:>7}{r['n_spans_already_scored']:>8}")
    print(f"\n  tier 3 {sum(1 for r in rows if r['tier'] == 3)}, "
          f"tier 1 {sum(1 for r in rows if r['tier'] == 1)}, "
          f"tier 2 {sum(1 for r in rows if r['tier'] == 2)}, "
          f"tier 0 {sum(1 for r in rows if r['tier'] == 0)}")

    # THE YIELD IS NOT GUARANTEED AND THE COUNT SAYS SO. A recording that needs
    # a NO gains its contrast only if one of its audited candidates actually
    # comes back NO; a packet that comes back all YES has cost an hour and
    # bought nothing. The ceiling is printed, never a projection.
    ceil_sem = len(both_sem) + sum(1 for r in rows
                                   if r["sem_gap"] in ("needs NO", "needs YES",
                                                       "unaudited"))
    ceil_bnd = len(both_bnd) + sum(1 for r in rows
                                   if r["bnd_gap"] != "complete")
    print(f"\n  CEILING if every packet lands the missing side: BOTH-class "
          f"recordings {len(both_sem)} -> {ceil_sem},\n  "
          f"BOTH-boundary recordings {len(both_bnd)} -> {ceil_bnd}. Targets "
          f"are 25-30 and 20-25.")
    print(f"  This is a ceiling, not a projection: a packet whose candidates "
          f"all come back YES\n  buys nothing, and nothing here predicts "
          f"which way one will go -- by design, since\n  selecting on a "
          f"model's opinion would make the evaluation circular.")

    if a.out:
        pack = []
        for r in rows:
            rid = r["recording_id"]
            sc = sorted(sem_cand.get(rid, ()),
                        key=lambda c: -len(c["why_selected"]))
            sp = sorted(span_cand.get(rid, ()),
                        key=lambda c: (not c["already_scored"], c["start"]))
            pack.append(dict(r, sem_targets=sc[:a.per_recording],
                             span_targets=sp[:a.per_recording]))
        json.dump({"generated_for": "bridge audit",
                   "both_class_today": sorted(both_sem),
                   "both_boundary_today": sorted(both_bnd),
                   "packets": pack},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {len(pack)} packets -> {a.out}")
        print(f"  Each packet is {a.per_recording} semantic and "
              f"{a.per_recording} span targets in ONE recording. The auditor "
              f"sees\n  the clip, the candidate label and the internal join "
              f"time -- never a model score,\n  a current classification, or "
              f"which failure bucket the example came from.")


if __name__ == "__main__":
    main()
