"""How far the reorder arm can be widened, counted in RECORDINGS.

reorder is the one kind whose excess interval contains zero, and the reason is
not the pair count: `drop_claim` has the same 29 pairs over the same 14
recordings with the same 4-per-recording cap, and half the interval width
(+-0.123 against +-0.238). The effect varies across recordings rather than
being thin, and the interval resamples recordings -- so more pairs inside the
same 14 change almost nothing. RECORDINGS is the number this audit reports.

THIS EMITS NOTHING. It answers whether a benchmark v2 is worth building before
the next model exists, because widening an evaluation arm after a model is
designed is the post-hoc move this whole line has been avoiding.

THREE POOLS, and they are not equally trustworthy:

    audited_yes         what the current arm draws from: segments of events a
                        human audited as claim_support=yes. The original is
                        KNOWN correct, which is what makes the pair clean.

    labelled_multiclause  every segment in the corpus whose stored label
                        decomposes into two clauses with a temporal
                        constraint, audited or not. Far more recordings, and
                        the cost is that the original is UNVERIFIED: if the
                        label is wrong about its video, both sides of the pair
                        are wrong and it contributes noise rather than signal.

    adjacent_spans      two consecutive segments joined into one span. The
                        order comes from the TIMESTAMPS, not from an
                        annotator's "after" phrase -- so this pool is the only
                        one whose temporal ground truth does not depend on the
                        annotation being right about order, even though each
                        label individually is still unaudited.

The third is the interesting one for H2 and the audit reports it separately
rather than pooling all three into one encouraging number.

Usage:
    python -m src.auditor.semantic.reorder_capacity \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --claims data/gold/atomic_claims_frozen.json \
        --benchmark data/gold/paired_semantic_benchmark.jsonl \
        --gold data/gold/semantic_ontology_gold_48.json \
        --gold data/gold/semantic_enrichment_gold_41.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from src.auditor.semantic.compose_supervision import resolve
from src.auditor.semantic.paired_benchmark import (
    SPLIT_CLAUSE, recase, well_formed)
from src.auditor.semantic.render_ontology_clips import get_segments


def clauses(label):
    return [p for p in SPLIT_CLAUSE.split(label or "") if p and p.strip()
            and not SPLIT_CLAUSE.fullmatch(p)]


def reorderable(label, dec, vset, skip):
    """Can this label produce a reorder counterfactual at all?

    The same four gates the benchmark applies, counted rather than silently
    dropped -- a pool that fails mostly on one gate is a pool that a change to
    that gate would unlock, and one that fails on all four is exhausted."""
    if dec is None:
        skip["no frozen decomposition"] += 1
        return False
    parts = clauses(label)
    if len(parts) < 2:
        skip["single clause"] += 1
        return False
    if len(dec.get("actions") or []) < 2:
        skip["one action: clauses share an object"] += 1
        return False
    if not dec.get("temporal_constraints"):
        skip["no temporal constraint"] += 1
        return False
    for i in range(len(parts) - 1):
        sw = list(parts)
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        cand = recase(label, " and ".join(sw))
        if well_formed(cand, vset) and cand.lower() != label.lower():
            return True
    skip["swap not well formed"] += 1
    return False


def report(name, per_rec, skip, current):
    recs = {r for r, n in per_rec.items() if n}
    new = recs - current
    print(f"\n  {name}")
    print(f"    candidate_recordings           {len(per_rec)}")
    print(f"    eligible_recordings            {len(recs)}")
    print(f"    eligible_pairs                 {sum(per_rec.values())}")
    print(f"    max_pairs_per_recording        "
          f"{max(per_rec.values()) if per_rec else 0}")
    print(f"    new_recordings_beyond_current  {len(new)}")
    print(f"    would take 14 ->               {len(current | recs)}")
    if skip:
        print(f"    rejected:")
        for k, v in skip.most_common():
            print(f"      {k:<38}{v:>6}")
    return recs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--max_gap_s", type=float, default=2.0,
                    help="two segments are adjacent only if less than this "
                         "much unlabelled time separates them. A long gap "
                         "means something happened that neither claim mentions "
                         "and the joined span is not what the video shows")
    a = ap.parse_args()

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    vset = {x["verb"] for d in claims.values() for x in d["actions"]
            if x.get("verb")}
    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    cur = {r["recording_id"] for r in bench if r["kind"] == "reorder"}
    audited = {r["segment_uid"] for r in bench}
    print(f"{len(claims)} frozen decompositions, {len(vset)} verbs")
    print(f"reorder today: {sum(1 for r in bench if r['kind'] == 'reorder')} "
          f"pairs over {len(cur)} recordings")

    # POOL 1, for calibration: re-derive the current arm from the benchmark's
    # own segments. If this does not come back at 14 the other two pools are
    # not measuring what they claim either.
    p1, s1 = Counter(), Counter()
    seen1 = set()
    for r in bench:
        if r["segment_uid"] in seen1:
            continue
        seen1.add(r["segment_uid"])
        if reorderable(r["original"], claims.get(r["original"]), vset, s1):
            p1[r["recording_id"]] += 1

    p2, s2 = Counter(), Counter()
    p3, s3 = Counter(), Counter()
    n_seg = n_rec = 0
    for path in resolve(a.recseg):
        blob = json.load(open(path, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid:
                continue
            n_rec += 1
            segs = sorted(([str(x[0]), float(x[1]), float(x[2])]
                           for x in get_segments(r)[0]), key=lambda x: x[1])
            n_seg += len(segs)
            for lab, st, _en in segs:
                uid = f"{rid}_s{st}"
                if uid in audited:
                    continue
                if reorderable(lab, claims.get(lab), vset, s2):
                    p2[rid] += 1
            # POOL 3. The order is the timestamps', so a swapped span is false
            # regardless of whether the annotator described order correctly --
            # the only pool here whose ground truth does not inherit the
            # annotation's own temporal claim.
            for i in range(len(segs) - 1):
                x, y = segs[i], segs[i + 1]
                if y[1] - x[2] > a.max_gap_s:
                    s3["gap too long"] += 1
                    continue
                if x[0] == y[0]:
                    s3["same label twice"] += 1
                    continue
                if x[0] not in claims or y[0] not in claims:
                    s3["label has no decomposition"] += 1
                    continue
                joined = recase(x[0], f"{x[0]} and "
                                      f"{y[0][0].lower() + y[0][1:]}")
                sw = recase(x[0], f"{y[0]} and "
                                  f"{x[0][0].lower() + x[0][1:]}")
                if not well_formed(joined, vset) or not well_formed(sw, vset):
                    s3["span not well formed"] += 1
                    continue
                if joined.lower() == sw.lower():
                    s3["swap is a no-op"] += 1
                    continue
                p3[rid] += 1
    print(f"\n{n_rec} recordings, {n_seg} segments in the recseg files")

    r1 = report("audited_yes (the current arm, re-derived)", p1, s1, cur)
    if len(r1) != len(cur):
        print(f"    !! re-derives {len(r1)} recordings, not {len(cur)}. The "
              f"other pools use the same gates, so they are not trustworthy "
              f"until this matches.")
    report("labelled_multiclause (unaudited originals)", p2, s2, cur)
    r3 = report("adjacent_spans (order from timestamps)", p3, s3, cur)

    print(f"\n  The decision this feeds: 14 -> 20-30+ recordings is worth "
          f"emitting a\n  benchmark v2 for and freezing H2 against. 14 -> "
          f"16 or 17 is not, and H2\n  then has to be written as large-effect "
          f"detection only.")
    print(f"\n  adjacent_spans is the pool to weigh most heavily: its temporal "
          f"ground truth\n  comes from the segment boundaries rather than "
          f"from the annotator's own\n  ordering claim, which is the thing "
          f"reorder is supposed to be testing.")
    print(f"  Both new pools leave the ORIGINAL unaudited -- a label that is "
          f"wrong about\n  its video makes both sides of its pair wrong, so "
          f"they add noise as well as\n  recordings, and that trade is only "
          f"worth taking for the recordings.")


if __name__ == "__main__":
    main()
