"""A reorder arm wide enough to resolve, built from segment boundaries.

The frozen arm has 28 pairs over 14 recordings and an excess interval of
+-0.238 -- twice `drop_claim`'s at identical 29/14/4 structure, so the effect
is heterogeneous across recordings rather than thin, and the interval resamples
recordings. More pairs inside the same 14 buy nothing. The capacity audit found
161 recordings' worth of adjacent spans; at the same heterogeneity, 150
clusters would put the half-width near 0.073 and today's +0.214 clearly clear
of zero.

THE ORDER COMES FROM THE TIMESTAMPS, NOT FROM THE ANNOTATION. Segment A ends
and segment B begins, so a text claiming B before A is false about the video
whether or not either annotator described order correctly. That is the whole
reason this pool beats "every multi-clause label in the corpus": reorder is
supposed to test whether a model reads order off the video, and a pool whose
ground truth is the annotator's own ordering phrase cannot test that cleanly.

"THEN", NOT "AND". `A and B` only weakly implies sequence in English, so a
model that called both orders acceptable would be right about the language and
would score a tie -- measuring an ambiguous question rather than a temporal
one. `A then B` asserts the order, which makes the swap unambiguously false.
The frozen 28 do not have this problem because they came from labels whose
annotator wrote an explicit temporal relation.

CAPPED PER RECORDING. One recording offers 195 eligible spans and most offer a
handful; taking them all would let a few recordings dominate a bootstrap that
resamples recordings, which is the opposite of the point. The cap is applied by
taking spans spread evenly through each recording rather than randomly, so the
sample also covers different parts of a session.

EMITTED AS A SEPARATE KIND. `reorder_span` rather than `reorder`, and the
frozen 28 are copied in unchanged, so one scoring run produces both rows and
neither construction is silently pooled into the other.

THE ORIGINALS ARE UNAUDITED. Neither label was checked for claim_support, so
some spans describe their video wrongly. For THIS kind that adds noise without
bias -- a content error is present on both sides of the pair, while the order,
the only thing the pair varies, is right by construction.

Usage:
    python -m src.auditor.semantic.reorder_v2 \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --claims data/gold/atomic_claims_frozen.json \
        --benchmark data/gold/paired_semantic_benchmark.jsonl \
        --out data/gold/reorder_span_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from src.auditor.semantic.compose_supervision import resolve
from src.auditor.semantic.paired_benchmark import recase, well_formed
from src.auditor.semantic.render_ontology_clips import get_segments, get_video


def lower_first(s):
    return s[0].lower() + s[1:] if s else s


def spread(items, k):
    """k items spread evenly through the list, not k random ones.

    A recording's spans are ordered in time, so an even sweep covers different
    parts of a session while a random draw can land four times in one minute."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl",
                    help="the frozen benchmark. Its 28 reorder pairs are "
                         "copied in unchanged so the frozen number keeps a "
                         "referent")
    ap.add_argument("--join", default="then",
                    help="the word between the two clauses. 'and' only weakly "
                         "implies sequence, so the swap stops being false and "
                         "the arm measures an ambiguity")
    ap.add_argument("--max_gap_s", type=float, default=2.0)
    ap.add_argument("--max_span_s", type=float, default=30.0,
                    help="a span longer than this is dropped. The scorer sees "
                         "8 frames of whatever window it is given, and order "
                         "inside a 90-second window is not a fair question at "
                         "that sampling rate")
    ap.add_argument("--per_recording", type=int, default=4,
                    help="matches the frozen arm's cap. The aim is clusters, "
                         "not pairs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    vset = {x["verb"] for d in claims.values() for x in d["actions"]
            if x.get("verb")}
    frozen = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
              if l.strip()]
    old = [r for r in frozen if r["kind"] == "reorder"]
    old_recs = {r["recording_id"] for r in old}
    print(f"{len(claims)} decompositions; frozen arm has {len(old)} reorder "
          f"pairs over {len(old_recs)} recordings")

    by_rec, skip = defaultdict(list), Counter()
    durs = []
    for path in resolve(a.recseg):
        blob = json.load(open(path, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid:
                continue
            vid = get_video(r)
            segs = sorted(([str(x[0]), float(x[1]), float(x[2])]
                           for x in get_segments(r)[0]), key=lambda x: x[1])
            for i in range(len(segs) - 1):
                x, y = segs[i], segs[i + 1]
                if y[1] - x[2] > a.max_gap_s:
                    skip["gap too long"] += 1
                    continue
                if y[2] - x[1] > a.max_span_s:
                    skip["span longer than max_span_s"] += 1
                    continue
                if x[0] == y[0]:
                    skip["same label twice"] += 1
                    continue
                if x[0] not in claims or y[0] not in claims:
                    skip["label has no decomposition"] += 1
                    continue
                orig = recase(x[0], f"{x[0]} {a.join} {lower_first(y[0])}")
                swap = recase(x[0], f"{y[0]} {a.join} {lower_first(x[0])}")
                if not (well_formed(orig, vset) and well_formed(swap, vset)):
                    skip["span not well formed"] += 1
                    continue
                if orig.lower() == swap.lower():
                    skip["swap is a no-op"] += 1
                    continue
                durs.append(y[2] - x[1])
                by_rec[rid].append({
                    "segment_uid": f"{rid}_span{x[1]:g}",
                    "recording_id": rid, "video": vid,
                    "start": x[1], "end": y[2],
                    "audit_key": f"{rid}_span{x[1]:g}",
                    "original": orig, "counterfactual": swap,
                    "kind": "reorder_span", "expected": "no",
                    "pool": "adjacent_span",
                    "note": f"segments swapped: {x[0]!r} | {y[0]!r}"})

    print(f"\n{sum(len(v) for v in by_rec.values())} eligible spans over "
          f"{len(by_rec)} recordings")
    for k, v in skip.most_common():
        print(f"  rejected, {k}: {v}")
    if durs:
        q = np.percentile(durs, [50, 90, 100])
        print(f"  span duration: median {q[0]:.1f}s, p90 {q[1]:.1f}s, "
              f"max {q[2]:.1f}s")

    out = [dict(r, pool="audited_yes") for r in old]
    for rid in sorted(by_rec):
        out += spread(by_rec[rid], a.per_recording)
    new = [r for r in out if r["kind"] == "reorder_span"]
    recs = {r["recording_id"] for r in new}
    per = Counter(r["recording_id"] for r in new)
    print(f"\nafter the cap: {len(new)} reorder_span pairs over {len(recs)} "
          f"recordings, max {max(per.values()) if per else 0} per recording")
    print(f"  {len(recs - old_recs)} of them are recordings the frozen arm "
          f"never saw; clusters go {len(old_recs)} -> {len(recs | old_recs)}")
    # WHAT THE CLUSTER COUNT BUYS, stated before anything is scored. The
    # interval scales roughly with 1/sqrt(clusters) if the heterogeneity is
    # unchanged, and that assumption is itself a prediction this arm tests.
    if recs:
        print(f"  at unchanged heterogeneity the +-0.238 half-width would go "
              f"to about "
              f"{0.238 * (len(old_recs) / len(recs | old_recs)) ** 0.5:.3f}")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out)} pairs -> {a.out}")
    print(f"  BOTH arms have to be rescored on this file. The frozen "
          f"+0.330 and +0.214\n  belong to the 28 audited pairs and say "
          f"nothing about reorder_span; H2's\n  target is whatever cosine "
          f"scores HERE.")


if __name__ == "__main__":
    main()
