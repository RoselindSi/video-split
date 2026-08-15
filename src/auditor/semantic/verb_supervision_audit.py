"""Is there enough verb-directed supervision to train on, and where does it sit?

The frame sweep settled that temporal sampling is not the verb bottleneck --
8 to 32 frames moved verb accuracy 0.690 to 0.701 and its separation not at
all. What is left is supervision that points at the verb, and the compound
supervision file already contains two variants that do exactly that:

    paraphrase      the verb replaced from the SAME cluster -> still YES
    replace_verb    the verb replaced from a DIFFERENT cluster -> NO

Everything else about the example is identical, so the pair isolates the verb.
The question this file answers is not whether that idea is sound; it is how
many verbs actually receive both sides, over how many recordings, and whether
a handful of frequent verbs carry all of it.

WHY THIS RUNS BEFORE THE ARCHITECTURE IS WRITTEN. A design that assumes 1706
examples of verb contrast will be written differently from one that discovers
afterwards that 30 verbs have both sides and 60 have one or none. The three
outcomes need three different losses:

    A  enough per verb          verb-conditioned contrastive verifier
    B  broad but long-tailed    cluster-level supervision on a shared
                                embedding; no per-verb head
    C  only a few verbs have    1706 cannot carry the main objective; it
       two-sided support        becomes an auxiliary loss or pretraining

CORRELATION IS PART OF THE ANSWER. compose_supervision emits one paraphrase and
one replace_verb per span, so the two sides of a verb's supervision share a
video and all but one word -- which is what makes them a clean contrast and
also means they are not two independent observations. Spans are built from
CONSECUTIVE segments, so span i and span i+1 share a segment outright. Both are
counted here rather than left for a training curve to reveal.

Usage:
    python -m src.auditor.semantic.verb_supervision_audit \
        --supervision /workspace/tr1/results/auditor/compound_supervision.jsonl \
        --claims data/gold/atomic_claims_frozen.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from src.auditor.semantic.claim_support_diagnostic import CLUSTERS

CLUSTER_OF = {w: i for i, c in enumerate(CLUSTERS) for w in c}


def verb_diff(orig, var):
    """(original verb, substitute) for the one action that changed.

    Read by diffing against the span's own `original` variant rather than by
    parsing the note string, so a change to how notes are worded cannot
    silently turn this into a count of zero."""
    changed = [(a["verb"], b["verb"]) for a, b in zip(orig, var)
               if a["verb"] != b["verb"]]
    return changed[0] if len(changed) == 1 else None


def other_fields_moved(orig, var):
    """Did anything but the verb move? It should not."""
    return sum(1 for a, b in zip(orig, var)
               if a.get("object") != b.get("object")
               or (a.get("qualifiers") or {}) != (b.get("qualifiers") or {}))


def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else \
        "2-4" if n <= 4 else "5-9" if n <= 9 else "10+"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supervision", required=True)
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    a = ap.parse_args()

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    vcount = Counter(x["verb"] for d in claims.values()
                     for x in d["actions"] if x.get("verb"))
    singles = {v for v, c in vcount.items() if c == 1}
    print(f"{len(claims)} frozen decompositions, {len(vcount)} distinct verbs, "
          f"{len(singles)} of them appearing once")

    rows = [json.loads(l) for l in open(a.supervision, encoding="utf-8")
            if l.strip()]
    spans = defaultdict(dict)
    for r in rows:
        spans[(r["recording_id"], r["start"], r["end"])][r["variant"]] = r
    print(f"{len(rows)} examples over {len(spans)} spans, "
          f"{len({k[0] for k in spans})} recordings")
    print(f"  variants: {dict(Counter(r['variant'] for r in rows))}")

    pos, neg = defaultdict(list), defaultdict(list)
    subs = {"paraphrase": Counter(), "replace_verb": Counter()}
    moved = Counter()
    same_cluster_check = Counter()
    for key, v in spans.items():
        o = v.get("original")
        if not o:
            continue
        for variant, store in (("paraphrase", pos), ("replace_verb", neg)):
            x = v.get(variant)
            if not x:
                continue
            d = verb_diff(o["actions"], x["actions"])
            if d is None:
                moved[f"{variant}: not exactly one verb changed"] += 1
                continue
            n_other = other_fields_moved(o["actions"], x["actions"])
            if n_other:
                moved[f"{variant}: object or qualifier also changed"] += 1
            store[d[0]].append((key[0], d[1]))
            subs[variant][d[1]] += 1
            same = CLUSTER_OF.get(d[0], -1) == CLUSTER_OF.get(d[1], -2)
            same_cluster_check[
                f"{variant}: {'same' if same else 'different'} cluster"] += 1

    def block(name, store):
        pairs = sum(len(v) for v in store.values())
        recs = {r for v in store.values() for r, _ in v}
        verbs = set(store)
        return (name, pairs, len(recs), len(verbs), len(verbs & singles))

    both = {v for v in pos if v in neg}
    both_pairs = sum(len(pos[v]) + len(neg[v]) for v in both)
    both_recs = {r for v in both for r, _ in pos[v] + neg[v]}
    table = [block("same-cluster positive", pos),
             block("cross-cluster negative", neg),
             ("both available for same verb", both_pairs, len(both_recs),
              len(both), len(both & singles))]
    print(f"\n  {'subset':<30}{'pairs':>7}{'recordings':>12}{'verbs':>7}"
          f"{'singleton verbs':>17}")
    for n, p, r, vv, sg in table:
        print(f"  {n:<30}{p:>7}{r:>12}{vv:>7}{sg:>17}")
    print(f"\n  of {len(vcount)} verbs and {len(singles)} singletons in the "
          f"frozen decompositions")

    allv = sorted(set(pos) | set(neg))
    print(f"\n  per-verb support")
    print(f"    {'bucket':<8}{'positive':>10}{'negative':>10}")
    bp = Counter(bucket(len(pos.get(v, ()))) for v in allv)
    bn = Counter(bucket(len(neg.get(v, ()))) for v in allv)
    for b in ("0", "1", "2-4", "5-9", "10+"):
        print(f"    {b:<8}{bp.get(b, 0):>10}{bn.get(b, 0):>10}")
    for name, store in (("positive", pos), ("negative", neg)):
        c = [len(store.get(v, ())) for v in allv]
        print(f"    {name}: median {np.median(c):.0f}, p90 "
              f"{np.percentile(c, 90):.0f}, max {max(c)}")
    print(f"    only positive {len(set(pos) - set(neg))}, only negative "
          f"{len(set(neg) - set(pos))}, both {len(both)}")

    print(f"\n  leakage and triviality")
    for k, v in sorted(same_cluster_check.items()):
        print(f"    {k}: {v}")
    if moved:
        for k, v in moved.most_common():
            print(f"    !! {k}: {v}")
    else:
        print(f"    every variant changed exactly one verb and nothing else")
    for variant in ("paraphrase", "replace_verb"):
        c = subs[variant]
        top = sum(n for _, n in c.most_common(5))
        tot = sum(c.values()) or 1
        print(f"    {variant}: {len(c)} distinct substitutes, top 5 carry "
              f"{100 * top / tot:.0f}% ({[w for w, _ in c.most_common(5)]})")

    # SPANS OVERLAP BY CONSTRUCTION. compose_supervision walks consecutive
    # segments, so span i and span i+1 share a segment and their examples share
    # video. Pairs that share a segment are not independent draws, and a
    # training curve would not say so.
    seg_use = Counter()
    for rid, st, en in spans:
        seg_use[(rid, st)] += 1
        seg_use[(rid, en)] += 1
    shared = sum(1 for v in seg_use.values() if v > 1)
    print(f"    {shared} of {len(seg_use)} segment endpoints are shared by two "
          f"spans, so adjacent spans overlap")

    print(f"\n  which branch this supports")
    print(f"    A  verb-conditioned contrastive verifier   needs most verbs "
          f"two-sided with 5+ each")
    print(f"    B  cluster-level supervision, no per-verb head   broad "
          f"coverage, long tail")
    print(f"    C  auxiliary loss only   few verbs two-sided")
    print(f"    -> {len(both)} verbs have both sides; median support is "
          f"{np.median([len(pos.get(v, ())) for v in allv]):.0f} positive and "
          f"{np.median([len(neg.get(v, ())) for v in allv]):.0f} negative")


if __name__ == "__main__":
    main()
