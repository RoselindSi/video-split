"""Compound supervision from CONSECUTIVE REAL segments, with counterfactuals.

25 natural multi-action labels is a limit on EVALUATION, not on training. The
recordings already contain multi-action sequences -- wipe, then discard, then
retrieve -- and each step already has a frozen atomic decomposition. Joining
adjacent segments gives a compound example whose video is real continuous
behaviour and whose temporal order comes from the data rather than from a
model's guess.

THE VIDEO IS NEVER SYNTHESISED. Only the TEXT side is perturbed. Segments are
joined only when they are adjacent in the same recording, so a positive is a
span that actually happened in that order.

FOUR COUNTERFACTUAL KINDS, and one of them is a POSITIVE:

    drop_claim      one atomic claim removed -> PARTIAL. This is the same
                    thing `claim_support=partial` means in the frozen schema:
                    some stated claims supported, some absent
    reorder         the temporal constraint reversed -> contradiction
    replace_verb    a verb from a DIFFERENT cluster -> NO
    paraphrase      a verb from the SAME cluster -> still YES

The last one is not decoration. Training on compositional hard negatives alone
teaches a model to fail on any wording change, so a semantics-PRESERVING
variation has to be present for the same claim. `rinse` -> `wash` must stay
YES while `rinse` -> `scrub` becomes NO, and the two differ only in which
cluster the substitute came from.

REPLACEMENTS COME FROM THE OBSERVED VOCABULARY, not from a generator. A verb
is swapped for another verb that appears in this corpus, so the negative is a
claim someone plausibly wrote rather than a string no annotator would produce.

POOL A RECORDINGS ARE EXCLUDED BY DEFAULT. The 89 audited events are the
evaluation, and constructing training spans from their recordings puts the
same scenes on both sides. Excluded, counted, and the flag to include them
says what it costs.

Usage:
    python -m src.auditor.semantic.compose_supervision \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --claims data/gold/atomic_claims_frozen.json \
        --exclude data/gold/semantic_ontology_gold_48.json \
        --exclude data/gold/semantic_enrichment_gold_41.csv \
        --out /workspace/tr1/results/auditor/compound_supervision.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

from src.auditor.semantic.claim_support_diagnostic import CLUSTERS
from src.auditor.semantic.render_ontology_clips import get_segments, get_video

CLUSTER_OF = {w: i for i, c in enumerate(CLUSTERS) for w in c}


def resolve(patterns):
    out = []
    for pat in patterns:
        hits = (sorted(glob.glob(os.path.join(pat, "*.json")))
                if os.path.isdir(pat) else
                ([pat] if os.path.exists(pat) else sorted(glob.glob(pat))))
        if not hits:
            print(f"  !! matched nothing: {pat}")
        for h in hits:
            if h not in out and not h.endswith(".manifest.json"):
                out.append(h)
    return out


def excluded_recordings(paths):
    out = set()
    for p in paths:
        if not os.path.exists(p):
            print(f"  !! --exclude {p} not found")
            continue
        if p.lower().endswith(".csv"):
            with open(p, newline="", encoding="utf-8-sig") as f:
                items = list(csv.DictReader(f))
        else:
            blob = json.load(open(p, encoding="utf-8-sig"))
            items = blob.get("events", blob if isinstance(blob, list) else [])
        for e in items:
            if not isinstance(e, dict):
                continue
            rid = e.get("recording_id")
            if not rid:
                m = re.match(r"^(recording_\d+)", str(e.get("event_id") or ""))
                rid = m.group(1) if m else None
            if rid:
                out.add(rid)
    return out


def claims_of(dec, offset):
    """Re-id one label's claims so several can be concatenated."""
    emap = {e["id"]: f"e{offset}_{e['id']}" for e in dec["entities"]}
    amap = {a["id"]: f"a{offset}_{a['id']}" for a in dec["actions"]}
    ents = [dict(e, id=emap[e["id"]]) for e in dec["entities"]]
    acts = [dict(a, id=amap[a["id"]],
                 object=emap.get(a["object"]) if a["object"] else None)
            for a in dec["actions"]]
    temp = [dict(t, first=amap.get(t["first"], t["first"]),
                 second=amap.get(t["second"], t["second"]))
            for t in dec["temporal_constraints"]]
    return ents, acts, temp


def verb_pool(claims):
    by_cluster = defaultdict(set)
    allv = Counter()
    for d in claims.values():
        for a in d["actions"]:
            v = a["verb"]
            allv[v] += 1
            by_cluster[CLUSTER_OF.get(v, -1)].add(v)
    return allv, by_cluster


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--include_pool_a", action="store_true",
                    help="build spans from the audited recordings too. They "
                         "are the evaluation, so this puts the same scenes on "
                         "both sides")
    ap.add_argument("--max_gap_s", type=float, default=2.0,
                    help="two segments are adjacent only if the unlabelled "
                         "time between them is under this. A long gap means "
                         "something happened that neither claim mentions")
    ap.add_argument("--span", type=int, default=2,
                    help="how many consecutive segments per compound example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    allv, by_cluster = verb_pool(claims)
    print(f"{len(claims)} frozen label decompositions; {len(allv)} distinct "
          f"verbs over {len(by_cluster)} clusters")
    excl = set() if a.include_pool_a else excluded_recordings(a.exclude)
    if a.exclude and not a.include_pool_a and not excl:
        raise SystemExit("--exclude matched no recordings; that would build "
                         "training spans on the evaluation recordings without "
                         "it showing in any count below.")
    print(f"  excluding {len(excl)} pool-A recordings")

    seen, spans = set(), []
    skip = Counter()
    for p in resolve(a.recseg):
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            if rid in excl:
                skip["pool-A recording"] += 1
                continue
            segs = sorted(([str(x[0]), float(x[1]), float(x[2])]
                           for x in get_segments(r)[0]), key=lambda x: x[1])
            vid = get_video(r)
            for i in range(len(segs) - a.span + 1):
                w = segs[i:i + a.span]
                if any(w[k + 1][1] - w[k][2] > a.max_gap_s
                       for k in range(len(w) - 1)):
                    skip["gap too long"] += 1
                    continue
                if any(s[0] not in claims for s in w):
                    skip["label has no decomposition"] += 1
                    continue
                if len({s[0] for s in w}) < len(w):
                    skip["repeated label in the span"] += 1
                    continue
                spans.append((rid, vid, w))

    print(f"\n{len(spans)} candidate spans of {a.span} consecutive segments "
          f"over {len({s[0] for s in spans})} recordings")
    for k, v in skip.most_common():
        print(f"  skipped, {k}: {v}")
    if not spans:
        raise SystemExit("no spans; nothing to construct")

    rng = random.Random(a.seed)
    out, kinds = [], Counter()
    for rid, vid, w in spans:
        ents, acts, temp = [], [], []
        for k, s in enumerate(w):
            e, ac, t = claims_of(claims[s[0]], k)
            ents += e
            acts += ac
            temp += t
        for k in range(len(w) - 1):
            temp.append({"relation": "before",
                         "first": f"a{k}_a1", "second": f"a{k + 1}_a1",
                         "importance": "stated"})
        base = {"recording_id": rid, "video": vid,
                "start": w[0][1], "end": w[-1][2],
                "source_labels": [s[0] for s in w],
                "entities": ents, "actions": acts,
                "temporal_constraints": temp}

        def emit(kind, target, acts_, temp_, note):
            out.append(dict(base, variant=kind, target=target,
                            actions=acts_, temporal_constraints=temp_,
                            note=note))
            kinds[(kind, target)] += 1

        emit("original", "yes", acts, temp, "the span as it happened")

        if len(acts) > 1:
            drop = rng.randrange(len(acts))
            kept = [x for i, x in enumerate(acts) if i != drop]
            kept_ids = {x["id"] for x in kept}
            emit("drop_claim", "partial", kept,
                 [t for t in temp if t["first"] in kept_ids
                  and t["second"] in kept_ids],
                 f"dropped {acts[drop]['verb']}; the video still shows it")

        if temp:
            t0 = rng.choice(temp)
            emit("reorder", "no", acts,
                 [dict(t, first=t["second"], second=t["first"])
                  if t is t0 else t for t in temp],
                 f"reversed {t0['first']} before {t0['second']}")

        tgt = rng.randrange(len(acts))
        v = acts[tgt]["verb"]
        far = [x for x in allv
               if CLUSTER_OF.get(x, -2) != CLUSTER_OF.get(v, -1) and x != v]
        if far:
            sub = rng.choice(far)
            emit("replace_verb", "no",
                 [dict(x, verb=sub) if i == tgt else x
                  for i, x in enumerate(acts)], temp,
                 f"{v} -> {sub}, different cluster")
        near = [x for x in by_cluster.get(CLUSTER_OF.get(v, -1), ())
                if x != v]
        if near:
            sub = rng.choice(near)
            emit("paraphrase", "yes",
                 [dict(x, verb=sub) if i == tgt else x
                  for i, x in enumerate(acts)], temp,
                 f"{v} -> {sub}, same cluster: meaning preserved")

    print(f"\n{len(out)} constructed examples:")
    print(f"  {'variant':<16}{'target':<10}{'n':>6}")
    for (k, t), n in kinds.most_common():
        print(f"  {k:<16}{t:<10}{n:>6}")
    ty = Counter(t for (_k, t) in kinds.elements())
    print(f"\n  targets: {dict(ty)}")
    npos = sum(n for (k, t), n in kinds.items() if t == "yes")
    print(f"  {npos} YES of {len(out)} -- and {kinds.get(('paraphrase','yes'),0)}"
          f" of those are PARAPHRASES, the semantics-preserving variation.\n"
          f"  Without them a verifier trained here learns that any wording "
          f"change means NO.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")
        print(f"  TRAINING ONLY. The 25 natural compound labels stay the "
              f"evaluation: these show\n  whether a compositional rule can be "
              f"learned, and only natural gold shows whether\n  it transfers "
              f"to annotation people actually wrote.")


if __name__ == "__main__":
    main()
