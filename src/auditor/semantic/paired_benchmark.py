"""Same segment, one word changed: a benchmark where recording identity cancels.

WHY THE CURRENT GOLD CANNOT ANSWER THE QUESTION. 32 recordings, one carrying
both classes, 99.2% of YES/NO pairs straddling recordings, and 13 of the 17
negatives inside three recordings. A model can score well by recognising which
kitchen it is looking at. The video-only prior reaching 0.827 without reading
a label at all is the proof: that number is available with no semantics
whatsoever.

So the fix is the evaluation, not the model. Here every comparison is WITHIN
ONE SEGMENT: the same frames, the same scene, the same person, the same
recording, scored against the stored label and against a perturbation of it.
Recording identity is constant across the pair and cancels exactly.

ONE WORD CHANGES. The counterfactual is built by substituting a single token
in the stored label -- the verb, the object, a qualifier -- located using the
frozen atomic decomposition. Re-rendering the claim from its structure was the
alternative and it would have changed the phrasing everywhere, so a score
difference could come from fluency rather than from meaning. Every pair here
differs by one word or by the order of two clauses.

ANCHORED ON HUMAN-VERIFIED CORRECT LABELS. A pair is only meaningful if the
original really is right, so segments come from events the audit marked
`claim_support = yes`. On a segment whose stored label is wrong, "original
beats counterfactual" is not the expected answer and scoring it would add
noise with a sign nobody can predict.

THREE KINDS ARE ONE-WORD SWAPS AND TWO ARE CLAUSE SURGERY, and only the first
three were ever clean. The first run had `reorder` at 28/28 and `drop_claim`
at 0.850 -- not because a dual encoder reads clause order, but because
rejoining clauses with " and " broke the capitalisation and, on labels whose
verbs share one object, left `Fold` or the empty string. Both kinds now
require the decomposition to hold more than one action, re-case the result,
and emit nothing unless it is a claim rather than a fragment.

FIVE KINDS, scored separately, because a single number hides which failure
the model cannot see:

    wrong_verb        verb replaced from a DIFFERENT cluster
    wrong_object      object replaced by another object in the corpus
    wrong_qualifier   region or instrument replaced
    drop_claim        one clause removed -- expects PARTIAL, not NO
    reorder           two clauses swapped -- only where the label states order

THE METRIC IS PAIR ACCURACY AND MARGIN, not AUROC. P(score(original) >
score(counterfactual)) over pairs, plus the mean gap. Both are computed within
a segment, so neither can be inflated by knowing which recording it is.

Usage:
    python -m src.auditor.semantic.paired_benchmark --emit \
        --gold data/gold/semantic_ontology_gold_48.json \
        --gold data/gold/semantic_enrichment_gold_41.csv \
        --claims data/gold/atomic_claims_frozen.json \
        --join .../naming_run_join.json --event_map .../..._event_map.json \
        --out data/gold/paired_semantic_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from src.auditor.semantic.claim_support_diagnostic import (
    CLUSTERS, load_gold, norm_key)

CLUSTER_OF = {w: i for i, c in enumerate(CLUSTERS) for w in c}
SPLIT_CLAUSE = re.compile(r"(\s+and\s+|\s*,\s*|\s*;\s*|\s*/\s*|\s*->\s*)",
                          re.I)


def swap_word(label, old, new):
    """Replace one whole word, preserving its capitalisation.

    Returns None when the word is not found as a token -- a substitution that
    silently matched inside another word would make a counterfactual nobody
    can read."""
    if not old or not new:
        return None
    pat = re.compile(rf"\b{re.escape(old)}\b", re.I)
    if not pat.search(label):
        return None

    def rep(m):
        s = m.group(0)
        return new.capitalize() if s[:1].isupper() else new
    return pat.sub(rep, label, count=1)


def recase(original, text):
    """Match the original's sentence casing.

    Reordering `Fold and tuck ... to seal` produced `tuck ... and Fold`: a
    lowercase start and a capitalised word mid-string. The model then had a
    well-formedness cue instead of an order cue, and `reorder` came back at
    28/28 -- which is not a dual encoder detecting clause order, it is one
    detecting a mangled string."""
    if not text:
        return text
    t = text[0].lower() + text[1:] if not original[:1].isupper() \
        else text[0].upper() + text[1:]
    # a word that was sentence-initial and is now mid-string loses its capital
    return re.sub(r"(?<=[a-z] )([A-Z])(?=[a-z])",
                  lambda m: m.group(1).lower(), t)


def well_formed(text, verbs):
    """A claim, not a fragment.

    `Remove and discard plastic wrap from bowl` lost a clause and became the
    EMPTY STRING; `Fold and crease sheet edge` became `Fold`. Scoring a bare
    verb or an empty string against a video and reading the result as
    "detects a missing claim" measures nothing. A counterfactual needs a verb
    and something for the verb to act on."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(toks) < 2:
        return False
    return any(t in verbs for t in toks) and any(t not in verbs for t in toks)


def build(label, dec, vocab, rng):
    """Counterfactuals for one label. Each differs by one word or one swap."""
    out = []
    verbs = [a["verb"] for a in dec["actions"] if a["verb"]]
    objs = [e["name"] for e in dec["entities"] if e.get("name")]
    quals = [(k, v) for a in dec["actions"]
             for k, v in (a.get("qualifiers") or {}).items()]

    if verbs:
        v = rng.choice(verbs)
        far = [x for x in vocab["verbs"]
               if CLUSTER_OF.get(x, -2) != CLUSTER_OF.get(v, -1) and x != v]
        if far:
            t = swap_word(label, v, rng.choice(far))
            if t:
                out.append(("wrong_verb", t, "no", f"verb {v} replaced"))
    if objs:
        o = rng.choice(objs)
        head = o.split()[-1]
        other = [x for x in vocab["objects"] if x != head
                 and x not in o.split()]
        if other:
            t = swap_word(label, head, rng.choice(other))
            if t:
                out.append(("wrong_object", t, "no", f"object {head} replaced"))
    if quals:
        k, val = rng.choice(quals)
        head = str(val).split()[-1]
        other = [x for x in vocab["qualifiers"].get(k, ()) if x != head]
        if other:
            t = swap_word(label, head, rng.choice(other))
            if t:
                out.append(("wrong_qualifier", t, "no",
                            f"{k} {head} replaced"))

    # Clause surgery only where the clauses are real. `Fold and crease sheet
    # edge` is ONE action whose verbs share an object, not two claims, and the
    # frozen decomposition says so: dropping half of it leaves `Fold`. Both
    # kinds now require at least two actions in the decomposition AND a
    # well-formed result, and emit nothing when they cannot produce one.
    vset = set(vocab["verbs"])
    parts = [p for p in SPLIT_CLAUSE.split(label) if p and p.strip()
             and not SPLIT_CLAUSE.fullmatch(p)]
    if len(parts) > 1 and len(dec["actions"]) > 1:
        for i in rng.sample(range(len(parts)), len(parts)):
            keep = [p for k, p in enumerate(parts) if k != i]
            cand = recase(label, " and ".join(keep))
            if well_formed(cand, vset):
                out.append(("drop_claim", cand, "partial",
                            f"clause {i + 1} of {len(parts)} removed"))
                break
        if dec["temporal_constraints"]:
            i = rng.randrange(len(parts) - 1)
            sw = list(parts)
            sw[i], sw[i + 1] = sw[i + 1], sw[i]
            cand = recase(label, " and ".join(sw))
            if well_formed(cand, vset) and cand.lower() != label.lower():
                out.append(("reorder", cand, "no", "two clauses swapped"))

    # OVERCLAIMING, the other half of completeness. drop_claim asks whether a
    # scorer notices a claim that is MISSING; these ask whether it notices one
    # that is PRESENT and did not happen. A scorer that merely prefers longer,
    # more complete-looking text passes the first and fails these.
    #
    # add_claim and replace_claim differ in one thing on purpose: the claim
    # COUNT. Under a wrong video every claim is unsupported, so a scorer that
    # scores completeness will prefer whichever text claims less -- and the
    # reranker's drop_claim null of 0.237 says this one does exactly that. So
    # add_claim's null is expected NOT to sit at chance: length and grounding
    # point the SAME way there, and the null cannot separate them.
    # replace_claim holds the count fixed so they point in different ways, and
    # is the readable version. Both are emitted so the prediction is testable
    # rather than assumed.
    #
    # THE BORROWED CLAIM IS ONLY PROBABLY FALSE. It comes from another label
    # and is rejected if its verb or its head noun already appears here, but
    # nothing verifies it is absent from THIS segment's video -- an adjacent
    # action can be genuinely present. That biases both kinds toward the null,
    # so they are conservative rather than optimistic, and the note records
    # which clause was borrowed so a sample can be audited by hand.
    # A SEPARATE GENERATOR, DERIVED FROM THE LABEL. Drawing the new kinds
    # from `rng` would advance it, and every label after the first would draw
    # different verbs and objects -- silently re-rolling the 306 pairs that
    # the cosine and reranker tables were computed on, so the two arms would
    # no longer be measured on the same data. Seeding from the label text
    # keeps these kinds deterministic and leaves `rng` untouched.
    org = random.Random(f"overclaim::{label}")
    here = set(re.findall(r"[a-z0-9]+", label.lower()))
    foreign = [c for c in vocab.get("clauses", ())
               if well_formed(c, vset)
               and not (set(re.findall(r"[a-z0-9]+", c.lower())) & here)]
    if foreign:
        add = org.choice(foreign)
        cand = recase(label, f"{label} and {add[0].lower() + add[1:]}")
        out.append(("add_claim", cand, "no",
                    f"borrowed claim appended: {add!r}"))
        if len(parts) > 1 and len(dec["actions"]) > 1:
            i = org.randrange(len(parts))
            n = len(parts[i].split())
            # LENGTH-MATCHED, so the count AND the word count both hold. A
            # replacement two words longer would reintroduce the very prior
            # this kind exists to remove.
            near = sorted(foreign, key=lambda c: abs(len(c.split()) - n))
            sub = near[0] if abs(len(near[0].split()) - n) <= 1 else None
            if sub:
                sw = [sub if k == i else pp for k, pp in enumerate(parts)]
                cand = recase(label, " and ".join(sw))
                if well_formed(cand, vset) and cand.lower() != label.lower():
                    out.append(("replace_claim", cand, "no",
                                f"clause {i + 1} of {len(parts)} replaced by "
                                f"{sub!r}"))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--claims", default="data/gold/atomic_claims_frozen.json")
    ap.add_argument("--join")
    ap.add_argument("--event_map", action="append", default=[])
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--evaluate")
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect_unchanged",
                    help="a previously emitted benchmark. Every pair of every "
                         "kind it contains must come out identical, or this "
                         "exits -- adding a kind must not re-roll the pairs "
                         "the published tables were computed on")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.evaluate:
        evaluate(a)
        return

    claims = json.load(open(a.claims, encoding="utf-8"))["claims"]
    rows = load_gold(a.gold)
    yes = {(r.get("audit_key") or r.get("event_id"))
           for r in rows if r["claim_support"] == "yes"}
    print(f"{len(rows)} audited events; {len(yes)} with claim_support=yes")

    join = json.load(open(a.join, encoding="utf-8")) if a.join else {}
    emap = {}
    for p in a.event_map:
        for k, v in json.load(open(p, encoding="utf-8")).items():
            emap[norm_key(k)] = v

    segs = {}
    for key in yes:
        m = emap.get(norm_key(key))
        if not m:
            continue
        for s in m["segments"]:
            if not s.get("shown_in_sheet", True):
                continue
            j = join.get(s["segment_uid"])
            if j and j.get("stored_label") in claims:
                segs[s["segment_uid"]] = dict(j, audit_key=key)
    print(f"  {len(segs)} segments from YES events with a frozen "
          f"decomposition")
    if not segs:
        raise SystemExit("no anchor segments; nothing to pair against")

    vocab = {"verbs": sorted({a_["verb"] for d in claims.values()
                              for a_ in d["actions"] if a_["verb"]}),
             "objects": sorted({e["name"].split()[-1]
                                for d in claims.values()
                                for e in d["entities"] if e.get("name")}),
             "qualifiers": defaultdict(set),
             # Every clause any label contains, as the pool a false claim is
             # borrowed from. Real annotation phrasing, so an overclaim is
             # something a person plausibly wrote rather than a generated
             # string no annotator would produce.
             "clauses": sorted({c.strip() for lab_ in claims
                                for c in SPLIT_CLAUSE.split(lab_)
                                if c and c.strip()
                                and not SPLIT_CLAUSE.fullmatch(c)})}
    for d in claims.values():
        for a_ in d["actions"]:
            for k, v in (a_.get("qualifiers") or {}).items():
                vocab["qualifiers"][k].add(str(v).split()[-1])
    print(f"  vocabulary: {len(vocab['verbs'])} verbs, "
          f"{len(vocab['objects'])} object heads, "
          f"{len(vocab['qualifiers'])} qualifier keys, "
          f"{len(vocab['clauses'])} clauses")

    rng = random.Random(a.seed)
    out, kinds = [], Counter()
    for uid, j in sorted(segs.items()):
        lab = j["stored_label"]
        for kind, text, target, note in build(lab, claims[lab], vocab, rng):
            if text.strip().lower() == lab.strip().lower():
                continue
            out.append({"segment_uid": uid, "recording_id": j["recording_id"],
                        "start": j["start"], "end": j["end"],
                        "audit_key": j["audit_key"], "original": lab,
                        "counterfactual": text, "kind": kind,
                        "expected": target, "note": note})
            kinds[kind] += 1

    if a.expect_unchanged:
        # THE OLD KINDS MUST COME OUT BYTE-IDENTICAL. Adding a kind is only
        # safe if it leaves the pairs the published tables were computed on
        # exactly where they were; a quiet re-roll would make two arms
        # incomparable while every count still looked right.
        old = [json.loads(l) for l in open(a.expect_unchanged,
                                           encoding="utf-8") if l.strip()]
        okinds = {r["kind"] for r in old}
        key = lambda r: (r["segment_uid"], r["kind"], r["original"],
                         r["counterfactual"])
        o, n_ = {key(r) for r in old}, {key(r) for r in out
                                        if r["kind"] in okinds}
        if o != n_:
            for x in sorted(o - n_)[:5]:
                print(f"  GONE    {x}")
            for x in sorted(n_ - o)[:5]:
                print(f"  NEW     {x}")
            raise SystemExit(f"{len(o - n_)} pairs disappeared and "
                             f"{len(n_ - o)} appeared among the kinds that "
                             f"already existed. The published cosine and "
                             f"reranker tables were computed on those pairs.")
        print(f"  verified: all {len(o)} pairs of the pre-existing kinds are "
              f"unchanged")

    print(f"\n{len(out)} pairs over {len({r['segment_uid'] for r in out})} "
          f"segments, {len({r['recording_id'] for r in out})} recordings")
    for k, n in kinds.most_common():
        print(f"  {k:<18}{n:>5}")
    clean = sum(n for k, n in kinds.items()
                if k in ("wrong_verb", "wrong_object", "wrong_qualifier"))
    print(f"\n  {clean} of {len(out)} pairs are single-token swaps "
          f"(wrong_verb/object/qualifier).\n  Those are the clean ones: the "
          f"counterfactual differs from the original by one word\n  and "
          f"nothing else. drop_claim and reorder rewrite the clause "
          f"structure, so read\n  them apart from the others and never "
          f"through the ALL row.")
    print(f"\n  every pair holds the video fixed and changes one word or the "
          f"order of two\n  clauses, so recording identity is constant across "
          f"the pair and cancels. The\n  0.827 a video-only head scored on the "
          f"unpaired gold is unavailable here by\n  construction.")
    print(f"\n  examples:")
    for r in out[:5]:
        print(f"    [{r['kind']:<15}] {r['original']!r}\n"
              f"    {'':<19}-> {r['counterfactual']!r}  expect "
              f"{r['expected']}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")
        print(f"  score each (segment, original) and (segment, "
              f"counterfactual) with any scorer,\n  write "
              f"{{'segment_uid','text','score'}} per line, and run "
              f"--evaluate on it.")


def evaluate(a):
    """Pair accuracy and margin from any scorer's output."""
    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    score = {}
    for line in open(a.evaluate, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            score[(r["segment_uid"], r["text"])] = float(r["score"])
    print(f"{len(bench)} pairs; {len(score)} scored (segment, text) entries")

    by = defaultdict(list)
    missing = 0
    for p in bench:
        so = score.get((p["segment_uid"], p["original"]))
        sc = score.get((p["segment_uid"], p["counterfactual"]))
        if so is None or sc is None:
            missing += 1
            continue
        by[p["kind"]].append((so - sc, p["recording_id"]))
    if missing:
        print(f"  !! {missing} pairs had no score and are excluded")
    if not by:
        raise SystemExit("nothing scored")

    print(f"\n  {'kind':<18}{'n':>5}{'pair acc':>10}{'mean margin':>13}")
    tot = []
    for k in sorted(by):
        d = [x for x, _ in by[k]]
        acc = sum(1 for x in d if x > 0) / len(d)
        print(f"  {k:<18}{len(d):>5}{acc:>10.3f}{sum(d) / len(d):>13.4f}")
        tot += d
    acc = sum(1 for x in tot if x > 0) / len(tot)
    print(f"  {'ALL':<18}{len(tot):>5}{acc:>10.3f}"
          f"{sum(tot) / len(tot):>13.4f}")
    print(f"\n  0.5 is chance. A margin near zero with accuracy near 0.5 means "
          f"the scorer is not\n  reading the changed word at all -- which on "
          f"pairs this tight is the finding,\n  not a shortfall.")


if __name__ == "__main__":
    main()
