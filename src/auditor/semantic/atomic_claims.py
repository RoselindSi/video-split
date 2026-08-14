"""Decompose stored labels into atomic claims. Video-blind, and a DRAFT.

WHY VIDEO-BLIND IS THE WHOLE POINT. The decomposition says what the annotation
CLAIMS. If it were produced with the video in view it would quietly become a
description of what happened, and a verifier scored against it would be
grading the video against itself. So this reads the label string and nothing
else -- no video, no features, no naming prediction.

WHAT IT MUST NOT DO: normalise the claim toward the canonical ontology.
`Wipe upper kettle body` decomposes to wipe(kettle) with region=upper_body,
NOT to wipe(kettle). Whether "upper body" is too fine is a GRANULARITY
question and it has its own head; deleting the qualifier here would destroy
the thing that question is about, and would also make the claim easier to
support than the annotation actually made it.

THIS PARSER IS A FIRST PASS, NOT THE DECOMPOSITION. It splits on surface
conjunctions and takes the first content token as the verb, which is right for
imperative labels and wrong for noun-initial ones -- `Slipper pair flip cycle`
parses as slipper(pair), and `verb_position_uncertain` marks it because a real
verb appears later. Roughly a third of these labels are not imperative. So the
output is a correction SHEET with the draft beside the label, and the frozen
artifact is what comes back from it: once, and reusable under any later
architecture, because it depends on the label alone.

TEMPORAL ORDER IS MARKED `stated` OR `inferred`. `then`, `after`, `->` state
an order. Plain `and` does not: "Rinse and seat" almost certainly happens in
that order, but the label does not say so, and a verifier that checks an order
the annotation never claimed will fail events for a claim nobody made. The
draft emits it as `inferred` so the correction pass decides.

Usage:
    python -m src.auditor.semantic.atomic_claims \
        --context data/gold/audit_188_context.jsonl \
        --out data/gold/atomic_claims_draft.json \
        --sheet data/gold/atomic_claims_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, OrderedDict

from src.auditor.semantic.claim_support_diagnostic import (
    CLUSTERS, STOP, ORD, tok)

VERBS = {w for c in CLUSTERS for w in c}
# surface splitters. `and` is included but its order is marked inferred.
SPLIT = re.compile(r"\s*(?:->|;|/|,|\band then\b|\bthen\b|\bafter\b|\band\b)\s*",
                   re.I)
ORDERED = re.compile(r"->|\bthen\b|\bafter\b|\band then\b", re.I)
# qualifiers we keep verbatim rather than folding into the object
REGION = {"upper", "lower", "top", "bottom", "inner", "interior", "outer",
          "exterior", "side", "front", "back", "left", "right", "rim", "edge",
          "opening", "surface", "body", "lid"}
INSTRUMENT = {"with", "using", "by"}


def clauses(label):
    parts = [p.strip() for p in SPLIT.split(label or "") if p and p.strip()]
    return parts or [(label or "").strip()]


def parse_clause(text, ent_ids):
    """verb + object + qualifiers from one clause, surface only.

    THE VERB IS THE FIRST CONTENT TOKEN, not the first token found in the
    CLUSTERS vocabulary. That vocabulary was built to group synonyms for
    scoring, not to parse, and looking verbs up in it left 144 of 288 labels
    with no verb at all -- `apply`, `unfold`, `crumple`, `squeeze` are simply
    not in it. These labels are imperative, so position identifies the verb
    far better than membership does; the vocabulary is kept only to FLAG which
    verbs are unfamiliar, which is information for the correction pass rather
    than a parse decision."""
    words = tok(text)
    verb = next((w for w in words if w not in STOP and w not in ORD), None)
    after = words[words.index(verb) + 1:] if verb else words
    instrument = None
    if any(w in INSTRUMENT for w in words):
        i = next(i for i, w in enumerate(words) if w in INSTRUMENT)
        instrument = " ".join(w for w in words[i + 1:]
                              if w not in STOP) or None
        after = [w for w in after if w not in words[i:]]
    region = [w for w in after if w in REGION]
    obj = [w for w in after if w not in STOP and w not in ORD
           and w not in REGION]
    name = " ".join(obj) or None
    q = {}
    if region:
        q["region"] = " ".join(region)
    if instrument:
        q["instrument"] = instrument
    return verb, name, q, (verb is not None and verb not in VERBS)


def decompose(label):
    cl = clauses(label)
    stated_order = bool(ORDERED.search(label or ""))
    ents, actions = OrderedDict(), []
    unknown = []
    for i, c in enumerate(cl):
        verb, obj, q, unk = parse_clause(c, ents)
        if unk:
            unknown.append(verb)
        eid = None
        if obj:
            if obj not in ents:
                ents[obj] = f"e{len(ents) + 1}"
            eid = ents[obj]
        actions.append({"id": f"a{i + 1}", "verb": verb, "object": eid,
                        "qualifiers": q, "importance": "core",
                        "source_clause": c})
    # COORDINATED VERBS SHARE AN OBJECT. "fold and unfold tissue" splits into
    # a clause with no object and one with `tissue`; the first verb acts on
    # the same thing. Inherited from the nearest clause that names one, and
    # marked, because an inherited object is a parse decision and not
    # something the label said twice.
    named = [x for x in actions if x["object"]]
    if named:
        for x in actions:
            if not x["object"]:
                near = min(named, key=lambda z: abs(int(z["id"][1:])
                                                    - int(x["id"][1:])))
                x["object"] = near["object"]
                x["object_inherited"] = True
    temporal = []
    for i in range(len(actions) - 1):
        temporal.append({"relation": "before", "first": actions[i]["id"],
                         "second": actions[i + 1]["id"],
                         "importance": "stated" if stated_order
                         else "inferred"})
    return {"label": label,
            "entities": [{"id": v, "name": k, "importance": "core"}
                         for k, v in ents.items()],
            "actions": actions,
            "spatial_relations": [],
            "temporal_constraints": temporal,
            "parser_flags": sorted(
                ([] if all(a["verb"] for a in actions) else ["missing_verb"])
                + ([] if all(a["object"] for a in actions) else
                   ["missing_object"])
                + (["object_inherited"]
                   if any(a.get("object_inherited") for a in actions) else [])
                + (["verb_position_uncertain"] if any(
                    a["verb"] not in VERBS
                    and any(w in VERBS for w in tok(a["source_clause"]))
                    for a in actions if a["verb"]) else [])
                + ([] if len(actions) == 1 or stated_order else
                   ["order_inferred_from_and"])),
            # a vocabulary summary, NOT a per-label flag. Reporting one flag
            # per unfamiliar verb produced 100 distinct "flags" and made the
            # flag counter useless for what it is for, which is telling
            # structural parse problems apart.
            "unknown_verbs": sorted(set(unknown))}


def labels_from(paths, join_path):
    out = Counter()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("prev_segment_label", "containing_segment_label",
                      "next_segment_label", "nearest_previous_segment_label",
                      "nearest_next_segment_label"):
                if r.get(k):
                    out[r[k]] += 1
    if join_path and os.path.exists(join_path):
        for v in json.load(open(join_path, encoding="utf-8")).values():
            if v.get("stored_label"):
                out[v["stored_label"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", action="append",
                    default=["data/gold/audit_188_context.jsonl"])
    ap.add_argument("--join", help="naming_run_join.json, whose stored_label "
                                   "is the exact string the audit judged")
    ap.add_argument("--out")
    ap.add_argument("--sheet")
    a = ap.parse_args()

    counts = labels_from(a.context, a.join)
    print(f"{len(counts)} distinct stored labels over "
          f"{sum(counts.values())} occurrences")

    dec = {lab: decompose(lab) for lab in counts}
    n_act = Counter(len(d["actions"]) for d in dec.values())
    print(f"\n  actions per label: {dict(sorted(n_act.items()))}")
    flags = Counter(f for d in dec.values() for f in d["parser_flags"])
    print(f"  parser flags: {dict(flags.most_common())}")
    unk = Counter(v for d in dec.values() for v in d["unknown_verbs"])
    print(f"  {len(unk)} verbs outside the CLUSTERS vocabulary, over "
          f"{sum(unk.values())} labels:\n    "
          f"{', '.join(k for k, _ in unk.most_common(14))} ...")
    print(f"  That vocabulary was built to group synonyms for scoring, so "
          f"being outside it is\n  not an error -- it is the list of verbs "
          f"the scorer cannot cluster, which the\n  correction pass and any "
          f"later verifier both need.")
    clean = [l for l, d in dec.items() if not d["parser_flags"]]
    print(f"  {len(clean)}/{len(dec)} parsed with no flag -- and a clean parse "
          f"is still a DRAFT,\n  because nothing here can tell a wrong verb "
          f"from a right one.")
    q = Counter(k for d in dec.values() for x in d["actions"]
                for k in x["qualifiers"])
    print(f"  qualifiers kept: {dict(q.most_common())}  "
          f"(region/instrument are NOT folded into the object)")
    multi = sum(1 for d in dec.values() if len(d["temporal_constraints"]) > 0)
    inf = sum(1 for d in dec.values()
              for t in d["temporal_constraints"]
              if t["importance"] == "inferred")
    print(f"  {multi} labels carry a temporal constraint; {inf} of those "
          f"constraints are INFERRED\n  from `and` rather than stated. A "
          f"verifier that checks an order the annotation never\n  claimed "
          f"fails events for a claim nobody made.")

    print(f"\n  examples:")
    for lab in sorted(counts, key=lambda x: -counts[x])[:4]:
        d = dec[lab]
        acts = "  ".join(f"{x['verb']}({x['object']})"
                         + (f"[{x['qualifiers']}]" if x["qualifiers"] else "")
                         for x in d["actions"])
        print(f"    {lab!r}\n      -> {acts}"
              + (f"   {d['parser_flags']}" if d["parser_flags"] else ""))

    if a.out:
        json.dump({"n_labels": len(dec), "claims": dec},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")
    if a.sheet:
        with open(a.sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["stored_label", "occurrences", "draft_actions",
                        "draft_entities", "draft_temporal", "parser_flags",
                        "corrected_actions", "corrected_entities",
                        "corrected_temporal", "note"])
            w.writerow(["", "", "", "", "", "",
                        "verb(object)[qualifiers], one per action",
                        "name only", "before(a1,a2) / none",
                        "why the draft was wrong"])
            for lab, c in counts.most_common():
                d = dec[lab]
                w.writerow([
                    lab, c,
                    " ; ".join(f"{x['verb']}({x['object']})"
                               + (f"[{x['qualifiers']}]"
                                  if x["qualifiers"] else "")
                               for x in d["actions"]),
                    " ; ".join(f"{e['id']}={e['name']}"
                               for e in d["entities"]),
                    " ; ".join(f"{t['relation']}({t['first']},{t['second']})"
                               f":{t['importance']}"
                               for t in d["temporal_constraints"]),
                    ",".join(d["parser_flags"]
                             + ([f"unknown_verbs:"
                                 f"{'/'.join(d['unknown_verbs'])}"]
                                if d["unknown_verbs"] else [])),
                    "", "", "", ""])
        print(f"wrote {a.sheet} -- the draft is what gets CORRECTED, not "
              f"what gets used.\n  Correct it once and the result is reusable "
              f"under any later architecture,\n  because it depends on the "
              f"label alone.")


if __name__ == "__main__":
    main()
