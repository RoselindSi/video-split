"""Merge the human corrections into the frozen atomic-claim decomposition.

171 of 288 labels were corrected, so the draft was wrong more often than it
was right and the frozen artifact is mostly human. Untouched rows keep the
draft, which is the annotator saying it was already correct rather than an
absence of review.

THE FALLBACK IS PER FIELD, NOT PER ROW. "Leave the cell blank if the draft is
right" was applied per COLUMN: a row can carry corrected actions and a blank
entities cell because the draft entity was already correct. Treating a blank
column as an empty list produced an action naming an entity that did not
exist, and `fields_from_draft` records which columns each label inherited.

WHAT THE FLAGS WERE WORTH, measured against the corrections rather than
assumed:

    verb_position_uncertain   12 of 13 corrected   92%
    unknown_verbs             97 of 155            63%
    order_inferred_from_and   39 of 92             42%
    object_inherited          23 of 65             35%
    NO FLAG AT ALL            61 of 104            59%

Unflagged drafts were corrected at 59%, the same as the overall rate. So the
flags did not triage: only `verb_position_uncertain` predicted anything, and
it covered 13 labels. I told you to work the flagged rows first and treat
unflagged ones as lowest priority, and that advice was wrong -- a clean
surface parse carries no information about whether the verb is the right verb.

VALIDATION IS THE POINT OF THIS FILE. Corrections are free text and the
schema has referential structure: every `eN` an action names must exist among
the entities, and every `aN` a temporal constraint names must exist among the
actions. A dangling reference produces a claim about nothing, which a verifier
would silently score. Both are checked and reported per label.

Usage:
    python -m src.auditor.semantic.atomic_claims_freeze \
        --filled data/gold/atomic_claims_filled_288.csv \
        --draft data/gold/atomic_claims_draft.json \
        --out data/gold/atomic_claims_frozen.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter

ACTION = re.compile(r"^\s*([A-Za-z_][A-Za-z_-]*)\s*\(([^)]*)\)\s*(\[.*\])?\s*$")
TEMPORAL = re.compile(r"^\s*([a-z_]+)\s*\(\s*(a\d+)\s*,\s*(a\d+)\s*\)\s*$",
                      re.I)


def lines(cell):
    return [x.strip() for x in re.split(r"[\n;]", cell or "") if x.strip()]


def parse_actions(cell):
    out, bad = [], []
    for i, ln in enumerate(lines(cell)):
        m = ACTION.match(ln)
        if not m:
            bad.append(ln)
            continue
        q = {}
        if m.group(3):
            try:
                q = ast.literal_eval(m.group(3))
                q = q[0] if isinstance(q, list) and q else q
            except (ValueError, SyntaxError):
                bad.append(f"qualifiers unparseable: {ln}")
        out.append({"id": f"a{i + 1}", "verb": m.group(1).lower(),
                    "object": (m.group(2).strip() or None),
                    "qualifiers": q if isinstance(q, dict) else {},
                    "importance": "core"})
    return out, bad


def parse_entities(cell):
    """Entities are one name per line, positional: line 1 is e1."""
    return [{"id": f"e{i + 1}", "name": n, "importance": "core"}
            for i, n in enumerate(lines(cell))]


def parse_temporal(cell):
    out, bad = [], []
    for ln in lines(cell):
        if ln.lower() == "none":
            continue
        m = TEMPORAL.match(ln)
        if not m:
            bad.append(ln)
            continue
        out.append({"relation": m.group(1).lower(), "first": m.group(2),
                    "second": m.group(3), "importance": "stated"})
    return out, bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filled",
                    default="data/gold/atomic_claims_filled_288.csv")
    ap.add_argument("--draft", default="data/gold/atomic_claims_draft.json")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.filled, newline="",
                                           encoding="utf-8-sig"))
            if (r.get("stored_label") or "").strip()]
    draft = json.load(open(a.draft, encoding="utf-8"))["claims"]
    print(f"{len(rows)} labels in the filled sheet; {len(draft)} in the draft")

    frozen, src = {}, Counter()
    problems = []
    for r in rows:
        lab = r["stored_label"]
        has = any((r.get(c) or "").strip()
                  for c in ("corrected_actions", "corrected_entities",
                            "corrected_temporal"))
        if not has:
            d = draft.get(lab)
            if d is None:
                problems.append((lab, "untouched but not in the draft"))
                continue
            frozen[lab] = dict(d, source="draft_accepted")
            src["draft_accepted"] += 1
            continue
        # PER-FIELD FALLBACK TO THE DRAFT. The sheet says "leave the cell
        # blank if the draft is right", and the annotator applied that per
        # COLUMN, not per row: `Rinse mug after scrubbing` has corrected
        # actions and a blank entities cell because `e1=mug` was already
        # correct. Reading a blank column as "no entities" produced an action
        # naming an entity that did not exist -- the one validation failure,
        # and it was my merge rule rather than the annotation.
        d = draft.get(lab) or {}
        if (r.get("corrected_actions") or "").strip():
            acts, bad_a = parse_actions(r.get("corrected_actions"))
        else:
            acts, bad_a = list(d.get("actions") or []), []
        if (r.get("corrected_entities") or "").strip():
            ents = parse_entities(r.get("corrected_entities"))
        else:
            ents = list(d.get("entities") or [])
        if (r.get("corrected_temporal") or "").strip():
            temp, bad_t = parse_temporal(r.get("corrected_temporal"))
        else:
            temp, bad_t = list(d.get("temporal_constraints") or []), []
        for b in bad_a:
            problems.append((lab, f"action not parseable: {b}"))
        for b in bad_t:
            problems.append((lab, f"temporal not parseable: {b}"))
        # REFERENTIAL CHECKS. A claim naming an entity that was never listed
        # is a claim about nothing, and a verifier would score it silently.
        eids = {e["id"] for e in ents}
        aids = {x["id"] for x in acts}
        for x in acts:
            if x["object"] and x["object"] not in eids:
                problems.append((lab, f"{x['id']} names {x['object']}, "
                                      f"entities are {sorted(eids) or 'none'}"))
        for t in temp:
            for k in ("first", "second"):
                if t[k] not in aids:
                    problems.append((lab, f"temporal names {t[k]}, actions "
                                          f"are {sorted(aids)}"))
        frozen[lab] = {"label": lab, "entities": ents, "actions": acts,
                       "spatial_relations": [], "temporal_constraints": temp,
                       "source": "human_corrected",
                       "fields_from_draft": [
                           k for k, c in (("actions", "corrected_actions"),
                                          ("entities", "corrected_entities"),
                                          ("temporal", "corrected_temporal"))
                           if not (r.get(c) or "").strip()],
                       "note": (r.get("note") or "").strip(),
                       "occurrences": int(r.get("occurrences") or 0)}
        src["human_corrected"] += 1

    print(f"\n  {dict(src)}")
    n_act = Counter(len(v["actions"]) for v in frozen.values())
    n_ent = Counter(len(v["entities"]) for v in frozen.values())
    print(f"  actions per label:  {dict(sorted(n_act.items()))}")
    print(f"  entities per label: {dict(sorted(n_ent.items()))}")
    q = Counter(k for v in frozen.values() for x in v["actions"]
                for k in x["qualifiers"])
    print(f"  qualifier keys: {dict(q.most_common())}")
    t = Counter(t["relation"] for v in frozen.values()
                for t in v["temporal_constraints"])
    print(f"  temporal relations: {dict(t.most_common())} over "
          f"{sum(1 for v in frozen.values() if v['temporal_constraints'])} "
          f"labels")
    verbs = Counter(x["verb"] for v in frozen.values() for x in v["actions"])
    print(f"  {len(verbs)} distinct verbs; top: "
          f"{[k for k, _ in verbs.most_common(12)]}")
    only1 = [k for k, c in verbs.items() if c == 1]
    print(f"  {len(only1)} verbs appear once -- an atomic verifier has one "
          f"example of each")

    print(f"\nVALIDATION: {len(problems)} problems")
    for lab, p in problems[:12]:
        print(f"    {lab[:44]:<46}{p}")
    if len(problems) > 12:
        print(f"    ... and {len(problems) - 12} more")
    if not problems:
        print("    every action's entity and every constraint's action "
              "resolve. Nothing dangles.")

    if a.out:
        json.dump({"n_labels": len(frozen),
                   "source_counts": dict(src),
                   "problems": [{"label": l, "problem": p}
                                for l, p in problems],
                   "claims": frozen},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
