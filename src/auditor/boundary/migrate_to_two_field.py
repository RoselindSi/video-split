"""Carry every existing human audit into (instance_relation, transition_shape).

The migration is deliberately lossy in ONE direction and the output says where.

`transition_shape` is largely recoverable: the old subtypes and the blind
topology audit both describe what the change looked like. `instance_relation`
is mostly NOT recoverable, because the old schema never asked it --
`sharp_visible_transition` is silent on whether the new thing is a new action
or the next repetition of the old one, and that is exactly the question 21 of
48 blind-audited batch3 events turn on. So the UNKNOWN count on
instance_relation is not a defect of this script. It is the measurement of how
much annotation the new schema still needs, and inventing a value would erase
the number the whole exercise is for.

FIVE SOURCES, in increasing order of precedence, so a later human pass
overrides an earlier derivation rather than being averaged with it:

  1 the canonical pair labels        seven-way subtype
  2 the 90-event batch3 relabel      seven-way subtype, human, from video
  3 the 37 blind topology calls      shape only: smooth_ramp, overlap,
                                     multi_step, point_like
  4 the 37 ontology re-audit         seven-way subtype, task-granularity
  5 the 48 blind double audit        call plus a free-text reason

SOURCE 5 IS PARSED FROM PROSE AND IS THE ONLY HEURISTIC HERE. The annotator
wrote why, in sentences, and the reasons fall into recognisable groups -- "两个
动作之间手短暂idle", "动作结束到idle", "画面外". Keyword rules turn those into
a relation, and every derived row carries the sentence it came from so the
derivation can be checked rather than trusted. Rows whose prose matches nothing
are left UNKNOWN.

NOTHING IS WRITTEN INTO THE CANONICAL LABELS. This produces a separate
two-field table plus a coverage report; deciding what a
`same_action_new_instance` should DO is a policy question and lives in the
ontology, not in the annotation.

Usage:
    python -m src.auditor.boundary.migrate_to_two_field \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --topology_37 data/gold/interval_audit_37_filled.csv \
        --ontology_37 data/gold/interval_audit_37_ontology_revised.csv \
        --double_48 data/gold/batch3_double_audit_annotator1_filled.csv \
        --out data/gold/pair_schema_v2_migrated.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

UNKNOWN = "UNKNOWN"

SHAPE_FROM_TOPOLOGY = {"point_like": "point", "smooth_ramp": "gradual",
                       "overlapping_transition": "overlap",
                       "multi_step_transition": "gap",
                       "ambiguous_interval": UNKNOWN}

# source 5 only. Ordered: the first group whose words appear wins, so the
# specific patterns precede the general ones.
PROSE_RULES = [
    ("cannot_determine", "not_observable",
     ["画面外", "画面里", "观测不足", "不可见", "无法判断"]),
    ("terminal_action_end", "gap",
     ["视频到这里结束", "动作结束到idle", "之后继续idle", "没有后续动作",
      "候选附近没有"]),
    ("same_action_new_instance", "gap",
     ["两个动作实例之间", "两个动作之间", "再开始下一次", "再次取出",
      "重新取", "循环", "反复", "依次取出再放回", "重复", "下一次"]),
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    return next((v for k, v in row.items()
                 if k and k.startswith(prefix)), "") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="configs/auditor/pair_schema_v2.yaml")
    ap.add_argument("--pair_labels", action="append", default=[])
    ap.add_argument("--topology_37")
    ap.add_argument("--ontology_37")
    ap.add_argument("--double_48")
    ap.add_argument("--relation_audit",
                    help="the 54-row sheet that asked ONLY the relation. "
                         "Highest precedence: it is the only pass that put the "
                         "question directly")
    ap.add_argument("--out")
    a = ap.parse_args()

    import yaml
    sch = yaml.safe_load(open(a.schema, encoding="utf-8"))
    legacy = sch["legacy"]
    valid_r = set(sch["instance_relation"]) | {UNKNOWN}
    # the relation audit introduced initial_action_start, so a stale schema
    # would reject the file rather than silently dropping the value
    valid_s = set(sch["transition_shape"]) | {UNKNOWN}

    # every claim is kept, not just the winner. A count that changes between
    # the parse and the output -- 26 prose matches becoming 24 -- has to be
    # explainable event by event, and it cannot be if only the winner survives.
    claims = defaultdict(list)

    def put(eid, rel, shape, src, evidence="", detail=""):
        claims[eid].append({"source": src, "instance_relation": rel or UNKNOWN,
                            "transition_shape": shape or UNKNOWN,
                            "evidence": (evidence or "")[:160],
                            "detail": detail})

    n_src = Counter()
    for p in a.pair_labels:
        if not os.path.exists(p):
            continue
        for row in read_csv(p):
            e, s = row.get("event_id"), (row.get("temporal_pair_subtype") or "").strip()
            if not e or s not in legacy:
                continue
            n_src[os.path.basename(p)] += 1
            put(e, legacy[s]["instance_relation"], legacy[s]["transition_shape"],
                f"legacy:{os.path.basename(p)}")

    if a.topology_37 and os.path.exists(a.topology_37):
        for row in read_csv(a.topology_37):
            e, c = row.get("event_id"), col(row, "your_call").strip()
            if e and c in SHAPE_FROM_TOPOLOGY:
                n_src["topology_37"] += 1
                # shape only: this audit never asked about the relation
                put(e, UNKNOWN, SHAPE_FROM_TOPOLOGY[c], "topology_37",
                    col(row, "why_one_line"))

    if a.ontology_37 and os.path.exists(a.ontology_37):
        for row in read_csv(a.ontology_37):
            e = row.get("event_id")
            s = (row.get("revised_temporal_pair_subtype") or "").strip()
            if e and s in legacy:
                n_src["ontology_37"] += 1
                put(e, legacy[s]["instance_relation"],
                    legacy[s]["transition_shape"], "ontology_37",
                    row.get("ontology_reason", ""))

    prose_hits = Counter()
    conflict_48 = []
    if a.double_48 and os.path.exists(a.double_48):
        for row in read_csv(a.double_48):
            e = row.get("event_id")
            c = col(row, "your_call").strip().lower()
            note = (row.get("notes") or "").strip()
            if not e:
                continue
            n_src["double_48"] += 1
            p_rel = p_shape = UNKNOWN
            for r_, s_, words in PROSE_RULES:
                if any(w in note for w in words):
                    p_rel, p_shape = r_, s_
                    prose_hits[r_] += 1
                    break
            else:
                prose_hits["no rule matched"] += 1
            # the explicit call and my keyword parse of the prose are two
            # different claims by the same annotator, and where they disagree
            # that disagreement is the finding -- three of the six
            # terminal_action_end notes carry an explicit `same`. The call
            # wins because it is what the person answered, but the conflict is
            # recorded rather than resolved away.
            rel, shape, why = p_rel, p_shape, "prose"
            if c == "same":
                rel, shape, why = "same_instance", "not_applicable", "call"
            elif c == "cannot":
                rel, shape, why = "cannot_determine", "not_observable", "call"
            elif c == "sharp" and shape == UNKNOWN:
                shape, why = "point", "call"
            if why == "call" and p_rel != UNKNOWN and p_rel != rel:
                conflict_48.append((e, p_rel, c, rel, note))
            put(e, rel, shape, "double_48", note,
                f"call={c} prose={p_rel} resolved_by={why}")

    if a.relation_audit and os.path.exists(a.relation_audit):
        for row in read_csv(a.relation_audit):
            e = row.get("event_id")
            rel = col(row, "your_call").strip()
            if not e or not rel:
                continue
            n_src["relation_audit_54"] += 1
            # relation only. This pass never looked at the shape, and writing
            # one here would put an unobserved value beside a directly asked
            # answer and make them indistinguishable later.
            put(e, rel, UNKNOWN, "relation_audit_54",
                col(row, "why_one_line"),
                f"confidence={col(row, 'confidence')}")

    print(f"rows read per source: {dict(n_src)}")
    if prose_hits:
        print(f"  double_48 prose rules: {dict(prose_hits)}")

    # -------------------------------------------------------- resolution
    # last source wins, which is the order the sources were added: legacy,
    # then each human pass in the order it was run
    rows = []
    cross = []
    for e, cl in sorted(claims.items()):
        r = {"event_id": e, "instance_relation": UNKNOWN,
             "transition_shape": UNKNOWN, "relation_source": "",
             "shape_source": "", "evidence": "", "detail": ""}
        for c_ in cl:
            if c_["instance_relation"] != UNKNOWN:
                r["instance_relation"] = c_["instance_relation"]
                r["relation_source"] = c_["source"]
            if c_["transition_shape"] != UNKNOWN:
                r["transition_shape"] = c_["transition_shape"]
                r["shape_source"] = c_["source"]
            if c_["evidence"]:
                r["evidence"] = c_["evidence"]
            if c_["detail"]:
                r["detail"] = c_["detail"]
        seen = {c_["instance_relation"] for c_ in cl
                if c_["instance_relation"] != UNKNOWN}
        if len(seen) > 1:
            cross.append((e, [(c_["source"], c_["instance_relation"])
                              for c_ in cl
                              if c_["instance_relation"] != UNKNOWN],
                          r["instance_relation"], r["relation_source"]))
        rows.append(r)

    print(f"\n{'=' * 78}\nWHERE THE COUNTS MOVE\n{'=' * 78}")
    print(f"  {len(claims)} events carry {sum(len(v) for v in claims.values())} "
          f"claims from {len(n_src)} sources; last source wins")
    if conflict_48:
        print(f"\n  WITHIN double_48, the explicit call contradicted my parse "
              f"of the annotator's own prose on {len(conflict_48)} rows.")
        print(f"  The call wins -- it is what the person answered -- and this "
              f"is where the prose-rule counts shrink:")
        for e, p_rel, c_, rel, note in conflict_48:
            print(f"    {e[-42:]:<43} prose {p_rel} vs call `{c_}` -> {rel}")
            print(f"      \"{note[:96]}\"")
        print(f"  Three of these are the terminal_action_end notes. One person, "
              f"one day, writing that the action ends into idle and\n  then "
              f"answering `same` -- which is what an undefined case looks like "
              f"from the inside, not a careless row.")
    if cross:
        print(f"\n  ACROSS sources, {len(cross)} events carry more than one "
              f"instance_relation:")
        for e, hist, sel, src in cross[:10]:
            print(f"    {e[-42:]:<43} {hist} -> {sel} ({src})")
    else:
        print(f"\n  No event received conflicting instance_relation values "
              f"from different sources.")
    bad = [r for r in rows if r["instance_relation"] not in valid_r
           or r["transition_shape"] not in valid_s]
    if bad:
        raise SystemExit(f"{len(bad)} rows carry a value outside the schema, "
                         f"e.g. {bad[0]}")
    # the constraints are checked rather than enforced, so a violation is
    # reported as a real disagreement between sources instead of being
    # silently rewritten
    viol = [r for r in rows
            if (r["instance_relation"] == "same_instance"
                and r["transition_shape"] not in ("not_applicable", UNKNOWN))
            or (r["instance_relation"] == "cannot_determine"
                and r["transition_shape"] not in ("not_observable", UNKNOWN))]

    print(f"\n{'=' * 78}\nCOVERAGE OF THE TWO FIELDS\n{'=' * 78}")
    for f in ("instance_relation", "transition_shape"):
        c = Counter(r[f] for r in rows)
        known = len(rows) - c[UNKNOWN]
        print(f"\n  {f}: {known}/{len(rows)} known ({known / len(rows):.1%})")
        for k, n in c.most_common():
            print(f"    {k:<28} {n:>5}")
    print(f"\n  Both fields known: "
          f"{sum(1 for r in rows if r['instance_relation'] != UNKNOWN and r['transition_shape'] != UNKNOWN)}"
          f"/{len(rows)}")
    print(f"  The instance_relation gap is the point of this table. The old "
          f"schema never asked it, so it cannot be derived --\n  "
          f"`sharp_visible_transition` is silent on whether the new thing is a "
          f"new action or the next repetition, and that is the\n  question the "
          f"repeated-instance cases turn on. That count is the annotation "
          f"still owed, not a bug here.")

    if viol:
        print(f"\n  !! {len(viol)} rows violate a schema constraint, which "
              f"means two sources disagree rather than that the migration is\n"
              f"     wrong. Reported rather than rewritten:")
        for r in viol[:5]:
            print(f"     {r['event_id'][-44:]:<45} "
                  f"{r['instance_relation']} + {r['transition_shape']} "
                  f"({r['relation_source']} / {r['shape_source']})")

    known_rel = [r for r in rows if r["instance_relation"] != UNKNOWN]
    if known_rel:
        print(f"\n{'=' * 78}\nWHERE instance_relation COMES FROM\n{'=' * 78}")
        for k, n in Counter(r["relation_source"] for r in known_rel).most_common():
            print(f"  {n:>5}  {k}")
        print(f"\n  `legacy:*` rows are all `same_instance`, which is the one "
              f"relation the old schema could express. Every\n  "
              f"new_action / same_action_new_instance / terminal_action_end "
              f"came from a pass that asked.")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["event_id", "instance_relation",
                                   "transition_shape", "relation_source",
                                   "shape_source", "evidence", "detail"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out} ({len(rows)} events). The canonical pair "
              f"labels are untouched: what a same_action_new_instance should "
              f"DO\n  is a policy decision and belongs in the ontology file.")


if __name__ == "__main__":
    main()
