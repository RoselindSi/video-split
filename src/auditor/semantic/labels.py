"""Semantic v1 label layer -- and what the gold cannot supervise.

The plan for the semantic side was claim-level verification: separate heads for
P(primary verb supported), P(object supported), P(each secondary supported),
then a fixed ontology over them. That plan does not survive contact with the
gold, and it is better to find out here than after building it.

WHAT THE GOLD ACTUALLY CONTAINS. Four fields look like independent axes and
are one judgement written four times:

    label_completeness   6 values, H = 2.00 bits   the real axis
    label_granularity    adds 0.41 bits            almost entirely the
                                                   too_fine split inside
                                                   `complete`
    label_support        adds 0.00 bits            a coarsening of
                                                   completeness; it carries
                                                   nothing completeness does
                                                   not already say
    object_relation      adds 0.61 bits            6 non-`same` events in 188

The completeness x granularity table is diagonal. `missing_secondary` occurs
only with `too_coarse`, `partially_correct` only with `mixed`, `incorrect` and
`wrong_object` only with `not_applicable`. The single genuine branch is
`complete` splitting into appropriate (67) and too_fine (25).

SO THERE ARE NO PER-CLAIM JUDGEMENTS TO TRAIN ON. Three heads fitted against
these columns would be three views of one label, and their agreement would be
a property of the schema rather than evidence about the video. That is the
same entanglement the boundary side spent months inside; it is visible here on
day one only because the boundary work taught us to look.

THE ONTOLOGY IS ALREADY A CLEAN FUNCTION, unlike the boundary one. Seven
distinct field combinations map to six statuses with zero conflicts, so the
mapping can be extracted and frozen rather than invented. It is verified on
load: a combination that maps to two statuses would mean the gold disagrees
with itself and would be reported rather than resolved.

WHAT REMAINS TRAINABLE, honestly:

    the AUTO_ACCEPT decision    `correct` against everything else, 67 vs 121
    the six-way status          67 / 36 / 34 / 25 / 23 / 3 -- the last class
                                cannot be scored at all
    too_fine                    25 events, and per the architecture note this
                                is an oversegmentation signal that belongs to
                                a boundary-semantic cross-check rather than to
                                a semantic head deciding alone

Usage:
    python -m src.auditor.semantic.labels \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --out data/gold/semantic_v1_labels.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

FIELDS = ["label_support", "label_completeness", "label_granularity",
          "object_relation"]
STATUS = "legacy_semantic_label_status"
ACCEPT = "correct"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def entropy(counter, n):
    return -sum(v / n * math.log2(v / n) for v in counter.values() if v)


def mutual_info(rows, a, b):
    n = len(rows)
    pair = Counter((r.get(a), r.get(b)) for r in rows)
    ca, cb = Counter(r.get(a) for r in rows), Counter(r.get(b) for r in rows)
    return sum(v / n * math.log2((v / n) / ((ca[k[0]] / n) * (cb[k[1]] / n)))
               for k, v in pair.items() if v), entropy(ca, n), entropy(cb, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = read_jsonl(a.gold)
    ctx = {r["event_id"]: r for r in read_jsonl(a.context)} \
        if os.path.exists(a.context) else {}
    n = len(rows)
    print(f"{n} audited events over "
          f"{len({r['recording_id'] for r in rows})} recordings")

    # ---------------------------------------------------- the ontology table
    print(f"\n{'=' * 84}\nTHE ONTOLOGY ALREADY IN THE GOLD\n{'=' * 84}")
    m = defaultdict(Counter)
    for r in rows:
        m[tuple(r.get(k) for k in FIELDS[:3])][r.get(STATUS)] += 1
    bad = {k: v for k, v in m.items() if len(v) > 1}
    print(f"  {'support':<14}{'completeness':<20}{'granularity':<16}"
          f"{'status':<28}{'n':>4}")
    table = {}
    for k in sorted(m, key=lambda x: tuple(str(v) for v in x)):
        st = m[k].most_common(1)[0][0]
        table["|".join(str(v) for v in k)] = st
        print(f"  {str(k[0]):<14}{str(k[1]):<20}{str(k[2]):<16}{str(st):<28}"
              f"{sum(m[k].values()):>4}")
    if bad:
        print(f"\n  !! {len(bad)} combinations map to more than one status. "
              f"The gold disagrees with itself and the mapping cannot be\n"
              f"     frozen until those are resolved:")
        for k, v in list(bad.items())[:5]:
            print(f"     {k} -> {dict(v)}")
    else:
        print(f"\n  {len(m)} combinations, {len(set(table.values()))} "
              f"statuses, no conflicts. The mapping is a function and can be "
              f"frozen as config\n  rather than reinvented -- which is more "
              f"than the boundary ontology managed.")

    # -------------------------------------------------------- collinearity
    print(f"\n{'=' * 84}\nHOW MUCH EACH FIELD ADDS\n{'=' * 84}")
    print(f"  {'pair':<52}{'H(b)':>8}{'MI':>8}{'b adds':>10}")
    for x, y in (("label_completeness", "label_granularity"),
                 ("label_completeness", "label_support"),
                 ("label_completeness", "object_relation"),
                 ("label_completeness", STATUS)):
        mi, ha, hb = mutual_info(rows, x, y)
        print(f"  {y + ' beyond ' + x:<52}{hb:>8.2f}{mi:>8.2f}"
              f"{hb - mi:>10.2f}")
    print(f"\n  A field adding near zero is the same judgement recorded twice. "
          f"Heads fitted against those columns separately would\n  agree "
          f"because the schema made them agree, not because the video did.")
    print(f"\n  completeness x granularity, the table that shows it:")
    gr = sorted({r.get("label_granularity") for r in rows}, key=str)
    t = defaultdict(Counter)
    for r in rows:
        t[r.get("label_completeness")][r.get("label_granularity")] += 1
    print("  " + f"{'':<20}" + "".join(f"{str(g)[:14]:>16}" for g in gr))
    for k in sorted(t, key=str):
        print("  " + f"{str(k):<20}"
              + "".join(f"{t[k].get(g, 0):>16}" for g in gr))

    # ------------------------------------------------------ what is trainable
    print(f"\n{'=' * 84}\nWHAT CAN BE SUPERVISED\n{'=' * 84}")
    st = Counter(r.get(STATUS) for r in rows)
    recs = {s: len({r["recording_id"] for r in rows if r.get(STATUS) == s})
            for s in st}
    print(f"  {'status':<30}{'n':>5}{'recordings':>12}  reportable")
    for s, c in st.most_common():
        ok = c >= a.n_folds and recs[s] >= a.n_folds
        print(f"  {str(s):<30}{c:>5}{recs[s]:>12}  "
              f"{'yes' if ok else 'NO -- fewer than folds'}")
    acc = sum(1 for r in rows if r.get(STATUS) == ACCEPT)
    acc_rec = len({r["recording_id"] for r in rows if r.get(STATUS) == ACCEPT})
    print(f"\n  the AUTO_ACCEPT binary, `{ACCEPT}` against everything else: "
          f"{acc} / {n - acc}, over {acc_rec} recordings")
    print(f"  That is the decision the product actually makes, and it is the "
          f"only semantic target here with both classes large enough\n  to "
          f"survive a recording-grouped split.")
    tf = sum(1 for r in rows if r.get("label_granularity") == "too_fine")
    print(f"\n  too_fine: {tf} events. It is the oversegmentation signal, and "
          f"it should NOT be decided by a semantic head alone --\n  whether a "
          f"segment was cut too finely is a claim about the boundary, and the "
          f"cross-check with the boundary side is\n  where it belongs.")

    obj = Counter(r.get("object_relation") for r in rows)
    print(f"\n  object_relation: {dict(obj.most_common())}. "
          f"{n - obj.get('same', 0)} events are not `same`, which cannot "
          f"support an\n  object-claim head however the plan described it.")

    if a.out:
        out = []
        for r in rows:
            e = r["event_id"]
            c = ctx.get(e, {})
            out.append({
                "event_id": e, "recording_id": r["recording_id"],
                "status": r.get(STATUS),
                "auto_accept_target": int(r.get(STATUS) == ACCEPT),
                "label_support": r.get("label_support"),
                "label_completeness": r.get("label_completeness"),
                "label_granularity": r.get("label_granularity"),
                "object_relation": r.get("object_relation"),
                "corrected_primary_verb": r.get("corrected_primary_verb"),
                "corrected_object": r.get("corrected_object"),
                "corrected_secondary_verbs": r.get("corrected_secondary_verbs"),
                "prev_segment_label": c.get("prev_segment_label"),
                "next_segment_label": c.get("next_segment_label"),
                "containing_segment_label": c.get("containing_segment_label"),
            })
        json.dump({"ontology": table, "n": n, "events": out},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}. The extracted ontology travels with it, so "
              f"the mapping is frozen alongside the labels it came from.")


if __name__ == "__main__":
    main()
