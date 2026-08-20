"""Human audit sheets -> joint_policy events. And what the sheets cannot supply.

`joint_policy` consumes EVIDENCE and emits a DECISION. Most of what this
project has collected is decisions, so the first job here is to say which
columns are which, out loud, before anything is converted.

    evidence   morphology, interaction_continuity, release_reset_restart,
               observability, and the two CROSS support terms
    decision   instance_relation, task_boundary, claim_support

BATCH4 CANNOT FEED THIS, and the reason is worth stating because it looks like
it can. Its `interaction_relation` is the answer -- instance_relation_policy_v2
maps it to a decision in one line -- so feeding it in and comparing the output
is grading an answer against itself. And its two semantic columns are the
DIAGONAL terms, each segment against its OWN label. The policy's continuity
evidence is the CROSS terms: the left segment against the RIGHT label and vice
versa. A diagonal term cannot be turned into a cross term, so batch4 supplies
neither side of the semantic input.

WHICH MEANS THE CURRENT SHEETS CANNOT PRODUCE AN AUTO_REJECT_CANDIDATE AT ALL.
That conjunction requires `semantic_evidence: compatible_with_continuity`, which
requires both cross terms, which no audit sheet has ever collected. The
all-REVIEW result below is therefore a property of the schema, not of the
videos, and it is cheap to fix in the remaining packets and impossible to
backfill.

THE v1 SPAN SHEET KEEPS ITS RESET EVIDENCE IN PROSE. Recordings 176/242/250
returned before `release_reset_restart` was a column; both YES cases say
"disengagement" in a notes field. Reading that with a regex and calling it data
is exactly the move this project has spent months undoing, so it is opt-in via
--extract_reset_from_notes, every extraction prints, and each converted event
records `reset_source` so a downstream reader can drop them.

Usage:
    python -m src.auditor.joint_events \
        --span data/gold/bridge_returned/bridge_span_176_242_250.csv \
        --out /tmp/joint_events.jsonl
    python -m src.auditor.joint_events --span ... --extract_reset_from_notes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter

from src.auditor.joint_policy import ABSENT, PRESENT, UNKNOWN, decide, load_config

# v2 sheet values -> the policy's tri-state. `uncertain` and `not_observable`
# are different information and the same decision: neither is an observation of
# absence, so neither licenses a reject.
RESET_MAP = {
    "observed_present": PRESENT,
    "observed_absent": ABSENT,
    "uncertain": UNKNOWN,
    "not_observable": UNKNOWN,
}

# Prose that indicates a reset in the v1 sheets. Deliberately narrow: a phrase
# list that grows until it matches everything is a way of deciding the answer.
#
# NEGATION IS CHECKED FIRST, and it is not a nicety. The first version matched
# `reset` inside "no task-level reset" and read three of its four hits
# backwards -- every one of them a human NO turned into KEEP_BOUNDARY, all in
# the direction of manufacturing a boundary. The flag's own printing is what
# caught it.
#
# And the negated form is the more valuable one: "no task-level reset", written
# by someone who watched, IS an observation of absence -- the OBSERVED_ABSENT
# that is the only value licensing a reject.
_RESET = r"(?:disengag\w*|hands? (?:withdraw|leave|come off)|reset|restart)"
RESET_ABSENT_PROSE = re.compile(
    r"\b(?:no|without|never|not)\b[^.;]{0,24}\b" + _RESET, re.I)
RESET_PRESENT_PROSE = re.compile(r"\b" + _RESET + r"\b", re.I)


def read_sheet(path):
    """Bridge sheets carry a title row above the header."""
    rows = list(csv.reader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        return []
    hdr = rows[0]
    body = rows[1:]
    # A title row has one filled cell and the rest empty.
    if sum(1 for c in hdr if c.strip()) <= 2 and len(rows) > 1:
        hdr, body = rows[1], rows[2:]
    return [dict(zip(hdr, r)) for r in body if r and any(c.strip() for c in r)]


def convert(row, extract_prose, missing):
    """One span row -> (boundary_evidence, semantic_evidence, human, source)."""
    v2 = "release_reset_restart" in row
    notes = " ".join(str(row.get(k) or "") for k in ("notes", "human_a",
                                                     "human_b"))

    if v2:
        reset = RESET_MAP.get((row.get("release_reset_restart") or "").strip()
                              .lower(), UNKNOWN)
        src = "column"
    elif extract_prose and RESET_ABSENT_PROSE.search(notes):
        reset = ABSENT
        src = "notes_regex_negated"
        print(f"    reset ABSENT from prose: {row.get('recording')}/"
              f"{row.get('span_id')}  <- "
              f"{RESET_ABSENT_PROSE.search(notes).group(0)!r}")
    elif extract_prose and RESET_PRESENT_PROSE.search(notes):
        reset = PRESENT
        src = "notes_regex"
        print(f"    reset PRESENT from prose: {row.get('recording')}/"
              f"{row.get('span_id')}  <- "
              f"{RESET_PRESENT_PROSE.search(notes).group(0)!r}")
    else:
        reset = UNKNOWN
        src = "absent_from_v1_schema"
        missing["release_reset_restart"] += 1

    # EVIDENCE THE SHEET HAS NEVER CARRIED. Recorded as missing rather than
    # defaulted: `continuous` would be an assertion nobody made, and it is one
    # of the five conditions an auto-reject needs.
    for k in ("morphology", "interaction_continuity", "observability"):
        missing[k] += 1

    bnd = {"release_reset_restart": reset}

    # The sheet's a_semantic / b_semantic are DIAGONAL. Left blank on purpose:
    # `semantic_compatible_with_continuity` then reports "a cross-support term
    # is missing", which is true, instead of a number derived from the wrong
    # quantity.
    sem = {"label_L": row.get("clause_a"), "label_R": row.get("clause_b")}
    missing["support_L_labelR"] += 1
    missing["support_R_labelL"] += 1

    human = (row.get("task_boundary") or row.get("join_relation") or "").strip()
    return bnd, sem, human, src


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--span", action="append", required=True)
    ap.add_argument("--extract_reset_from_notes", action="store_true",
                    help="read reset evidence out of the v1 prose. OFF by "
                         "default: a regex over a notes field is not an "
                         "observation, and every match prints so it can be "
                         "checked")
    ap.add_argument("--config", default="configs/auditor/joint_policy_v1.yaml")
    ap.add_argument("--out")
    a = ap.parse_args()

    cfg = load_config(a.config)
    rows = []
    for p in a.span:
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found")
        r = read_sheet(p)
        print(f"{p}: {len(r)} rows, "
              f"{'v2 (structured)' if r and 'release_reset_restart' in r[0] else 'v1 (prose)'}")
        rows += r
    if not rows:
        raise SystemExit("no rows")

    if a.extract_reset_from_notes:
        print("\n  !! --extract_reset_from_notes is ON. Reset evidence below "
              "comes from a regex\n     over prose, not from an observation "
              "anyone recorded as one.")

    missing = Counter()
    out, tab = [], Counter()
    print()
    for r in rows:
        bnd, sem, human, src = convert(r, a.extract_reset_from_notes, missing)
        d = decide(bnd, sem, cfg)
        tab[(human or "(blank)", d["action"])] += 1
        out.append({"recording_id": r.get("recording"),
                    "span_id": r.get("span_id"),
                    "boundary": bnd, "semantic": sem,
                    "reset_source": src, "human_verdict": human,
                    "decision": d})

    print(f"\n{len(out)} events converted\n")
    print(f"  {'human':<12}{'joint_policy':<26}{'n':>4}")
    for (h, act), n in sorted(tab.items(), key=lambda x: -x[1]):
        print(f"  {h:<12}{act:<26}{n:>4}")

    print(f"\n  evidence the sheet never carried, per event:")
    for k, v in missing.most_common():
        print(f"    {k:<26}{v:>4}/{len(out)}")

    acts = Counter(o["decision"]["action"] for o in out)
    if not acts.get("AUTO_REJECT_CANDIDATE"):
        print(f"\n  NO EVENT REACHED AUTO_REJECT_CANDIDATE, and that is the "
              f"schema, not the video.\n  The conjunction needs "
              f"`semantic_evidence: compatible_with_continuity`, which needs "
              f"both\n  CROSS support terms -- the left segment against the "
              f"RIGHT label and the reverse.\n  No audit sheet in this project "
              f"has ever collected them; the sheets collect the\n  DIAGONAL "
              f"terms, each segment against its own label, and a diagonal term "
              f"cannot\n  be turned into a cross term.")
        print(f"\n  That is cheap to add to the remaining packets and "
              f"impossible to backfill:\n  two extra judgements per span, "
              f"`does label_R describe segment A` and the reverse.")

    keep = [o for o in out if o["decision"]["action"] == "KEEP_BOUNDARY"]
    if keep:
        print(f"\n  {len(keep)} reached KEEP_BOUNDARY through precedence-1 "
              f"(reset observed):")
        for o in keep:
            print(f"    {o['recording_id']}/{o['span_id']}  human said "
                  f"{o['human_verdict']!r}  reset from {o['reset_source']}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
