"""Two independent annotators and the stored label, compared three ways.

The single most informative comparison this project can make, and it needed no
new data: two people answered sharp/same/cannot on the same 36 REVIEW-band
events, blind, and the taxonomy label already exists for all of them.

WHY THE THREE-WAY MATTERS AND A PAIRWISE NUMBER DOES NOT. Annotator 1 agreed
with the label on 21 of 33 and the reading was "the target is not
reproducible". That reading has a competitor: the label is the outlier. Only
the third view separates them, and it separates them decisively --

    if the humans agree with each other LESS than each agrees with the label,
    the decision is genuinely hard and the label is a fair summary of it

    if the humans agree with each other MORE than either agrees with the
    label, the humans are reproducing something the label does not encode,
    and the label is the thing to fix

The events where BOTH annotators agree with each other AND differ from the
label are reported by name. Two independent votes against a stored label is
the strongest evidence about that label this data can produce, and it is not
an aggregate -- those specific rows are actionable.

DIRECTION IS REPORTED, NOT JUST COUNTS. Disagreements that run one way are a
calibration difference: one annotator's threshold for "a transition happened"
sits elsewhere, which is fixable by a definition. Disagreements that run both
ways are a convention conflict, which is not. The two demand different
remedies, so a bare agreement rate is not enough to choose one.

Cohen's kappa is reported beside raw agreement because raw agreement on a
near-balanced binary already starts at 0.5 by chance, and 0.64 sounds far
better than the kappa 0.27 it corresponds to.

`cannot` is a third category, never silently dropped. Dropping it inflates
agreement by removing exactly the events people found hardest.

Usage:
    python -m src.boundary.c3_annotator_agreement \
        --sheet data/gold/observable_audit_your_call_36.csv \
        --sheet data/gold/observable_audit_annotator2_36.csv \
        --key /workspace/tr1/results/hal/c3/observable_audit/audit_key.csv \
        --out /workspace/tr1/results/hal/c3/observable_audit/annotator_agreement.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.c3_selective_policy import wilson

DECIDED = ("sharp", "same")


def read_csv(p):
    # utf-8-sig: a reviewer-edited CSV carries a BOM and the first column name
    # arrives as "﻿your_call(...)", so a prefix lookup silently misses.
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    return next((k for k in row if k.startswith(prefix)), None)


def load_sheet(p):
    rows = read_csv(p)
    c_call, c_conf = col(rows[0], "your_call"), col(rows[0], "confidence")
    c_ans = col(rows[0], "answer")
    if not c_call:
        raise SystemExit(f"{p}: no your_call column")
    return {r["event_id"]: {
        "call": (r.get(c_call) or "").strip().lower(),
        "conf": (r.get(c_conf) or "").strip(),
        "ans": (r.get(c_ans) or "").strip() if c_ans else "",
    } for r in rows}


def kappa(a, b):
    """Cohen's kappa over the events both sides decided."""
    both = [(x, y) for x, y in zip(a, b) if x in DECIDED and y in DECIDED]
    if not both:
        return float("nan")
    n = len(both)
    po = sum(1 for x, y in both if x == y) / n
    ca, cb = Counter(x for x, _ in both), Counter(y for _, y in both)
    pe = sum(ca[k] * cb[k] for k in DECIDED) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def pair(name_a, a, name_b, b, ids):
    both = [e for e in ids if a[e] in DECIDED and b[e] in DECIDED]
    ag = [e for e in both if a[e] == b[e]]
    r = len(ag) / len(both) if both else float("nan")
    lo, hi = wilson(len(ag), len(both)) if both else (float("nan"),) * 2
    k = kappa([a[e] for e in ids], [b[e] for e in ids])
    d = Counter((a[e], b[e]) for e in both if a[e] != b[e])
    one_way = len(d) <= 1
    print(f"  {name_a:<12} vs {name_b:<12} {len(ag):>3}/{len(both):<3} = "
          f"{r:.3f} [{lo:.2f}, {hi:.2f}]   kappa {k:+.3f}")
    if d:
        parts = ", ".join(f"{x}->{y}: {c}" for (x, y), c in sorted(d.items()))
        print(f"  {'':<12}    disagreements {parts}"
              + ("   ONE-WAY: a threshold difference, fixable by a definition"
                 if one_way else
                 "   BOTH WAYS: a convention conflict, not a threshold"))
    return {"n": len(both), "n_agree": len(ag), "agreement": r,
            "wilson": [lo, hi], "kappa": k,
            "directions": {f"{x}->{y}": c for (x, y), c in d.items()},
            "one_way": bool(one_way)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", action="append", required=True,
                    help="repeatable; give two")
    ap.add_argument("--key", help="audit_key.csv, for the stored label")
    ap.add_argument("--out")
    a = ap.parse_args()
    if len(a.sheet) != 2:
        raise SystemExit("give exactly two --sheet files")

    s1, s2 = load_sheet(a.sheet[0]), load_sheet(a.sheet[1])
    ids = sorted(set(s1) & set(s2))
    print(f"{len(s1)} and {len(s2)} rows, {len(ids)} shared event ids")
    if len(ids) < min(len(s1), len(s2)):
        print(f"  !! ids do not fully overlap -- a spreadsheet round trip "
              f"rewrites ids like t112.5, and a shrunken intersection would "
              f"be reported as a result")

    A1 = {e: s1[e]["call"] for e in ids}
    A2 = {e: s2[e]["call"] for e in ids}
    print(f"  annotator 1: {dict(Counter(A1.values()))}")
    print(f"  annotator 2: {dict(Counter(A2.values()))}")

    lab = None
    if a.key:
        key = {r["event_id"]: r for r in read_csv(a.key)}
        miss = [e for e in ids if e not in key]
        if miss:
            print(f"  !! {len(miss)} ids missing from the key")
        lab = {e: ("sharp" if str(key.get(e, {}).get("y", "")).strip()
                   in ("1", "1.0") else "same") for e in ids if e in key}
        ids = [e for e in ids if e in lab]

    print(f"\n{'=' * 72}\nPAIRWISE AGREEMENT\n{'=' * 72}")
    out = {"n_shared": len(ids),
           "A1_vs_A2": pair("annotator 1", A1, "annotator 2", A2, ids)}
    if lab:
        out["A1_vs_label"] = pair("annotator 1", A1, "stored label", lab, ids)
        out["A2_vs_label"] = pair("annotator 2", A2, "stored label", lab, ids)

        hh = out["A1_vs_A2"]["agreement"]
        hl = max(out["A1_vs_label"]["agreement"], out["A2_vs_label"]["agreement"])
        print(f"\n{'=' * 72}\nWHICH IS THE OUTLIER\n{'=' * 72}")
        if hh > hl:
            print(f"  The two annotators agree with EACH OTHER ({hh:.3f}) more "
                  f"than either agrees with the stored label ({hl:.3f}).")
            print("  They are reproducing something the label does not encode. "
                  "The earlier reading -- 'a target humans cannot reproduce' --"
                  "\n  does not survive this: humans reproduce each other. It "
                  "is the LABEL that is out of line, so the next work is on "
                  "the stored\n  labels and the process that produced them, "
                  "not on abandoning the target and not on a new "
                  "representation.")
        else:
            print(f"  The annotators agree with each other ({hh:.3f}) no more "
                  f"than with the label ({hl:.3f}). The decision itself is "
                  f"hard\n  and the label is a fair summary of it -- a target "
                  f"this unstable cannot be learned, and the definitions are "
                  f"the thing to fix.")

        # two independent votes against a stored label
        both_vs = [e for e in ids if A1[e] in DECIDED and A2[e] in DECIDED
                   and A1[e] == A2[e] and A1[e] != lab[e]]
        print(f"\n  {len(both_vs)} events where BOTH annotators agree and BOTH "
              f"differ from the label. Two independent votes against a stored "
              f"label\n  is the strongest evidence this data can give about "
              f"that label; these rows are actionable individually:")
        for e in both_vs:
            print(f"    {e:<50} both said {A1[e]:<5} label {lab[e]}")
        out["both_annotators_against_label"] = both_vs

        split = [e for e in ids if A1[e] in DECIDED and A2[e] in DECIDED
                 and A1[e] != A2[e]]
        print(f"\n  {len(split)} events where the annotators split. These are "
              f"the genuinely ambiguous ones and no vote settles them:")
        for e in split:
            print(f"    {e:<50} A1 {A1[e]:<5} A2 {A2[e]:<5} label {lab[e]}")
        out["annotators_split"] = split

    ans = Counter(s2[e]["ans"] for e in ids if s2[e]["ans"])
    if ans:
        print(f"\n{'=' * 72}\nOBSERVABILITY ANSWERS (annotator 2)\n{'=' * 72}")
        for k, v in sorted(ans.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<24} {v:>3}  {v / sum(ans.values()):.0%}")
        if not any(k.startswith("3_") for k in ans):
            print("  Nobody answered semantic/long-context. The window is not "
                  "the limitation -- which is evidence against extending it "
                  "and for\n  the object-relative hypothesis, on a stratified "
                  "sample of 36 that does not carry the band's natural "
                  "proportions.")
        out["observability_answers"] = dict(ans)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
