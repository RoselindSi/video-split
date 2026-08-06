"""How well does a human do on the REVIEW band? The first ceiling this project
has measured.

Every result on this band has been read as if the ceiling were high: P1 at
0.528, hand trajectories at 0.513, and each round concluded "the
representation is inadequate". The competing explanation was never tested --
that these events carry too little evidence for anyone, in which case no
representation would help and the work belongs in the annotation protocol.

The audit's `your_call` column tests it directly. It was recorded blind: no
score, no subtype, no stratum, and the reviewer answered before being asked
what evidence they used, so a confident call here is a real decision rather
than a reconstruction.

WHAT THIS IS NOT. Human accuracy against the taxonomy label is not an upper
bound on achievable accuracy in the strict sense -- the label was itself
assigned by a human, so agreement measures reproducibility of a convention as
much as difficulty of a decision. Where the two disagree, this cannot say
which is right. It is reported as agreement, and the disagreements are listed
by name so they can be looked at rather than assumed to be human error.

`cannot` ROWS ARE NOT SCORED AS WRONG, and they are not dropped either. A
reviewer declining to guess is evidence about the band, so they are reported
as their own category with the rate stated. Scoring them as errors would
understate the ceiling; dropping them silently would overstate it.

Comparison to a model number needs care and the report makes it explicit: a
human produces one hard decision while P1 produces a ranking, so its AUROC and
this accuracy are different quantities. The comparable model figure is its
accuracy at its own best threshold, which is computed here from the same
events when a score column is available.

Usage:
    python -m src.boundary.c3_human_ceiling \
        --sheet /workspace/tr1/results/hal/c3/observable_audit/audit_sheet_with_your_call_complete_36.csv \
        --key /workspace/tr1/results/hal/c3/observable_audit/audit_key.csv \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --out /workspace/tr1/results/hal/c3/observable_audit/human_ceiling.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.c3_selective_policy import wilson


# utf-8-sig, not utf-8: a spreadsheet writes a BOM and the FIRST column
# name comes back as "\ufeffyour_call(...)", so a prefix match on it
# fails and the file looks like it is missing the column it plainly has.
def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    return next((k for k in row if k.startswith(prefix)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--decisions")
    ap.add_argument("--out")
    a = ap.parse_args()

    sheet = read_csv(a.sheet)
    key = {r["event_id"]: r for r in read_csv(a.key)}
    c_call = col(sheet[0], "your_call")
    c_conf = col(sheet[0], "confidence")
    c_ans = col(sheet[0], "answer")
    if not c_call:
        raise SystemExit("no your_call column in the sheet")

    # event ids must match EXACTLY. A spreadsheet round trip turns t112.5 into
    # 112.5 or a date, and a silent 0-row join would be reported as a result.
    miss = [r["event_id"] for r in sheet if r["event_id"] not in key]
    print(f"{len(sheet)} sheet rows, {len(key)} key rows, "
          f"{len(sheet) - len(miss)} joined")
    if miss:
        print(f"  !! {len(miss)} sheet ids are not in the key -- a spreadsheet "
              f"round trip rewrites ids like t112.5. Fix the join before "
              f"reading anything below:")
        for m in miss[:8]:
            print(f"     {m}")
        if len(miss) == len(sheet):
            raise SystemExit("nothing joined")

    rows = []
    for r in sheet:
        k = key.get(r["event_id"])
        if not k:
            continue
        call = (r.get(c_call) or "").strip().lower()
        truth = "sharp" if str(k.get("y", "")).strip() in ("1", "1.0") else "same"
        rows.append({"event_id": r["event_id"], "call": call, "truth": truth,
                     "conf": (r.get(c_conf) or "").strip(),
                     "answer": (r.get(c_ans) or "").strip() if c_ans else "",
                     "stratum": k.get("stratum", ""),
                     "coverage": k.get("hand_detect_coverage", "")})

    n = len(rows)
    dec = [r for r in rows if r["call"] in ("sharp", "same")]
    cannot = [r for r in rows if r["call"] == "cannot"]
    other = [r for r in rows if r["call"] not in ("sharp", "same", "cannot")]
    if other:
        print(f"  !! {len(other)} rows have an unrecognised call: "
              f"{sorted({r['call'] for r in other})}")
    right = [r for r in dec if r["call"] == r["truth"]]
    acc = len(right) / len(dec) if dec else float("nan")
    lo, hi = wilson(len(right), len(dec)) if dec else (float("nan"),) * 2

    print(f"\n{'=' * 68}\nHUMAN AGREEMENT WITH THE TAXONOMY LABEL, REVIEW BAND"
          f"\n{'=' * 68}")
    print(f"  {n} events: {len(dec)} decided, {len(cannot)} declined "
          f"({len(cannot) / n:.0%})")
    print(f"  agreement on the decided ones: {len(right)}/{len(dec)} = "
          f"{acc:.3f}  95% Wilson [{lo:.2f}, {hi:.2f}]")
    print(f"  over ALL {n} events, counting declines as not-automated: "
          f"{len(right)}/{n} = {len(right) / n:.3f}")

    print(f"\n  by stated confidence:")
    by = defaultdict(list)
    for r in dec:
        by[r["conf"] or "(blank)"].append(r)
    for c in sorted(by):
        g = by[c]
        k = sum(1 for r in g if r["call"] == r["truth"])
        w = wilson(k, len(g))
        print(f"    {c:<12} {k:>3}/{len(g):<3} = {k / len(g):.3f}  "
              f"[{w[0]:.2f}, {w[1]:.2f}]")
    if len(by) == 1:
        print("    Only one confidence value was used, so this column cannot "
              "separate confident from uncertain calls and the ceiling is a "
              "single number rather than a curve.")

    wrong = [r for r in dec if r["call"] != r["truth"]]
    if wrong:
        print(f"\n  the {len(wrong)} disagreements, by name -- these are NOT "
              f"assumed to be human error. The label was assigned by a human "
              f"too, and a disagreement at high confidence is a candidate "
              f"annotation problem:")
        for r in wrong:
            print(f"    {r['event_id']:<50} said {r['call']:<5} label "
                  f"{r['truth']:<5} conf {r['conf']}")
    if cannot:
        print(f"\n  the {len(cannot)} declined:")
        for r in cannot:
            print(f"    {r['event_id']:<50} label {r['truth']:<5} "
                  f"coverage {r['coverage']}")

    out = {"n": n, "n_decided": len(dec), "n_declined": len(cannot),
           "agreement": acc, "agreement_wilson": [lo, hi],
           "disagreements": [r["event_id"] for r in wrong],
           "declined": [r["event_id"] for r in cannot]}

    if a.decisions:
        sc = {}
        d0 = read_csv(a.decisions)
        skey = next((k for k in ("score", "p1_score", "primary_score",
                                 "fused_score", "oof_score") if d0 and k in d0[0]),
                    None)
        if skey:
            for r in d0:
                try:
                    sc[r["event_id"]] = float(r[skey])
                except (KeyError, ValueError):
                    pass
            s = np.array([sc.get(r["event_id"], np.nan) for r in rows])
            y = np.array([r["truth"] == "sharp" for r in rows])
            m = np.isfinite(s)
            if m.sum() >= 10:
                # P1's accuracy at ITS OWN BEST threshold on these very events.
                # That is generous to the model -- the threshold is chosen with
                # the answers in hand, which no deployment can do -- and it is
                # the right comparison to make anyway: if a human beats an
                # oracle-thresholded model, the gap is not about calibration.
                best = max((np.mean((s[m] >= t) == y[m]) for t in s[m]),
                           default=float("nan"))
                print(f"\n  P1 on the same {int(m.sum())} events, accuracy at "
                      f"its own BEST threshold chosen with the answers "
                      f"visible: {best:.3f}")
                print(f"  Human {acc:.3f} vs an oracle-thresholded {best:.3f}. "
                      f"A human ahead of a model that was allowed to pick its "
                      f"threshold after seeing the labels is a gap in what is "
                      f"being SEEN, not in how it is being calibrated.")
                out["p1_oracle_threshold_accuracy"] = float(best)

    print(f"\n{'=' * 68}")
    if np.isfinite(acc) and acc >= 0.85 and len(cannot) / n <= 0.2:
        print("  The REVIEW band is resolvable by a human. The two failed "
              "rounds cannot be explained by these events carrying no "
              "evidence, so looking for a better observable is justified --\n"
              "  but this says nothing about WHICH observable, and the "
              "four-way answer column is what would.")
    elif np.isfinite(acc):
        print("  A human is not clearly ahead of the models on this band. "
              "Before building another representation, the annotation "
              "protocol is the thing to examine: a target humans cannot "
              "reproduce is not a modelling problem.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
