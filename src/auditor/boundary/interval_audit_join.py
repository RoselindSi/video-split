"""After the 37 are relabelled: does the topology explain the score, or not?

Run only on a FROZEN sheet. The whole design of the blind audit is spent if
the labels move after the scores are seen.

Two hypotheses, and this table separates them. They are written down here
before the numbers exist.

  PATH A, a label problem. The subtypes line up with the score modes --
  point_like sits high on P(POINT), smooth_ramp sits low, and the spread
  within each subtype is much smaller than the spread across all 37. Then
  `gradual` was a semantic umbrella, INTERVAL was never one visual class, and
  the fix is to the ontology: a hierarchical target, or three sibling classes,
  or point_like events revised outright.

  PATH B, a representation problem. The same subtype is still strongly
  bimodal. Then the frozen global and local features do not carry temporal
  morphology and no relabelling reaches it; the next work is the
  representation, and only then.

A THIRD READING IS AVAILABLE AND EASY TO MISS. The per-fold POINT-vs-INTERVAL
AUROC ran 0.629 / 0.726 / 0.871 / 0.558 / 0.741, a spread of 0.313. If the
folds differ in subtype composition, most of that spread is which events
landed where rather than transportability, and the composition table below
says so directly. That is a different diagnosis from either path and it does
not require the score-mode alignment to hold.

WITH 37 EVENTS EVERY PER-SUBTYPE NUMBER IS FRAGILE. A subtype spanning fewer
recordings than folds gets counts and a median and no AUROC, on the same rule
the rest of this project uses: a per-class rate decided by which fold a
handful of events fell into is not a rate.

Usage:
    python -m src.auditor.boundary.interval_audit_join \
        --sheet .../interval_audit_sheet_filled.csv \
        --predictions .../boundary_v1_oof.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.state_adapter import _auroc
from src.boundary.pairwise_verifier import stratified_grouped_folds

POINT, INTERVAL, NONE = ("POINT_TRANSITION", "INTERVAL_TRANSITION",
                         "NO_TRANSITION")
SUBTYPES = ["smooth_ramp", "overlapping_transition", "multi_step_transition",
            "point_like", "ambiguous_interval"]


def read_sheet(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            call = next((v for k, v in r.items()
                         if k and k.startswith("your_call")), "")
            conf = next((v for k, v in r.items()
                         if k and k.startswith("confidence")), "")
            e = r.get("event_id")
            if e and (call or "").strip():
                out[e] = {"subtype": call.strip(), "confidence": conf.strip(),
                          "why": (r.get("why_one_line") or "").strip()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    calls = read_sheet(a.sheet)
    blob = json.load(open(a.predictions, encoding="utf-8"))
    by = {r["event_id"]: r for r in blob["events"]}
    iv = [dict(by[e], **{"_sub": v["subtype"], "_conf": v["confidence"]})
          for e, v in calls.items() if e in by]
    print(f"{len(calls)} rows filled, {len(iv)} joined to a prediction")
    bad = sorted({v["subtype"] for v in calls.values()} - set(SUBTYPES))
    if bad:
        print(f"  !! unrecognised subtype(s) {bad}; fix the sheet rather than "
              f"guessing what was meant")
    unfilled = [r["event_id"] for r in blob["events"]
                if r.get("morphology_true") == INTERVAL
                and r["event_id"] not in calls]
    if unfilled:
        print(f"  !! {len(unfilled)} INTERVAL events are still blank. A "
              f"partial sheet biases every row below, because the ones\n"
              f"     left for later are rarely a random half.")

    pt = np.array([r["morphology"][POINT] for r in blob["events"]
                   if r.get("morphology_true") == POINT])
    nn = np.array([r["morphology"][POINT] for r in blob["events"]
                   if r.get("morphology_true") == NONE])
    p_med, n_med = float(np.median(pt)), float(np.median(nn))
    print(f"\n  anchors on P(POINT): POINT median {p_med:.3f}, NONE median "
          f"{n_med:.3f}")

    print(f"\n{'=' * 92}\nSUBTYPE vs P(POINT)\n{'=' * 92}")
    print(f"  {'subtype':<24} {'n':>3} {'recs':>5} {'p25':>7} {'median':>8} "
          f"{'p75':>7} {'>=POINT':>8} {'<=NONE':>7} {'IQR':>7}")
    rows_out = []
    for s in SUBTYPES + [x for x in sorted({r["_sub"] for r in iv})
                         if x not in SUBTYPES]:
        g = [r for r in iv if r["_sub"] == s]
        if not g:
            continue
        v = np.array([r["morphology"][POINT] for r in g])
        q = np.percentile(v, [25, 50, 75])
        nrec = len({r["recording_id"] for r in g})
        hi = int((v >= p_med).sum())
        lo = int((v <= n_med).sum())
        print(f"  {s:<24} {len(g):>3} {nrec:>5} {q[0]:>7.3f} {q[1]:>8.3f} "
              f"{q[2]:>7.3f} {hi:>8} {lo:>7} {q[2] - q[0]:>7.3f}")
        rows_out.append({"subtype": s, "n": len(g), "recordings": nrec,
                         "median_p_point": float(q[1]),
                         "iqr": float(q[2] - q[0]),
                         "at_or_above_point_median": hi,
                         "at_or_below_none_median": lo})
    all_v = np.array([r["morphology"][POINT] for r in iv])
    all_iqr = float(np.percentile(all_v, 75) - np.percentile(all_v, 25))
    print(f"  {'ALL 37 together':<24} {len(iv):>3} "
          f"{len({r['recording_id'] for r in iv}):>5} "
          f"{np.percentile(all_v, 25):>7.3f} {np.median(all_v):>8.3f} "
          f"{np.percentile(all_v, 75):>7.3f} "
          f"{int((all_v >= p_med).sum()):>8} {int((all_v <= n_med).sum()):>7} "
          f"{all_iqr:>7.3f}")
    tight = [r for r in rows_out if r["n"] >= 4 and r["iqr"] < all_iqr / 2]
    print(f"\n  PATH A wants the per-subtype IQRs well inside the pooled "
          f"{all_iqr:.3f}: {len(tight)} of {len(rows_out)} subtypes with n>=4 "
          f"manage it.")
    print(f"  PATH B is a subtype that is still split between the two ends -- "
          f"read the >=POINT and <=NONE columns on the same row.")

    print(f"\n{'=' * 92}\nPOINT vs EACH SUBTYPE\n{'=' * 92}")
    pts = [r for r in blob["events"] if r.get("morphology_true") == POINT]
    print(f"  {'subtype':<24} {'n':>3} {'recs':>5} {'AUROC vs POINT':>16}")
    for s in sorted({r["_sub"] for r in iv}):
        g = [r for r in iv if r["_sub"] == s]
        nrec = len({r["recording_id"] for r in g})
        if len(g) < a.n_folds or nrec < a.n_folds:
            print(f"  {s:<24} {len(g):>3} {nrec:>5} {'WITHHELD':>16}   "
                  f"(fewer events or recordings than folds)")
            continue
        sel = pts + g
        y = np.array([1.0] * len(pts) + [0.0] * len(g))
        p = np.array([r["morphology"][POINT] for r in sel])
        print(f"  {s:<24} {len(g):>3} {nrec:>5} {_auroc(y, p):>16.3f}")
    print("  A subtype the model already separates from POINT is not what "
          "broke the four-way head; one it cannot separate is.")

    print(f"\n{'=' * 92}\nFOLD COMPOSITION -- is the 0.313 spread just which "
          f"events landed where?\n{'=' * 92}")
    sel = [r for r in blob["events"]
           if r.get("morphology_true") in (POINT, INTERVAL)]
    y = np.array([r["morphology_true"] == POINT for r in sel], float)
    g = [r["recording_id"] for r in sel]
    folds = stratified_grouped_folds(g, y, a.n_folds, seed=a.seed)
    sub_of = {r["event_id"]: r["_sub"] for r in iv}
    print(f"  {'fold':>4} {'n_INTERVAL':>11}  composition")
    for fi, f in enumerate(folds):
        te = [r for r, gg in zip(sel, g) if gg in f
              and r["morphology_true"] == INTERVAL]
        c = Counter(sub_of.get(r["event_id"], "unlabelled") for r in te)
        print(f"  {fi:>4} {len(te):>11}  {dict(c)}")
    print("  If the folds that scored 0.871 and 0.558 differ mainly in which "
          "subtypes they hold, the spread is composition,\n  not "
          "transportability, and the per-fold numbers should not be read as "
          "instability of the model.")

    pl = [r for r in iv if r["_sub"] == "point_like"]
    if pl:
        print(f"\n{'=' * 92}\nLABEL REVISIONS\n{'=' * 92}")
        print(f"  {len(pl)} of {len(iv)} were called point_like on review. "
              f"Those are candidates for gradual -> sharp in the pair labels,")
        print(f"  which changes the morphology target and therefore every "
              f"number above. Revise them in the label file and rerun; do NOT")
        print(f"  patch them in here, or the training target and the "
              f"evaluation target stop being the same thing.")
        for r in pl:
            print(f"    {r['event_id'][-46:]:<47} P(POINT) "
                  f"{r['morphology'][POINT]:.3f}  conf {r['_conf']}")

    if a.out:
        json.dump({"n_joined": len(iv), "point_median": p_med,
                   "none_median": n_med, "pooled_iqr": all_iqr,
                   "subtypes": rows_out,
                   "events": [{"event_id": r["event_id"], "subtype": r["_sub"],
                               "confidence": r["_conf"],
                               "p_point": r["morphology"][POINT]}
                              for r in iv]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
