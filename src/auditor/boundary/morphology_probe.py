"""Three pairwise questions, four scorers, one table. No training, no policy.

Boundary v1 produced a deployment AUTO_KEEP of 10 events at 0.800 precision
and a P(POINT) AUROC of 0.751 against a fused frozen score of about 0.821. The
reformulation is conceptually right and has not yet produced a better learned
representation, and the aggregate numbers cannot say which part is missing.

Splitting it into three pairs can, because they fail differently:

    POINT vs NONE       does the encoder still see what the old clean binary
                        saw? A loss here means the temporal student threw away
                        discrimination the frozen scores already had.
    POINT vs INTERVAL   does it see the SHAPE of the change -- a step against
                        a ramp? This is the whole claim of the reformulation
                        and the source of 10 of the 27 high-confidence errors.
    INTERVAL vs NONE    does it see that a change happened at all, setting
                        aside its shape?

The diagnostic reading is the pattern, not any single cell. POINT-vs-NONE and
INTERVAL-vs-NONE strong with POINT-vs-INTERVAL near chance means the
representation encodes change EXISTENCE and not change MORPHOLOGY -- which is
what a small TCN would produce if it had quietly rediscovered the pre/post
contrast that the sequence input was meant to replace. Feeding frames in does
not force a model to use them.

THE STUDENT'S SCORE ON A PAIR IS ITS RENORMALISED BINARY, P(A)/(P(A)+P(B)).
Reading P(POINT) on INTERVAL-vs-NONE would ask a head about a class neither
event belongs to and score it for answering a third question.

THE OLD SCORERS GET THE SAME COLUMN ON EVERY PAIR, because they only have one
-- they were trained on sharp versus same_action and have no notion of an
interval. That is the point: their number on POINT-vs-INTERVAL says how much
of the new question the old formulation could already answer, and if it is
also near chance then the gap is in the features, not in the head.

DELTAS ARE PAIRED AND BOOTSTRAPPED BY RECORDING. Two separate intervals on
correlated estimates cannot be compared by whether they overlap.

Usage:
    python -m src.auditor.boundary.morphology_probe \
        --predictions .../boundary_v1_oof.json \
        --decisions .../policy_decisions_v4...csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from src.boundary.state_adapter import _auroc
from src.boundary.pairwise_verifier import stratified_grouped_folds

POINT, INTERVAL, NONE = ("POINT_TRANSITION", "INTERVAL_TRANSITION",
                         "NO_TRANSITION")
PAIRS = [(POINT, NONE), (POINT, INTERVAL), (INTERVAL, NONE)]
OLD_ARMS = ["P1 (global) alone", "local alone", "P1 + local, feature-level"]


def au(rows, score):
    y = np.array([r["_y"] for r in rows], float)
    if len(set(y.tolist())) < 2:
        return None
    p = np.array([score(r) for r in rows], float)
    if not np.isfinite(p).all():
        return None
    return _auroc(y, p)


def paired_bootstrap(rows, fa, fb, n_boot, seed):
    """CI on AUROC(a) - AUROC(b) resampling recordings, not events."""
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        s = [x for i in rng.integers(0, len(keys), len(keys)) for x in by[keys[i]]]
        y = np.array([r["_y"] for r in s], float)
        if len(set(y.tolist())) < 2:
            continue
        a, b = au(s, fa), au(s, fb)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        out.append(a - b)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--decisions", help="the frozen policy decisions csv, for "
                                        "the three old scorers")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    blob = json.load(open(a.predictions, encoding="utf-8"))
    rows = [r for r in blob["events"]
            if r.get("morphology_true")
            and np.isfinite(list(r["morphology"].values())[0])]
    old = {}
    if a.decisions and os.path.exists(a.decisions):
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = {}
                for c in OLD_ARMS:
                    try:
                        d[c] = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        pass
                if d:
                    old[r["event_id"]] = d
    print(f"{os.path.basename(a.predictions)}: {len(rows)} events with a "
          f"morphology target; {sum(1 for r in rows if r['event_id'] in old)} "
          f"also carry the frozen scores")

    print(f"\n{'=' * 96}\nPAIRWISE AUROC, recording-grouped, same events for "
          f"every column\n{'=' * 96}")
    hdr = ["temporal student"] + OLD_ARMS
    print(f"  {'pair':<34} {'n':>5} {'recs':>5} " + "".join(f"{h[:18]:>20}" for h in hdr))
    results = {}
    for pos, neg in PAIRS:
        sel = [dict(r, _y=float(r["morphology_true"] == pos))
               for r in rows if r["morphology_true"] in (pos, neg)]
        common = [r for r in sel if r["event_id"] in old]
        nrec = len({r["recording_id"] for r in sel})
        # the student's score on this pair is its renormalised binary
        stu = lambda r, p=pos, n=neg: (
            r["morphology"][p] / max(r["morphology"][p] + r["morphology"][n],
                                     1e-12))
        cells = [au(sel, stu)]
        for c in OLD_ARMS:
            cells.append(au(common, lambda r, c=c: old[r["event_id"]][c])
                         if common else None)
        results[(pos, neg)] = (sel, common, stu, cells)
        label = f"{pos.split('_')[0]} vs {neg.split('_')[0]}"
        print(f"  {label:<34} {len(sel):>5} {nrec:>5} "
              + "".join(f"{(v if v is not None else float('nan')):>20.3f}"
                        for v in cells))
    print("  The old scorers have ONE score and get the same column on every "
          "row: they were fitted on sharp versus same_action\n  and have no "
          "notion of an interval. Their POINT-vs-INTERVAL cell says how much "
          "of the new question the old\n  formulation could already answer.")

    print(f"\n{'=' * 96}\nSTUDENT MINUS THE BEST OLD ARM, paired and "
          f"bootstrapped by recording\n{'=' * 96}")
    for (pos, neg), (sel, common, stu, cells) in results.items():
        if not common or all(c is None for c in cells[1:]):
            continue
        best_i = int(np.nanargmax([c if c is not None else -1
                                   for c in cells[1:]]))
        best = OLD_ARMS[best_i]
        s_c = au(common, stu)
        o_c = au(common, lambda r, c=best: old[r["event_id"]][c])
        lo, hi = paired_bootstrap(
            common, stu, lambda r, c=best: old[r["event_id"]][c],
            a.n_boot, a.seed)
        # both AUROCs recomputed on `common`, so the delta is not contaminated
        # by the student being scored on a larger set than the baseline
        d = s_c - o_c
        verdict = ("no detectable difference" if lo <= 0 <= hi else
                   "student is WORSE" if hi < 0 else "student is better")
        print(f"  {pos.split('_')[0]} vs {neg.split('_')[0]:<14} on the "
              f"{len(common)} shared events: student {s_c:.3f}, {best} "
              f"{o_c:.3f}")
        print(f"      delta {d:+.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    print(f"\n{'=' * 96}\nPER-FOLD STABILITY of the student\n{'=' * 96}")
    for (pos, neg), (sel, _, stu, _) in results.items():
        y = np.array([r["_y"] for r in sel], float)
        g = [r["recording_id"] for r in sel]
        folds = stratified_grouped_folds(g, y, a.n_folds, seed=a.seed)
        per = []
        for f in folds:
            te = [r for r, gg in zip(sel, g) if gg in f]
            v = au(te, stu) if te else None
            per.append(v if v is not None else float("nan"))
        print(f"  {pos.split('_')[0]} vs {neg.split('_')[0]:<14} "
              + " ".join(f"{v:.3f}" for v in per)
              + f"   spread {np.nanmax(per) - np.nanmin(per):.3f}")

    # ------------------------------------------------------- score geometry
    print(f"\n{'=' * 96}\nSCORE GEOMETRY: where INTERVAL actually sits"
          f"\n{'=' * 96}")
    byc = {c: np.array([r["morphology"][POINT] for r in rows
                        if r["morphology_true"] == c])
           for c in (POINT, INTERVAL, NONE)}
    print(f"  P(POINT) by true class")
    print(f"  {'class':<22} {'n':>4} {'p10':>7} {'p25':>7} {'median':>8} "
          f"{'p75':>7} {'p90':>7}")
    for c, v in byc.items():
        if len(v) == 0:
            continue
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f"  {c:<22} {len(v):>4} " + "".join(f"{x:>7.3f}" for x in q[:2])
              + f"{q[2]:>8.3f}" + "".join(f"{x:>7.3f}" for x in q[3:]))
    iv, pt, nn = byc[INTERVAL], byc[POINT], byc[NONE]
    if len(iv) and len(pt) and len(nn):
        p_med, n_med = float(np.median(pt)), float(np.median(nn))
        if p_med <= n_med:
            print(f"\n  the POINT median ({p_med:.3f}) is not above the NONE "
                  f"median ({n_med:.3f}), so there is no ordering to place "
                  f"INTERVAL\n  inside. The geometry question only has an "
                  f"answer once the two anchor classes separate at all.")
            return
        # the three regions are disjoint by construction; the earlier version
        # counted >= p_med and <= n_med independently and produced a negative
        # remainder whenever the two medians crossed
        inside_p = int((iv >= p_med).sum())
        inside_n = int((iv <= n_med).sum())
        between = int(((iv > n_med) & (iv < p_med)).sum())
        print(f"\n  of the {len(iv)} INTERVAL events: {inside_p} sit at or "
              f"above the POINT median, {inside_n} at or below the NONE "
              f"median,\n  {between} in between.")
        print(f"  A single intermediate class would sit mostly in between. "
              f"Mass at BOTH ends means `gradual` is not one visual\n  "
              f"morphology -- a smooth ramp, an overlap where one action has "
              f"not ended before the next begins, and a multi-step\n  "
              f"transition with no unique instant look nothing alike, and "
              f"splitting them is a labelling question, not a model one.")


if __name__ == "__main__":
    main()
