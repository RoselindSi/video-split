"""Score the out-of-fold predictions, and answer the one question v1 asks.

THE HEADLINE IS NOT AN AUROC. The reformulation claims that POINT / INTERVAL /
NO_TRANSITION / UNOBSERVABLE is better posed than sharp-versus-same_action. A
higher AUROC on a different target proves nothing, so the comparison is made
on the SAME events and the SAME folds by reading the old binary off the new
head: P(POINT) against the old sharp label, restricted to the events the old
target admitted. If that is no better than the old head's own number, the
reformulation has not paid for itself on discrimination -- and it may still be
worth keeping for what it does to the classes the old target could not
represent at all, which is reported separately rather than folded in.

A CLASS WITH FEWER EVENTS THAN FOLDS IS NOT REPORTED. Its per-class recall is
decided by which fold its handful of events landed in, and printing it invites
exactly the reading the number cannot support. It is named and withheld.

Usage:
    python -m src.auditor.boundary.evaluate \
        --predictions .../boundary_v1_oof.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.state_adapter import _auroc
from src.boundary.c3_selective_policy import wilson

MORPHOLOGY = ["POINT_TRANSITION", "INTERVAL_TRANSITION", "NO_TRANSITION",
              "UNOBSERVABLE"]
RELATION = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID"]


def grouped_bootstrap(fn, rows, n_boot, seed):
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [x for i in pick for x in by[keys[i]]]
        v = fn(sample)
        if v is not None and np.isfinite(v):
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    blob = json.load(open(a.predictions, encoding="utf-8"))
    rows = [r for r in blob["events"]
            if r["morphology"] and np.isfinite(list(r["morphology"].values())[0])]
    print(f"{os.path.basename(a.predictions)}: {len(blob['events'])} events, "
          f"{len(rows)} with an out-of-fold prediction")

    sup = [r for r in rows if r["morphology_true"]]
    print(f"\n{'=' * 78}\nMORPHOLOGY, {len(sup)} supervised events\n{'=' * 78}")
    pred = {r["event_id"]: max(MORPHOLOGY, key=lambda k: r["morphology"][k])
            for r in sup}
    counts = Counter(r["morphology_true"] for r in sup)
    print(f"  {'class':<22} {'n':>4} {'precision':>10} {'recall':>8} "
          f"{'reported':>9}")
    for c in MORPHOLOGY:
        g = [r for r in sup if r["morphology_true"] == c]
        p = [r for r in sup if pred[r["event_id"]] == c]
        tp = sum(1 for r in p if r["morphology_true"] == c)
        ok = counts.get(c, 0) >= a.n_folds
        prec = tp / len(p) if p else float("nan")
        rec = tp / len(g) if g else float("nan")
        print(f"  {c:<22} {counts.get(c, 0):>4} "
              + (f"{prec:>10.3f} {rec:>8.3f} {'yes':>9}" if ok else
                 f"{'--':>10} {'--':>8} {'withheld':>9}"))
    withheld = [c for c in MORPHOLOGY if counts.get(c, 0) < a.n_folds]
    if withheld:
        print(f"  withheld: {withheld} have fewer events than folds, so a "
              f"per-class rate is decided by fold assignment")

    print(f"\n  confusion, rows are truth:")
    print(f"  {'':<22}" + "".join(f"{c[:12]:>14}" for c in MORPHOLOGY))
    for t in MORPHOLOGY:
        g = [r for r in sup if r["morphology_true"] == t]
        cc = Counter(pred[r["event_id"]] for r in g)
        print(f"  {t:<22}" + "".join(f"{cc.get(c, 0):>14}" for c in MORPHOLOGY))

    acc = sum(1 for r in sup if pred[r["event_id"]] == r["morphology_true"])
    lo, hi = wilson(acc, len(sup))
    print(f"\n  accuracy {acc}/{len(sup)} = {acc / len(sup):.3f}   "
          f"Wilson [{lo:.3f}, {hi:.3f}]")
    maj = counts.most_common(1)[0]
    print(f"  always predicting {maj[0]} would give "
          f"{maj[1] / len(sup):.3f}   <- the number to beat, not 0.25")

    # ------------------------------------------------- the actual v1 question
    print(f"\n{'=' * 78}\nDOES THE REFORMULATION PAY? Same events, same folds."
          f"\n{'=' * 78}")
    old = [r for r in sup if r["morphology_true"] in
           ("POINT_TRANSITION", "NO_TRANSITION")]
    y = np.array([r["morphology_true"] == "POINT_TRANSITION" for r in old],
                 float)
    p = np.array([r["morphology"]["POINT_TRANSITION"] for r in old])
    au = _auroc(y, p)
    lo, hi = grouped_bootstrap(
        lambda s: _auroc(
            np.array([r["morphology_true"] == "POINT_TRANSITION" for r in s],
                     float),
            np.array([r["morphology"]["POINT_TRANSITION"] for r in s]))
        if len({r["morphology_true"] for r in s}) > 1 else None,
        old, a.n_boot, a.seed)
    print(f"  the OLD binary read off the new head: POINT vs NO_TRANSITION on "
          f"{len(old)} events")
    print(f"  AUROC {au:.3f}   recording-grouped bootstrap [{lo:.3f}, {hi:.3f}]")
    print(f"  The old CLEAN_BINARY head is the comparison, on these same "
          f"events. A gain inside this interval is not a gain.")

    extra = [r for r in sup if r["morphology_true"] in
             ("INTERVAL_TRANSITION", "UNOBSERVABLE")]
    if extra:
        right = sum(1 for r in extra if pred[r["event_id"]] == r["morphology_true"])
        print(f"\n  WHAT THE OLD TARGET COULD NOT REPRESENT: {len(extra)} "
              f"INTERVAL and UNOBSERVABLE events, {right} placed in their own "
              f"class.")
        mis = Counter(pred[r["event_id"]] for r in extra
                      if pred[r["event_id"]] != r["morphology_true"])
        print(f"  the rest go to {dict(mis)}. Under the old target every one "
              f"of these {len(extra)} was a negative or excluded,\n  so there "
              f"is no old number to compare against -- this is capability the "
              f"old formulation did not have, not an improvement on it.")

    # ------------------------------------------------------------- relation
    # scored on the relation head's OWN mask, not on the morphology-supervised
    # subset: 6 events are masked for morphology and still carry a relation
    # target, and scoring one head through another's mask is the coupling the
    # whole reformulation exists to remove
    sr = [r for r in rows if r["relation_true"] in RELATION]
    print(f"\n{'=' * 78}\nCANDIDATE RELATION, {len(sr)} supervised events"
          f"\n{'=' * 78}")
    rc = Counter(r["relation_true"] for r in sr)
    thin = [c for c in RELATION if 0 < rc.get(c, 0) < a.n_folds]
    absent = [c for c in RELATION if rc.get(c, 0) == 0]
    if sr:
        rpred = {r["event_id"]: max(RELATION, key=lambda k: r["relation"][k])
                 for r in sr}
        for c in RELATION:
            n = rc.get(c, 0)
            if n < a.n_folds:
                continue
            g = [r for r in sr if r["relation_true"] == c]
            pp = [r for r in sr if rpred[r["event_id"]] == c]
            tp = sum(1 for r in pp if r["relation_true"] == c)
            print(f"  {c:<22} {n:>4} precision "
                  f"{(tp / len(pp) if pp else float('nan')):.3f}   recall "
                  f"{tp / len(g):.3f}")
    if thin or absent:
        print(f"  withheld {thin}, absent {absent}. EARLY and LATE are the "
              f"retime signal and there are not enough of them to\n  measure; "
              f"that is why SUGGEST_RETIME is disabled in the ontology rather "
              f"than tuned.")

    off = [r for r in rows if r["offset_true"] is not None
           and r["offset"] is not None]
    if off:
        err = np.array([r["offset"] - r["offset_true"] for r in off])
        base = np.array([r["offset_true"] for r in off])
        print(f"\n  offset MAE {np.abs(err).mean():.3f}s over {len(off)} "
              f"events; predicting 0 everywhere gives "
              f"{np.abs(base).mean():.3f}s")
        print(f"  {int((np.abs(base) <= 0.5).sum())}/{len(off)} targets are "
              f"already within 0.5s, so an MAE near the constant baseline "
              f"means the head learned nothing.")


if __name__ == "__main__":
    main()
