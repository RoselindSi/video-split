"""Is there ANY threshold on P(POINT) that earns an automatic keep? No training.

The ontology's AUTO_KEEP gate -- argmax is POINT and confidence >= 0.95 -- was
written by hand, not selected. On the deployment simulation it admits 148
events at 0.789 precision over the ones that have a truth, and 0.682 counting
the untargeted. That is a rule, not an operating point, and the question it
leaves open is whether the score is capable of better anywhere.

EVERY EVENT WITHOUT A MORPHOLOGY TARGET IS A NEGATIVE HERE. In training,
annotation_convention and camera are masked because no perception target
exists for them. At deployment they still arrive, and admitting one puts an
event into the dataset with nothing behind it. Excluding them from this
measurement would be scoring the model on a population it will not meet -- the
exclusion that made clean-binary precision read 0.967 while full-taxonomy read
0.784. Both columns are printed so the size of that gap stays visible, and the
one that decides is the one counting them.

NESTED. The threshold is chosen inside each outer fold's training recordings
and applied only to its held-out ones. Choosing on pooled out-of-fold scores
is how every earlier operating point in this project was picked and why they
kept breaching afterwards.

WHAT A PASS HERE WOULD AND WOULD NOT MEAN. AUTO_KEEP needs the candidate to be
on the boundary, not merely near a real transition -- morphology POINT plus
relation EXACT. Relation has 172 supervised events over 47 recordings and no
gradient, so a morphology threshold that reaches precision here has cleared
one of the two conditions, and the second is not measurable yet on 240 of the
415 events. A pass is a necessary step and not an operating point.

Usage:
    python -m src.auditor.boundary.operating_point \
        --predictions .../boundary_v1_oof.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.state_adapter import _auroc
from src.boundary.c3_selective_policy import wilson

MIN_PRECISION = 0.95
MIN_N = 20
POINT = "POINT_TRANSITION"


def buffered(tp, fp):
    return tp / (tp + fp + 1) if (tp + fp + 1) else float("nan")


def pick_threshold(scores, y, target=MIN_PRECISION):
    """Deepest cut down the ranking whose prefix still meets the BUFFERED
    precision. Selected on inner-fold scores only."""
    m = np.isfinite(scores)
    if m.sum() < 10 or len(set(y[m].tolist())) < 2:
        return None
    order = np.argsort(-scores[m])
    s, yy = scores[m][order], y[m][order]
    tp = fp = 0
    th = None
    for i in range(len(s)):
        tp += int(yy[i] == 1)
        fp += int(yy[i] == 0)
        if buffered(tp, fp) >= target:
            th = float(s[i])
    return th


def run(rows, y, groups, n_folds, seed, tag):
    folds = stratified_grouped_folds(groups, y, n_folds, seed=seed)
    sc = np.array([r["morphology"][POINT] for r in rows])
    kept = np.zeros(len(rows), bool)
    print(f"\n  {tag}: {len(rows)} events, {int(y.sum())} positive "
          f"({y.mean():.3f}), AUROC {_auroc(y, sc):.3f}")
    print(f"  {'fold':>4} {'n_test':>7} {'thr':>8} {'kept':>6} {'TP':>4} "
          f"{'FP':>4} {'prec':>6} {'buff':>6}")
    for fi, f in enumerate(folds):
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 20:
            continue
        th = pick_threshold(sc[tr], y[tr])
        if th is None:
            print(f"  {fi:>4} {int(te.sum()):>7} {'none':>8}   no cut on the "
                  f"training recordings reaches the buffered precision")
            continue
        hit = te & (sc >= th)
        kept |= hit
        tp = int(y[hit].sum())
        fp = int(hit.sum() - tp)
        print(f"  {fi:>4} {int(te.sum()):>7} {th:>8.4f} {int(hit.sum()):>6} "
              f"{tp:>4} {fp:>4} {(tp / max(1, tp + fp)):>6.3f} "
              f"{buffered(tp, fp):>6.3f}")
    tp = int(y[kept].sum())
    fp = int(kept.sum() - tp)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    lo, hi = wilson(tp, tp + fp) if tp + fp else (float("nan"),) * 2
    ok = (int(kept.sum()) >= MIN_N and prec >= MIN_PRECISION
          and buffered(tp, fp) >= MIN_PRECISION)
    print(f"  AUTO_KEEP n {int(kept.sum())}   correct {tp}   wrong {fp}   "
          f"precision {prec:.3f} Wilson [{lo:.3f}, {hi:.3f}]   buffered "
          f"{buffered(tp, fp):.3f}")
    print(f"  coverage {kept.sum() / len(rows):.1%}   BAR "
          f"(n>={MIN_N}, prec>={MIN_PRECISION}, buffered>={MIN_PRECISION}): "
          f"{'MET' if ok else 'NOT MET'}")
    if fp:
        bad = Counter(rows[i].get("morphology_true") or
                      f"no target ({rows[i].get('subtype')})"
                      for i in range(len(rows)) if kept[i] and not y[i])
        print(f"  what the {fp} wrong keeps actually are: {dict(bad)}")
    return kept, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    blob = json.load(open(a.predictions, encoding="utf-8"))
    rows = [r for r in blob["events"]
            if r["morphology"]
            and np.isfinite(list(r["morphology"].values())[0])]
    groups = [r["recording_id"] for r in rows]
    print(f"{os.path.basename(a.predictions)}: {len(rows)} events with an "
          f"out-of-fold P(POINT)")

    print(f"\n{'=' * 78}\nDEPLOYMENT POPULATION: untargeted events count as "
          f"negatives\n{'=' * 78}")
    y_all = np.array([r.get("morphology_true") == POINT for r in rows], float)
    kept, ok_all = run(rows, y_all, groups, a.n_folds, a.seed,
                       "all events the model will meet")

    print(f"\n{'=' * 78}\nTARGETED SUBSET ONLY: the flattering column, shown "
          f"so the gap is visible\n{'=' * 78}")
    sub = [r for r in rows if r.get("morphology_true")]
    y_sub = np.array([r["morphology_true"] == POINT for r in sub], float)
    _, ok_sub = run(sub, y_sub, [r["recording_id"] for r in sub],
                    a.n_folds, a.seed, "events that have a morphology target")
    print(f"\n  The difference between these two blocks is the same exclusion "
          f"that made clean-binary precision read 0.967 while\n  the full "
          f"taxonomy read 0.784. Only the first block describes deployment.")

    print(f"\n{'=' * 78}\nWHAT A PASS WOULD STILL NOT BUY\n{'=' * 78}")
    rel = [r for r in rows if r.get("relation_true")
           and r["relation_true"] != "UNDECIDABLE"]
    print(f"  AUTO_KEEP needs POINT and the candidate ON the boundary. "
          f"Relation is supervised on {len(rel)} of {len(rows)} events\n  over "
          f"{len({r['recording_id'] for r in rel})} recordings and receives no "
          f"gradient, so the second condition is not measurable here.")
    if kept is not None and kept.any():
        k = [rows[i] for i in range(len(rows)) if kept[i]]
        known = [r for r in k if r.get("relation_true")
                 and r["relation_true"] != "UNDECIDABLE"]
        if known:
            ex = sum(1 for r in known if r["relation_true"] == "EXACT")
            print(f"  Of the {len(k)} events this threshold keeps, "
                  f"{len(known)} have a relation truth and {ex} of those are "
                  f"EXACT.\n  The other {len(known) - ex} are real transitions "
                  f"at the wrong timestamp -- admitted, and wrong.")

    if a.out:
        json.dump({"bar_met_deployment": bool(ok_all),
                   "bar_met_targeted_only": bool(ok_sub),
                   "kept": [rows[i]["event_id"] for i in range(len(rows))
                            if kept[i]]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
