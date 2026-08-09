"""What AUROC does a PERSON get on these events? The only free way to tell
whether 0.73 is a weak model or a hard label.

Every scorer this project has -- P1, local, fused, and the temporal student --
sits between 0.73 and 0.76 on batch3 and cannot be told apart from the others
by a paired bootstrap. A shared ceiling like that is either the features or the
labels, and nothing measured so far separates them. A second annotator does:
if one person predicts another's calls no better than the model does, the
model is at the label's ceiling and a new representation buys nothing.

THE SCORE IS ORDINAL, not binary. Each annotator gave a call and a confidence,
so `sharp` at 3_sure ranks above `sharp` at 1_guess, and `cannot` sits in the
middle -- an abstention is not a vote for `same`. Collapsing to a binary
throws away the ranking that AUROC is computed on and would understate the
human by construction.

THE POPULATION IS THE HARDEST SLICE THERE IS, and that is the main limit on
what this can conclude. These 36 events were drawn from the REVIEW band --
the ones the policy could not decide -- so a human ceiling measured here is a
LOWER bound on the human ceiling over the whole population, and the model's
0.731 comes from a broader draw. The two numbers are not directly comparable
and the comparison that IS valid is the one made on these same 36 events, in
the last block.

WHAT WOULD SETTLE IT EITHER WAY:

  model ~= human on these events   the labels are the ceiling here; a new
                                   representation has nothing to reach for
                                   and the annotation work is what pays
  human >> model                   a person extracts something from the video
                                   the features do not carry, which is the
                                   representation case, and the gap is its size

Usage:
    python -m src.auditor.boundary.human_ceiling_auroc \
        --a data/gold/observable_audit_your_call_36.csv \
        --b data/gold/observable_audit_annotator2_36.csv \
        --labels data/gold/boundary_v1_labels_ontology_v2.json \
        --predictions .../boundary_v1_oof_ontology_v2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

POINT = "POINT_TRANSITION"
# ordinal, least to most boundary-like. `cannot` is an abstention and belongs
# between the two votes rather than with either
GRADE = {("same", "3_sure"): 0.0, ("same", "2_lean"): 1.0,
         ("same", "1_guess"): 2.0, ("cannot", ""): 3.0,
         ("sharp", "1_guess"): 4.0, ("sharp", "2_lean"): 5.0,
         ("sharp", "3_sure"): 6.0}


def read_calls(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            call = next((v for k, v in r.items()
                         if k and k.startswith("your_call")), "") or ""
            conf = next((v for k, v in r.items()
                         if k and k.startswith("confidence")), "") or ""
            e = r.get("event_id")
            call, conf = call.strip().lower(), conf.strip()
            if not e or not call:
                continue
            g = GRADE.get((call, conf))
            if g is None:
                g = GRADE.get((call, ""), 3.0)
            out[e] = {"call": call, "conf": conf, "grade": g}
    return out


def auroc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    o = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[o] = np.arange(len(p)) + 1.0
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    pos, neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) /
                 max(pos * neg, 1))


def boot(y, p, groups, n_boot=2000, seed=0):
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = [i for k in rng.integers(0, len(keys), len(keys))
               for i in by[keys[k]]]
        v = auroc([y[i] for i in idx], [p[i] for i in idx])
        if np.isfinite(v):
            out.append(v)
    if len(out) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="annotator 1 sheet")
    ap.add_argument("--b", required=True, help="annotator 2 sheet")
    ap.add_argument("--labels", help="boundary_v1_labels json, for the stored "
                                     "morphology target")
    ap.add_argument("--predictions", help="boundary_v1_oof json")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A, B = read_calls(a.a), read_calls(a.b)
    ev = sorted(set(A) & set(B))
    print(f"{len(A)} calls in {os.path.basename(a.a)}, {len(B)} in "
          f"{os.path.basename(a.b)}, {len(ev)} events in both")
    print(f"  annotator 1: {dict(Counter(A[e]['call'] for e in ev))}")
    print(f"  annotator 2: {dict(Counter(B[e]['call'] for e in ev))}")

    stored, pred = {}, {}
    if a.labels and os.path.exists(a.labels):
        for r in json.load(open(a.labels, encoding="utf-8"))["events"]:
            if r.get("morphology"):
                stored[r["event_id"]] = r["morphology"]
    if a.predictions and os.path.exists(a.predictions):
        for r in json.load(open(a.predictions, encoding="utf-8"))["events"]:
            if r.get("morphology"):
                pred[r["event_id"]] = r["morphology"][POINT]

    rec = {e: e.split("_t")[0] for e in ev}
    import re
    rx = re.compile(r"^(recording_\d+)")
    rec = {e: (rx.match(e).group(1) if rx.match(e) else e) for e in ev}
    print(f"  over {len(set(rec.values()))} recordings")

    def run(name, target_fn, score_fn, pool=ev):
        sel = [e for e in pool
               if target_fn(e) is not None and score_fn(e) is not None]
        y = [target_fn(e) for e in sel]
        p = [score_fn(e) for e in sel]
        if len(set(y)) < 2:
            print(f"  {name:<46} {len(sel):>4}   only one class, withheld")
            return
        v = auroc(y, p)
        lo, hi = boot(y, p, [rec[e] for e in sel], a.n_boot, a.seed)
        print(f"  {name:<46} {len(sel):>4}   {v:.3f}  [{lo:.3f}, {hi:.3f}]")

    bin_a = lambda e: (1.0 if A[e]["call"] == "sharp"
                       else 0.0 if A[e]["call"] == "same" else None)
    bin_b = lambda e: (1.0 if B[e]["call"] == "sharp"
                       else 0.0 if B[e]["call"] == "same" else None)

    print(f"\n{'=' * 84}\nHUMAN AGAINST HUMAN\n{'=' * 84}")
    print(f"  {'comparison':<46} {'n':>4}   AUROC  [95% CI, grouped]")
    run("annotator 2's graded call vs annotator 1's call",
        bin_a, lambda e: B[e]["grade"])
    run("annotator 1's graded call vs annotator 2's call",
        bin_b, lambda e: A[e]["grade"])
    both = [e for e in ev if bin_a(e) is not None and bin_b(e) is not None]
    agree = sum(1 for e in both if bin_a(e) == bin_b(e))
    print(f"  plain agreement on the {len(both)} events where both voted: "
          f"{agree}/{len(both)} = {agree / max(len(both), 1):.3f}")

    if stored:
        print(f"\n{'=' * 84}\nEACH AGAINST THE STORED LABEL\n{'=' * 84}")
        tgt = lambda e: (1.0 if stored.get(e) == POINT else
                         0.0 if stored.get(e) == "NO_TRANSITION" else None)
        print(f"  {'comparison':<46} {'n':>4}   AUROC  [95% CI, grouped]")
        run("annotator 1 vs stored morphology", tgt,
            lambda e: A[e]["grade"])
        run("annotator 2 vs stored morphology", tgt,
            lambda e: B[e]["grade"])
        if pred:
            run("MODEL P(POINT) vs stored morphology", tgt,
                lambda e: pred.get(e))

    if pred:
        print(f"\n{'=' * 84}\nTHE COMPARISON THAT IS ACTUALLY VALID"
              f"\n{'=' * 84}")
        print("  Same 36 events, same target, model against human. The "
              "batch3 0.731 came from a broader draw and cannot be\n  set "
              "beside a number computed on the REVIEW band.")
        print(f"\n  {'comparison':<46} {'n':>4}   AUROC  [95% CI, grouped]")
        run("annotator 2 predicting annotator 1", bin_a,
            lambda e: B[e]["grade"])
        run("MODEL predicting annotator 1", bin_a, lambda e: pred.get(e))
        run("annotator 1 predicting annotator 2", bin_b,
            lambda e: A[e]["grade"])
        run("MODEL predicting annotator 2", bin_b, lambda e: pred.get(e))
        print("\n  If the model's rows sit inside the human rows' intervals, "
              "then on the hardest slice a person is no better\n  than the "
              "features are, and the annotation work is what pays. If the "
              "human rows are clearly higher, the gap is\n  what a new "
              "representation would have to recover, and its size is the "
              "estimate of what that work is worth.")
        print("\n  Either way this is 36 events from one band. It bounds the "
              "hardest slice and says nothing directly about the\n  "
              "population the deployment number is computed on.")


if __name__ == "__main__":
    main()
