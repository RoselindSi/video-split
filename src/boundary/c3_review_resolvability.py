"""Can any part of the REVIEW band be automated safely?

c3_error_separability.py asks whether the 161 AUTOMATIC decisions contain
detectable errors. This asks the different question that actually bears on the
goal: of the 251 events sent to a human, is there a subregion that could be
decided automatically at the required precision? Those are not the same
question and the first cannot answer the second.

Split by the reason the event was diverted, because the three have nothing in
common:

  insufficient_margin        both branches uncertain. The only one where more
                             review could plausibly be recovered, and the one
                             examined most closely here.
  low_local_reliability      the local score is not trustworthy, so any
                             analysis of it on this subset is analysis of an
                             artefact. What matters is whether P1 ALONE can
                             still decide the extremes.
  global_local_disagreement  measured at 11 of 15 wrong had P1 alone decided
                             them, against an 8.7% error rate in the automatic
                             stream. These are the hardest events in the set
                             and are not a review-reduction target; the
                             analysis reports them so that stays visible.

TWO TARGETS, kept apart:

  A. clean-binary. Among reviewed sharp/same-action events, is there a
     high-precision automatable region? Thresholds are chosen by
     RECORDING-GROUPED cross-validation and scored on held-out recordings --
     picking a threshold on the same events it is then evaluated on would
     manufacture whatever precision was asked for.
  B. full-taxonomy observability. Can any feature separate clean-observable
     events from gradual/offscreen/camera/annotation/ambiguous? If the old
     features score near chance here too, that is the evidence that a new
     observable is required rather than a new classifier.

Both report grouped-bootstrap intervals and a permutation baseline, because an
AUROC quoted bare at these sample sizes cannot be told from what shuffling
produces.

Usage:
    python -m src.boundary.c3_review_resolvability \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --events /workspace/tr1/results/hal/c3/local_events.csv \
        --events /workspace/tr1/results/hal/c3/local_events_batch3.csv \
        --min_precision 0.95 \
        --out /workspace/tr1/results/hal/c3/review_resolvability.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from src.boundary.c3_error_separability import auroc_stats
from src.boundary.c3_selective_policy import wilson

SHARP = "sharp_visible_transition"
SAME = "same_action_internal_motion"
CLEAN = (SHARP, SAME)
SKIP = {"event_id", "recording_id", "source", "y", "subtype", "decision",
        "reason", "policy_role"}


def grouped_folds(recs, k, seed=0):
    u = sorted(set(recs))
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    return [set(u[i::k]) for i in range(k)]


def best_threshold(scores, is_pos, target, side):
    """Threshold on the TRAINING split giving the most coverage while holding
    precision >= target. side='keep' takes from the top, 'reject' from the
    bottom. Returns None when no threshold reaches the target at all, which is
    itself the answer for that fold."""
    order = np.argsort(-scores if side == "keep" else scores)
    want = is_pos if side == "keep" else ~is_pos
    best = None
    hit = 0
    for n, i in enumerate(order, start=1):
        hit += bool(want[i])
        if hit / n >= target:
            best = scores[i]
    return best


def evaluate_side(rows, score_col, target, side, k=5, seed=0):
    """Out-of-fold: choose the threshold on training recordings, apply to the
    held-out ones, pool. Never evaluates a threshold on the events that chose
    it."""
    s = np.array([float(r[score_col]) if r.get(score_col) not in (None, "") else np.nan
                  for r in rows])
    pos = np.array([r["subtype"] == SHARP for r in rows])
    recs = [r["recording_id"] for r in rows]
    m = np.isfinite(s)
    if m.sum() < 20 or len(set(pos[m].tolist())) < 2:
        return None
    folds = grouped_folds([r for r, k_ in zip(recs, m) if k_], k, seed)
    n_auto = n_right = 0
    n_folds_with_threshold = 0
    for f in folds:
        te = np.array([r in f for r in recs]) & m
        tr = (~np.array([r in f for r in recs])) & m
        if te.sum() < 2 or tr.sum() < 10:
            continue
        th = best_threshold(s[tr], pos[tr], target, side)
        if th is None:
            continue
        n_folds_with_threshold += 1
        sel = (s[te] >= th) if side == "keep" else (s[te] <= th)
        want = pos[te] if side == "keep" else ~pos[te]
        n_auto += int(sel.sum())
        n_right += int((sel & want).sum())
    return {"n": int(m.sum()), "n_auto": n_auto, "n_right": n_right,
            "precision": (n_right / n_auto) if n_auto else float("nan"),
            "precision_wilson": wilson(n_right, n_auto),
            "coverage": n_auto / int(m.sum()) if m.sum() else float("nan"),
            "folds_with_threshold": n_folds_with_threshold, "n_folds": len(folds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--events", action="append")
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_auto", type=int, default=20,
                    help="a region must automate at least this many events "
                         "out-of-fold before its precision is worth quoting")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    extra = {}
    for p in a.events or []:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                extra[r["event_id"]] = r
    rows = list(csv.DictReader(open(a.decisions, newline="", encoding="utf-8")))
    for r in rows:
        r.update({k: v for k, v in extra.get(r["event_id"], {}).items()
                  if k not in r or r.get(k) in (None, "")})
    rev = [r for r in rows if r["decision"] == "REVIEW"]
    print(f"{len(rows)} events, {len(rev)} in REVIEW ({len(rev) / len(rows):.1%})")
    print(f"  by reason:  {dict(Counter(r['reason'] for r in rev))}")
    print(f"  by subtype: {dict(Counter(r['subtype'] or '(none)' for r in rev))}")

    score_cols = [c for c in rows[0] if c not in SKIP]
    report = {"n_review": len(rev), "min_precision": a.min_precision, "reasons": {}}

    print(f"\n{'=' * 72}\nA. CLEAN-BINARY: is there an automatable region inside REVIEW?"
          f"\n   thresholds chosen on training recordings, scored on held-out ones,"
          f"\n   target precision {a.min_precision}\n{'=' * 72}")
    for reason in sorted({r["reason"] for r in rev}):
        sub = [r for r in rev if r["reason"] == reason and r["subtype"] in CLEAN]
        n_sharp = sum(1 for r in sub if r["subtype"] == SHARP)
        print(f"\n  {reason}: {len(sub)} clean events "
              f"({n_sharp}+ / {len(sub) - n_sharp}-), "
              f"{len({r['recording_id'] for r in sub})} recordings")
        if len(sub) < 20:
            print("    too few to cross-validate a threshold on")
            continue
        res = {}
        print(f"    {'score':<32} {'side':>7} {'auto':>10} {'precision':>10} "
              f"{'95% CI':>11} {'cover':>7} {'folds':>5}")
        for c in score_cols:
            for side in ("keep", "reject"):
                r_ = evaluate_side(sub, c, a.min_precision, side, a.folds, a.seed)
                if not r_ or r_["n_auto"] == 0:
                    continue
                res[f"{c}::{side}"] = r_
                lo, hi = r_["precision_wilson"]
                print(f"    {c:<32} {side:>7} {r_['n_auto']:>4}/{r_['n']:<5} "
                      f"{r_['precision']:>10.3f} [{lo:.2f},{hi:.2f}] "
                      f"{r_['coverage']:>7.3f} "
                      f"{r_['folds_with_threshold']}/{r_['n_folds']:>3}")
        if not res:
            print("    NOTHING reached the target precision on any held-out fold. "
                  "For this subset the reviewed events are not separable at the "
                  "precision required, with any of these scores.")
        else:
            # Two failure modes to avoid at once. A precision of 1.000 over
            # five events is not evidence -- its Wilson lower bound is near
            # 0.57 -- so a minimum region size is required. But demanding the
            # lower BOUND clear the target is the other extreme: at 30 of 31
            # correct the bound is 0.84, and clearing 0.95 needs roughly 73
            # consecutive correct, so that guard rejects genuinely good
            # regions too (it did, on a fixture with a planted separator).
            # So: point precision must clear the target on a region of at
            # least --min_auto events, and the interval is REPORTED for
            # judgement rather than used as a gate.
            eligible = {k: v for k, v in res.items()
                        if v["n_auto"] >= a.min_auto
                        and v["precision"] >= a.min_precision}
            big = {k: v for k, v in res.items() if v["n_auto"] >= a.min_auto}
            if not eligible:
                if big:
                    nb = max(big.items(), key=lambda kv: kv[1]["precision"])
                    print(f"    Nothing reaches {a.min_precision} on a region of "
                          f"at least {a.min_auto} events. Best that is large "
                          f"enough to mean anything: {nb[0]}, {nb[1]['n_auto']} "
                          f"events at {nb[1]['precision']:.3f} "
                          f"[{nb[1]['precision_wilson'][0]:.2f}, "
                          f"{nb[1]['precision_wilson'][1]:.2f}].")
                else:
                    print(f"    No region reaches {a.min_auto} automated events "
                          f"out-of-fold at all -- whatever thresholds held on "
                          f"training recordings selected almost nothing on new "
                          f"ones.")
            else:
                b = max(eligible.items(), key=lambda kv: kv[1]["coverage"])
                lo, hi = b[1]["precision_wilson"]
                print(f"    {b[0]} reaches {b[1]['precision']:.3f} "
                      f"[{lo:.2f}, {hi:.2f}] at coverage {b[1]['coverage']:.3f} "
                      f"out-of-fold on {b[1]['n_auto']} events -- worth pursuing. "
                      f"It recovers {b[1]['coverage'] * len(sub):.0f} of "
                      f"{len(rev)} reviewed events, "
                      f"{b[1]['coverage'] * len(sub) / len(rows):.1%} of the set, "
                      f"and the interval is what a held-out run would have to "
                      f"beat, not the point estimate.")
        report["reasons"][reason] = res

    print(f"\n{'=' * 72}\nB. OBSERVABILITY: can any feature tell clean-observable from "
          f"non-clean?\n{'=' * 72}")
    lab = np.array([1.0 if r["subtype"] in CLEAN else 0.0 for r in rev])
    recs = [r["recording_id"] for r in rev]
    print(f"  {int(lab.sum())} clean vs {int((1 - lab).sum())} non-clean, "
          f"{len(set(recs))} recordings")
    if len(set(lab.tolist())) < 2:
        print("  only one class present -- nothing to separate")
    else:
        print(f"  {'feature':<32} {'AUROC':>7} {'95% CI':>18} {'chance p95':>11} "
              f"{'dir':>7}")
        obs, best = {}, None
        for c in sorted(score_cols):
            v = np.array([float(r[c]) if r.get(c) not in (None, "") else np.nan
                          for r in rev])
            m_ = np.isfinite(v)
            if m_.sum() < len(rev) * 0.5 or len(set(v[m_].tolist())) < 3:
                continue    # a constant or near-constant column has nothing to say
            st = auroc_stats(lab, v, recs, a.n_boot, a.n_boot, a.seed)
            if not st:
                continue
            obs[c] = st
            flag = "" if st["folded"] > st["perm_folded_p95"] else "  (below chance)"
            ci = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]"
            print(f"  {c:<32} {st['folded']:>7.3f} {ci:>18} "
                  f"{st['perm_folded_p95']:>11.3f} {st['direction']:>7}{flag}")
            if best is None or st["folded"] > obs[best]["folded"]:
                best = c
        report["observability"] = obs
        if best and obs[best]["folded"] <= obs[best]["perm_folded_p95"]:
            print("\n  No feature beats its own permutation baseline. Telling "
                  "clean-observable events from nuisance ones is not something "
                  "these scores can do, which is the direct evidence that a "
                  "Stage 0 gate needs a NEW observable rather than a better "
                  "classifier over the existing ones.")
        elif best:
            print(f"\n  {best} reaches {obs[best]['folded']:.3f} against a chance "
                  f"baseline of {obs[best]['perm_folded_p95']:.3f}. That is a "
                  f"starting point for a Stage 0 gate, though separating the "
                  f"classes is not the same as reaching a deployable precision "
                  f"-- section A's out-of-fold thresholds are what would decide "
                  f"that.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
