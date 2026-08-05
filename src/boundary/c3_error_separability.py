"""Is there ANY available feature that separates the policy's automatic errors
from its automatic successes?

This is the computed half of the `separable_by` column in the wrong-keep
review sheet, and it decides whether a Stage 0 observability gate is learnable
from what already exists. The visual audit answers what is happening on
screen; this answers whether anything we can measure knows about it. Neither
replaces the other -- a mechanism visible only to a human is a reason to build
a new feature, while a mechanism no feature sees is a reason not to build a
gate on these features at all.

The scope is deliberately narrow: only events the policy decided
AUTOMATICALLY. Review events are not errors, and including them would measure
the policy's confidence rather than its correctness.

Two things it reports, and the first is the more important:

  WHAT THE DISAGREEMENT RULE ACTUALLY BUYS. Note first what is NOT worth
  reporting: under an agreement policy every automatically-decided event has
  both branches on the same side, because that is what the rule requires. So
  "the branches agree on all 14 errors" is a restatement of the rule, not an
  observation about the data -- it was briefly mistaken for one here, which is
  why the point is written down. The answerable question is about the events
  the rule DIVERTED: of those sent to REVIEW for disagreement, how many would
  have been wrong had P1 alone decided them, against the automatic stream's
  own error rate? That measures the protection the second branch buys, and
  the size of that band bounds how much it could ever matter.

  PER-FEATURE SEPARABILITY. For each numeric column, the AUROC of that feature
  against the error label, plus what gating on it would actually cost. An
  AUROC near 0.5 everywhere is a real answer, not a failed experiment: it says
  the information needed is not in these features, and a Stage 0 gate built on
  them would be fitting noise.

Usage:
    python -m src.boundary.c3_error_separability \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --events /workspace/tr1/results/hal/c3/local_events.csv \
        --events /workspace/tr1/results/hal/c3/local_events_batch3.csv \
        --out /workspace/tr1/results/hal/c3/error_separability.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from src.boundary.state_adapter import _auroc

CORRECT = "sharp_visible_transition"
SKIP = {"event_id", "recording_id", "source", "y", "subtype", "decision",
        "reason", "policy_role"}


def is_error(r):
    """An automatic decision is wrong when it accepts something that is not a
    sharp transition, or rejects something that is."""
    sharp = (r["subtype"] == CORRECT) or (not r["subtype"] and r["y"] == "1")
    if r["decision"] == "AUTO_KEEP":
        return not sharp
    if r["decision"] == "AUTO_REJECT":
        return sharp
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--events", action="append",
                    help="event dumps, for columns the decisions file lacks")
    ap.add_argument("--score_a", default="P1 (global) alone")
    ap.add_argument("--score_b", default="local alone")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()

    extra = {}
    for p in a.events or []:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                extra[r["event_id"]] = r

    rows = list(csv.DictReader(open(a.decisions, newline="", encoding="utf-8")))
    auto = [r for r in rows if r["decision"] in ("AUTO_KEEP", "AUTO_REJECT")]
    for r in auto:
        r.update({k: v for k, v in extra.get(r["event_id"], {}).items()
                  if k not in r or r.get(k) in (None, "")})
    err = np.array([1.0 if is_error(r) else 0.0 for r in auto])
    print(f"{len(rows)} events, {len(auto)} decided automatically, "
          f"{int(err.sum())} of them wrong")
    print(f"  wrong by decision: "
          f"{dict(Counter(r['decision'] for r, e in zip(auto, err) if e))}")
    print(f"  wrong by subtype:  "
          f"{dict(Counter(r['subtype'] or '(none)' for r, e in zip(auto, err) if e))}")
    if err.sum() == 0 or err.sum() == len(err):
        raise SystemExit("all automatic decisions are the same -- nothing to separate")

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, "") else np.nan
                         for r in auto])

    print("\n=== is the disagreement rule protecting anything? ===")
    A, B = col(a.score_a), col(a.score_b)
    # For an agreement policy, EVERY automatically-decided event has both
    # branches on the same side -- that is what the rule requires, so
    # "the branches agree on all the errors" is a restatement of the rule and
    # says nothing about the data. (Recorded because it was briefly mistaken
    # for a finding.) The informative question is about the events the rule
    # DID divert: of those routed to REVIEW for disagreement, how many would
    # have been errors had P1 alone decided them? That is the protection the
    # second branch actually buys.
    dis = [r for r in rows if r["reason"] == "global_local_disagreement"]
    print(f"  {len(dis)} events routed to REVIEW because the branches disagreed")
    if dis:
        def sharp(r):
            return (r["subtype"] == CORRECT) or (not r["subtype"] and r["y"] == "1")
        pa = np.array([float(r[a.score_a]) if r.get(a.score_a) else np.nan for r in dis])
        would_keep = pa >= 0.75
        would_err = sum(1 for r, k in zip(dis, would_keep)
                        if (k and not sharp(r)) or (not k and sharp(r)))
        print(f"  had P1 alone decided them at the same thresholds, "
              f"{would_err} of {len(dis)} would have been WRONG "
              f"({would_err / len(dis):.3f})")
        base_err = err.mean()
        print(f"  the automatic stream's own error rate is {base_err:.3f}")
        if would_err / len(dis) > base_err:
            print("  -> the diverted events ARE harder than average, so the "
                  "second branch is buying real protection, just not enough of "
                  "it to matter at this volume")
        else:
            print("  -> the diverted events are NO harder than the ones kept "
                  "automatic, so routing them to REVIEW costs coverage without "
                  "buying safety")
        print(f"  ({len(dis)} events is {len(dis) / len(rows):.1%} of the set -- "
              f"even perfect protection here cannot move the headline numbers)")
    ew = (err == 1) & np.isfinite(A) & np.isfinite(B)
    if ew.sum():
        print(f"\n  errors sit at {a.score_a} in "
              f"[{np.nanmin(A[ew]):.3f}, {np.nanmax(A[ew]):.3f}] and "
              f"{a.score_b} in [{np.nanmin(B[ew]):.3f}, {np.nanmax(B[ew]):.3f}] "
              f"-- whether that differs from the CORRECT decisions is the "
              f"per-feature table below, not something the ranges alone show")

    print("\n=== per-feature separability of ERRORS from correct decisions ===")
    print("  AUROC 0.5 = the feature knows nothing. Values are folded above 0.5 "
          "with the direction shown, since a feature that predicts correctness "
          "is as usable as one that predicts error.")
    print(f"  {'feature':<34} {'AUROC':>7} {'dir':>5} {'err median':>11} "
          f"{'ok median':>10}")
    feats, results = [c for c in auto[0] if c not in SKIP], {}
    for c in sorted(feats):
        v = col(c)
        m = np.isfinite(v)
        if m.sum() < len(auto) * 0.5 or len(set(v[m].tolist())) < 3:
            continue
        au = _auroc(err[m], v[m])
        folded = max(au, 1 - au)
        d = "higher" if au >= 0.5 else "lower"
        em, om = np.median(v[m & (err == 1)]), np.median(v[m & (err == 0)])
        results[c] = {"auroc": au, "folded": folded, "err_median": float(em),
                      "ok_median": float(om)}
        print(f"  {c:<34} {folded:>7.3f} {d:>5} {em:>11.3f} {om:>10.3f}")

    best = max(results.items(), key=lambda kv: kv[1]["folded"]) if results else None
    print()
    if best is None or best[1]["folded"] < 0.65:
        print("  NOTHING separates the errors: the best feature reaches "
              f"{best[1]['folded']:.3f} if any. The information a Stage 0 gate "
              "would need is not in these features, so building one on them "
              "would be fitting noise. That is a result -- it says the next "
              "move is a NEW observable, not a new classifier over the old "
              "ones.")
    else:
        print(f"  best separator: {best[0]} at AUROC {best[1]['folded']:.3f} "
              f"(error median {best[1]['err_median']:.3f} vs correct "
              f"{best[1]['ok_median']:.3f}). "
              f"Worth checking what gating on it costs before believing it: with "
              f"{int(err.sum())} errors this is a small-sample AUROC and one "
              f"event moves it noticeably.")

    print("\n=== what a stricter reliability gate would cost ===")
    rel = col("reliability")
    if np.isfinite(rel).any():
        print(f"  {'threshold':>10} {'errors caught':>14} {'correct lost':>13} "
              f"{'auto coverage':>14}")
        for th in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
            below = rel < th
            print(f"  {th:>10.2f} {int((below & (err == 1)).sum()):>7}/"
                  f"{int(err.sum()):<6} {int((below & (err == 0)).sum()):>13} "
                  f"{1 - below.mean():>14.3f}")
        print("  Each row sends everything below the threshold to REVIEW. "
              "Catching errors by raising it only helps if the correct-lost "
              "column stays small -- and choosing a row here because it looks "
              "good is threshold tuning on development data, which is what the "
              "pre-registration exists to prevent. The table is for sizing the "
              "trade, not for picking one.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"n_auto": len(auto), "n_error": int(err.sum()),
                       "n_disagreement_routed": len(dis),
                       "features": results}, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
