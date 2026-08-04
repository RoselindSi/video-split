"""Read-only recheck of the `worst_fold_gain` criterion in an ALREADY-WRITTEN
pairwise_verifier report. No retraining, no refitting, no modification of any
frozen artifact -- it only re-reads per_fold_auroc arrays that the report
already stores and recomputes one number from them.

WHY: three gates in this project computed the "worst fold" criterion as
    min(candidate_per_fold) - min(baseline_per_fold)
i.e. the change in the MINIMUM fold AUROC, where the two minima need not come
from the same fold. What `max_worst_fold_drop` was always meant to bound is how
far any INDIVIDUAL fold regresses:
    min_i (candidate_i - baseline_i)

These are not the same number, and the difference is one-directional:

    min(c) - min(b) >= min_i (c_i - b_i)   for all c, b

    Proof: let j = argmin_i c_i and k = argmin_i b_i. Then
    min(c) - min(b) = c_j - b_k >= c_j - b_j >= min_i (c_i - b_i),
    using b_k = min(b) <= b_j.

So the quantity that was computed is ALWAYS >= the true worst per-fold delta:
the criterion was systematically LENIENT and can only ever have produced a
false PASS, never a false FAIL. That is why C1 and C2 need no recheck -- both
were DO-NOT-ADOPT, and a too-lenient criterion cannot cause a wrong rejection
(both also failed on the gain criterion independently). P1 is the one decision
at risk, because P1 PASSED its gate, and that pass is what authorized freezing
the config and collecting batch3.

Usage (server, read-only):
    python -m src.boundary.gate_worstfold_recheck \
        --report /workspace/tr1/results/hal/pairwise/report_clean_v1.json
"""
from __future__ import annotations

import argparse
import json


def recheck(report, baseline_arm=None, candidate_arm=None, max_drop=None):
    gate = report.get("gate", {})
    cfg = gate.get("config", {})
    base_key = baseline_arm or cfg.get("baseline_arm", "P0_v1_logreg")
    cand_key = candidate_arm or cfg.get("candidate_arm", "P1_pairwise_proj")
    if max_drop is None:
        max_drop = cfg.get("max_worst_fold_drop", 0.05)

    for k in (base_key, cand_key):
        if k not in report:
            raise SystemExit(f"arm {k!r} not in report (arms present: "
                             f"{[k2 for k2 in report if isinstance(report.get(k2), dict) and 'per_fold_auroc' in report[k2]]})")
    bf = report[base_key]["per_fold_auroc"]
    cf = report[cand_key]["per_fold_auroc"]
    paired = [(c, b) for c, b in zip(cf, bf)
              if c is not None and b is not None and c == c and b == b]
    if not paired:
        raise SystemExit("no folds with both arms scored")

    deltas = [c - b for c, b in paired]
    as_computed = min(c for c, _ in paired) - min(b for _, b in paired)
    corrected = min(deltas)
    return {
        "baseline_arm": base_key, "candidate_arm": cand_key,
        "max_worst_fold_drop": max_drop,
        "baseline_per_fold": [b for _, b in paired],
        "candidate_per_fold": [c for c, _ in paired],
        "per_fold_delta": deltas,
        "as_computed_min_fold_change": as_computed,
        "corrected_worst_per_fold_delta": corrected,
        "as_computed_pass": as_computed >= -max_drop,
        "corrected_pass": corrected >= -max_drop,
        "stored_value": gate.get("checks", {}).get("worst_fold_gain", {}).get("value"),
        "stored_gate_passed": gate.get("passed"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="pairwise_verifier report JSON (read-only)")
    ap.add_argument("--baseline_arm")
    ap.add_argument("--candidate_arm")
    ap.add_argument("--max_worst_fold_drop", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()

    with open(a.report, encoding="utf-8") as f:
        report = json.load(f)
    r = recheck(report, a.baseline_arm, a.candidate_arm, a.max_worst_fold_drop)

    print(f"candidate {r['candidate_arm']} vs baseline {r['baseline_arm']}")
    print(f"{'fold':>5} {'baseline':>9} {'candidate':>10} {'delta':>8}")
    for i, (b, c, d) in enumerate(zip(r["baseline_per_fold"], r["candidate_per_fold"],
                                      r["per_fold_delta"])):
        print(f"{i:>5} {b:>9.3f} {c:>10.3f} {d:>+8.3f}")

    if r["stored_value"] is not None:
        print(f"\nvalue stored in the report:      {r['stored_value']:+.4f}"
              f"  (gate recorded as {'PASSED' if r['stored_gate_passed'] else 'NOT PASSED'})")
    print(f"as computed (min-of-minima):     {r['as_computed_min_fold_change']:+.4f}  "
          f"-> {'PASS' if r['as_computed_pass'] else 'FAIL'}")
    print(f"CORRECTED (worst per-fold delta):{r['corrected_worst_per_fold_delta']:+.4f}  "
          f"-> {'PASS' if r['corrected_pass'] else 'FAIL'}  "
          f"(needs >= {-r['max_worst_fold_drop']:+.3f})")

    if r["as_computed_pass"] and not r["corrected_pass"]:
        print("\n  !! THIS CRITERION FLIPS. The gate's worst-fold check passed only "
              "because of the min-of-minima formulation; a single fold regressed "
              "by more than the pre-registered bound. Whether the GATE overall "
              "flips depends on the other criteria, which are unaffected -- but "
              "this one was not actually satisfied, and any decision that cited "
              "it (freezing the config, collecting batch3) rests on a criterion "
              "that was not met.")
    elif r["as_computed_pass"]:
        print("\n  -> this criterion holds under the corrected formulation too: "
              "the lenient computation did not change the outcome here, so the "
              "decision that cited it stands as recorded.")
    else:
        print("\n  -> the criterion failed even as originally computed; the "
              "correction only makes it fail by more.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
