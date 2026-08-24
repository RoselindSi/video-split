"""The boundary arm's first risk-coverage curve on real predictions.

`auditor_v1 --calibrate` has only ever been run on a synthetic fixture. The
val-split logits saved during head training carry per-frame probabilities, the
annotated boundaries and the segments, which is everything needed to pick peaks,
match them at the current tolerance and ask what an automatic KEEP would buy.

TWO LIMITS, STATED BEFORE THE NUMBERS RATHER THAN UNDER THEM:

  THE DEFAULT ASSUMPTION IS NOT INDEPENDENT, and `--independent_because`
  overturns it only by making someone write a sentence. b2_logits.pt comes from
  `train_head_multi --val feat_val_full_noblur_multi.pt`, and that split drove
  early stopping. A threshold chosen here is chosen on data the model was tuned
  against, which is the exact overlap `auditor_v1`'s certificate refuses. So
  this run emits a certificate marked `independent: false`, and --run will
  decline to automate from it.

  It is still worth measuring, because the bias has a known direction. These
  numbers are OPTIMISTIC. If no threshold reaches useful coverage at high
  precision HERE, none will on held-out data either, and that conclusion
  transfers even though the rate does not.

  THE SCORE IS THE DETECTOR'S, NOT THE AUDITOR'S. A peak probability is not a
  morphology judgement, so no ontology veto can be applied and this is
  `--veto none` -- score-only, the mode documented as a diagnostic rather than
  an ontology auditor. The morphology head would add vetoes, which can only
  remove candidates, so it moves precision up and coverage down from here.

MATCHING IS GREEDY BY SCORE AND EACH BOUNDARY IS CLAIMED ONCE. Two peaks 0.4s
apart both sit within 1.0s of one boundary; counting both as correct would
report a duplicate as a success, and duplicates were 22 of the errors in the
July audit.

Usage:
    python -m src.auditor.boundary.detector_calibration \
        --logits ~/Downloads/tr1_audits/results/boundary/b2_logits.pt \
        --emit_certificate results/boundary_cert_valsplit.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TOL_S = 1.0        # 2026-08-19; see memory/tolerance-is-1s.md
MIN_GAP_S = 1.0    # as deployed in the July error audit
BASE_THR = 0.45    # the candidate pool: what the detector would propose at all


def peaks(prob, times, base_thr, min_gap_s):
    """Local maxima above base_thr, thinned by min_gap, highest score first.

    Thinning by score rather than by time is what makes the kept peak the
    strongest of a cluster; taking the earliest would hand the evaluation a
    weaker score for the same event and understate every threshold."""
    p, t = np.asarray(prob, float), np.asarray(times, float)
    hi = np.where(p >= base_thr)[0]
    loc = [i for i in hi
           if (i == 0 or p[i] >= p[i - 1]) and (i == len(p) - 1 or p[i] >= p[i + 1])]
    out = []
    for i in sorted(loc, key=lambda j: -p[j]):
        if all(abs(t[i] - t[j]) >= min_gap_s for j in out):
            out.append(i)
    return sorted(out, key=lambda j: t[j]), p, t


def match(idx, p, t, gt, tol):
    """(time, score, is_true) per peak. Each boundary may be claimed once."""
    gt = sorted(float(g) for g in gt)
    taken = set()
    rows = []
    for i in sorted(idx, key=lambda j: -p[j]):
        cand = [k for k, g in enumerate(gt)
                if k not in taken and abs(t[i] - g) <= tol]
        k = min(cand, key=lambda k: abs(t[i] - gt[k])) if cand else None
        if k is not None:
            taken.add(k)
        rows.append((float(t[i]), float(p[i]), k is not None))
    return sorted(rows), len(taken), len(gt)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--tol_s", type=float, default=TOL_S)
    ap.add_argument("--base_thr", type=float, default=BASE_THR)
    ap.add_argument("--min_gap_s", type=float, default=MIN_GAP_S)
    ap.add_argument("--gate", default="configs/auditor/auto_keep_gate_v1.yaml")
    ap.add_argument("--emit_certificate")
    ap.add_argument("--independent_because",
                    help="mark the certificate independent, and say WHY in one "
                         "sentence that goes into it. A boolean flag would let "
                         "the claim be made by habit; a sentence has to be "
                         "written by someone who checked. Without it the "
                         "certificate is independent:false and --run refuses "
                         "to automate from it.")
    a = ap.parse_args()

    import torch
    from src.auditor.auditor_v1 import load_gate, review_lift, risk_coverage

    recs = torch.load(a.logits, map_location="cpu", weights_only=False)
    print(f"{len(recs)} recordings from {os.path.basename(a.logits)}")
    print(f"  tolerance {a.tol_s}s | candidate pool = peaks >= {a.base_thr} "
          f"thinned at {a.min_gap_s}s")

    items, hit, tot, ids = [], 0, 0, []
    for r in recs:
        idx, p, t = peaks(r["prob"], r["times"], a.base_thr, a.min_gap_s)
        rows, h, g = match(idx, p, t, r["gt"], a.tol_s)
        hit, tot = hit + h, tot + g
        for tt, sc, ok in rows:
            items.append((r["recording_id"], sc, ok))
            ids.append(f"{r['recording_id']}@{tt:.1f}")

    ntp = sum(1 for _, _, ok in items if ok)
    print(f"\n  {len(items)} candidate peaks, {ntp} on a boundary "
          f"({ntp / len(items):.1%} precision at the pool threshold)")
    print(f"  {hit} of {tot} annotated boundaries recovered "
          f"({hit / tot:.1%} recall)")
    print(f"\n  !! val split -- the head was SELECTED on these recordings, so "
          f"every number\n     below is optimistic. It bounds what held-out "
          f"data can do; it does not\n     estimate it.")

    gate = load_gate(a.gate) if os.path.exists(a.gate) else None
    rows = risk_coverage(items, gate=gate)
    lift = review_lift(items)

    if a.emit_certificate:
        from src.auditor.auditor_v1 import event_fingerprint
        fp, n = event_fingerprint(ids)
        json.dump({
            "auditor_version": "v1", "veto_mode": "none",
            "tolerance_s": a.tol_s,
            "independent": bool(a.independent_because),
            "independent_because": a.independent_because,
            "not_independent_because": None if a.independent_because else
                "no --independent_because was given; the default assumption is "
                "that the scores come from a split the model was selected on",
            "score_is": "detector peak probability, not a morphology judgement",
            "gold": os.path.abspath(a.logits),
            "gate_config": os.path.abspath(a.gate),
            "gate": (gate or {}).get("gate"),
            "n_events": n, "event_fingerprint": fp,
            "event_ids": sorted(set(ids)), "rows": rows,
            "review_lift": lift,
        }, open(a.emit_certificate, "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        print(f"\nwrote {a.emit_certificate}  "
              f"(independent: {str(bool(a.independent_because)).lower()})")
        if a.independent_because:
            print(f"  claimed independent because: {a.independent_because}")
            print(f"  --run will accept this as backing for a threshold IF a "
                  f"row also passes the\n  pre-registered gate, which ships "
                  f"with its three targets null.")
        else:
            print(f"  Recorded so a later reader cannot mistake this for a "
                  f"deployable operating\n  point. It is the shape of the "
                  f"curve, measured where the model had an advantage.")


if __name__ == "__main__":
    main()
