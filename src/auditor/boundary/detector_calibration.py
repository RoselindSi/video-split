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

def load_logits(path):
    """Read a torch.save'd list of dicts, WITHOUT requiring torch.

    Everything this file needs -- prob, times, gt, segments, valid_mask -- was
    saved as a plain Python list, so the pickle inside the archive can be read
    directly. torch is tried first so the normal path is unchanged; the
    fallback exists because installing a CUDA torch to read four lists is
    several gigabytes for nothing, and on a slow link that is the difference
    between a result today and a result tomorrow.

    Anything under `torch.` unpickles to None rather than raising. A tensor
    would arrive as None and be visible as such at the call site, which is the
    honest failure -- silently substituting an empty array would produce a
    curve out of nothing."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        pass
    import io
    import pickle
    import zipfile

    class _NoTorch(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            return None

    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    d = _NoTorch(io.BytesIO(z.read(pk))).load()
    need = ("prob", "times", "gt")
    bad = [k for k in need if not isinstance((d[0] if d else {}).get(k), list)]
    if bad:
        raise SystemExit(
            f"read {path} without torch, but {bad} did not come back as plain "
            f"lists -- they were probably tensors and are now None. Install "
            f"torch (CPU is enough) rather than trusting this.")
    print(f"  read without torch ({len(d)} recordings)")
    return d


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


VETO_SWEEP = (0.50, 0.60, 0.70, 0.80, 0.90)
RELEASE_BUDGETS = (0.01, 0.03, 0.05)


def review_budget_table(cands, morph, target, sweep=VETO_SWEEP,
                        oracle=None):
    """The three numbers a review budget actually costs, per veto threshold.

    NOT precision, and not AUROC. At a fixed review budget the question is
    what the budget BUYS and what it destroys, and precision answers neither:
    a system that hits 10% review by rejecting half the real boundaries has an
    excellent precision on what it kept.

        human_review_rate          share of candidates a person still sees
        true_boundary_loss_rate    annotated boundaries thrown away
        false_boundaries_released  wrong candidates admitted unreviewed

    THE OPERATING POINT IS CHOSEN BY A STATED RULE, not by looking. Among
    every (t_lo, t_hi) whose review rate fits the budget, take the one with
    the lowest true-boundary loss, breaking ties on fewer false releases. The
    same rule runs on every arm, so the comparison is between arms rather than
    between two searches.

    MORPHOLOGY ENTERS ONLY AS NEGATIVE EVIDENCE. A confident NO_TRANSITION
    removes a candidate whatever its score; a confident POINT never admits
    one. Admission needs relation EXACT and an adequate view, and those heads
    have 10 and 0 usable events -- letting P(POINT) admit would be automating
    on a head with no supervision behind it."""
    sc = np.array([c["detector_score"] for c in cands], float)
    ok = np.array([c["is_true_boundary"] for c in cands], bool)
    N, T = len(cands), int(ok.sum())
    los = np.quantile(sc, np.linspace(0.0, 0.95, 40))
    his = np.quantile(sc, np.linspace(0.05, 1.0, 40))

    # THE SAME RELEASE BUDGETS ON EVERY ARM. Minimising a single scalar
    # collapses immediately: rank on true-boundary loss alone and the search
    # returns "reject nothing", which loses 0% and releases 845 wrong
    # candidates. It is a two-objective problem, so the honest presentation
    # fixes one objective at levels a person can choose between and reports
    # the other. Identical levels on every arm is what makes the arms
    # comparable rather than two separate searches.
    def one(vetoed, tag):
        rows = []
        for cap_frac in RELEASE_BUDGETS:
            cap = int(cap_frac * N)
            best = None
            for lo in los:
                for hi in his:
                    if hi <= lo:
                        continue
                    rej = vetoed | (sc < lo)
                    keep = (~vetoed) & (sc >= hi)
                    rev = ~rej & ~keep
                    if rev.mean() > target:
                        continue
                    rel = int((keep & ~ok).sum())
                    if rel > cap:
                        continue
                    lost = int((rej & ok).sum())
                    if best is None or lost < best[0]:
                        best = (lost, rel, float(rev.mean()), float(lo),
                                float(hi), int(keep.sum()))
            if best is None:
                rows.append({"release_budget": cap_frac, "feasible": False})
                print(f"  {tag:<24}{cap_frac:>8.0%}{'infeasible':>28}")
                continue
            lost, rel, rev, lo, hi, nkeep = best
            rows.append({"release_budget": cap_frac, "feasible": True,
                         "review_rate": rev,
                         "true_boundary_loss_rate": lost / T,
                         "false_boundaries_released": rel,
                         "n_kept": nkeep, "t_lo": lo, "t_hi": hi})
            print(f"  {tag:<24}{cap_frac:>8.0%}{rev:>10.1%}"
                  f"{lost / T:>14.1%}{rel:>11}{nkeep:>8}")
        return {"veto": tag, "points": rows}

    print(f"\n{'=' * 82}\nAT A REVIEW BUDGET OF {target:.0%}\n{'=' * 82}")
    print(f"  {'arm':<24}{'release':>8}{'REVIEW':>10}{'true lost':>14}"
          f"{'released':>11}{'kept':>8}")
    out = [one(np.zeros(N, bool), "score only")]
    if oracle is not None:
        out.append(one(oracle, "ORACLE no_transition"))
    if morph:
        pnt = np.array([morph.get(c["candidate_id"], {}).get(
            "p_no_transition", np.nan) for c in cands], float)
        miss = int(np.isnan(pnt).sum())
        if miss:
            print(f"  ({miss} candidates carry no morphology prediction; "
                  f"they are never vetoed)")
        for thr in sweep:
            out.append(one(np.nan_to_num(pnt, nan=0.0) >= thr,
                           f"+morph p_nt>={thr:.2f}"))
    print(f"\n  {T} annotated boundaries are recoverable in this pool of {N}. "
          f"`release` caps the\n  wrong candidates admitted unreviewed; "
          f"`true lost` is what that cap costs in real\n  boundaries. Read "
          f"the arms DOWN a release column, never across rows.")
    return [o for o in out if o]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--tol_s", type=float, default=TOL_S)
    ap.add_argument("--base_thr", type=float, default=BASE_THR)
    ap.add_argument("--min_gap_s", type=float, default=MIN_GAP_S)
    ap.add_argument("--gate", default="configs/auditor/auto_keep_gate_v1.yaml")
    ap.add_argument("--emit_candidates",
                    help="write the candidate pool as JSONL. THE ONLY "
                         "authoritative pool: any experiment that re-derives "
                         "peaks is comparing two different populations, and "
                         "the whole point of a veto experiment is that the "
                         "candidates are identical on both arms.")
    ap.add_argument("--morphology_predictions",
                    help="JSONL from morphology_external, keyed by "
                         "candidate_id. Used ONLY as negative evidence: a "
                         "confident NO_TRANSITION removes a candidate. "
                         "P(POINT) never admits one -- an admission needs the "
                         "relation and observability heads, which have 10 and "
                         "0 usable events.")
    ap.add_argument("--oracle_audit",
                    help="the error-audit predictions.jsonl for THESE "
                         "recordings. Adds an ORACLE arm that vetoes exactly "
                         "the false_mid_segment candidates -- a perfect "
                         "NO_TRANSITION head. It is the ceiling the learned "
                         "head is measured against, and putting it in the same "
                         "harness is what makes `fraction of oracle gain` a "
                         "quantity rather than a ratio of two tables.")
    ap.add_argument("--veto", choices=("none", "morphology_only"),
                    default="none")
    ap.add_argument("--review_target", type=float, default=0.10)
    ap.add_argument("--emit_certificate")
    ap.add_argument("--independent_because",
                    help="mark the certificate independent, and say WHY in one "
                         "sentence that goes into it. A boolean flag would let "
                         "the claim be made by habit; a sentence has to be "
                         "written by someone who checked. Without it the "
                         "certificate is independent:false and --run refuses "
                         "to automate from it.")
    a = ap.parse_args()

    from src.auditor.auditor_v1 import load_gate, review_lift, risk_coverage

    recs = load_logits(a.logits)
    print(f"{len(recs)} recordings from {os.path.basename(a.logits)}")
    print(f"  tolerance {a.tol_s}s | candidate pool = peaks >= {a.base_thr} "
          f"thinned at {a.min_gap_s}s")

    items, hit, tot, ids, cands = [], 0, 0, [], []
    for r in recs:
        idx, p, t = peaks(r["prob"], r["times"], a.base_thr, a.min_gap_s)
        rows, h, g = match(idx, p, t, r["gt"], a.tol_s)
        hit, tot = hit + h, tot + g
        for tt, sc, ok in rows:
            items.append((r["recording_id"], sc, ok))
            ids.append(f"{r['recording_id']}@{tt:.1f}")
            cands.append({"candidate_id": f"{r['recording_id']}@{tt:.3f}",
                          "recording_id": r["recording_id"],
                          "candidate_time": tt, "detector_score": sc,
                          "is_true_boundary": bool(ok)})

    if a.emit_candidates:
        with open(a.emit_candidates, "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        print(f"\nwrote {a.emit_candidates}  ({len(cands)} candidates over "
              f"{len({c['recording_id'] for c in cands})} recordings)")
        print(f"  Every later arm must consume THIS file. Re-deriving peaks "
              f"elsewhere would compare\n  two candidate pools and call the "
              f"difference an effect of the veto.")

    # THE PROPERTY INDEPENDENCE DOES NOT GIVE YOU. Two calibration sets from
    # this project share no recordings and differ 5.15 vs 8.31 annotated
    # boundaries per hundred frames; precision at any coverage rises with that
    # density mechanically, because a denser GT makes any candidate likelier to
    # land near one. A threshold calibrated on the dense set and deployed on
    # the sparse one would underperform, and nothing about `independent: true`
    # warns of it. Recorded so a reader can compare rather than assume.
    n_frames = sum(len(r["times"]) for r in recs)
    gt_density = tot / n_frames * 100 if n_frames else float("nan")
    ntp = sum(1 for _, _, ok in items if ok)
    print(f"\n  {len(items)} candidate peaks, {ntp} on a boundary "
          f"({ntp / len(items):.1%} precision at the pool threshold)")
    print(f"  {hit} of {tot} annotated boundaries recovered "
          f"({hit / tot:.1%} recall)")
    print(f"\n  POPULATION, which independence does not fix:")
    print(f"    {len(recs)} recordings, {tot} annotated boundaries, "
          f"{n_frames} frames")
    print(f"    GT density        {gt_density:.2f} per 100 frames")
    print(f"    candidate base rate {ntp / len(items):.3f}")
    print(f"    Precision at every coverage below scales with these. Compare "
          f"them against the\n    deployment population before reading any "
          f"row as a deployable operating point.")
    print(f"\n  !! val split -- the head was SELECTED on these recordings, so "
          f"every number\n     below is optimistic. It bounds what held-out "
          f"data can do; it does not\n     estimate it.")

    morph = None
    if a.morphology_predictions:
        morph = {}
        for l in open(a.morphology_predictions, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                morph[r["candidate_id"]] = r
        hit = sum(1 for c in cands if c["candidate_id"] in morph)
        if not hit:
            raise SystemExit(
                f"--morphology_predictions matched 0 of {len(cands)} "
                f"candidate_ids. Both files must come from the same "
                f"--emit_candidates run.")
        print(f"\n  morphology predictions joined onto {hit}/{len(cands)} "
              f"candidates ({hit / len(cands):.1%})")
    oracle = None
    if a.oracle_audit:
        bad = set()
        for l in open(a.oracle_audit, encoding="utf-8"):
            if not l.strip():
                continue
            e = json.loads(l)
            for x in e.get("predicted_peaks", []):
                if x.get("status") == "false_mid_segment":
                    bad.add((e["recording_id"], round(x["pred_time"], 1)))
        oracle = np.array([(c["recording_id"],
                            round(c["candidate_time"], 1)) in bad
                           for c in cands])
        print(f"\n  oracle audit: {len(bad)} false_mid_segment in the audit, "
              f"{int(oracle.sum())} matched onto the {len(cands)} candidates")
        if not oracle.sum():
            raise SystemExit(
                "--oracle_audit matched 0 candidates. The audit and the "
                "candidate pool must come from the same recordings and the "
                "same peak picking.")
    budget = review_budget_table(cands, morph, a.review_target,
                                 oracle=oracle) \
        if (morph or oracle is not None or a.veto == "none") else None

    gate = load_gate(a.gate) if os.path.exists(a.gate) else None
    rows = risk_coverage(items, gate=gate)
    lift = review_lift(items)

    if a.emit_certificate:
        from src.auditor.auditor_v1 import event_fingerprint
        fp, n = event_fingerprint(ids)
        json.dump({
            "auditor_version": "v1", "veto_mode": "none",
            "tolerance_s": a.tol_s,
            # TWO INDEPENDENCES, AND ONLY ONE OF THEM IS EVER TRUE HERE.
            # A single `independent: true` reads as "independent test of the
            # operating point", and this curve cannot be that: the moment a
            # threshold is chosen from these rows, the rows become its
            # selection set. Keeping the fields apart is what stops a later
            # reader from citing a calibration curve as its own validation --
            # which is how the pooled out-of-fold threshold that breached got
            # justified.
            "model_training_independent": bool(a.independent_because),
            "model_training_independent_because": a.independent_because,
            "not_independent_because": None if a.independent_because else
                "no --independent_because was given; the default assumption is "
                "that the scores come from a split the model was selected on",
            "operating_point_independent_test": "not_run",
            "operating_point_note":
                "a threshold selected from these rows is calibrated, not "
                "validated. Its independent test needs a further set that was "
                "not used to choose it.",
            # kept for older readers; means training independence only
            "independent": bool(a.independent_because),
            "score_is": "detector peak probability, not a morphology judgement",
            "gold": os.path.abspath(a.logits),
            "gate_config": os.path.abspath(a.gate),
            "gate": (gate or {}).get("gate"),
            "population": {
                "n_recordings": len(recs), "n_gt_boundaries": tot,
                "n_frames": n_frames, "gt_density_per_100_frames": gt_density,
                "candidate_base_rate": ntp / len(items),
                "note": "precision at any coverage scales with these; "
                        "independence does not make a calibration set "
                        "representative",
            },
            "n_events": n, "event_fingerprint": fp,
            "event_ids": sorted(set(ids)), "rows": rows,
            "review_lift": lift,
            "review_budget": budget,
        }, open(a.emit_certificate, "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        print(f"\nwrote {a.emit_certificate}")
        print(f"  model_training_independent      "
              f"{str(bool(a.independent_because)).lower()}")
        print(f"  operating_point_independent_test not_run")
        if a.independent_because:
            print(f"    because: {a.independent_because}")
            print(f"\n  THOSE ARE DIFFERENT CLAIMS. Every prediction here "
                  f"comes from a model that never\n  saw its recording. A "
                  f"threshold CHOSEN from these rows is calibrated on them, "
                  f"so\n  this curve is not that threshold's independent "
                  f"test -- that needs a further set.")
            print(f"  --run accepts this as backing IF a row also passes the "
                  f"pre-registered gate,\n  which ships with its three "
                  f"targets null.")
        else:
            print(f"  Recorded so a later reader cannot mistake this for a "
                  f"deployable operating\n  point. It is the shape of the "
                  f"curve, measured where the model had an advantage.")


if __name__ == "__main__":
    main()
