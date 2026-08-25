"""Level C0: put morphology into the ranking, with zero free parameters.

The question this isolates is not "what coefficients work" but "does the
ontology belong in the score at all". So nothing here is fitted:

    E = logit(p_detector) + log( (P(POINT)+eps) / (P(NO_TRANSITION)+eps) )

Both coefficients are fixed at 1 and eps is 1e-8 for numerical safety only. No
eta is learned, no reranker is added, no temperature is tuned, no morphology
threshold is chosen. If a fusion with ZERO free parameters lifts a real
boundary above an internal motion, that is evidence about where the ontology
belongs -- not about a coefficient search.

IT IS NOT CLAIMED TO BE BAYES-OPTIMAL, and the name says so. Adding log-odds
has a probabilistic reading only if the detector and the morphology head are
calibrated AND conditionally independent given the truth, and neither has been
shown here. This is a PRE-REGISTERED PARAMETER-FREE MONOTONIC FUSION whose sign
is the ontology -- POINT raises the energy, NO_TRANSITION lowers it -- and
whose magnitude is a stated default rather than a derived optimum.

WHY THE 79 TRAINING PAIRS ARE NOT THE VALIDATION SET. They would be, for the
morphology head's own ordering, because train.py produces recording-grouped
out-of-fold morphology predictions. They are NOT a validation of the FUSED
score: the detector scores on those recordings may come from a detector that
trained on them, so the fused number would carry an advantage the evaluation
recordings do not give it. Building a nested detector OOF to fix that would
introduce four new variables at once, so the pairs stay a diagnostic resource.

WHAT THIS CANNOT DO. It reorders the candidate pool; it cannot add to it. A
boundary that never became a peak is untouched by any reranking, so a large
ordering gain beside a small recall gain is consistent rather than
contradictory. Level C must not be held responsible for proposal recall.

Usage:
    python -m src.auditor.boundary.ontology_fusion \
        --candidates results/auditor/oof_candidates.jsonl \
        --morphology results/auditor/morphology_external_oof.jsonl \
        --oracle_audit results/boundary/oof/audit/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

EPS = 1e-8
ETA_POINT = 1.0            # pre-registered, not fitted
ETA_NO_TRANSITION = 1.0    # pre-registered, not fitted


def ontology_energy(det_score, p_point, p_no_transition, eps=EPS):
    d = np.asarray(det_score, float).clip(1e-4, 1 - 1e-4)
    logit = np.log(d / (1 - d))
    # The two terms are applied separately even though both coefficients are
    # 1, so the sign of each stays visible in the code. Collapsing them into a
    # single log-ratio hides that POINT is the one that raises and
    # NO_TRANSITION the one that lowers, and a later edit to one coefficient
    # would then silently change both.
    return (logit
            + ETA_POINT * np.log(np.asarray(p_point, float) + eps)
            - ETA_NO_TRANSITION * np.log(np.asarray(p_no_transition, float)
                                         + eps))


def _auroc(sc, pos, neg):
    a, b = sc[pos], sc[neg]
    if not len(a) or not len(b):
        return float("nan")
    allv = np.concatenate([a, b])
    r = allv.argsort().argsort() + 1.0
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def _pair_acc(sc, ok, fm, rec):
    """Within recording: a true boundary against an audited false_mid_segment.

    Within, because across recordings the comparison is separable by
    recognising the recording, and the failure being attacked is a real
    boundary ranked below an internal motion in the SAME one."""
    by = defaultdict(lambda: {"p": [], "n": []})
    for i, r in enumerate(rec):
        if ok[i]:
            by[r]["p"].append(sc[i])
        elif fm[i]:
            by[r]["n"].append(sc[i])
    w, n = 0.0, 0
    for v in by.values():
        for x in v["p"]:
            for y in v["n"]:
                w += 1.0 if x > y else (0.5 if x == y else 0.0)
                n += 1
    return (w / n if n else float("nan")), n


def _pctile(sc, mask, rec):
    """Mean within-recording rank percentile of the masked candidates."""
    by = defaultdict(list)
    for i, r in enumerate(rec):
        by[r].append(i)
    out = []
    for idx in by.values():
        v = sc[np.array(idx)]
        rk = v.argsort().argsort() / max(len(v) - 1, 1)
        out += [rk[j] for j, i in enumerate(idx) if mask[i]]
    return float(np.mean(out)) if out else float("nan")


def ranking_diagnostics(cands, new_score, fm_mask, label="E_onto"):
    """Ordering, reported before any policy.

    A policy compares two systems at an operating point; this compares them as
    orderings. If the ordering does not move, no threshold on it can, and the
    review-budget table would only be showing that at three release caps."""
    det = np.array([c["detector_score"] for c in cands], float)
    ok = np.array([c["is_true_boundary"] for c in cands], bool)
    rec = [c["recording_id"] for c in cands]
    fm = np.asarray(fm_mask, bool)
    new = np.asarray(new_score, float)

    print("=" * 78)
    print("RANKING, BEFORE ANY POLICY")
    print("=" * 78)
    print(f"  {'':<34}{'detector':>12}{label:>12}{'delta':>10}")
    out = {}
    checks = (
        ("AUROC true vs false_mid", lambda s: _auroc(s, ok, fm)),
        ("same-recording pair accuracy", lambda s: _pair_acc(s, ok, fm, rec)[0]),
        ("mean pctile of true boundaries", lambda s: _pctile(s, ok, rec)),
        ("mean pctile of false_mid", lambda s: _pctile(s, fm, rec)),
    )
    for name, fn in checks:
        a, b = fn(det), fn(new)
        out[name] = {"detector": a, "fused": b, "delta": b - a}
        print(f"  {name:<34}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")
    _, npair = _pair_acc(det, ok, fm, rec)
    print(f"    {npair} within-recording true x false_mid pairs, "
          f"{int(ok.sum())} true and {int(fm.sum())} audited false_mid")

    print()
    print("  true boundaries among the top of the ordering:")
    print(f"  {'top':>6}{'detector':>12}{label:>12}{'delta':>10}")
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(frac * len(cands)))
        a = float(ok[np.argsort(-det)[:k]].mean())
        b = float(ok[np.argsort(-new)[:k]].mean())
        out[f"top_{int(frac * 100)}"] = {"detector": a, "fused": b,
                                         "delta": b - a}
        print(f"  {frac:>5.0%}{a:>12.3f}{b:>12.3f}{b - a:>+10.3f}")
    print(f"    pool base rate {ok.mean():.3f}")
    print()
    print("  An ordering gain here is about ordering only. A boundary that")
    print("  never became a candidate cannot be reranked into existence, so a")
    print("  small recall change beside a large ordering change is consistent.")
    return out


def load_fusion(cand_path, morph_path):
    cands = [json.loads(l) for l in open(cand_path, encoding="utf-8")
             if l.strip()]
    morph = {}
    for l in open(morph_path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            morph[r["candidate_id"]] = r
    hit = [c for c in cands if c["candidate_id"] in morph]
    if len(hit) != len(cands):
        raise SystemExit(
            f"morphology covers {len(hit)} of {len(cands)} candidates. The "
            f"fused arm must score the SAME pool as the detector arm; a "
            f"partial join would compare two populations.")
    pp = np.array([morph[c["candidate_id"]]["p_point"] for c in cands])
    pn = np.array([morph[c["candidate_id"]]["p_no_transition"] for c in cands])
    det = np.array([c["detector_score"] for c in cands])
    return cands, ontology_energy(det, pp, pn)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--oracle_audit", required=True,
                    help="supplies the audited false_mid_segment mask. It "
                         "labels the negatives of the comparison and is never "
                         "an input to the score.")
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.ontology_constitution import Constitution
    C = Constitution()
    C.check_level("C_ontology_rerank")
    C.check_energy_signs({"eta_point": ETA_POINT,
                          "eta_no_transition": ETA_NO_TRANSITION})
    C.check_oracle_use("headroom")

    cands, energy = load_fusion(a.candidates, a.morphology)
    C.check_candidate_pool(cands, a.candidates)
    print(f"{len(cands)} candidates over "
          f"{len({c['recording_id'] for c in cands})} recordings")
    print(f"  eta_point {ETA_POINT}  eta_no_transition {ETA_NO_TRANSITION}  "
          f"eps {EPS}  -- pre-registered, none of them fitted")

    bad = set()
    for l in open(a.oracle_audit, encoding="utf-8"):
        if not l.strip():
            continue
        e = json.loads(l)
        for x in e.get("predicted_peaks", []):
            if x.get("status") == "false_mid_segment":
                bad.add((e["recording_id"], round(x["pred_time"], 1)))
    fm = np.array([(c["recording_id"], round(c["candidate_time"], 1)) in bad
                   for c in cands])
    print(f"  audited false_mid_segment matched onto {int(fm.sum())} "
          f"candidates")
    if not fm.sum():
        raise SystemExit("the audit matched no candidate; the pools differ")

    diag = ranking_diagnostics(cands, energy, fm)

    if a.out:
        json.dump({"eta_point": ETA_POINT,
                   "eta_no_transition": ETA_NO_TRANSITION, "eps": EPS,
                   "fitted": False,
                   "name": "pre-registered parameter-free monotonic fusion",
                   "not_claimed": "Bayes-optimal; log-odds addition has a "
                                  "probabilistic reading only under "
                                  "calibration and conditional independence, "
                                  "neither shown",
                   "n_candidates": len(cands),
                   "n_false_mid_matched": int(fm.sum()),
                   "diagnostics": diag},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
