"""Do hand trajectories solve the REVIEW band?

The primary question, fixed in advance, is the REVIEW-band clean verifier:
169 clean events (94 sharp, 75 same-action, 110 recordings) that the frozen
policy sends to a human. Everything else here is diagnostic.

THREE ARMS, because "P1 + trajectory does not beat P1" has two very different
causes and the combined arm alone cannot tell them apart:

  P1 alone            the baseline, refit from raw side vectors in the same
                      folds -- never taken from a stored OOF column, which
                      would leak the test fold through a model trained on the
                      other folds
  trajectory alone    29 features. Answers "is there signal" independently of
                      whether a shared L2 lets 29 columns compete with 256
                      PCA-derived ones. The side_change arm could not
                      distinguish those, 3 scalars against 260, and said so.
  P1 + trajectory     the question that decides adoption

A PER-FEATURE table with a FAMILY-WISE permutation baseline runs alongside.
With 29 features, comparing each against its own p95 expects about one and a
half spurious crossings per task, so the baseline is the best AUROC obtainable
by chance ACROSS ALL 29 -- the same correction the concentration probe needed
after a no-signal fixture produced a false hit.

Pass conditions are the ones already used for side_change, unchanged:
  delta AUROC grouped-bootstrap CI lower bound > 0
  at least 20 additional events automated out-of-fold
  those additions holding >= 0.95 precision
  sharp false-reject not above 0.05
  at least 4 of 5 folds moving the same way

Usage:
    python -m src.boundary.c3_hand_trajectory_probe \
        --trajectory /workspace/tr1/data_recseg/hand_trajectory_dev_10fps.pt \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --batch3_manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --features_out /workspace/tr1/results/hal/c3/hand_trajectory_features.csv \
        --out /workspace/tr1/results/hal/c3/hand_trajectory_probe.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import stratified_grouped_folds, build_matrices
from src.boundary.hand_traj_features import all_features
from src.boundary.c3_sidechange_arm import run_cv, automatable, report
from src.boundary.pairwise_verifier import _impute_scale_fit, _impute_scale_apply
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.c3_error_separability import auroc_stats


def run_cv_extra_only(extra, y, groups, folds, l2=5.0):
    """Grouped CV on the extra feature matrix ALONE.

    A separate path rather than passing zeroed side vectors through run_cv:
    zero-variance input makes the PCA fit meaningless and whatever came out
    would not have been the trajectory features' own performance."""
    oof = np.full(len(y), np.nan)
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
            per_fold.append(float("nan"))
            continue
        st = _impute_scale_fit(extra[tr])
        w, b = fit_logreg(_impute_scale_apply(st, extra[tr]), y[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(st, extra[te]) @ w + b)
        per_fold.append(_auroc(y[te], oof[te])
                        if len(set(y[te].tolist())) == 2 else float("nan"))
    return oof, per_fold


SHARP = "sharp_visible_transition"
SAME = "same_action_internal_motion"
CLEAN = (SHARP, SAME)


MIN_OBS, MIN_UNIQUE = 30, 5


def eligible(mat, names):
    """Columns the per-feature table can evaluate, and why the rest are out.

    The SAME set must be used for the family-wise baseline. The first version
    filtered the table on >=30 observations and >=5 distinct values but let the
    baseline range over every column at >=20 observations, so the printed
    "baseline over 26 features" was a maximum taken over 29. More columns means
    a higher chance maximum, so the threshold every feature was judged against
    was raised by columns that were never shown -- conservative, but not the
    quantity it claimed to be."""
    keep, drop = [], {}
    for j, nm in enumerate(names):
        m = np.isfinite(mat[:, j])
        u = len(set(mat[m, j].tolist()))
        if m.sum() < MIN_OBS:
            drop[nm] = f"only {int(m.sum())} finite values (need {MIN_OBS})"
        elif u < MIN_UNIQUE:
            drop[nm] = f"only {u} distinct values (need {MIN_UNIQUE})"
        else:
            keep.append(j)
    return keep, drop


def family_baseline(lab, mat, cols, n_perm, seed):
    """Best folded AUROC obtainable by chance across the evaluated columns."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_perm):
        p = rng.permutation(lab)
        best = 0.0
        for j in cols:
            v = mat[:, j]
            m = np.isfinite(v)
            if len(set(p[m].tolist())) < 2:
                continue
            au = _auroc(p[m], v[m])
            best = max(best, max(au, 1 - au))
        out.append(best)
    return float(np.percentile(out, 95)) if out else float("nan")


def per_feature(label, lab, mat, names, groups, n_boot, seed):
    print(f"\n  --- per-feature separability: {label} ---")
    cols, dropped = eligible(mat, names)
    if dropped:
        print(f"    {len(dropped)} of {len(names)} features not evaluable here "
              f"(and excluded from the baseline too):")
        for nm, why in sorted(dropped.items()):
            print(f"      {nm:<36} {why}")
    print(f"    {'feature':<34} {'AUROC':>7} {'95% CI':>18} {'dir':>7}")
    res = {}
    for j in cols:
        v = mat[:, j]
        m = np.isfinite(v)
        st = auroc_stats(lab[m], v[m], [g for g, k in zip(groups, m) if k],
                         n_boot, n_boot, seed)
        if st:
            res[names[j]] = st
    fam = family_baseline(lab, mat, cols, min(n_boot, 500), seed)
    for nm, st in sorted(res.items(), key=lambda kv: -kv[1]["folded"])[:10]:
        flag = "" if st["folded"] > fam else "  (not above chance)"
        ci = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]"
        print(f"    {nm:<34} {st['folded']:>7.3f} {ci:>18} "
              f"{st['direction']:>7}{flag}")
    print(f"    family-wise chance baseline over the {len(cols)} evaluated "
          f"features: {fam:.3f}")
    win = {k: v for k, v in res.items() if v["folded"] > fam}
    if not win:
        print("    nothing beats the family-wise baseline")
    else:
        b = max(win.items(), key=lambda kv: kv[1]["folded"])
        marg = b[1]["ci95"][0] <= fam
        print(f"    -> {b[0]} at {b[1]['folded']:.3f}"
              + ("  MARGINAL: its CI contains the baseline" if marg
                 else "  its CI clears the baseline"))
    for k, v in res.items():
        v["family_wise_p95"] = fam
        v["beats_family_wise"] = bool(v["folded"] > fam)
    return res


def observability(traj, feats, names, a):
    """SECONDARY, pre-registered: is an event CLEAN-OBSERVABLE at all?

    A different question from the primary one and it must not be allowed to
    rescue it. The primary asks whether trajectories separate sharp from
    same-action among events a human already has to watch. This asks whether
    they separate those clean events from the ones the taxonomy excludes --
    offscreen, camera motion, ambiguous. A positive result here buys routing
    (send the unobservable ones somewhere else), never a verifier.

    It runs on ALL cached events, not the taxonomy-kept subset, so it needs no
    P1 features: the trajectory arm alone is the whole question. Recordings are
    the CV group here as everywhere else, because one recording contributes
    many events and its camera behaviour is shared across them."""
    ids = sorted(traj)
    sub = [str(traj[i].get("subtype") or "") for i in ids]
    grp = [traj[i]["recording_id"] for i in ids]
    M = np.array([[feats[i].get(n, np.nan) for n in names] for i in ids])
    print(f"\n{'#' * 72}\n### SECONDARY: is the event clean-observable?"
          f"\n{'#' * 72}")
    print(f"  {len(ids)} cached events by subtype: {dict(Counter(sub))}")

    res = {}
    for tag, pos in [("clean vs everything else", lambda s: s in CLEAN)] + \
            [(f"clean vs {k}", (lambda kk: (lambda s: s in CLEAN if s in CLEAN
                                            else (None if s != kk else False)))(k))
             for k in sorted({x for x in sub if x and x not in CLEAN})]:
        lab, sel = [], []
        for s_ in sub:
            v = pos(s_)
            sel.append(v is not None)
            lab.append(1.0 if v else 0.0)
        sel = np.array(sel)
        y = np.array(lab)[sel]
        if sel.sum() < 40 or len(set(y.tolist())) < 2:
            print(f"  {tag:<34} too few events ({int(sel.sum())})")
            continue
        g = [x for x, k in zip(grp, sel) if k]
        folds = stratified_grouped_folds(g, y, 5, seed=a.seed)
        oof, pf = run_cv_extra_only(M[sel], y, g, folds)
        m = np.isfinite(oof)
        au = (_auroc(y[m], oof[m]) if len(set(y[m].tolist())) == 2
              else float("nan"))
        print(f"  {tag:<34} n={int(sel.sum()):>4}  "
              f"{int(y.sum())}+/{int((1 - y).sum())}-  AUROC {au:.3f}   "
              f"per-fold {[round(x, 3) for x in pf]}")
        res[tag] = {"n": int(sel.sum()), "n_pos": int(y.sum()),
                    "auroc": float(au), "per_fold": pf}
    print("  Routing only. A separation here does NOT license using these "
          "features on the sharp-vs-same-action task, which is what the "
          "primary block already answered on its own population.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--batch3_manifest")
    ap.add_argument("--batch3_pair_labels", default="data/gold/batch3_pair_labels_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features_out")
    ap.add_argument("--out")
    a = ap.parse_args()

    traj = torch.load(a.trajectory, weights_only=False, map_location="cpu")
    print(f"trajectory cache: {len(traj)} events")
    feats = {eid: all_features(e) for eid, e in traj.items()}
    names = sorted({k for v in feats.values() for k in v})
    print(f"  {len(names)} features per event")
    cov = np.array([f.get("hand_detect_coverage", np.nan) for f in feats.values()])
    print(f"  hand_detect_coverage over the candidate window: "
          f"median {np.nanmedian(cov):.3f}, below 0.5 in "
          f"{int((cov < 0.5).sum())}/{len(cov)} events")

    dec = {}
    with open(a.decisions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dec[r["event_id"]] = r
    by_rid = load_feature_caches(a.feat_cache)
    events = T.apply_to_events(
        build_events(S.load_gold(a.gold), S.load_context(a.context), by_rid),
        T.load_pair_labels(a.pair_labels))
    if a.batch3_manifest:
        from src.boundary.batch3_dev_events import build_events as build_b3
        events += T.apply_to_events(build_b3(a.batch3_manifest, by_rid),
                                    T.load_pair_labels(a.batch3_pair_labels))
    events = [e for e in events if e["event_id"] in feats]
    print(f"clean events with both P1 features and a trajectory: {len(events)}")

    X_v1, Ls, Rs, X_rel, keep, _ = build_matrices(events, False)
    ev = [events[i] for i in keep]
    y = np.array([e["y"] for e in ev], float)
    groups = [e["recording_id"] for e in ev]
    Tm = np.array([[feats[e["event_id"]].get(n, np.nan) for n in names] for e in ev])
    is_sharp = np.array([e.get("temporal_pair_subtype") == SHARP for e in ev])
    print(f"{len(ev)} poolable, {len(set(groups))} recordings, "
          f"trajectory matrix {Tm.shape}")

    out = {"n_events": len(ev), "features": names}

    def block(title, sel, tag):
        if sel.sum() < 40 or len(set(y[sel].tolist())) < 2:
            print(f"\n### {title}: too few events ({int(sel.sum())})")
            return None
        g = [x for x, k in zip(groups, sel) if k]
        folds = stratified_grouped_folds(g, y[sel], 5, seed=a.seed)
        oa, pa = run_cv(Ls[sel], Rs[sel], X_rel[sel], y[sel], g, folds, None, a.pca_dim)
        ot, pt = run_cv_extra_only(Tm[sel], y[sel], g, folds)
        ob, pb = run_cv(Ls[sel], Rs[sel], X_rel[sel], y[sel], g, folds, Tm[sel], a.pca_dim)
        print(f"\n{'#' * 72}\n### {title}\n{'#' * 72}")
        m = np.isfinite(ot)
        print(f"  trajectory ALONE: AUROC "
              f"{_auroc(y[sel][m], ot[m]) if len(set(y[sel][m].tolist())) == 2 else float('nan'):.3f}"
              f"   per-fold {[round(x, 3) for x in pt]}")
        print("    (a separate arm because 29 columns sharing an L2 with 256 "
              "PCA-derived ones can be swamped; this says whether the signal "
              "exists at all)")
        r = report(title, y[sel], oa, ob, pa, pb, g, folds, is_sharp[sel],
                   a.n_boot, a.seed, arm_name="hand_trajectory")
        pf = per_feature(title, np.where(is_sharp[sel], 1.0, 0.0), Tm[sel], names,
                         g, a.n_boot, a.seed)
        out[tag] = {"combined": r, "per_feature": pf,
                    "trajectory_alone_auroc": float(_auroc(y[sel][m], ot[m]))
                    if len(set(y[sel][m].tolist())) == 2 else None}
        return r

    rev = np.array([dec.get(e["event_id"], {}).get("decision") == "REVIEW" for e in ev])
    primary = block(f"PRIMARY: REVIEW band clean verifier", rev, "primary_review_band")
    out["secondary_observability"] = observability(traj, feats, names, a)
    block("DIAGNOSTIC: all clean events (not an adoption reason)",
          np.ones(len(ev), bool), "diagnostic_all_clean")

    print(f"\n{'=' * 72}\nPRE-REGISTERED PASS CONDITIONS (primary task only)\n{'=' * 72}")
    if not primary:
        print("  primary task could not be evaluated")
    else:
        aa, ab = primary["automatable_p1"], primary["automatable_arm"]
        gained = ab["n_auto"] - aa["n_auto"]
        d = primary["per_fold_delta"]
        checks = {
            "delta CI lower bound > 0": primary["ci95"][0] > 0,
            ">= 20 additional events automated": gained >= 20,
            "additions hold >= 0.95 precision": (np.isfinite(ab["precision"])
                                                 and ab["precision"] >= 0.95),
            ">= 4/5 folds move the same way": sum(1 for x in d if x > 0) >= 4,
        }
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"    (delta {primary['delta']:+.3f} CI "
              f"[{primary['ci95'][0]:+.3f}, {primary['ci95'][1]:+.3f}]; automated "
              f"{aa['n_auto']} -> {ab['n_auto']} ({gained:+d}); precision "
              f"{ab['precision']:.3f}; folds {sum(1 for x in d if x > 0)}/{len(d)})")
        out["pass_conditions"] = {k: bool(v) for k, v in checks.items()}
        if all(checks.values()):
            print("\n  Hand trajectories resolve part of the REVIEW band. Next is a "
                  "policy component, developed on 145 + batch3 + batch4 and frozen "
                  "before batch5.")
        else:
            print("\n  NOT met on the primary task. Before concluding, read the "
                  "per-feature table and the trajectory-alone arm: no feature "
                  "above the family-wise baseline means the observables are not "
                  "there, while features above it plus no combined gain means "
                  "they are there and this fusion cannot use them -- different "
                  "problems with different next moves.")

    if a.features_out:
        with open(a.features_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "subtype", "y", "decision"] + names)
            for i, e in enumerate(ev):
                w.writerow([e["event_id"], e["recording_id"],
                            e.get("temporal_pair_subtype", ""), int(y[i]),
                            dec.get(e["event_id"], {}).get("decision", "")]
                           + [f"{Tm[i, j]:.6f}" if np.isfinite(Tm[i, j]) else ""
                              for j in range(len(names))])
        print(f"\nwrote {a.features_out}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
