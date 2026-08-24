"""Stage 0: can a candidate with no decidable boundary be detected automatically?

64 of the 252 REVIEW events carry no visual boundary to find -- the GT
segmentation cuts where nothing happens (`annotation_convention`), the camera
moves rather than the hands (`camera_or_viewpoint_shift`), or the moment is
off-frame (`visibility_or_offscreen`). A verifier asked to call those is being
asked to reproduce a labelling rule from pixels that do not encode it.

Removing them upstream is the cheapest review reduction available, worth about
10.7 points with no model change. But that arithmetic assumes they can be
IDENTIFIED, and nothing has tested that. Their subtypes were assigned by a
human watching video; a gate has only the signals already computed. This
measures whether that is enough.

TWO TARGETS, NOT ONE, BECAUSE THEY EARN DIFFERENT DECISIONS.

  reject   annotation_convention + camera_or_viewpoint_shift. Nothing is
           there to find and nothing would be gained by looking harder, so a
           confident detection can auto-reject.
  abstain  visibility_or_offscreen + ambiguous. "The evidence is not visible"
           is not "there is no boundary". Auto-rejecting these would trade a
           real uncertainty for a better-looking review rate, so they are
           measured separately and route to REVIEW either way.

`gradual_phase_transition` is in NEITHER. It is a real change without an
instant, and forcing it into a binary is the framing problem, not a nuisance
to filter.

THE HEADLINE NUMBER IS NOT AUROC. Three quantities decide whether this is
deployable, and they are the ones a reviewer's workload actually depends on:
removal coverage (how many nuisance candidates are caught), removal precision
(how many of the caught ones really were nuisance) and the SHARP FALSE-DROP
RATE (real boundaries thrown away by the gate). A gate that removes 90% of
nuisance while discarding 5% of real boundaries has made the system worse, and
an AUROC would hide that.

REPORTED PER CANDIDATE SOURCE, because the gain is not portable.
`annotation_convention` is 80% GT-derived, and deployment generates candidates
from raw change peaks rather than from GT boundaries. So the development-set
gain is an overestimate of the deployed one by construction, and the raw-peak
column is the honest figure.

Usage:
    python -m src.boundary.c3_validity_gate \
        --clean145 \
        --batch3_manifest .../batch3_manifest.jsonl \
        --batch3_pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --feat_cache ... --local_cache ... \
        --out /workspace/tr1/results/hal/c3/validity_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from src.boundary.hal_features import load_feature_caches
from src.boundary.pairwise_verifier import (
    stratified_grouped_folds, build_matrices, pca_fit, pca_apply, pair_block,
    _impute_scale_fit, _impute_scale_apply,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.state_adapter import _auroc
from src.boundary.c3_local_eval import detect_coverage, detect_longest_gap_s
from src.boundary.c3_selective_policy import wilson
from src.boundary.frozen_scorer import gather, matrices

REJECT = ("annotation_convention", "camera_or_viewpoint_shift")
ABSTAIN = ("visibility_or_offscreen", "ambiguous")
DECIDABLE = ("sharp_visible_transition", "same_action_internal_motion")
SHARP = "sharp_visible_transition"

SOURCE = re.compile(r"_(gt_boundary|raw_change_peak|false_mid_segment|false_gap|"
                    r"false_near_edge|missed_[a-z_]+?|late|early|exact|duplicate)_t")


def cand_source(eid):
    m = SOURCE.search(eid)
    return m.group(1) if m else "other"


def run_cv(Lg, Rg, Ll, Rl, X_rel, extra, y, groups, folds, pca_dim=64, l2=5.0):
    oof = np.full(len(y), np.nan)
    for f in folds:
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
            continue
        pg = pca_fit(np.concatenate([Lg[tr], Rg[tr]], 0), pca_dim)
        pl = pca_fit(np.concatenate([Ll[tr], Rl[tr]], 0), pca_dim)
        st_rel = _impute_scale_fit(X_rel[tr])
        st_ex = _impute_scale_fit(extra[tr])

        def build(m):
            return np.concatenate([
                pair_block(pca_apply(pg, Lg[m]), pca_apply(pg, Rg[m])),
                pair_block(pca_apply(pl, Ll[m]), pca_apply(pl, Rl[m])),
                _impute_scale_apply(st_rel, X_rel[m]),
                _impute_scale_apply(st_ex, extra[m])], 1)

        Ptr = build(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), y[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(stP, build(te)) @ w + b)
    return oof


def operating_point(oof, y, is_sharp, groups, folds, target=0.95):
    """Out-of-fold: the threshold is chosen on training recordings and applied
    to held-out ones, so no event is judged by a threshold it helped pick."""
    n_rm = n_right = n_sharp_lost = 0
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in groups]) & np.isfinite(oof)
        tr = (~np.array([g in f for g in groups])) & np.isfinite(oof)
        if te.sum() < 2 or tr.sum() < 10:
            continue
        order = np.argsort(-oof[tr])
        st, hit, th = oof[tr][order], 0, None
        want = y[tr][order]
        for i in range(len(st)):
            hit += bool(want[i])
            if hit / (i + 1) >= target:
                th = st[i]
        if th is None:
            # No prefix of this fold's ranking reaches the target precision, so
            # the gate removes NOTHING on the held-out recordings rather than
            # removing something at a lower bar. That is the desired failure:
            # a detector that ranks real boundaries at the top cannot find a
            # threshold, and therefore cannot throw them away.
            per_fold.append(0)
            continue
        sel = oof[te] >= th
        n_rm += int(sel.sum())
        n_right += int((sel & (y[te] == 1)).sum())
        n_sharp_lost += int((sel & is_sharp[te]).sum())
        per_fold.append(int(sel.sum()))
    return {"n_removed": n_rm, "n_correct": n_right,
            "precision": n_right / n_rm if n_rm else float("nan"),
            "precision_wilson": wilson(n_right, n_rm),
            "n_sharp_dropped": n_sharp_lost, "per_fold_n": per_fold}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--clean145", action="store_true")
    ap.add_argument("--batch3_manifest", action="append", default=[])
    ap.add_argument("--batch3_pair_labels", action="append", default=[])
    ap.add_argument("--hand_traj",
                    help="hand_trajectory_features.csv. THE FEATURE FAMILY "
                         "THIS GATE WAS NEVER GIVEN. It ran on frozen ViT "
                         "global+local only, one day after the trajectories "
                         "existed. On the same events the trajectory probe "
                         "separates visibility_or_offscreen at AUROC 0.900 "
                         "with every fold in [0.853, 0.946]; this gate's "
                         "abstain arm reached 0.729 without it.")
    ap.add_argument("--per_class", action="store_true",
                    help="report each nuisance subtype on its own as well as "
                         "pooled. annotation_convention has no visual "
                         "signature BY DEFINITION -- the GT cut where nothing "
                         "happens -- so pooling it with camera scores a "
                         "detectable class together with an undetectable one "
                         "and then reports that no operating point exists.")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    loc_rid = load_feature_caches(a.local_cache)
    sources = ([(None, None)] if a.clean145 else []) \
        + list(zip(a.batch3_manifest, a.batch3_pair_labels))
    if not sources:
        raise SystemExit("no event source: pass --clean145 and/or a manifest")

    allev = []
    for man, pl in sources:
        sub = argparse.Namespace(**vars(a))
        sub.batch3_manifest, sub.batch3_pair_labels = man, pl
        ev, ex = gather(sub, by_rid, loc_rid)
        # clean AND non-clean together: the gate's whole job is telling them
        # apart, so a run on either alone has no target
        allev += ev + ex
    ev, Lg, Rg, Ll, Rl, X_rel = matrices(allev, loc_rid)
    sub = [e.get("temporal_pair_subtype") or "" for e in ev]
    print(f"{len(ev)} events with both branches; "
          f"{dict(Counter(sub))}")

    extra = np.array([[detect_coverage(loc_rid[e["recording_id"]], e["t"]),
                       detect_longest_gap_s(loc_rid[e["recording_id"]], e["t"])]
                      for e in ev])

    # HAND TRAJECTORIES, APPENDED RATHER THAN SUBSTITUTED. The ViT branches
    # stay so the comparison is with-versus-without on identical folds; a run
    # that swapped the features would confound the family with everything else
    # that changed.
    if a.hand_traj:
        import csv as _csv
        rows = list(_csv.DictReader(open(a.hand_traj, newline="",
                                         encoding="utf-8-sig")))
        cols = [c for c in (rows[0] if rows else {})
                if c not in ("event_id", "recording_id", "t", "subtype")]
        tab = {}
        for r in rows:
            v = []
            for c in cols:
                try:
                    v.append(float(r[c]))
                except (TypeError, ValueError):
                    v.append(np.nan)
            tab[r.get("event_id", "")] = v
        hit = sum(1 for e in ev if e["event_id"] in tab)
        if not hit:
            raise SystemExit(
                f"--hand_traj matched 0 of {len(ev)} events by event_id. "
                f"An unmatched join here would silently add a column of NaN "
                f"and report the ViT-only result as if the trajectories had "
                f"been used.\n  csv has {len(rows)} rows, first id "
                f"{rows[0].get('event_id') if rows else None!r}")
        print(f"  hand trajectories: {len(cols)} features, joined onto "
              f"{hit}/{len(ev)} events ({hit / len(ev):.1%})")
        H = np.array([tab.get(e["event_id"], [np.nan] * len(cols))
                      for e in ev], float)
        extra = np.concatenate([extra, H], 1)
    groups = [e["recording_id"] for e in ev]
    is_sharp = np.array([s == SHARP for s in sub])
    src = np.array([cand_source(e["event_id"]) for e in ev])
    out = {"n": len(ev), "by_subtype": dict(Counter(sub))}

    arms = [("REJECTABLE (annotation_convention + camera)", REJECT),
            ("ABSTAIN-ONLY (offscreen + ambiguous)", ABSTAIN)]
    if a.per_class:
        arms += [(f"per-class: {c}", (c,))
                 for c in sorted(set(REJECT + ABSTAIN))]
    for tag, pos in arms:
        keep = np.array([s in pos or s in DECIDABLE for s in sub])
        y = np.array([s in pos for s in sub], float)[keep]
        if y.sum() < 15:
            print(f"\n### {tag}: only {int(y.sum())} positives, not evaluated")
            continue
        g = [x for x, k in zip(groups, keep) if k]
        folds = stratified_grouped_folds(g, y, 5, seed=a.seed)
        oof = run_cv(Lg[keep], Rg[keep], Ll[keep], Rl[keep], X_rel[keep],
                     extra[keep], y, g, folds, a.pca_dim)
        m = np.isfinite(oof)
        au = _auroc(y[m], oof[m]) if len(set(y[m].tolist())) == 2 else float("nan")
        op = operating_point(oof, y, is_sharp[keep], g, folds)
        lo, hi = op["precision_wilson"]
        print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")
        print(f"  {int(keep.sum())} events, {int(y.sum())} nuisance / "
              f"{int((1 - y).sum())} decidable, AUROC {au:.3f}")
        print(f"  at a >=0.95-precision threshold chosen out-of-fold:")
        print(f"    removed          {op['n_removed']} of {int(y.sum())} "
              f"nuisance candidates  (coverage "
              f"{op['n_removed'] / max(y.sum(), 1):.3f})")
        print(f"    removal precision {op['precision']:.3f} [{lo:.2f}, {hi:.2f}]")
        print(f"    SHARP DROPPED    {op['n_sharp_dropped']}"
              + ("   <- real boundaries thrown away; this is the number that "
                 "decides deployability" if op["n_sharp_dropped"] else
                 "   <- none, which is the only acceptable value"))
        print(f"    per-fold removed {op['per_fold_n']}")

        # by candidate source: the deployed gain is the raw-peak column
        print(f"\n  by candidate source (the deployed mix is raw_change_peak, "
              f"not gt_boundary):")
        print(f"    {'source':<22} {'n':>4} {'nuisance':>9} {'AUROC':>7}")
        for s in sorted(set(src[keep])):
            sm = src[keep] == s
            if sm.sum() < 15:
                continue
            ys, os_ = y[sm], oof[sm]
            k = np.isfinite(os_)
            asrc = (_auroc(ys[k], os_[k])
                    if len(set(ys[k].tolist())) == 2 else float("nan"))
            print(f"    {s:<22} {int(sm.sum()):>4} {int(ys.sum()):>9} "
                  f"{asrc:>7.3f}")
        out[tag] = {"auroc": float(au), **{k: v for k, v in op.items()
                                           if k != "precision_wilson"}}

    print(f"\n{'=' * 70}")
    print("  A gate is worth deploying only if it removes a useful share of "
          "nuisance at high precision AND drops no real boundaries. Removal "
          "coverage below\n  roughly half, or any non-zero sharp drop, means "
          "the 10.7-point review reduction is not available from this signal "
          "and the nuisance events\n  have to keep going to a human.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
