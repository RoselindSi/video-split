"""P1 under the original labels and under the corrected ones, same folds.

Every number this project has reported for P1 -- 0.778 on all clean events,
0.528 in the REVIEW band -- was measured against labels that two independent
annotators disagree with on roughly a third of the audited sample. This
re-runs the identical model and the identical folds with the corrected labels
substituted, and reports nothing else.

WHAT A MOVE HERE WOULD AND WOULD NOT MEAN. Eight labels change out of 313.
If the AUROC rises, part of what looked like model error was label error, and
the rise is a LOWER BOUND on what full relabelling would give -- the other
mislabelled events are still in the set, still scored as failures. If it does
not rise, that is not evidence the labels are fine: eight corrections out of
313 is a 2.6% perturbation, and the confidence interval on a 313-event AUROC
is far wider than any effect that size can produce.

So the honest reading is asymmetric, and stated as such rather than left to
the reader. This run cannot vindicate the label set. It can only show whether
a small measured correction moves things in the direction the audit predicts.

THE EVENT SET CHANGES, AND THAT IS HANDLED RATHER THAN FORBIDDEN. Two-annotator
corrections were all sharp<->same, so the population stayed fixed and the run
refused to continue otherwise. A full relabel is different by construction: on
the first 40 rows of the batch3 relabel, 15 of 40 events crossed the clean-set
boundary while only 3 flipped sign inside it -- the repair is mostly about
WHICH events belong, not how the ones that belong are signed. Refusing to run
would refuse to measure the main effect.

So two arms, and they answer different questions:

  MATCHED    only events in both clean sets. Same events, same folds, only the
             target differs -- the one arm whose delta means "the model got
             better", and the only one the null and the decomposition apply to.
  FULL       each label set on its own population. The two AUROCs are computed
             on DIFFERENT events, so their difference is NOT a gain and is
             never printed as a delta. It answers "what will the number be from
             now on", which is a separate and also necessary thing to know.

The composition change is reported first, before either arm. The batch3
held-out failure was diagnosed as a base-rate shift, so a relabel that moves
events in and out of the clean set perturbs the exact quantity that diagnosis
rested on.

FOLDS ARE BUILT ONCE, FROM THE ORIGINAL LABELS, AND REUSED. Stratified
grouped folds depend on the labels, so rebuilding them under the corrected set
would change the split as well as the target and confound the comparison.

Usage:
    python -m src.boundary.c3_p1_relabel_impact \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --batch3_pair_labels data/gold/batch3_pair_labels_v1.csv \
        --pair_labels_corrected data/gold/pair_labels_v1_corrected_v1.csv \
        --batch3_pair_labels_corrected data/gold/batch3_pair_labels_v1_corrected_v1.csv \
        --batch3_manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --out /workspace/tr1/results/hal/c3/p1_relabel_impact.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import stratified_grouped_folds, build_matrices
from src.boundary.c3_sidechange_arm import run_cv, automatable

SHARP = "sharp_visible_transition"


def build(gold, ctx, by_rid, pl, b3_manifest, b3_pl):
    events = T.apply_to_events(build_events(S.load_gold(gold), S.load_context(ctx),
                                            by_rid), T.load_pair_labels(pl))
    if b3_manifest:
        from src.boundary.batch3_dev_events import build_events as build_b3
        events += T.apply_to_events(build_b3(b3_manifest, by_rid),
                                    T.load_pair_labels(b3_pl))
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--batch3_pair_labels",
                    default="data/gold/batch3_pair_labels_v1.csv")
    ap.add_argument("--pair_labels_corrected", required=True)
    ap.add_argument("--batch3_pair_labels_corrected", required=True)
    ap.add_argument("--batch3_manifest")
    ap.add_argument("--decisions")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_null", type=int, default=50,
                    help="random-flip draws; the scale for the observed delta")
    ap.add_argument("--allow_population_change", action="store_true",
                    help="expected for a full relabel, a bug for a two-annotator "
                         "correction; adds the matched and full arms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    ev_o = build(a.gold, a.context, by_rid, a.pair_labels,
                 a.batch3_manifest, a.batch3_pair_labels)
    ev_c = build(a.gold, a.context, by_rid, a.pair_labels_corrected,
                 a.batch3_manifest, a.batch3_pair_labels_corrected)
    id_o = {e["event_id"] for e in ev_o}
    id_c = {e["event_id"] for e in ev_c}
    only_o, only_c = sorted(id_o - id_c), sorted(id_c - id_o)
    rate = lambda evs: float(np.mean([e["y"] for e in evs])) if evs else float("nan")
    print(f"\n{'=' * 68}\nCOMPOSITION\n{'=' * 68}")
    print(f"  original clean set {len(id_o)} events, base rate {rate(ev_o):.3f}")
    print(f"  relabelled         {len(id_c)} events, base rate {rate(ev_c):.3f}")
    print(f"  {len(only_o)} left the clean set, {len(only_c)} entered, "
          f"{len(id_o & id_c)} in both")
    for e in only_o:
        print(f"    LEFT   {e}")
    for e in only_c:
        print(f"    ENTER  {e}")
    if only_o or only_c:
        print("  The batch3 held-out failure was diagnosed as a base-rate "
              "shift, so this movement perturbs the quantity that diagnosis "
              "rested on.")
    if not a.allow_population_change and (only_o or only_c):
        raise SystemExit(
            "the populations differ. For a two-annotator correction that is a "
            "bug -- the arms would be scored on different events. For a full "
            "relabel it is the point. Pass --allow_population_change to get "
            "the matched and full arms.")

    shared = [e["event_id"] for e in ev_o if e["event_id"] in id_c]
    by_o = {e["event_id"]: e for e in ev_o}
    by_c = {e["event_id"]: e for e in ev_c}

    # MATCHED: same events, one feature build, only the target differs
    ev_m = [by_o[i] for i in shared]
    X_v1, Ls, Rs, X_rel, keep, _ = build_matrices(ev_m, False)
    ev = [ev_m[i] for i in keep]
    y_o = np.array([e["y"] for e in ev], float)
    y_c = np.array([by_c[e["event_id"]]["y"] for e in ev], float)
    groups = [e["recording_id"] for e in ev]
    flipped = [e["event_id"] for e, a_, b_ in zip(ev, y_o, y_c) if a_ != b_]
    print(f"\nmatched population: {len(ev)} poolable, {len(set(groups))} "
          f"recordings, {len(flipped)} labels differ")
    for f in flipped:
        print(f"    {f}")
    if not flipped:
        print("  !! no label differs inside the matched population, so the "
              "matched arm has nothing to measure. The whole effect of this "
              "relabel is the population change, which the full arm reports.")

    dec = {}
    if a.decisions:
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dec[r["event_id"]] = r.get("decision")

    out = {"n_events": len(ev), "n_flipped": len(flipped), "flipped": flipped}

    def arm(title, sel):
        if sel.sum() < 40:
            print(f"\n### {title}: too few events ({int(sel.sum())})")
            return None
        g = [x for x, k in zip(groups, sel) if k]
        nf = int(sum(1 for e, k in zip(ev, sel)
                     if k and e["event_id"] in flipped))
        print(f"\n{'=' * 68}\n{title}  ({int(sel.sum())} events, "
              f"{nf} of them relabelled)\n{'=' * 68}")
        res = {}
        for tag, y in (("original", y_o), ("corrected", y_c)):
            # folds from the ORIGINAL labels for BOTH arms. Stratification
            # depends on y, so building them from the corrected target would
            # change the split as well as the label and confound the two.
            fo = stratified_grouped_folds(g, y_o[sel], 5, seed=a.seed)
            oof, pf = run_cv(Ls[sel], Rs[sel], X_rel[sel], y[sel], g, fo,
                             None, a.pca_dim)
            m = np.isfinite(oof)
            au = (_auroc(y[sel][m], oof[m])
                  if len(set(y[sel][m].tolist())) == 2 else float("nan"))
            isp = y[sel].astype(bool)
            aut = automatable(oof, isp, g, fo)
            print(f"  {tag:<10} AUROC {au:.3f}   per-fold "
                  f"{[round(x, 3) for x in pf]}")
            print(f"  {'':<10} automatable at >=0.95: {aut['n_auto']} events, "
                  f"precision {aut['precision']:.3f}")
            res[tag] = {"auroc": float(au), "per_fold": pf, "automatable": aut,
                        "oof": oof}
        d = res["corrected"]["auroc"] - res["original"]["auroc"]
        print(f"  delta {d:+.3f}")

        # DECOMPOSITION. A relabelled event whose score already sat on the
        # annotators' side raises AUROC the moment its label flips, with no
        # change in the model. That is real information -- the label was wrong
        # and the model was not -- but it is not evidence that cleaner labels
        # produce a better model. The events whose labels did NOT change
        # separate the two: their targets are identical in both arms, so any
        # movement there comes only from the corrected rows sitting in other
        # folds' TRAINING data.
        unch = np.array([e["event_id"] not in flipped
                         for e, k in zip(ev, sel) if k])
        if unch.sum() >= 40 and len(set(y_o[sel][unch].tolist())) == 2:
            u = {}
            for tag in ("original", "corrected"):
                o = np.asarray(res[tag]["oof"])[unch]
                yy = y_o[sel][unch]
                m = np.isfinite(o)
                u[tag] = (_auroc(yy[m], o[m])
                          if len(set(yy[m].tolist())) == 2 else float("nan"))
            print(f"  on the {int(unch.sum())} events whose labels did NOT "
                  f"change: {u['original']:.3f} -> {u['corrected']:.3f} "
                  f"({u['corrected'] - u['original']:+.3f})")
            print(f"    identical targets in both arms, so this movement is "
                  f"the model learning from cleaner TRAINING labels -- the "
                  f"part that generalises. The rest of the "
                  f"{d:+.3f} is the corrected rows scoring against their own "
                  f"fixed labels.")
            res["unchanged_only"] = {"original": float(u["original"]),
                                     "corrected": float(u["corrected"]),
                                     "delta": float(u["corrected"] - u["original"])}

        # NULL. Flipping the same number of labels at random should not help.
        # Without this, "+0.066 after changing 8 labels" has no scale: any
        # perturbation of a 169-event AUROC moves it somewhat.
        rng = np.random.RandomState(a.seed)
        idx = np.nonzero(sel)[0]
        null = []
        for _ in range(a.n_null):
            yr = y_o.copy()
            pick = rng.choice(idx, min(len(flipped), len(idx)), replace=False)
            yr[pick] = 1 - yr[pick]
            oof, _ = run_cv(Ls[sel], Rs[sel], X_rel[sel], yr[sel], g, fo,
                            None, a.pca_dim)
            m = np.isfinite(oof)
            if len(set(yr[sel][m].tolist())) == 2:
                null.append(_auroc(yr[sel][m], oof[m])
                            - res["original"]["auroc"])
        if null:
            null = np.array(null)
            p = float((null >= d).mean())
            print(f"  null: flipping {len(flipped)} labels at random moves it "
                  f"{null.mean():+.3f} on average "
                  f"[{np.percentile(null, 5):+.3f}, "
                  f"{np.percentile(null, 95):+.3f}] over {len(null)} draws; "
                  f"p = {p:.3f}")
            res["null"] = {"mean": float(null.mean()), "p": p,
                           "p05": float(np.percentile(null, 5)),
                           "p95": float(np.percentile(null, 95))}
        if nf:
            print(f"  {nf} of {int(sel.sum())} labels changed "
                  f"({nf / sel.sum():.1%}). A rise is a LOWER BOUND on full "
                  f"relabelling; a flat result does not clear the label set, "
                  f"because a perturbation this small cannot move an AUROC "
                  f"beyond its own interval.")
        for k in ("original", "corrected"):
            res[k].pop("oof", None)
        res["delta"] = float(d)
        return res

    if flipped:
        out["matched_all_clean"] = arm("MATCHED: ALL CLEAN EVENTS",
                                       np.ones(len(ev), bool))
        if dec:
            rev = np.array([dec.get(e["event_id"]) == "REVIEW" for e in ev])
            out["matched_review_band"] = arm("MATCHED: REVIEW BAND", rev)

    # FULL: each label set on its own population. NOT a delta -- the two
    # numbers are computed on different events, and subtracting them would
    # report a change in which events are being scored as a change in model
    # quality. It answers what the baseline becomes from here on.
    print(f"\n{'=' * 68}\nFULL POPULATIONS (NOT a like-for-like comparison)"
          f"\n{'=' * 68}")
    full = {}
    for tag, evs in (("original", ev_o), ("relabelled", ev_c)):
        _, L, R, Xr, kp, _ = build_matrices(evs, False)
        e2 = [evs[i] for i in kp]
        yy = np.array([e["y"] for e in e2], float)
        gg = [e["recording_id"] for e in e2]
        ff = stratified_grouped_folds(gg, yy, 5, seed=a.seed)
        oof, pf = run_cv(L, R, Xr, yy, gg, ff, None, a.pca_dim)
        m = np.isfinite(oof)
        au = (_auroc(yy[m], oof[m]) if len(set(yy[m].tolist())) == 2
              else float("nan"))
        aut = automatable(oof, yy.astype(bool), gg, ff)
        print(f"  {tag:<12} {len(e2):>4} events, base rate {yy.mean():.3f}, "
              f"AUROC {au:.3f}   per-fold {[round(x, 3) for x in pf]}")
        print(f"  {'':<12} automatable at >=0.95: {aut['n_auto']} events, "
              f"precision {aut['precision']:.3f}")
        full[tag] = {"n": len(e2), "base_rate": float(yy.mean()),
                     "auroc": float(au), "per_fold": pf, "automatable": aut}
    print("  These two AUROCs are computed on DIFFERENT events. Their "
          "difference is not a gain and is deliberately not printed as one;\n"
          "  the matched arm above is the comparison that means the model "
          "changed.")
    out["full_populations"] = full

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
