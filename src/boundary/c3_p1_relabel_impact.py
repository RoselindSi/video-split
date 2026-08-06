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

THE EVENT SET MUST NOT CHANGE. A correction that turned a clean subtype into
gradual or ambiguous would drop the event from the population, and the two
arms would then be scored on different events -- a difference in the number of
hard events, reported as a difference in model quality. Every correction here
is sharp<->same and both stay in the clean binary, but that is asserted rather
than assumed, and the run stops if the populations differ.

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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    ev_o = build(a.gold, a.context, by_rid, a.pair_labels,
                 a.batch3_manifest, a.batch3_pair_labels)
    ev_c = build(a.gold, a.context, by_rid, a.pair_labels_corrected,
                 a.batch3_manifest, a.batch3_pair_labels_corrected)
    id_o = [e["event_id"] for e in ev_o]
    id_c = [e["event_id"] for e in ev_c]
    print(f"original clean events {len(id_o)}, corrected {len(id_c)}")
    if set(id_o) != set(id_c):
        only_o, only_c = set(id_o) - set(id_c), set(id_c) - set(id_o)
        raise SystemExit(
            f"the populations differ: {len(only_o)} dropped, {len(only_c)} "
            f"added. A correction moved an event in or out of the clean set, "
            f"so the two arms would be scored on different events and the "
            f"difference in how many hard events each contains would be "
            f"reported as model quality. Examples: "
            f"{sorted(only_o)[:3]} / {sorted(only_c)[:3]}")

    # align the corrected events to the original order so one feature build
    # serves both arms and only the target vector differs
    by_id = {e["event_id"]: e for e in ev_c}
    ev_c = [by_id[i] for i in id_o]

    X_v1, Ls, Rs, X_rel, keep, _ = build_matrices(ev_o, False)
    ev = [ev_o[i] for i in keep]
    evc = [ev_c[i] for i in keep]
    y_o = np.array([e["y"] for e in ev], float)
    y_c = np.array([e["y"] for e in evc], float)
    groups = [e["recording_id"] for e in ev]
    flipped = [e["event_id"] for e, a_, b_ in zip(ev, y_o, y_c) if a_ != b_]
    print(f"{len(ev)} poolable, {len(set(groups))} recordings, "
          f"{len(flipped)} labels differ between the arms")
    for f in flipped:
        print(f"    {f}")
    if not flipped:
        raise SystemExit("no label differs inside the poolable set -- the "
                         "corrections did not reach the events P1 is scored "
                         "on, so this comparison has nothing to measure")

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

    out["all_clean"] = arm("ALL CLEAN EVENTS", np.ones(len(ev), bool))
    if dec:
        rev = np.array([dec.get(e["event_id"]) == "REVIEW" for e in ev])
        out["review_band"] = arm("REVIEW BAND", rev)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
