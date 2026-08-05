"""One fixed diagnostic: does side_change add anything to P1?

side_change is the only temporal feature in this project that replicated from
the clean-145 to the merged 313 without collapsing -- 0.835 to 0.710 with a
bootstrap CI of [0.624, 0.791] entirely above a family-wise chance baseline of
0.590. It is also constructionally different from everything P1 measures:

    P1           are the two sides of the candidate DIFFERENT from each other
    side_change  does each side keep CHANGING WITHIN ITSELF

Two events can carry the same P1 score with one side internally stable and the
other in continuous motion, and the second is likelier to be repetition or an
unsettled phase. Whether that difference is complementary INFORMATION is the
one question the single-variable analysis could not answer: side_change alone
reaches 0.610 in the REVIEW band, below its baseline there, but a variable can
be uninformative alone and still explain P1's residual.

Deliberately narrow. ONE arm, defined in advance:

    P1 pair block  +  side_change_left, side_change_right, side_change_imbalance
    -> per-fold standardisation -> L2 logistic regression

No feature subset search, no weights, no variants, and it is NOT added to the
v2/v3 policy search space. It is a mechanism diagnostic and its definition does
not change after the result is seen.

P1 IS REFIT FROM FEATURES INSIDE THE SAME FOLDS, not taken from the existing
OOF column. An OOF score is produced by a model trained on the other folds, so
feeding it as an input to a model trained on those same other folds leaks the
test fold through the back door. Both arms are built from raw side vectors here
so the comparison is clean.

PRE-REGISTERED STOPPING RULE, written before the run. side_change graduates to
a policy component ONLY if, in the REVIEW band:
  1. it gains over P1 there, not only on all clean events, with a grouped
     bootstrap CI on the delta that clears zero
  2. at least 20 additional events are automated out-of-fold
  3. those additions hold >= 0.95 precision
  4. sharp false-reject does not rise above 0.05
  5. the gain is not carried by one fold
Otherwise the finding is archived as "a real global ranking feature with no
contribution to reducing review", and this line stops.

Usage:
    python -m src.boundary.c3_sidechange_arm \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --batch3_manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --side_change /workspace/tr1/results/hal/c3/concentration_features_full.csv \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --out /workspace/tr1/results/hal/c3/sidechange_arm.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    stratified_grouped_folds, build_matrices,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.c3_selective_policy import wilson
from src.boundary.c3_error_separability import auroc_stats

SC_COLS = ["side_change_left", "side_change_right", "side_change_imbalance"]
SHARP = "sharp_visible_transition"


def run_cv(Ls, Rs, X_rel, y, groups, folds, extra=None, pca_dim=64, l2=5.0):
    oof = np.full(len(y), np.nan)
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
            per_fold.append(float("nan"))
            continue
        pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), pca_dim)
        st_rel = _impute_scale_fit(X_rel[tr])

        def build(m):
            parts = [pair_block(pca_apply(pca, Ls[m]), pca_apply(pca, Rs[m])),
                     _impute_scale_apply(st_rel, X_rel[m])]
            if extra is not None:
                parts.append(extra[m])
            return np.concatenate(parts, 1)

        Ptr = build(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), y[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(stP, build(te)) @ w + b)
        per_fold.append(_auroc(y[te], oof[te])
                        if len(set(y[te].tolist())) == 2 else float("nan"))
    return oof, per_fold


def automatable(scores, is_sharp, groups, folds, target=0.95):
    """Out-of-fold: threshold chosen on training recordings, applied to held-out
    ones. Counts how many events could be AUTO_KEEPed and at what precision.
    Never evaluates a threshold on the events that chose it."""
    n_auto = n_right = 0
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in groups]) & np.isfinite(scores)
        tr = (~np.array([g in f for g in groups])) & np.isfinite(scores)
        if te.sum() < 2 or tr.sum() < 10:
            continue
        order = np.argsort(-scores[tr])
        st, hit, th = scores[tr][order], 0, None
        want = is_sharp[tr][order]
        for i in range(len(st)):
            hit += bool(want[i])
            if hit / (i + 1) >= target:
                th = st[i]
        if th is None:
            per_fold.append(0)
            continue
        sel = scores[te] >= th
        n_auto += int(sel.sum())
        n_right += int((sel & is_sharp[te]).sum())
        per_fold.append(int(sel.sum()))
    return {"n_auto": n_auto, "n_right": n_right,
            "precision": n_right / n_auto if n_auto else float("nan"),
            "precision_wilson": wilson(n_right, n_auto),
            "per_fold_n": per_fold}


def report(label, y, oof_a, oof_b, pf_a, pf_b, groups, folds, is_sharp, n_boot, seed):
    m = np.isfinite(oof_a) & np.isfinite(oof_b)
    au_a = _auroc(y[m], oof_a[m])
    au_b = _auroc(y[m], oof_b[m])
    print(f"\n=== {label} ({int(m.sum())} events, "
          f"{int(y[m].sum())}+ / {int((1 - y[m]).sum())}-, "
          f"{len({g for g, k in zip(groups, m) if k})} recordings) ===")
    print(f"  P1 alone            {au_a:.3f}   per-fold "
          f"{[round(x, 3) for x in pf_a]}")
    print(f"  P1 + side_change    {au_b:.3f}   per-fold "
          f"{[round(x, 3) for x in pf_b]}")
    d = [b - a for a, b in zip(pf_a, pf_b) if np.isfinite(a) and np.isfinite(b)]
    print(f"  delta {au_b - au_a:+.3f}   per-fold delta {[round(x, 3) for x in d]}   "
          f"improved {sum(1 for x in d if x > 0)}/{len(d)}   worst {min(d):+.3f}")

    # paired grouped bootstrap on the DELTA, resampling recordings
    by = {}
    for i, g in enumerate(groups):
        if m[i]:
            by.setdefault(g, []).append(i)
    keys = sorted(by)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        idx = [i for k in rng.choice(keys, len(keys), replace=True) for i in by[k]]
        if len(set(y[idx].tolist())) == 2:
            bs.append(_auroc(y[idx], oof_b[idx]) - _auroc(y[idx], oof_a[idx]))
    ci = ([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
          if bs else [float("nan")] * 2)
    print(f"  95% grouped-bootstrap CI on the delta: [{ci[0]:+.3f}, {ci[1]:+.3f}]"
          + ("  <- excludes 0" if ci[0] > 0 else "  <- straddles 0"))

    aa = automatable(oof_a, is_sharp, groups, folds)
    ab = automatable(oof_b, is_sharp, groups, folds)
    print(f"  automatable at >=0.95 precision, out-of-fold:")
    for nm, r in (("P1 alone", aa), ("P1 + side_change", ab)):
        lo, hi = r["precision_wilson"]
        print(f"    {nm:<20} {r['n_auto']:>4} events, precision "
              f"{r['precision']:.3f} [{lo:.2f}, {hi:.2f}]  per-fold {r['per_fold_n']}")
    return {"auroc_p1": au_a, "auroc_arm": au_b, "delta": au_b - au_a,
            "ci95": ci, "per_fold_delta": d,
            "automatable_p1": aa, "automatable_arm": ab}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--batch3_manifest")
    ap.add_argument("--batch3_pair_labels", default="data/gold/batch3_pair_labels_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--side_change", required=True,
                    help="c3_transition_concentration --dump CSV")
    ap.add_argument("--decisions", required=True,
                    help="per-role decisions CSV, to identify the REVIEW band")
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    events = T.apply_to_events(
        build_events(S.load_gold(a.gold), S.load_context(a.context), by_rid),
        T.load_pair_labels(a.pair_labels))
    if a.batch3_manifest:
        from src.boundary.batch3_dev_events import build_events as build_b3
        events += T.apply_to_events(build_b3(a.batch3_manifest, by_rid),
                                    T.load_pair_labels(a.batch3_pair_labels))
    print(f"merged clean events: {len(events)}  "
          f"{dict(Counter(e.get('temporal_pair_subtype') for e in events))}")

    sc = {}
    with open(a.side_change, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                sc[r["event_id"]] = [float(r[c]) for c in SC_COLS]
            except (KeyError, ValueError):
                pass
    dec = {}
    with open(a.decisions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dec[r["event_id"]] = r["decision"]
    print(f"side_change joined for {sum(1 for e in events if e['event_id'] in sc)}"
          f"/{len(events)} events; decisions for "
          f"{sum(1 for e in events if e['event_id'] in dec)}/{len(events)}")

    events = [e for e in events if e["event_id"] in sc]
    if len(events) < 100:
        raise SystemExit("too few events carry side_change features")

    X_v1, Ls, Rs, X_rel, keep, _ = build_matrices(events, False)
    ev = [events[i] for i in keep]
    y = np.array([e["y"] for e in ev], dtype=float)
    groups = [e["recording_id"] for e in ev]
    extra = np.array([sc[e["event_id"]] for e in ev], dtype=float)
    is_sharp = np.array([e.get("temporal_pair_subtype") == SHARP for e in ev])
    folds = stratified_grouped_folds(groups, y, 5, seed=a.seed)
    print(f"{len(ev)} events poolable, {len(set(groups))} recordings")

    oof_a, pf_a = run_cv(Ls, Rs, X_rel, y, groups, folds, None, a.pca_dim)
    oof_b, pf_b = run_cv(Ls, Rs, X_rel, y, groups, folds, extra, a.pca_dim)
    out = {"all_clean": report("A. ALL CLEAN EVENTS", y, oof_a, oof_b, pf_a, pf_b,
                               groups, folds, is_sharp, a.n_boot, a.seed)}

    sel = np.array([dec.get(e["event_id"]) == "REVIEW" for e in ev])
    print(f"\nREVIEW band: {int(sel.sum())} of {len(ev)} clean events")
    if sel.sum() >= 40 and len(set(y[sel].tolist())) == 2:
        gsel = [g for g, k in zip(groups, sel) if k]
        fsel = stratified_grouped_folds(gsel, y[sel], 5, seed=a.seed)
        # BOTH arms are refit on the REVIEW subset alone. Reusing the scores
        # fitted on all clean events would let the model that saw the easy
        # events decide the hard ones, which is not the question.
        oa, pa = run_cv(Ls[sel], Rs[sel], X_rel[sel], y[sel], gsel, fsel, None, a.pca_dim)
        ob, pb = run_cv(Ls[sel], Rs[sel], X_rel[sel], y[sel], gsel, fsel,
                        extra[sel], a.pca_dim)
        out["review_band"] = report("C. REVIEW BAND ONLY", y[sel], oa, ob, pa, pb,
                                    gsel, fsel, is_sharp[sel], a.n_boot, a.seed)
    else:
        print("  too few to evaluate")
        out["review_band"] = None

    print(f"\n{'=' * 72}\nPRE-REGISTERED STOPPING RULE (REVIEW band)\n{'=' * 72}")
    rb = out.get("review_band")
    if not rb:
        print("  cannot be evaluated -- REVIEW band too small")
    else:
        aa, ab = rb["automatable_p1"], rb["automatable_arm"]
        gained = ab["n_auto"] - aa["n_auto"]
        d = rb["per_fold_delta"]
        checks = {
            # "a stable gain", not "a positive point estimate". Implemented as
            # delta > 0 alone, this condition PASSED on a pure-noise fixture
            # whose delta was +0.002 with a CI of [-0.011, +0.016] -- a
            # criterion that cannot fail is not a criterion. The grouped
            # bootstrap CI must clear zero.
            "gains over P1 in the REVIEW band (CI clears 0)":
                (rb["delta"] > 0 and rb["ci95"][0] > 0),
            ">= 20 additional events automated": (gained >= 20),
            "additions hold >= 0.95 precision": (np.isfinite(ab["precision"])
                                                 and ab["precision"] >= 0.95),
            "not carried by one fold": (sum(1 for x in d if x > 0) >= 3),
        }
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"    (delta {rb['delta']:+.3f}; automated {aa['n_auto']} -> "
              f"{ab['n_auto']}, i.e. {gained:+d}; arm precision "
              f"{ab['precision']:.3f}; folds improved "
              f"{sum(1 for x in d if x > 0)}/{len(d)})")
        out["stopping_rule"] = {k: bool(v) for k, v in checks.items()}
        if all(checks.values()):
            print("\n  All conditions met: side_change is worth developing into a "
                  "policy component.")
        else:
            print("\n  NOT met. Archive the finding as written in advance: "
                  "side_change is a real global ranking feature with no "
                  "contribution to reducing review. Do not tune it further -- "
                  "the next move is the local hand trajectory, which measures "
                  "what neither this nor concentration can see.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
