"""C3-lite evaluation: does a local hand-crop branch add anything over P1?

Consumes extract_features_local.py's cache alongside the global one. Adds NO
trained parameters: local features go through the same frozen PCA + logistic
regression P1 uses, refit per grouped fold on that fold's training split. C2
established what a from-scratch encoder does at this data scale (train AUROC
0.99 against OOF 0.77, a +0.21 gap on 110-121 training events), so the local
branch deliberately buys its shot at the hypothesis without buying capacity.

ARMS
  P1 (global) alone            the baseline every other number in this project
                               is measured against
  local alone                  is there ANY boundary signal in the hand crop
  P1 + local, feature-level    both pair blocks into one logreg. The natural
                               "add a branch" form and it introduces no tuning
                               knob -- but it doubles the feature dimension,
                               and the stereo block probe showed extra blocks
                               can cost PCA component budget, so it is not
                               assumed to be the better fusion.
  P1 + local, score-level      fixed 0.5/0.5, comparable to how C1 and C2 were
                               reported
  P1 + disagreement            P1 plus ONE scalar: d_global - d_local, where
                               each d is the cosine distance between that
                               branch's left and right side vectors. This is
                               the mechanism stated plainly: when the whole
                               scene changes but the hand-object region does
                               not, that is camera or body motion, not an
                               action boundary. One extra number cannot
                               overfit the way a doubled feature block can.

PER-EVENT GATING, not per-recording. The detector's rate varies enormously by
recording (0.95 down to 0.11 in the smoke test, with gaps up to 64.5s), but
features are only ever consumed in windows around candidates, and detection AT
candidate moments measured 0.87 against 0.71 over all frames. So coverage is
computed per event over [t-4, t+4] (the widest of pairwise_verifier's SCALES)
from the cache's per-frame `detected` mask, and results are reported both on
all events and on the well-detected subset.

THE BASELINE IS RECOMPUTED ON WHATEVER SUBSET IS BEING SCORED. If gating drops
events, comparing local-on-the-easy-subset against P1's 0.872-on-all-145 would
be a manufactured win. Every table here recomputes P1 on exactly the events
the other arms saw.

Primary criterion is the pre-registered one in configs/local_gate_c3.json:
rescues minus harms AT A MATCHED OPERATING POINT. Each arm's threshold is set
so it flags the SAME NUMBER of events as P1 does at the median of its
true-positive scores -- the same review budget -- so a rescue and a harm are
real net error changes rather than an artefact of two arms sitting at
different points on their curves. C2's postmortem had to attach that caveat
after the fact; here it is built in.

Usage:
    python -m src.boundary.c3_local_eval \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --local_cache /workspace/tr1/data_recseg/feat_local_dev.pt \
        --same_action_subtype data/gold/same_action_subtype_v1.csv \
        --out /workspace/tr1/results/hal/c3/local_eval.json \
        --dump_events /workspace/tr1/results/hal/c3/local_events.csv
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
    stratified_grouped_folds, build_matrices, side_vectors, SCALES,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid

WIDEST = max(SCALES)


def detect_coverage(rec, t, half=WIDEST):
    """Fraction of frames in [t-half, t+half] that were a REAL detection.

    Interpolated frames are not observations. A recording can run at 0.11
    overall and still be fine at the candidates, or run high overall and be
    interpolated exactly where it matters; only the per-event number
    distinguishes those."""
    d = rec.get("detected")
    if d is None:
        return 1.0
    times = rec["times"].numpy() if hasattr(rec["times"], "numpy") else np.asarray(rec["times"])
    d = d.numpy() if hasattr(d, "numpy") else np.asarray(d)
    n = min(len(times), len(d))
    m = (times[:n] >= t - half) & (times[:n] <= t + half)
    return float(d[:n][m].mean()) if m.any() else 0.0


def detect_longest_gap_s(rec, t, half=WIDEST):
    """Longest run of CONSECUTIVE undetected frames inside the window, in
    seconds. Computable from the existing caches, unlike box jitter, and it
    separates two cases that the coverage fraction alone conflates: a detector
    missing scattered single frames around a stable trajectory, versus one
    that lost the hand for a continuous stretch and had the whole stretch
    interpolated across. The extraction now also stores per-frame boxes so
    centre and scale jitter become available too, but only for caches built
    after that change -- using them would mean re-extracting both development
    sets, and a reliability feature present on one set and absent on the other
    is worse than a coarse one present on both."""
    d = rec.get("detected")
    if d is None:
        return 0.0
    times = rec["times"].numpy() if hasattr(rec["times"], "numpy") else np.asarray(rec["times"])
    d = d.numpy() if hasattr(d, "numpy") else np.asarray(d)
    n = min(len(times), len(d))
    m = (times[:n] >= t - half) & (times[:n] <= t + half)
    if not m.any():
        return float(2 * half)
    seq, tt = d[:n][m], times[:n][m]
    best = run = 0
    for i, v in enumerate(seq):
        run = 0 if v else run + 1
        best = max(best, run)
    step = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.5
    return float(best * step)


def cos_dist(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return 1.0 - (a * b).sum(1) / np.maximum(na * nb, 1e-8)


def matched_threshold(scores, n_positive_calls):
    """Threshold flagging exactly n_positive_calls events (ties resolved
    upward). Matching the ALERT BUDGET is what makes a rescue and a harm
    comparable across arms."""
    s = np.sort(scores[np.isfinite(scores)])[::-1]
    if n_positive_calls <= 0 or len(s) == 0:
        return np.inf
    k = min(n_positive_calls, len(s)) - 1
    return float(s[k])


def rescues_harms(y, base, cand, cutoff_base, cutoff_cand):
    pb = (base >= cutoff_base).astype(int)
    pc = (cand >= cutoff_cand).astype(int)
    ok_b, ok_c = pb == y, pc == y
    resc = int(((~ok_b) & ok_c).sum())
    harm = int((ok_b & (~ok_c)).sum())
    tp_harm = int((ok_b & (~ok_c) & (y == 1)).sum())
    fp_resc = int(((~ok_b) & ok_c & (y == 0)).sum())
    return {"rescues": resc, "harms": harm, "tp_harms": tp_harm,
            "fp_rescues": fp_resc, "net": resc - harm}


def grouped_bootstrap(y, base, cand, recs, n_boot=2000, seed=0):
    by = {}
    for i, r in enumerate(recs):
        if np.isfinite(base[i]) and np.isfinite(cand[i]):
            by.setdefault(r, []).append(i)
    keys = sorted(by)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        idx = [i for k in rng.choice(keys, len(keys), replace=True) for i in by[k]]
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        d.append(_auroc(yy, cand[idx]) - _auroc(yy, base[idx]))
    if not d:
        return float("nan"), float("nan")
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_folds(blocks, X_rel, y, groups, folds, extra=None, pca_dim=64, l2=5.0,
              oos=None):
    """blocks: list of raw side-vector pairs [(L, R), ...] each PCA'd
    separately and concatenated as pair_blocks. extra: [n, k] extra scalar
    columns. Everything is fit on the training split of each fold only.

    oos: optional (blocks, X_rel, groups, extra) for events that must be
    SCORED but never fitted -- the non-clean-binary taxonomy, which has no y.
    Each such event is scored by the fold whose test split contains its
    recording, i.e. by a model that never saw that recording, exactly like the
    clean events' OOF scores. Returns (oof, oof_oos)."""
    oof = np.full(len(y), np.nan)
    oos_oof = np.full(len(oos[2]), np.nan) if oos else np.zeros(0)
    for f in folds:
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
            continue
        pcas = [pca_fit(np.concatenate([L[tr], R[tr]], 0), pca_dim) for L, R in blocks]
        st_rel = _impute_scale_fit(X_rel[tr])

        def build(m):
            parts = [pair_block(pca_apply(p, L[m]), pca_apply(p, R[m]))
                     for p, (L, R) in zip(pcas, blocks)]
            parts.append(_impute_scale_apply(st_rel, X_rel[m]))
            if extra is not None:
                parts.append(extra[m])
            return np.concatenate(parts, 1)

        Ptr = build(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), y[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(stP, build(te)) @ w + b)
        if oos:
            ob, orel, ogr, oex = oos
            om = np.array([g in f for g in ogr])
            if om.any():
                parts = [pair_block(pca_apply(p, L[om]), pca_apply(p, R[om]))
                         for p, (L, R) in zip(pcas, ob)]
                parts.append(_impute_scale_apply(st_rel, orel[om]))
                if oex is not None:
                    parts.append(oex[om])
                oos_oof[om] = _sigmoid(
                    _impute_scale_apply(stP, np.concatenate(parts, 1)) @ w + b)
    return (oof, oos_oof) if oos else oof


def per_fold_auroc(y, oof, groups, folds):
    out = []
    m = np.isfinite(oof)
    for f in folds:
        te = np.array([g in f for g in groups]) & m
        if te.sum() >= 2 and len(set(y[te].tolist())) == 2:
            out.append(_auroc(y[te], oof[te]))
        else:
            out.append(float("nan"))
    return out


def evaluate(label, events_g, events_l, y, groups, sub_map, gate, a,
             oos_g=None, oos_l=None):
    """One full comparison on whatever event subset is passed in."""
    print(f"\n{'=' * 72}\n{label}: {len(y)} events "
          f"({int(y.sum())}+ / {int((1 - y).sum())}-), "
          f"{len(set(groups))} recordings\n{'=' * 72}")
    Xg, Lg, Rg, X_rel, keep_g, _ = build_matrices(events_g, False)
    Xl, Ll, Rl, _, keep_l, _ = build_matrices(events_l, False)
    keep = sorted(set(keep_g) & set(keep_l))
    if len(keep) < 20:
        print(f"  !! only {len(keep)} events are poolable in BOTH branches -- "
              f"too few to evaluate")
        return None
    gi = {k: i for i, k in enumerate(keep_g)}
    li = {k: i for i, k in enumerate(keep_l)}
    gsel = np.array([gi[k] for k in keep])
    lsel = np.array([li[k] for k in keep])
    if len(keep) < len(keep_g):
        print(f"  {len(keep_g) - len(keep)} events dropped: poolable globally "
              f"but not locally (the local cache has its own time grid)")

    Lg, Rg, X_rel = Lg[gsel], Rg[gsel], X_rel[gsel]
    Ll, Rl = Ll[lsel], Rl[lsel]
    yk = y[keep]
    gk = [groups[k] for k in keep]
    ev = [events_g[k] for k in keep]
    folds = stratified_grouped_folds(gk, yk, 5, seed=a.seed)

    disagree = (cos_dist(Lg, Rg) - cos_dist(Ll, Rl)).reshape(-1, 1)

    oos = None
    oos_events = []
    if oos_g:
        _, oLg, oRg, oXrel, ok_g, _ = build_matrices(oos_g, False)
        _, oLl, oRl, _, ok_l, _ = build_matrices(oos_l, False)
        okeep = sorted(set(ok_g) & set(ok_l))
        if okeep:
            ogi = {k: i for i, k in enumerate(ok_g)}
            oli = {k: i for i, k in enumerate(ok_l)}
            og = np.array([ogi[k] for k in okeep]); ol = np.array([oli[k] for k in okeep])
            oLg, oRg, oXrel = oLg[og], oRg[og], oXrel[og]
            oLl, oRl = oLl[ol], oRl[ol]
            oos_events = [oos_g[k] for k in okeep]
            oos = ([(oLg, oRg), (oLl, oRl)], oXrel,
                   [e["recording_id"] for e in oos_events],
                   (cos_dist(oLg, oRg) - cos_dist(oLl, oRl)).reshape(-1, 1))
            print(f"  scoring {len(okeep)} non-clean-binary events alongside "
                  f"(never fitted on)")

    # Block INDICES, not object identity: the out-of-sample scoring path has to
    # pick the same blocks for an arm as the in-sample path, and matching them
    # by `is` breaks the moment two arms share an array.
    ALL_BLOCKS = [(Lg, Rg), (Ll, Rl)]
    arms = {
        "P1 (global) alone": ([0], False),
        "local alone": ([1], False),
        "P1 + local, feature-level": ([0, 1], False),
        "P1 + disagreement scalar": ([0], True),
    }
    oofs, res, oos_oofs = {}, {}, {}
    for name, (bidx, use_disagree) in arms.items():
        blocks = [ALL_BLOCKS[i] for i in bidx]
        extra = disagree if use_disagree else None
        if oos:
            ob, orel, ogr, oex = oos
            r = run_folds(blocks, X_rel, yk, gk, folds, extra, a.pca_dim,
                          oos=([ob[i] for i in bidx], orel, ogr,
                               oex if use_disagree else None))
            oofs[name], oos_oofs[name] = r
        else:
            oofs[name] = run_folds(blocks, X_rel, yk, gk, folds, extra, a.pca_dim)

    # score-level fusion, min-max normalised on each fold's training scores
    base, loc = oofs["P1 (global) alone"], oofs["local alone"]
    fin = loc[np.isfinite(loc)]
    lo, hi = (float(fin.min()), float(fin.max())) if len(fin) else (0.0, 1.0)
    oofs["P1 + local, score 0.5/0.5"] = np.where(
        np.isfinite(loc), 0.5 * base + 0.5 * np.clip((loc - lo) / max(hi - lo, 1e-6), 0, 1), base)

    m = np.isfinite(base)
    n_calls = int((base[m] >= float(np.nanmedian(base[yk == 1]))).sum())
    cut_base = matched_threshold(base, n_calls)
    watch = {"regrasp_reposition", "direction_reversal"}

    # FPresc is split out from resc because the two mean completely different
    # things for this branch. C3 exists to fix same-action false positives; a
    # run whose rescues are almost all recovered false NEGATIVES has improved
    # the ranking among positives and left the target errors alone, which "net
    # +2" on its own would hide.
    print(f"\n{'arm':<30} {'AUROC':>7} {'d':>7} {'95% CI':>18} "
          f"{'resc':>5} {'FPresc':>7} {'harm':>5} {'TPharm':>7} {'net':>5} {'sameFP':>7}")
    for name, oof in oofs.items():
        mm = np.isfinite(oof)
        au = _auroc(yk[mm], oof[mm]) if len(set(yk[mm].tolist())) == 2 else float("nan")
        au_b = _auroc(yk[mm], base[mm]) if len(set(yk[mm].tolist())) == 2 else float("nan")
        rh = rescues_harms(yk, base, oof, cut_base, matched_threshold(oof, n_calls))
        neg = [i for i, e in enumerate(ev) if yk[i] == 0 and np.isfinite(oof[i])]
        fp = sum(oof[i] >= matched_threshold(oof, n_calls) for i in neg)
        sfp = fp / len(neg) if neg else float("nan")
        if name == "P1 (global) alone":
            ci = ("", "")
        else:
            ci = grouped_bootstrap(yk, base, oof, gk, a.n_boot, a.seed)
        res[name] = {"auroc": au, "delta": au - au_b, "ci95": list(ci) if ci[0] != "" else None,
                     "per_fold": per_fold_auroc(yk, oof, gk, folds),
                     "same_action_fp_rate": sfp, **rh}
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] != "" else "-- baseline --"
        print(f"{name:<30} {au:>7.3f} {au - au_b:>+7.3f} {cis:>18} "
              f"{rh['rescues']:>5} {rh['fp_rescues']:>7} {rh['harms']:>5} "
              f"{rh['tp_harms']:>7} {rh['net']:>+5} {sfp:>7.3f}")

    base_sfp = res["P1 (global) alone"]["same_action_fp_rate"]
    pf_b = res["P1 (global) alone"]["per_fold"]
    print(f"\n{'arm':<30} {'per-fold delta':<40} {'worst':>7} {'improved':>9}")
    for name, r in res.items():
        if name == "P1 (global) alone":
            continue
        d = [c - b for c, b in zip(r["per_fold"], pf_b)
             if np.isfinite(c) and np.isfinite(b)]
        r["per_fold_delta"] = d
        r["worst_per_fold_delta"] = min(d) if d else float("nan")
        print(f"{name:<30} {str([round(x, 3) for x in d]):<40} "
              f"{(min(d) if d else float('nan')):>+7.3f} "
              f"{sum(1 for x in d if x > 0)}/{len(d)}")

    print("\n=== pre-registered gate (configs/local_gate_c3.json) ===")
    for name, r in res.items():
        if name == "P1 (global) alone":
            continue
        checks = {
            "net rescues-harms >= min": (r["net"], gate["min_rescues_minus_harms"],
                                  r["net"] >= gate["min_rescues_minus_harms"]),
            "TP harms <= max": (r["tp_harms"], gate["max_true_positive_harms"],
                                    r["tp_harms"] <= gate["max_true_positive_harms"]),
            "AUROC gain >= min": (r["delta"], gate["min_auroc_gain"],
                           r["delta"] >= gate["min_auroc_gain"]),
            "worst per-fold >= -max": (r["worst_per_fold_delta"], -gate["max_worst_per_fold_drop"],
                               r["worst_per_fold_delta"] >= -gate["max_worst_per_fold_drop"]),
            # BOTH readings are reported because they disagree and the
            # disagreement flips the verdict. The config's 0.074 is P1's
            # same-action FP rate ON THE CLEAN-145 (2/27); its own comment
            # states the INTENT as "must not exceed P1's current rate". On
            # batch3, P1 itself sits at 0.207 -- so the absolute threshold is
            # unsatisfiable there by ANY arm, P1 included, which cannot have
            # been the intent. The absolute check is kept exactly as
            # pre-registered rather than quietly rewritten; the relative one is
            # added beside it, and a run is only called a pass when it is
            # stated which reading is meant.
            "same-action FP (absolute, as pre-registered)":
                (r["same_action_fp_rate"], gate["max_same_action_fp_rate"],
                 r["same_action_fp_rate"] <= gate["max_same_action_fp_rate"]),
            "same-action FP (vs P1 on THIS data)":
                (r["same_action_fp_rate"], base_sfp,
                 r["same_action_fp_rate"] <= base_sfp + 1e-9),
            # Named as the REQUIREMENT, not as a conclusion. The failure list
            # prints these names verbatim, so "CI excludes 0" appearing after
            # "failed:" read as the opposite of what happened.
            "CI lower bound > 0": (r["ci95"][0] if r["ci95"] else float("nan"), 0.0,
                                   bool(r["ci95"]) and r["ci95"][0] > 0),
        }
        r["gate_checks"] = {k: {"value": v, "threshold": t, "pass": bool(p)}
                            for k, (v, t, p) in checks.items()}
        abs_key = "same-action FP (absolute, as pre-registered)"
        rel_key = "same-action FP (vs P1 on THIS data)"
        strict = all(p for k, (_, _, p) in checks.items() if k != rel_key)
        relative = all(p for k, (_, _, p) in checks.items() if k != abs_key)
        r["gate_passed"] = strict
        r["gate_passed_relative_sameaction"] = relative
        fails_s = [k for k, (_, _, p) in checks.items() if not p and k != rel_key]
        fails_r = [k for k, (_, _, p) in checks.items() if not p and k != abs_key]
        print(f"  {name:<30} strict {'ADOPT' if strict else 'NO'}"
              f" / relative-sameFP {'ADOPT' if relative else 'NO'}")
        if fails_s or fails_r:
            print(f"    {'failed(strict): ' + ', '.join(fails_s) if fails_s else ''}")
            if fails_r != fails_s:
                print(f"    {'failed(relative): ' + ', '.join(fails_r) if fails_r else 'failed(relative): none'}")

    if sub_map:
        n_tagged = sum(1 for i, e in enumerate(ev)
                       if yk[i] == 0 and sub_map.get(e["event_id"]) in watch)
        print(f"\n  note: {n_tagged} of {int((1 - yk).sum())} negatives carry a fine "
              f"subtype tag; batch3's negatives have only the coarse label, so the "
              f"same-action column above is coarse by design")
    return {"n_events": len(keep), "arms": res,
            "oofs": {k: v.tolist() for k, v in oofs.items()},
            "event_ids": [e["event_id"] for e in ev], "y": yk.tolist(),
            "recording_ids": gk,
            "oos_oofs": {k: v.tolist() for k, v in oos_oofs.items()},
            "oos_events": oos_events}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--same_action_subtype", default="data/gold/same_action_subtype_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--batch3_manifest",
                    help="evaluate on batch3's events instead of the clean-145. "
                         "batch3 is 101 DIFFERENT recordings, so this is the "
                         "replication check -- the stereo block probe scored "
                         "+0.054 on the clean-145 and +0.007 on batch3.")
    ap.add_argument("--batch3_pair_labels", default="data/gold/batch3_pair_labels_v1.csv")
    ap.add_argument("--gate_config", default="configs/local_gate_c3.json")
    ap.add_argument("--score_all_taxonomy", action="store_true",
                    help="ALSO score and dump events whose supervision is not "
                         "clean binary (soft_transition, exclude, ...). They "
                         "take no part in fitting -- they have no y -- but they "
                         "are scored by the fold model that did not see their "
                         "recording, so the decision layer can be told what it "
                         "does with camera/offscreen/gradual candidates. Without "
                         "this they are dropped by apply_to_events long before "
                         "the dump, and c3_selective_policy's taxonomy safety "
                         "table is structurally empty -- it can never show a "
                         "policy auto-accepting nuisance candidates.")
    ap.add_argument("--min_detect_frac", type=float, default=0.5,
                    help="an event is 'well detected' when at least this "
                         "fraction of frames in [t-4, t+4] were real detections")
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--dump_events")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    loc_rid = load_feature_caches(a.local_cache)
    if a.batch3_manifest:
        from src.boundary.batch3_dev_events import build_events as build_b3
        raw_events = build_b3(a.batch3_manifest, by_rid)
        labels = T.load_pair_labels(a.batch3_pair_labels)
        print(f"event source: batch3 manifest ({a.batch3_manifest})")
    else:
        raw_events = build_events(S.load_gold(a.gold), S.load_context(a.context), by_rid)
        labels = T.load_pair_labels(a.pair_labels)
        print("event source: clean-145")
    events = T.apply_to_events(raw_events, labels)
    if a.score_all_taxonomy:
        clean_ids = {e["event_id"] for e in events}
        extra = []
        for e in raw_events:
            if e["event_id"] in clean_ids:
                continue
            lab = labels.get(e["event_id"])
            if lab is None or not lab.get("pair_supervision"):
                continue
            extra.append(dict(e, y=None, pair_supervision=lab["pair_supervision"],
                              temporal_pair_subtype=lab.get("temporal_pair_subtype")))
        print(f"  --score_all_taxonomy: +{len(extra)} non-clean-binary events "
              f"{dict(Counter(e['pair_supervision'] for e in extra))} "
              f"-- scored but NEVER fitted on (they have no y)")
    else:
        extra = []
    print(f"clean events: {len(events)}  "
          f"subtypes={dict(Counter(e.get('temporal_pair_subtype') for e in events))}")

    missing = [e for e in events if e["recording_id"] not in loc_rid]
    if missing:
        print(f"  !! {len(missing)} events have NO local features "
              f"({len(set(m['recording_id'] for m in missing))} recordings missing "
              f"from --local_cache) and are excluded from every arm, baseline "
              f"included, so the comparison stays like-for-like")
    events = [e for e in events if e["recording_id"] in loc_rid]
    if not events:
        raise SystemExit("no events have local features")

    extra = [e for e in extra if e["recording_id"] in loc_rid]
    events_l = [dict(e, rec=loc_rid[e["recording_id"]]) for e in events]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = [e["recording_id"] for e in events]
    cov = np.array([detect_coverage(loc_rid[e["recording_id"]], e["t"]) for e in events])
    print(f"per-event detection coverage over [t-{WIDEST:.0f}, t+{WIDEST:.0f}]: "
          f"median {np.median(cov):.3f}  "
          f">={a.min_detect_frac}: {int((cov >= a.min_detect_frac).sum())}/{len(cov)}")

    sub_map = {}
    if os.path.exists(a.same_action_subtype):
        with open(a.same_action_subtype, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sub_map[r["event_id"]] = r["subtype"]
    gate = json.load(open(a.gate_config, encoding="utf-8"))

    report = {"min_detect_frac": a.min_detect_frac,
              "coverage_median": float(np.median(cov))}
    extra_l = [dict(e, rec=loc_rid[e["recording_id"]]) for e in extra]
    report["all_events"] = evaluate("ALL EVENTS WITH LOCAL FEATURES",
                                    events, events_l, y, groups, sub_map, gate, a,
                                    oos_g=extra or None, oos_l=extra_l or None)

    sel = cov >= a.min_detect_frac
    if sel.sum() >= 20 and sel.sum() < len(cov):
        report["well_detected"] = evaluate(
            f"WELL-DETECTED SUBSET (coverage >= {a.min_detect_frac})",
            [events[i] for i in np.nonzero(sel)[0]],
            [events_l[i] for i in np.nonzero(sel)[0]],
            y[sel], [groups[i] for i in np.nonzero(sel)[0]], sub_map, gate, a)
        print("\n  ^ the baseline in this table is P1 REFIT on exactly these "
              "events. Comparing a local arm here against P1's 0.872 on the "
              "full 145 would be a manufactured win, since this subset is "
              "the one where hands were visible and is easier by construction.")
    elif sel.sum() == len(cov):
        print(f"\nevery event clears coverage {a.min_detect_frac}; no subset table needed")
    else:
        print(f"\nonly {int(sel.sum())} events clear coverage {a.min_detect_frac} -- "
              f"too few for a separate table")

    if a.dump_events and report.get("all_events"):
        r = report["all_events"]
        with open(os.path.expanduser(a.dump_events), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            names = list(r["oofs"])
            # subtype is required by c3_selective_policy: without it every
            # event is treated as clean-binary and the taxonomy table -- the
            # one that shows whether camera/offscreen candidates are being
            # auto-accepted -- comes out empty.
            w.writerow(["event_id", "recording_id", "y", "subtype",
                        "detect_coverage", "detect_longest_gap_s"] + names)
            covmap = {e["event_id"]: c for e, c in zip(events, cov)}
            submap = {e["event_id"]: (e.get("temporal_pair_subtype") or "")
                      for e in events}
            gapmap = {e["event_id"]: detect_longest_gap_s(loc_rid[e["recording_id"]], e["t"])
                      for e in events}
            for i, eid in enumerate(r["event_ids"]):
                w.writerow([eid, r["recording_ids"][i], int(r["y"][i]),
                            submap.get(eid, ""),
                            f"{covmap.get(eid, float('nan')):.3f}",
                            f"{gapmap.get(eid, float('nan')):.2f}"]
                           + [f"{r['oofs'][n][i]:.6f}" if np.isfinite(r["oofs"][n][i]) else ""
                              for n in names])
            # non-clean-binary rows: y is left EMPTY rather than filled with 0.
            # They have no boundary label; writing 0 would silently enter them
            # into every precision denominator as negatives.
            for i, e in enumerate(r.get("oos_events", [])):
                cv = detect_coverage(loc_rid[e["recording_id"]], e["t"])
                w.writerow([e["event_id"], e["recording_id"], "",
                            e.get("temporal_pair_subtype") or "", f"{cv:.3f}",
                            f"{detect_longest_gap_s(loc_rid[e['recording_id']], e['t']):.2f}"]
                           + [f"{r['oos_oofs'][n][i]:.6f}"
                              if n in r["oos_oofs"] and np.isfinite(r["oos_oofs"][n][i])
                              else "" for n in names])
        print(f"\nwrote {a.dump_events}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        for k in ("all_events", "well_detected"):
            if report.get(k):
                report[k].pop("oofs", None)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
