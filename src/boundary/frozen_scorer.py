"""Fit the scorers once, save everything, and apply them without fitting.

Until now nothing in this repository persisted a fitted model. Every score was
out-of-fold from a fit performed inside the evaluation, and c3_local_eval fits
on either the original pool or batch3, never both. Two consequences, both
blocking:

  BATCH4 COULD NOT BE A HELD-OUT TEST. Any score it received would come from a
  model fitted with batch4's own labels in its training folds. That measures
  whether the method works on batch4, not whether this artefact transfers to
  it, and only the second is what a reserved set is for. Freezing the policy
  config freezes thresholds and nothing that produces the numbers they apply
  to.

  THE TWO DEVELOPMENT SUB-POPULATIONS WERE NOT COMPARABLE. Their scores came
  from separate fits, so the gap between them (median 0.734 against 0.315)
  mixes prevalence, conditional shift and score scale, and no amount of
  arithmetic on those numbers separates the three. One scorer applied to both
  is the only thing that does.

WHAT IS SAVED. Not just weights: the PCA basis, the imputer and scaler
statistics for every stage, the logistic weights and intercept for each named
score, the reliability definition, the identity of every feature cache (path,
size, mtime and optionally sha256) and the commit that produced them. A
threshold applied to scores from a different feature build is not the frozen
artefact, and without the cache identities nothing detects that.

IN-SAMPLE SCORES ARE REFUSED BY DEFAULT. The frozen model is fitted on ALL
development events, so its scores on those events are in-sample and are not
development performance. The out-of-fold path in c3_local_eval remains the
only source of that. `--apply` therefore errors when asked to score an event
that was in the fit set, unless --allow_in_sample is passed, and it stamps
`in_sample` on every such row so a file cannot be mistaken for a held-out one
after the fact.

Usage:
    # once, before any held-out data is touched
    python -m src.boundary.frozen_scorer --fit \
        --batch3_manifest .../batch3_manifest.jsonl \
        --batch3_pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --feat_cache ... --local_cache ... \
        --out /workspace/tr1/results/hal/c3/frozen_scorer_v1.pt

    # later, on held-out events
    python -m src.boundary.frozen_scorer --apply \
        --model /workspace/tr1/results/hal/c3/frozen_scorer_v1.pt \
        --batch3_manifest .../batch4_manifest.jsonl \
        --batch3_pair_labels .../batch4_pair_labels.csv \
        --feat_cache ... --local_cache ... \
        --dump_events /workspace/tr1/results/hal/c3/scored_batch4.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter

import numpy as np
import torch

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events
from src.boundary.pairwise_verifier import (
    build_matrices, pca_fit, pca_apply, pair_block,
    _impute_scale_fit, _impute_scale_apply,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.c3_local_eval import detect_coverage, detect_longest_gap_s

# the names the policy config references; they must not drift
P1_NAME = "P1 (global) alone"
LOCAL_NAME = "local alone"


def file_identity(path, sha=True):
    st = os.stat(path)
    out = {"path": os.path.abspath(path), "size": st.st_size,
           "mtime": int(st.st_mtime)}
    if sha:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def fit_one(blocks, X_rel, y, pca_dim=64, l2=5.0):
    """One scorer, fitted on everything passed in. Mirrors run_folds' per-fold
    body exactly -- same PCA, same two imputer/scaler stages, same L2 -- so a
    frozen score and an out-of-fold score differ only in what they were fitted
    on and never in how they are computed."""
    pcas = [pca_fit(np.concatenate([L, R], 0), pca_dim) for L, R in blocks]
    st_rel = _impute_scale_fit(X_rel)
    parts = [pair_block(pca_apply(p, L), pca_apply(p, R))
             for p, (L, R) in zip(pcas, blocks)]
    parts.append(_impute_scale_apply(st_rel, X_rel))
    P = np.concatenate(parts, 1)
    stP = _impute_scale_fit(P)
    w, b = fit_logreg(_impute_scale_apply(stP, P), y, l2=l2)
    return {"pcas": pcas, "st_rel": st_rel, "stP": stP, "w": w, "b": float(b),
            "pca_dim": pca_dim, "l2": l2, "n_blocks": len(blocks)}


def apply_one(m, blocks, X_rel):
    if len(blocks) != m["n_blocks"]:
        raise SystemExit(f"this scorer was fitted on {m['n_blocks']} block(s) "
                         f"and was given {len(blocks)} -- the arms do not match")
    parts = [pair_block(pca_apply(p, L), pca_apply(p, R))
             for p, (L, R) in zip(m["pcas"], blocks)]
    parts.append(_impute_scale_apply(m["st_rel"], X_rel))
    P = np.concatenate(parts, 1)
    return _sigmoid(_impute_scale_apply(m["stP"], P) @ m["w"] + m["b"])


def gather(a, by_rid, loc_rid):
    """Every event with both global and local features, plus the non-clean
    taxonomy rows, which are SCORED but carry no y."""
    events, extra = [], []
    if a.batch3_manifest:
        from src.boundary.batch3_dev_events import build_events as build_b3
        raw = build_b3(a.batch3_manifest, by_rid)
        labels = T.load_pair_labels(a.batch3_pair_labels)
    else:
        raw = build_events(S.load_gold(a.gold), S.load_context(a.context), by_rid)
        labels = T.load_pair_labels(a.pair_labels)
    clean = T.apply_to_events(raw, labels)
    ids = {e["event_id"] for e in clean}
    for e in raw:
        if e["event_id"] in ids:
            continue
        lab = labels.get(e["event_id"])
        if lab and lab.get("pair_supervision"):
            extra.append(dict(e, y=None,
                              pair_supervision=lab["pair_supervision"],
                              temporal_pair_subtype=lab.get("temporal_pair_subtype")))
    events = [e for e in clean if e["recording_id"] in loc_rid]
    extra = [e for e in extra if e["recording_id"] in loc_rid]
    return events, extra


def matrices(events, loc_rid):
    """Aligned global and local matrices, dropping events poolable in only one
    branch -- the local cache has its own time grid, and an event present in
    one and absent in the other would silently shift the row alignment."""
    ev_l = [dict(e, rec=loc_rid[e["recording_id"]]) for e in events]
    _, Lg, Rg, X_rel, keep_g, _ = build_matrices(events, False)
    _, Ll, Rl, _, keep_l, _ = build_matrices(ev_l, False)
    keep = sorted(set(keep_g) & set(keep_l))
    gi = {k: i for i, k in enumerate(keep_g)}
    li = {k: i for i, k in enumerate(keep_l)}
    g = np.array([gi[k] for k in keep], int)
    l = np.array([li[k] for k in keep], int)
    return ([events[k] for k in keep], Lg[g], Rg[g], Ll[l], Rl[l], X_rel[g])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--model", help="frozen scorer, for --apply")
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--clean145", action="store_true",
                    help="include the original gold+context pool. OPT-IN and "
                         "not implied by anything: the first version added it "
                         "only when no manifest was given, so passing a "
                         "manifest silently dropped it and the 'one fit over "
                         "both populations' this file exists for became a fit "
                         "over one")
    ap.add_argument("--batch3_manifest", action="append", default=[])
    ap.add_argument("--batch3_pair_labels", action="append", default=[])
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--no_sha", action="store_true",
                    help="skip cache hashing; faster, and the artefact can no "
                         "longer prove which features produced it")
    ap.add_argument("--allow_in_sample", action="store_true")
    ap.add_argument("--selfcheck", action="store_true",
                    help="does the frozen PATH reproduce the out-of-fold score "
                         "distribution the thresholds were chosen on?")
    ap.add_argument("--dump_events")
    ap.add_argument("--out")
    a = ap.parse_args()
    # three modes, not two. The first version tested `a.fit == a.apply`, which
    # is True when neither is given -- so --selfcheck, added later, was
    # rejected by a guard that predated it.
    modes = [m for m in ("fit", "apply", "selfcheck") if getattr(a, m)]
    if len(modes) != 1:
        raise SystemExit(f"give exactly one of --fit / --apply / --selfcheck "
                         f"(got {modes or 'none'})")
    if len(a.batch3_manifest) != len(a.batch3_pair_labels):
        raise SystemExit("--batch3_manifest and --batch3_pair_labels must pair up")

    by_rid = load_feature_caches(a.feat_cache)
    loc_rid = load_feature_caches(a.local_cache)

    sources = ([(None, None)] if a.clean145 else []) \
        + list(zip(a.batch3_manifest, a.batch3_pair_labels))
    if not sources:
        raise SystemExit("no event source: pass --clean145 and/or "
                         "--batch3_manifest/--batch3_pair_labels")
    print(f"event sources ({len(sources)}): "
          + ", ".join("clean-145" if m is None else os.path.basename(m)
                      for m, _ in sources))

    all_ev, all_extra = [], []
    for man, pl in sources:
        sub = argparse.Namespace(**vars(a))
        sub.batch3_manifest, sub.batch3_pair_labels = man, pl
        ev, ex = gather(sub, by_rid, loc_rid)
        tag = "clean-145" if man is None else os.path.basename(man)
        for e in ev:
            e["_source"] = tag
        for e in ex:
            e["_source"] = tag
        print(f"  {'clean-145' if man is None else os.path.basename(man)}: "
              f"{len(ev)} clean + {len(ex)} non-clean")
        all_ev += ev
        all_extra += ex
    if not all_ev:
        raise SystemExit("no events with both global and local features")

    ev, Lg, Rg, Ll, Rl, X_rel = matrices(all_ev, loc_rid)
    y = np.array([e["y"] for e in ev], float)
    print(f"\n{len(ev)} events poolable in BOTH branches "
          f"({int(y.sum())}+ / {int((1 - y).sum())}-), "
          f"{len({e['recording_id'] for e in ev})} recordings, "
          f"base rate {y.mean():.3f}")

    if a.selfcheck:
        # THE QUESTION THIS ANSWERS. pair_block turns PCA-64 into 4*64 = 256
        # columns, plus 7 scalars: 263 features for ~300 samples. In-sample the
        # fit separates the classes perfectly (PPV 1.000, FPR 0.000), which is
        # what that much capacity does. The thresholds in the frozen policy
        # were chosen on OUT-OF-FOLD scores, so they are only meaningful on
        # batch4 if the frozen path -- fit on training recordings, applied
        # cold to held-out ones -- produces the same kind of distribution the
        # OOF path did. Nothing about batch4 is needed to check that.
        from src.boundary.pairwise_verifier import stratified_grouped_folds
        from src.boundary.state_adapter import _auroc
        groups = [e["recording_id"] for e in ev]
        folds = stratified_grouped_folds(groups, y, 5, seed=0)
        frozen_oof = {}
        for nm, blocks in ((P1_NAME, (Lg, Rg)), (LOCAL_NAME, (Ll, Rl))):
            L, R = blocks
            ins = apply_one(fit_one([(L, R)], X_rel, y, a.pca_dim),
                            [(L, R)], X_rel)
            oof = np.full(len(y), np.nan)
            for f in folds:
                te = np.array([g in f for g in groups])
                tr = ~te
                if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
                    continue
                m = fit_one([(L[tr], R[tr])], X_rel[tr], y[tr], a.pca_dim)
                oof[te] = apply_one(m, [(L[te], R[te])], X_rel[te])
            frozen_oof[nm] = oof
            k = np.isfinite(oof)
            print(f"\n  {nm}")
            print(f"    in-sample        AUROC {_auroc(y, ins):.3f}  "
                  f"median {np.median(ins):.3f}  >=0.75 {np.mean(ins >= 0.75):.3f}  "
                  f"in (0.2,0.8) {np.mean((ins > 0.2) & (ins < 0.8)):.3f}")
            print(f"    frozen path, OOF AUROC {_auroc(y[k], oof[k]):.3f}  "
                  f"median {np.median(oof[k]):.3f}  >=0.75 {np.mean(oof[k] >= 0.75):.3f}  "
                  f"in (0.2,0.8) {np.mean((oof[k] > 0.2) & (oof[k] < 0.8)):.3f}")
        # PER SOURCE, on the frozen path's out-of-fold scores. This is the
        # comparison that was impossible before: the two pools were previously
        # scored by two separate fits, so their medians (0.734 against 0.315)
        # confounded prevalence, conditional shift and score scale. One fit
        # removes the third, and whatever gap survives here is the other two.
        src = [e.get("_source", "?") for e in ev]
        if len(set(src)) > 1:
            print(f"\n  PER SOURCE at the policy's 0.75, frozen path OOF "
                  f"(scale confound removed)")
            print(f"    {'source':<24} {'n':>4} {'pi':>6} {'median':>7} "
                  f"{'TPR':>6} {'FPR':>6} {'PPV':>6} {'cov':>6}")
            for t in sorted(set(src)):
                sel = np.array([x == t for x in src]) & np.isfinite(oof)
                if sel.sum() < 10:
                    continue
                yy, ss = y[sel], oof[sel]
                k = ss >= 0.75
                tp = int(yy[k].sum()); fp = int(k.sum()) - tp
                P, N = int(yy.sum()), int((1 - yy).sum())
                print(f"    {t:<24} {int(sel.sum()):>4} {yy.mean():>6.3f} "
                      f"{np.median(ss):>7.3f} {tp / max(P, 1):>6.3f} "
                      f"{fp / max(N, 1):>6.3f} "
                      f"{tp / max(int(k.sum()), 1):>6.3f} "
                      f"{k.mean():>6.3f}")
            print("    A gap that survives here is prevalence plus conditional "
                  "shift only. Compare it against the separate-fit numbers "
                  "(median 0.734 vs 0.315):\n    whatever closed is score "
                  "scale, and that part was never a property of the data.")
        if a.dump_events:
            # THE THRESHOLDS HAVE TO BE RE-SELECTED ON THESE. The frozen policy
            # config was selected on out-of-fold scores from two SEPARATE fits,
            # whose scale differs from a single combined fit's -- so those
            # thresholds are not applicable to any cold-scoring path built on
            # this model, whatever form that path takes. These are single-fold
            # models from one combined fit, produced by the same code that will
            # score batch4, which makes them the right scores to choose
            # thresholds on.
            with open(a.dump_events, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                names = [P1_NAME, LOCAL_NAME]
                w.writerow(["event_id", "recording_id", "y", "subtype",
                            "detect_coverage", "detect_longest_gap_s",
                            "source"] + names)
                for i, e in enumerate(ev):
                    if not all(np.isfinite(frozen_oof[n][i]) for n in names):
                        continue
                    w.writerow([e["event_id"], e["recording_id"], int(y[i]),
                                e.get("temporal_pair_subtype") or "",
                                f"{detect_coverage(loc_rid[e['recording_id']], e['t']):.3f}",
                                f"{detect_longest_gap_s(loc_rid[e['recording_id']], e['t']):.2f}",
                                e.get("_source", "")]
                               + [f"{frozen_oof[n][i]:.6f}" for n in names])
            print(f"\n  wrote {a.dump_events}")
            print("  These are out-of-fold scores from ONE combined fit. The "
                  "existing frozen policy config was selected on out-of-fold "
                  "scores from TWO\n  separate fits, at a different scale, so "
                  "it does not carry over -- re-select on this file and freeze "
                  "the result before batch4.")
            print("  NOTE the residual: an out-of-fold score comes from a "
                  "model fitted on 4/5 of the data, and batch4 will be scored "
                  "by one fitted on 5/5,\n  which is slightly sharper. No "
                  "cold-scoring path reproduces the out-of-fold statistic "
                  "exactly -- a 5-model ensemble is sharper still. This is the "
                  "closest\n  available match and the direction of the "
                  "remaining mismatch is known.")
        print("\n  The SECOND row of each pair is what batch4 will look like: "
              "a model fitted without those recordings, applied cold. If its "
              "median and its\n  fraction above 0.75 are close to the numbers "
              "the policy thresholds were selected on, the thresholds carry "
              "over. If the in-sample row is\n  far more extreme -- it will be "
              "-- that is capacity, not evidence, and it is why the frozen "
              "scorer's own development scores are refused by default.")
        return

    if a.fit:
        model = {
            "scorers": {
                P1_NAME: fit_one([(Lg, Rg)], X_rel, y, a.pca_dim),
                LOCAL_NAME: fit_one([(Ll, Rl)], X_rel, y, a.pca_dim),
            },
            "fit_event_ids": [e["event_id"] for e in ev],
            "fit_recordings": sorted({e["recording_id"] for e in ev}),
            "fit_base_rate": float(y.mean()),
            "reliability": {"column": "detect_coverage",
                            "definition": "fraction of frames in [t-4, t+4] "
                                          "that were a real hand detection "
                                          "rather than an interpolated box"},
            "feat_cache": [file_identity(p, not a.no_sha) for p in a.feat_cache],
            "local_cache": [file_identity(p, not a.no_sha) for p in a.local_cache],
            "pair_labels": [file_identity(p, not a.no_sha)
                            for p in [a.pair_labels] + list(a.batch3_pair_labels)
                            if os.path.exists(p)],
            "pca_dim": a.pca_dim,
            "commit": git_commit(),
        }
        if len(sources) == 1:
            print("\n  !! FITTED ON ONE SOURCE. A scorer fitted on a single "
                  "sub-population inherits its prevalence and its score scale, "
                  "which is exactly the\n     confound this file exists to "
                  "remove. Pass every development source, or accept that this "
                  "artefact is not comparable across them.")
        if not a.out:
            raise SystemExit("--out is required for --fit")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        torch.save(model, a.out)
        print(f"\nwrote {a.out}")
        print(f"  commit {model['commit']}")
        for k in ("feat_cache", "local_cache"):
            for f in model[k]:
                print(f"  {k:<12} {os.path.basename(f['path'])}  "
                      f"{f['size'] / 1e6:.0f}MB  "
                      f"{f.get('sha256', '(unhashed)')[:16]}")
        print("\nThese scores are IN-SAMPLE on the events just fitted. They are "
              "not development performance -- the out-of-fold path in "
              "c3_local_eval remains the only\nsource of that. Commit this "
              "file's identity before any held-out data is labelled.")
        return

    # ------------------------------------------------------------- apply
    model = torch.load(a.model, weights_only=False, map_location="cpu")
    print(f"\nfrozen scorer from commit {model.get('commit')}, fitted on "
          f"{len(model['fit_event_ids'])} events across "
          f"{len(model['fit_recordings'])} recordings, base rate "
          f"{model['fit_base_rate']:.3f}")
    for k in ("feat_cache", "local_cache"):
        now = {os.path.abspath(p): file_identity(p, not a.no_sha)
               for p in (a.feat_cache if k == "feat_cache" else a.local_cache)}
        for f in model[k]:
            cur = now.get(f["path"])
            if cur is None:
                print(f"  !! {k} {os.path.basename(f['path'])} was used at fit "
                      f"time and is NOT among the caches given now")
            elif "sha256" in f and "sha256" in cur and f["sha256"] != cur["sha256"]:
                raise SystemExit(
                    f"{f['path']} has changed since the fit "
                    f"({f['sha256'][:12]} -> {cur['sha256'][:12]}). Scores from "
                    f"different features are not this frozen artefact, and a "
                    f"threshold carried across that change means nothing.")

    fit_ids = set(model["fit_event_ids"])
    fit_recs = set(model["fit_recordings"])
    inside = [e["event_id"] for e in ev if e["event_id"] in fit_ids]
    rec_overlap = sorted({e["recording_id"] for e in ev} & fit_recs)
    if inside and not a.allow_in_sample:
        raise SystemExit(
            f"{len(inside)} of {len(ev)} events were in the fit set, so their "
            f"scores would be in-sample and are not held-out performance. Pass "
            f"--allow_in_sample to score them anyway; every such row is stamped "
            f"in_sample=1. e.g. {inside[:3]}")
    if rec_overlap:
        print(f"  !! {len(rec_overlap)} recording(s) appear in BOTH the fit set "
              f"and this data. Even events that are individually new share a "
              f"recording with training data, so this is not a clean "
              f"recording-level held-out set: {rec_overlap[:4]}")

    scores = {P1_NAME: apply_one(model["scorers"][P1_NAME], [(Lg, Rg)], X_rel),
              LOCAL_NAME: apply_one(model["scorers"][LOCAL_NAME], [(Ll, Rl)], X_rel)}
    for n, s in scores.items():
        print(f"  {n:<22} median {np.median(s):.3f}  "
              f">=0.75 {np.mean(s >= 0.75):.3f}")

    # non-clean taxonomy rows: scored, y left EMPTY. Writing 0 would enter them
    # into every precision denominator as negatives.
    ex_rows = []
    if all_extra:
        ex, eLg, eRg, eLl, eRl, eX = matrices(all_extra, loc_rid)
        es = {P1_NAME: apply_one(model["scorers"][P1_NAME], [(eLg, eRg)], eX),
              LOCAL_NAME: apply_one(model["scorers"][LOCAL_NAME], [(eLl, eRl)], eX)}
        ex_rows = list(zip(ex, es[P1_NAME], es[LOCAL_NAME]))
        print(f"  scored {len(ex_rows)} non-clean-binary events (no y)")

    if a.dump_events:
        names = [P1_NAME, LOCAL_NAME]
        with open(a.dump_events, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "y", "subtype",
                        "detect_coverage", "detect_longest_gap_s",
                        "in_sample"] + names)
            for i, e in enumerate(ev):
                w.writerow([e["event_id"], e["recording_id"], int(y[i]),
                            e.get("temporal_pair_subtype") or "",
                            f"{detect_coverage(loc_rid[e['recording_id']], e['t']):.3f}",
                            f"{detect_longest_gap_s(loc_rid[e['recording_id']], e['t']):.2f}",
                            int(e["event_id"] in fit_ids)]
                           + [f"{scores[n][i]:.6f}" for n in names])
            for e, s1, s2 in ex_rows:
                w.writerow([e["event_id"], e["recording_id"], "",
                            e.get("temporal_pair_subtype") or "",
                            f"{detect_coverage(loc_rid[e['recording_id']], e['t']):.3f}",
                            f"{detect_longest_gap_s(loc_rid[e['recording_id']], e['t']):.2f}",
                            int(e["event_id"] in fit_ids), f"{s1:.6f}", f"{s2:.6f}"])
        print(f"\nwrote {a.dump_events}")
        print("  `in_sample` is per row. A file with any 1 in that column is "
              "not a held-out result, whatever the aggregate says.")


if __name__ == "__main__":
    main()
