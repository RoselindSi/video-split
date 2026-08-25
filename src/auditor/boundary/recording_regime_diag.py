"""What separates the recordings where morphology works from the rest.

Phase 3. The per-recording pair accuracies are heterogeneous well beyond
sampling noise (Cochran Q p ~ 1e-9, I^2 ~ 69% for the morphology term), and
that heterogeneity is the finding -- .870 in recording_000179 against .212 in
recording_000019. This asks what those recordings differ in, BEFORE anything
new is trained, because a difference in how the window was assembled would be
inherited by every ranking loss built on top of it.

THE HYPOTHESIS WITH A PAPER TRAIL. `feature_loader.window` marks a grid
position invalid when the nearest cached frame is further than
max(grid_step, cache_spacing)/2. On a 0.5s grid over a 2 fps local cache those
two are equal, so a candidate whose phase sits a quarter second off the cache
grid misses EVERY grid point and loses that entire stream -- and the event is
not dropped, it keeps its mask and trains on the other stream alone. The
loader's own docstring says the training events clear this by 0.05s and calls
that luck rather than margin. Those were the 415 morphology events. The 3707
evaluation candidates have never been checked, and a recording whose
candidates are systematically off-phase would be a recording where morphology
silently ran on half its evidence.

So `coverage_l`, `frac_no_local` and the realised phase offset are computed
here per candidate and aggregated per recording, next to the ordinary
suspects -- densities, feature norms, head entropy.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

It does not test a tri-grouped GOOD / NEUTRAL / INVERTED split as the primary
analysis. Grouping on a noisy outcome and then comparing groups invites
regression to the mean: the recordings that look best are partly the ones
whose noise ran up. The primary analysis correlates the CONTINUOUS
per-recording accuracy against each variable; the grouping is printed after,
as a picture.

And it does not report a bare p per variable. Fifteen variables over 34
recordings will hand you two or three at p < .05 with nothing behind them, so
every p is a permutation p and every column carries a Benjamini-Hochberg q.
Read the q. This stage generates hypotheses; it does not confirm them.

Usage:
    python -m src.auditor.boundary.recording_regime_diag \
        --candidates results/auditor/oof_candidates.jsonl \
        --morphology results/auditor/morphology_external_oof.jsonl \
        --oracle_audit results/boundary/oof/audit/predictions.jsonl \
        --feat_cache CACHE.pt [--feat_cache ...] \
        --local_cache LOCAL.pt [--local_cache ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src.auditor.boundary.ontology_fusion import (ontology_energy,
                                                  morphology_logratio)
from src.auditor.boundary.recording_shortcut_diag import pair_counts

PERM = 20000
SEED = 0
GOOD, INVERTED = 0.65, 0.40


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def _perm_p(x, y, perm=PERM, seed=SEED):
    """Permutation p for Spearman. n = 34 is small enough that the asymptotic
    formula is a guess, and cheap enough that it does not have to be."""
    rho = _spearman(x, y)
    if not np.isfinite(rho):
        return rho, float("nan")
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    cnt = sum(abs(_spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12
              for _ in range(perm))
    return rho, (cnt + 1) / (perm + 1)


def bh(pvals):
    """Benjamini-Hochberg q. Fifteen variables will give you three at p<.05."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * m / (m - rank + 1))
        q[i] = prev
    return q


# --- the window, replayed exactly as the model saw it ---------------------
def coverage_and_phase(rec_cache, t0, half_s=6.0, n_frames=25):
    """Re-run feature_loader.window's validity rule and also report WHY.

    Returns (coverage, mean nearest-frame distance over in-range grid points,
    cache spacing). The distance is the quantity the tolerance is compared
    against, so a recording sitting just under the threshold and one sitting
    just over it are visibly different here rather than only in the outcome."""
    if rec_cache is None:
        return 0.0, float("nan"), float("nan")
    times = rec_cache["times"]
    if hasattr(times, "detach"):
        times = times.detach().cpu().numpy()
    times = np.asarray(times, float)
    if len(times) == 0:
        return 0.0, float("nan"), float("nan")
    grid = np.linspace(-half_s, half_s, n_frames)
    step = grid[1] - grid[0] if n_frames > 1 else half_s
    spacing = float(np.median(np.diff(np.sort(times)))) if len(times) > 1 \
        else step
    tol = max(step, spacing) / 2 * (1 + 1e-9)
    lo, hi = times.min(), times.max()
    t = t0 + grid
    inr = (t >= lo) & (t <= hi)
    if not inr.any():
        return 0.0, float("nan"), spacing
    j = np.abs(times[None, :] - t[inr, None]).argmin(1)
    d = np.abs(times[j] - t[inr])
    return float((d <= tol).sum() / n_frames), float(d.mean()), spacing


def cache_stats(rec_cache, step):
    """The gaps in a recording's cache, and the tolerance they buy it.

    THE TOLERANCE IS ADAPTIVE, AND THAT IS THE PROBLEM. tol is
    max(grid_step, median_spacing)/2, so a recording that lost frames gets a
    median spacing of 1.0s instead of 0.5s and therefore a tolerance of 0.5s
    instead of 0.25s. Its coverage then looks FINE -- better than a recording
    that lost fewer frames -- while its window is assembled from frames up to
    half a second away from the instant they are standing in for.

    Coverage cannot see this. A smeared window still fills every grid slot.
    But morphology is precisely the question of where inside the window the
    change sits and how wide it is, so a half-second mismatch attacks the
    morphology signal while leaving a whole-window change score largely
    intact -- which is the shape of the split actually observed."""
    if rec_cache is None:
        return {}
    t = rec_cache["times"]
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    t = np.sort(np.asarray(t, float))
    if len(t) < 2:
        return {}
    d = np.diff(t)
    med = float(np.median(d))
    span = float(t[-1] - t[0])
    return {"spacing_med": med,
            "spacing_p90": float(np.quantile(d, 0.90)),
            "spacing_max": float(d.max()),
            "gap_ratio_p90": float(np.quantile(d, 0.90) / med) if med else
            float("nan"),
            "tol_used": max(step, med) / 2,
            # frames actually cached against what an unbroken stream at the
            # FINEST observed spacing would have held. Blur and black-frame
            # filtering remove frames without changing anything else, so this
            # is where that shows up.
            "frame_retention": float(len(t) / (span / max(np.quantile(d, 0.05),
                                                          1e-6) + 1)),
            "frames_per_s": float(len(t) / span) if span else float("nan")}


def recording_variables(cands, morph, gcache, lcache, half_s, n_frames):
    per = defaultdict(lambda: defaultdict(list))
    for c in cands:
        r, t0 = c["recording_id"], c["candidate_time"]
        cg, dg, sg = coverage_and_phase(gcache.get(r), t0, half_s, n_frames)
        cl, dl, sl = coverage_and_phase(lcache.get(r), t0, half_s, n_frames)
        m = morph[c["candidate_id"]]
        pp = float(m["p_point"])
        pn = float(m["p_no_transition"])
        s = pp + pn
        ent = float("nan")
        if s > 0:
            a, b = pp / s, pn / s
            ent = -(a * np.log(a + 1e-12) + b * np.log(b + 1e-12))
        per[r]["coverage_g"].append(cg)
        per[r]["coverage_l"].append(cl)
        per[r]["no_local"].append(1.0 if cl == 0 else 0.0)
        per[r]["no_global"].append(1.0 if cg == 0 else 0.0)
        per[r]["phase_offset_g"].append(dg)
        per[r]["phase_offset_l"].append(dl)
        per[r]["cache_spacing_g"].append(sg)
        per[r]["cache_spacing_l"].append(sl)
        per[r]["detector_score"].append(float(c["detector_score"]))
        per[r]["true_density"].append(float(bool(c["is_true_boundary"])))
        per[r]["p_point"].append(pp)
        per[r]["p_no_transition"].append(pn)
        per[r]["morph_entropy"].append(ent)
        per[r]["_t"].append(float(t0))

    out = {}
    for r, d in per.items():
        row = {k: float(np.nanmean(v)) for k, v in d.items()
               if not k.startswith("_")}
        t = np.array(d["_t"])
        row["n_candidates"] = float(len(t))
        row["span_s"] = float(t.max() - t.min()) if len(t) > 1 else 0.0
        row["cand_per_min"] = (len(t) / (row["span_s"] / 60)
                               if row["span_s"] else float("nan"))
        step = 2 * half_s / (n_frames - 1) if n_frames > 1 else half_s
        for tag, cache in (("g", gcache), ("l", lcache)):
            rc = cache.get(r)
            if rc is None:
                continue
            for k, v in cache_stats(rc, step).items():
                row[f"{k}_{tag}"] = v
            f = rc["feats"]
            if hasattr(f, "detach"):
                f = f.detach().cpu().numpy()
            f = np.asarray(f, np.float32)
            row[f"feat_norm_{tag}"] = float(np.linalg.norm(f, axis=1).mean())
            row[f"n_frames_cached_{tag}"] = float(len(f))
        out[r] = row
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--oracle_audit", required=True)
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="global stream cache(s); APPEND, one flag each")
    ap.add_argument("--local_cache", action="append", required=True,
                    help="local stream cache(s); APPEND, one flag each")
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.ontology_constitution import Constitution
    from src.auditor.common.feature_loader import load_caches
    C = Constitution()
    C.check_oracle_use("headroom")

    cands = [json.loads(l) for l in open(a.candidates, encoding="utf-8")
             if l.strip()]
    C.check_candidate_pool(cands, a.candidates)
    morph = {}
    for l in open(a.morphology, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            morph[r["candidate_id"]] = r
    if any(c["candidate_id"] not in morph for c in cands):
        raise SystemExit("morphology does not cover the frozen pool")

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
    ok = np.array([c["is_true_boundary"] for c in cands], bool)
    rec = [c["recording_id"] for c in cands]
    det = np.array([c["detector_score"] for c in cands], float)
    pp = np.array([morph[c["candidate_id"]]["p_point"] for c in cands])
    pn = np.array([morph[c["candidate_id"]]["p_no_transition"] for c in cands])
    logit = np.log(det.clip(1e-4, 1 - 1e-4) / (1 - det.clip(1e-4, 1 - 1e-4)))
    cols = [("detector", logit), ("morph only", morphology_logratio(pp, pn)),
            ("E_onto", ontology_energy(det, pp, pn))]
    counts = {n: pair_counts(np.asarray(s, float), ok, fm, rec)
              for n, s in cols}

    print(f"loading caches ...")
    gc = load_caches(a.feat_cache)
    lc = load_caches(a.local_cache)
    print(f"  {len(gc)} global recordings, {len(lc)} local")
    miss = sorted({r for r in rec if r not in gc or r not in lc})
    if miss:
        print(f"  {len(miss)} of {len({*rec})} evaluation recordings are "
              f"absent from a cache: {miss[:4]}")
        print(f"  Their coverage columns are 0 by construction and would "
              f"manufacture the\n  correlation this module is looking for. "
              f"They are excluded from the tests.")

    rv = recording_variables(cands, morph, gc, lc, a.half_s, a.n_frames)

    keys = [r for r in sorted(counts["morph only"])
            if r not in miss and r in rv]
    print(f"\n{len(keys)} recordings carry both a pair accuracy and cache "
          f"coverage\n")

    print("=" * 78)
    print("WINDOW ASSEMBLY, PER RECORDING")
    print("=" * 78)
    print(f"  {'recording':<22}{'cov_l':>7}{'no_l':>6}{'off_l':>7}"
          f"{'sp_l':>6}{'tol_l':>7}{'gapP90':>8}{'ret_l':>7}"
          f"{'morph':>8}{'det':>7}")
    for r in sorted(keys, key=lambda k: -rv[k].get("tol_used_l", 0)):
        v = rv[r]
        ma = counts["morph only"][r][0] / counts["morph only"][r][1]
        da = counts["detector"][r][0] / counts["detector"][r][1]
        print(f"  {str(r)[:22]:<22}{v['coverage_l']:>7.3f}"
              f"{v['no_local']:>6.2f}{v['phase_offset_l']:>7.3f}"
              f"{v.get('spacing_med_l', float('nan')):>6.2f}"
              f"{v.get('tol_used_l', float('nan')):>7.3f}"
              f"{v.get('gap_ratio_p90_l', float('nan')):>8.2f}"
              f"{v.get('frame_retention_l', float('nan')):>7.2f}"
              f"{ma:>8.3f}{da:>7.3f}")
    print(f"\n  `no_l` is the fraction of a recording's candidates whose LOCAL\n"
          f"  stream resolved to nothing. Those candidates were not dropped -- "
          f"they\n  kept their mask and ran on the global stream alone.\n"
          f"\n  `tol_l` is sorted first because it is the one that hides. It "
          f"is\n  max(grid_step, median_spacing)/2, so a recording that LOST "
          f"frames gets\n  a wider tolerance and its coverage looks fine while "
          f"its window is\n  built from frames up to that far from the instant "
          f"they stand for.\n  Coverage cannot see a smeared window; "
          f"morphology is exactly the\n  question a smeared window destroys.")

    names = sorted({k for r in keys for k in rv[r]})
    print(f"\n{'=' * 78}\nWHAT TRACKS A RECORDING'S PAIR ACCURACY\n{'=' * 78}")
    res = {}
    for cname, _ in cols:
        acc = np.array([counts[cname][r][0] / counts[cname][r][1]
                        for r in keys])
        rows = []
        for n in names:
            x = np.array([rv[r].get(n, np.nan) for r in keys], float)
            if np.isnan(x).any() or np.allclose(x, x[0]):
                continue
            rho, p = _perm_p(x, acc)
            rows.append([n, rho, p])
        qs = bh([r[2] for r in rows])
        for row, q in zip(rows, qs):
            row.append(q)
        rows.sort(key=lambda r: r[3])
        res[cname] = [{"variable": r[0], "rho": r[1], "p": r[2], "q": r[3]}
                      for r in rows]
        print(f"\n  {cname}   (n = {len(keys)} recordings)")
        print(f"  {'variable':<24}{'rho':>9}{'perm p':>10}{'BH q':>9}")
        for r in rows:
            mark = "   <-" if r[3] < 0.10 else ""
            print(f"  {r[0]:<24}{r[1]:>+9.3f}{r[2]:>10.4f}{r[3]:>9.3f}{mark}")
    print(f"\n  READ THE q COLUMN. With {len(names)} variables over "
          f"{len(keys)} recordings,\n  two or three will reach p < .05 with "
          f"nothing behind them, which is why\n  no bare p appears above. "
          f"Nothing here is confirmatory.")

    acc = np.array([counts["morph only"][r][0] / counts["morph only"][r][1]
                    for r in keys])
    grp = np.where(acc >= GOOD, "GOOD",
                   np.where(acc <= INVERTED, "INVERTED", "NEUTRAL"))
    print(f"\n{'=' * 78}\nTHE SAME THING AS A PICTURE, GROUPED BY MORPHOLOGY"
          f"\n{'=' * 78}")
    print(f"  GOOD >= {GOOD}, INVERTED <= {INVERTED}; "
          + ", ".join(f"{g} n={int((grp == g).sum())}"
                      for g in ("GOOD", "NEUTRAL", "INVERTED")))
    print(f"\n  {'variable':<24}{'GOOD':>12}{'NEUTRAL':>12}{'INVERTED':>12}")
    pic = {}
    for n in names:
        x = np.array([rv[r].get(n, np.nan) for r in keys], float)
        if np.isnan(x).any():
            continue
        med = {g: float(np.median(x[grp == g])) if (grp == g).any()
               else float("nan") for g in ("GOOD", "NEUTRAL", "INVERTED")}
        pic[n] = med
        print(f"  {n:<24}{med['GOOD']:>12.3f}{med['NEUTRAL']:>12.3f}"
              f"{med['INVERTED']:>12.3f}")
    print(f"\n  This panel is a picture, not a test. The recordings called "
          f"GOOD are\n  partly the ones whose sampling noise ran up, so the "
          f"gap between the\n  columns is biased outward -- the q column "
          f"above is the evidence.")

    if a.out:
        json.dump({"per_recording": {r: rv[r] for r in keys},
                   "pair_accuracy": {n: {r: counts[n][r][0] / counts[n][r][1]
                                         for r in keys} for n, _ in cols},
                   "correlations": res, "grouped_medians": pic,
                   "excluded_recordings": miss},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
