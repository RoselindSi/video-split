"""Is the visual change around a candidate CONCENTRATED or SPREAD OUT?

The cheapest testable piece of the proposed C3-T branch, and the only one
computable without a new extraction. It runs on the 10 fps GLOBAL caches that
already exist, so it costs minutes and no GPU.

The hypothesis is specific and falsifiable: a sharp transition should show its
change packed into a short interval around the candidate, while a gradual
phase transition spreads the same total change over seconds, and same-action
motion may be large but need not form a single concentrated switch. Nothing in
the current feature set expresses that -- every existing score is some form of
"how different are the two sides", which is blind to whether the difference
arrived all at once.

Why global features rather than the hand crop, when C3 established the local
region is where the complementary information lives: this is a screening step.
If concentration carries no signal even on global features, computing it on
local crops requires a 10 fps local extraction over 147 recordings and would be
spending that on an idea the cheap version already failed. If it does carry
signal, the local version has evidence behind it rather than an argument.

Three tasks, kept separate because the decision layer needs them separately:

  A  sharp vs same-action over all clean events -- the verifier question.
  B  clean-observable vs gradual/offscreen/camera/annotation/ambiguous -- the
     observability question, and the only place anything has beaten chance so
     far (detect_longest_gap_s at 0.673 against a 0.576 baseline).
  C  sharp vs same-action restricted to the REVIEW band, where any review
     reduction would actually have to come from. A feature that separates on
     all clean events but not here cannot reduce review.

Every number carries a recording-grouped bootstrap interval and a permutation
baseline, because at these sample sizes a bare AUROC cannot be told from what
shuffling produces -- which is exactly how the previous round's 0.637 turned
out to sit below chance.

Usage:
    python -m src.boundary.c3_transition_concentration \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_10fps_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_10fps_missing_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_10fps_refill_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_10fps_eval_clean_multi.pt \
        --out /workspace/tr1/results/hal/c3/transition_concentration.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter

import numpy as np

from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import _auroc
from src.boundary.c3_error_separability import auroc_stats

SHARP = "sharp_visible_transition"
SAME = "same_action_internal_motion"
CLEAN = (SHARP, SAME)
WIN = 2.0
N_BLOCKS = 5


def event_time(eid):
    m = re.search(r"_t(\d+(?:\.\d+)?)$", eid)
    return float(m.group(1)) if m else None


def change_trajectory(feats, times, t, win=WIN):
    """(dt, change) for consecutive frames inside [t-win, t+win].

    Change is cosine distance between consecutive frames, matching the
    direction-only convention the rest of the pipeline uses: _pool L2-
    normalises every pooled vector, so a uniform rescaling of the features is
    invisible downstream and should be invisible here too."""
    m = (times >= t - win) & (times <= t + win)
    if int(m.sum()) < 6:
        return None, None
    f = feats[m].astype(np.float64)
    tt = times[m]
    n = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    d = 1.0 - (n[1:] * n[:-1]).sum(1)
    mid = (tt[1:] + tt[:-1]) / 2.0
    return mid - t, d


def concentration_features(dt, d):
    """Shape descriptors of one change trajectory. All are ratios, so a
    recording with globally more motion does not score differently for that
    reason alone."""
    tot = float(d.sum())
    if tot <= 0:
        return None
    out = {}
    for w in (0.25, 0.5, 1.0):
        frac = float(d[np.abs(dt) <= w].sum()) / tot
        uniform = min(1.0, w / WIN)      # what a flat trajectory would give
        out[f"conc_{w}s"] = frac
        out[f"conc_{w}s_over_uniform"] = frac / uniform if uniform > 0 else np.nan
    peak = float(d.max())
    med = float(np.median(d))
    out["peak_over_median"] = peak / med if med > 0 else np.nan
    out["peak_width_s"] = float((d > peak / 2).sum()) * float(np.median(np.diff(dt))) \
        if len(dt) > 1 else np.nan
    out["peak_offset_s"] = float(dt[int(np.argmax(d))])
    left, right = d[dt < 0].sum(), d[dt >= 0].sum()
    out["asymmetry"] = float((right - left) / tot)
    # side stability, the one quantity C2 measured that nothing here carries:
    # how much each side changes WITHIN itself, as opposed to across the
    # candidate. C2-v2 found sharp events sitting at d_l ~ 0.27, i.e. far from
    # internally stable, which is why subtracting it there destroyed real
    # signal -- kept as its own feature rather than folded into a score.
    out["side_change_left"] = float(left / max(1, (dt < 0).sum()))
    out["side_change_right"] = float(right / max(1, (dt >= 0).sum()))
    out["side_change_imbalance"] = abs(out["side_change_left"] - out["side_change_right"])
    return out


def run_task(name, question, rows, label_fn, feats, recs, n_boot, seed,
             n_perm_family=2000):
    lab = np.array([label_fn(r) for r in rows], dtype=float)
    keep = np.isfinite(lab)
    if len(set(lab[keep].tolist())) < 2 or keep.sum() < 30:
        print(f"\n  {name}: too few labelled events ({int(keep.sum())})")
        return {}
    print(f"\n  {name}: {question}")
    print(f"    {int(lab[keep].sum())} positive / {int((1 - lab[keep]).sum())} negative, "
          f"{len({r for r, k in zip(recs, keep) if k})} recordings")
    print(f"    {'feature':<28} {'AUROC':>7} {'95% CI':>18} {'chance':>8} {'dir':>7}")
    out, usable = {}, []
    for f in sorted(feats):
        v = np.array([r["_feat"].get(f, np.nan) if r.get("_feat") else np.nan
                      for r in rows])
        m = keep & np.isfinite(v)
        if m.sum() < 30 or len(set(v[m].tolist())) < 5:
            continue
        st = auroc_stats(lab[m], v[m], [r for r, k in zip(recs, m) if k], n_boot, n_boot, seed)
        if not st:
            continue
        usable.append((f, v, m))
        out[f] = st
        flag = "" if st["folded"] > st["perm_folded_p95"] else "  (below chance)"
        ci = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]"
        print(f"    {f:<28} {st['folded']:>7.3f} {ci:>18} "
              f"{st['perm_folded_p95']:>8.3f} {st['direction']:>7}{flag}")
    # FAMILY-WISE baseline. Comparing each of ~11 features against its OWN
    # permutation p95 expects roughly half a false crossing per task at
    # alpha=0.05, and a fixture with no planted signal duly produced one
    # (conc_1.0s at 0.606 against 0.595) which the summary then reported as a
    # hit. The question that matters is whether the BEST feature beats the
    # best you would get by chance ACROSS THIS MANY FEATURES, so each
    # permutation is scored on every feature at once and the maximum is kept.
    rng = np.random.RandomState(seed + 1)
    fam = []
    for _ in range(n_perm_family):
        perm = rng.permutation(lab[keep])
        pl = np.full(len(lab), np.nan)
        pl[keep] = perm
        best = 0.0
        for f, v, m in usable:
            au = _auroc(pl[m], v[m])
            best = max(best, max(au, 1 - au))
        fam.append(best)
    fam_p95 = float(np.percentile(fam, 95)) if fam else float("nan")
    b = max(out.items(), key=lambda kv: kv[1]["folded"]) if out else None
    print(f"    family-wise chance baseline over {len(usable)} features: "
          f"{fam_p95:.3f}  (per-feature baselines above are NOT corrected for "
          f"testing this many)")
    if b is None or b[1]["folded"] <= fam_p95:
        print(f"    -> nothing beats it: best is "
              f"{b[0] if b else 'n/a'} at {b[1]['folded'] if b else float('nan'):.3f}")
        for k in out:
            out[k]["beats_family_wise"] = False
    else:
        print(f"    -> {b[0]} at {b[1]['folded']:.3f} beats the family-wise "
              f"baseline {fam_p95:.3f}, CI [{b[1]['ci95'][0]:.3f}, "
              f"{b[1]['ci95'][1]:.3f}]")
        for k, v in out.items():
            v["beats_family_wise"] = v["folded"] > fam_p95
    for v in out.values():
        v["family_wise_p95"] = fam_p95
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True,
                    help="a c3_selective_policy per-role decisions CSV, which "
                         "carries subtype and the REVIEW/AUTO decision")
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="10 fps caches -- the frame rate is the point: at 2 fps "
                         "a +-0.25s window is a single frame")
    ap.add_argument("--block", default="global",
                    choices=["global", "all"],
                    help="which part of a --pool multi vector to measure change "
                         "on. 'global' avoids the `center` block, which on these "
                         "packed-stereo frames straddles the seam between the "
                         "two cameras.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--dump")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    d_total = next(iter(by_rid.values()))["feats"].shape[1]
    sl = slice(0, d_total // N_BLOCKS) if (a.block == "global" and d_total % N_BLOCKS == 0) \
        else slice(0, d_total)
    print(f"feature dim {d_total}, measuring change on {sl.stop - sl.start} dims "
          f"({a.block})")

    rows = list(csv.DictReader(open(a.decisions, newline="", encoding="utf-8")))
    n_no_t = n_no_rec = n_no_traj = 0
    for r in rows:
        t = event_time(r["event_id"])
        rec = by_rid.get(r["recording_id"])
        if t is None:
            n_no_t += 1
            continue
        if rec is None:
            n_no_rec += 1
            continue
        times = rec["times"].numpy() if hasattr(rec["times"], "numpy") else np.asarray(rec["times"])
        f = rec["feats"]
        f = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
        dt, d = change_trajectory(f[:, sl], times, t)
        if dt is None:
            n_no_traj += 1
            continue
        r["_feat"] = concentration_features(dt, d)
    ok = [r for r in rows if r.get("_feat")]
    print(f"{len(rows)} events -> {len(ok)} with a trajectory "
          f"(no time in id: {n_no_t}, no 10fps cache: {n_no_rec}, "
          f"too few frames: {n_no_traj})")
    if len(ok) < 50:
        raise SystemExit("too few events with trajectories -- check the caches "
                         "cover these recordings at 10 fps")
    print(f"  median frames per window: "
          f"{np.median([len(r['_feat']) for r in ok]):.0f} features each")
    print(f"  by subtype: {dict(Counter(r['subtype'] or '(none)' for r in ok))}")

    feats = sorted({k for r in ok for k in r["_feat"]})
    recs = [r["recording_id"] for r in ok]
    report = {"n": len(ok), "block": a.block, "tasks": {}}

    clean = [r for r in ok if r["subtype"] in CLEAN]
    report["tasks"]["A_verifier_all_clean"] = run_task(
        "A", "sharp vs same-action, all clean events", clean,
        lambda r: 1.0 if r["subtype"] == SHARP else 0.0,
        feats, [r["recording_id"] for r in clean], a.n_boot, a.seed)

    report["tasks"]["B_observability"] = run_task(
        "B", "clean-observable vs non-clean taxonomy", ok,
        lambda r: 1.0 if r["subtype"] in CLEAN else 0.0,
        feats, recs, a.n_boot, a.seed)

    rev = [r for r in ok if r["decision"] == "REVIEW" and r["subtype"] in CLEAN]
    report["tasks"]["C_verifier_review_band"] = run_task(
        "C", "sharp vs same-action, REVIEW band only", rev,
        lambda r: 1.0 if r["subtype"] == SHARP else 0.0,
        feats, [r["recording_id"] for r in rev], a.n_boot, a.seed)

    print("\n" + "=" * 72)
    def _any(task):
        return any(v.get("beats_family_wise") for v in report["tasks"][task].values())
    anyA = _any("A_verifier_all_clean")
    anyB = _any("B_observability")
    anyC = _any("C_verifier_review_band")
    if not (anyA or anyB or anyC):
        print("Concentration carries no signal above chance on ANY task, measured "
              "on global features. Computing it on hand crops needs a 10 fps "
              "local extraction over 147 recordings; this says do not spend that "
              "on this idea. It does NOT rule out the other C3-T components "
              "(hand motion reversal, periodicity, contact continuity), which "
              "measure something concentration cannot see.")
    else:
        got = [n for n, f in (("A verifier", anyA), ("B observability", anyB),
                              ("C review band", anyC)) if f]
        print(f"Concentration beats chance on: {', '.join(got)}. That is the "
              f"evidence for computing it on LOCAL crops, where C3 showed the "
              f"complementary information lives -- with the caveat that beating "
              f"a permutation baseline is a long way from a deployable "
              f"threshold, and task C is the one that bears on review "
              f"reduction.")

    if a.dump:
        with open(a.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "subtype", "y", "decision"] + feats)
            for r in ok:
                w.writerow([r["event_id"], r["recording_id"], r["subtype"], r["y"],
                            r["decision"]]
                           + [f"{r['_feat'].get(k, float('nan')):.6f}" for k in feats])
        print(f"\nwrote {a.dump}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
