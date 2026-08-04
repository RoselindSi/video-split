"""C2-v1 postmortem: answers mentor's four questions before any C2-v2 work
or a move to C3 is considered. Consumes slow_latent_c2.py's --dump_events
CSV. Does NOT retrain or retune anything -- purely diagnostic on the
already-computed OOF scores from report_v1.

  A. Rescues: events P1 misclassified that the P1+C2 fusion gets right.
  B. Harms:   events P1 got right that the fusion gets wrong (the risk case
              mentor flagged -- does C2 smooth over real transitions like
              release->idle or a tool switch?).
  C. Spearman(P1, C2) -- redundant signal vs genuinely different-but-useless
     signal vs genuinely different-and-useful signal.
  D. Grouped bootstrap CI for ΔAUROC = AUROC(fused) - AUROC(P1), resampling
     RECORDINGS (not raw events) with replacement -- events from the same
     recording are correlated, so an event-level bootstrap would understate
     the true uncertainty. If the CI straddles 0, +0.003 is noise.
  E. Geometry by subtype: same-action cross-side distance (oof_d_lr) and
     both intra-window stability distances (oof_d_l, oof_d_r), split by
     same_action_subtype (regrasp_reposition / direction_reversal /
     periodic_repetition / spatial_phase_shift) plus sharp_visible_transition
     as the positive reference -- checks whether the encoder is actually
     "slow" (low d_l/d_r everywhere, low d_lr specifically for same-action)
     rather than just checking the final classification AUROC, which cannot
     distinguish "learned real invariance" from "found an unrelated
     classifier-head shortcut".

Usage:
    python -m src.boundary.c2_postmortem \
        --events_dump /workspace/tr1/results/hal/slow_latent_c2/events_v1.csv \
        --out /workspace/tr1/results/hal/slow_latent_c2/postmortem.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from src.boundary.state_adapter import _auroc
from src.boundary.predictive_continuity import _spearman


def _f(v):
    return float(v) if v not in (None, "") else float("nan")


def load_events(path, score_col="oof_fixed"):
    """score_col selects which C2xP1 combination to run the postmortem
    against: 'oof_fixed' (the pre-registered 0.5/0.5 fusion, comparable
    across v1->v2) or 'oof_nested' (v2's diagnostic nested-LR fusion).
    Falls back gracefully if the dump predates a column (v1 dumps have
    oof_fused/oof_c2 instead of oof_fixed/oof_q -- both are accepted)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        fused_col = score_col if score_col in fields else (
            "oof_fused" if "oof_fused" in fields else "oof_fixed")
        c2_col = "oof_q" if "oof_q" in fields else "oof_c2"
        for r in reader:
            rows.append({
                "event_id": r["event_id"], "recording_id": r["recording_id"],
                "y": int(r["y"]), "dev_pair_subtype": r["dev_pair_subtype"],
                "same_action_subtype": r["same_action_subtype"],
                "d_lr": _f(r["oof_d_lr"]), "d_l": _f(r["oof_d_l"]), "d_r": _f(r["oof_d_r"]),
                "p1": _f(r["oof_p1"]), "c2": _f(r[c2_col]), "fused": _f(r[fused_col]),
            })
    return rows


def rescues_and_harms(events, cutoff_p1, cutoff_fused):
    rescues, harms = [], []
    for e in events:
        if not (np.isfinite(e["p1"]) and np.isfinite(e["fused"])):
            continue
        pred_p1 = int(e["p1"] >= cutoff_p1)
        pred_fused = int(e["fused"] >= cutoff_fused)
        correct_p1 = pred_p1 == e["y"]
        correct_fused = pred_fused == e["y"]
        kind = "false_positive_fixed" if e["y"] == 0 else "false_negative_fixed"
        kind_h = "true_negative_broken" if e["y"] == 0 else "true_positive_broken"
        row = {**e, "pred_p1": pred_p1, "pred_fused": pred_fused}
        if not correct_p1 and correct_fused:
            row["kind"] = kind
            rescues.append(row)
        elif correct_p1 and not correct_fused:
            row["kind"] = kind_h
            harms.append(row)
    return rescues, harms


def grouped_bootstrap_delta_auroc(events, n_boot=2000, seed=0):
    """Resamples RECORDINGS with replacement (not raw events) -- events
    sharing a recording are correlated, so this is the appropriate unit for
    the CI, matching every grouped-fold discipline used elsewhere in this
    project (stratified_grouped_folds, etc.)."""
    by_rec = defaultdict(list)
    for e in events:
        if np.isfinite(e["p1"]) and np.isfinite(e["fused"]):
            by_rec[e["recording_id"]].append(e)
    recs = sorted(by_rec)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        sample_recs = rng.choice(recs, len(recs), replace=True)
        sample = [e for r in sample_recs for e in by_rec[r]]
        y = np.array([e["y"] for e in sample], dtype=float)
        if len(set(y.tolist())) < 2:
            continue
        p1 = np.array([e["p1"] for e in sample])
        fused = np.array([e["fused"] for e in sample])
        au_p1 = _auroc(y, p1)
        au_f = _auroc(y, fused)
        deltas.append(au_f - au_p1)
    deltas = np.array(deltas)
    return deltas, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def geometry_by_subtype(events):
    groups = defaultdict(list)
    for e in events:
        key = "sharp_visible_transition" if e["y"] == 1 else (e["same_action_subtype"] or "untagged")
        groups[key].append(e)
    out = {}
    for key, evs in groups.items():
        d_lr = np.array([e["d_lr"] for e in evs]); d_lr = d_lr[np.isfinite(d_lr)]
        d_l = np.array([e["d_l"] for e in evs]); d_l = d_l[np.isfinite(d_l)]
        d_r = np.array([e["d_r"] for e in evs]); d_r = d_r[np.isfinite(d_r)]
        out[key] = {
            "n": len(evs),
            "d_lr_mean": float(d_lr.mean()) if len(d_lr) else None,
            "d_l_mean": float(d_l.mean()) if len(d_l) else None,
            "d_r_mean": float(d_r.mean()) if len(d_r) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events_dump", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_col", default="oof_fixed",
                    help="which fusion column to analyze: oof_fixed (default, "
                         "pre-registered) or oof_nested (v2 diagnostic)")
    ap.add_argument("--out")
    a = ap.parse_args()

    events = load_events(a.events_dump, score_col=a.score_col)
    print(f"analyzing score column: {a.score_col}")
    y = np.array([e["y"] for e in events], dtype=float)
    print(f"events: {len(events)} ({int(y.sum())} positive / {int((1-y).sum())} negative)  "
          f"recordings: {len(set(e['recording_id'] for e in events))}")

    p1 = np.array([e["p1"] for e in events])
    fused = np.array([e["fused"] for e in events])
    c2 = np.array([e["c2"] for e in events])
    cutoff_p1 = float(np.nanmedian(p1[y == 1]))
    cutoff_fused = float(np.nanmedian(fused[y == 1]))
    print(f"cutoffs (median of true-positive OOF score): P1={cutoff_p1:.3f} fused={cutoff_fused:.3f}")

    print("\n=== A/B: rescues and harms ===")
    rescues, harms = rescues_and_harms(events, cutoff_p1, cutoff_fused)
    print(f"rescues (P1 wrong -> fused right): {len(rescues)}")
    for r in rescues:
        print(f"  {r['event_id']}  y={r['y']}  {r['kind']}  subtype={r['same_action_subtype'] or r['dev_pair_subtype']}  "
              f"p1={r['p1']:.3f} fused={r['fused']:.3f}")
    print(f"harms (P1 right -> fused wrong): {len(harms)}")
    for h in harms:
        print(f"  {h['event_id']}  y={h['y']}  {h['kind']}  subtype={h['same_action_subtype'] or h['dev_pair_subtype']}  "
              f"p1={h['p1']:.3f} fused={h['fused']:.3f}")
    if any(h["y"] == 1 for h in harms):
        print("  !! at least one TRUE POSITIVE (real transition) broken by fusion -- "
              "this is exactly the risk mentor flagged (release->idle, tool switch, "
              "new-object interaction smoothed over)")

    print("\n=== C: Spearman(P1, C2) ===")
    finite = np.isfinite(p1) & np.isfinite(c2)
    rho = _spearman(p1[finite], c2[finite])
    print(f"rho = {rho:.3f}  n={int(finite.sum())}")
    if np.isfinite(rho) and rho > 0.6:
        print("  -> high correlation: C2 is largely redundant with P1")
    elif np.isfinite(rho):
        print("  -> low/moderate correlation: C2 carries different signal "
              "(whether that signal is USEFUL is what D checks)")

    print(f"\n=== D: grouped bootstrap CI for delta-AUROC (n_boot={a.n_boot}) ===")
    deltas, lo, hi = grouped_bootstrap_delta_auroc(events, n_boot=a.n_boot, seed=a.seed)
    point_gain = _auroc(y[np.isfinite(p1)&np.isfinite(fused)], fused[np.isfinite(p1)&np.isfinite(fused)]) - \
                 _auroc(y[np.isfinite(p1)&np.isfinite(fused)], p1[np.isfinite(p1)&np.isfinite(fused)])
    print(f"point estimate delta-AUROC: {point_gain:+.4f}")
    print(f"95% grouped-bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0 <= hi:
        # The gain is interpolated from the actual point estimate, never
        # hardcoded -- an earlier version printed v1's "+0.003" verbatim and
        # kept printing it after v2 moved the point estimate to +0.0065.
        print(f"  -> CI STRADDLES 0: the {point_gain:+.4f} gain is not distinguishable "
              f"from noise at this sample size. This is the expected, honest reading "
              f"given {len(events)} events / {int((1 - y).sum())} negatives across "
              f"{len(set(e['recording_id'] for e in events))} recordings.")
    else:
        print("  -> CI excludes 0: the gain, while small, is not pure noise "
              "under recording-level resampling.")

    print("\n=== E: slow-latent geometry by subtype ===")
    geo = geometry_by_subtype(events)
    print(f"{'subtype':<28} {'n':>4} {'d_lr(L-R)':>10} {'d_l(L1-L2)':>11} {'d_r(R1-R2)':>11}")
    for key in sorted(geo, key=lambda k: -geo[k]["n"]):
        g = geo[key]
        def fmt(v): return f"{v:.4f}" if v is not None else "n/a"
        print(f"{key:<28} {g['n']:>4} {fmt(g['d_lr_mean']):>10} "
              f"{fmt(g['d_l_mean']):>11} {fmt(g['d_r_mean']):>11}")
    sharp_dlr = geo.get("sharp_visible_transition", {}).get("d_lr_mean")
    if sharp_dlr:
        for key, g in geo.items():
            if key != "sharp_visible_transition" and g["d_lr_mean"] is not None:
                ratio = g["d_lr_mean"] / sharp_dlr
                if ratio > 0.5:
                    print(f"  !! {key}: d_lr is {ratio:.2f}x the sharp-transition mean -- "
                          f"the encoder has NOT learned phase-invariance for this subtype, "
                          f"even if final AUROC looks acceptable")

    if a.out:
        report = {
            "n_events": len(events), "cutoff_p1": cutoff_p1, "cutoff_fused": cutoff_fused,
            "rescues": rescues, "harms": harms, "spearman_p1_c2": rho,
            "delta_auroc_point": point_gain, "delta_auroc_ci95": [lo, hi],
            "geometry_by_subtype": geo,
        }
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
