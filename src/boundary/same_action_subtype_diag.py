"""Cross-tab C1's per-event continuity scores against the same_action_internal_motion
motion-subtype tags, to decide between C2 (slow/motif latent) and C3 (hand-object
local crop) -- see same_action_subtype.py's docstring for the full rationale.

Answers, per subtype: is P1's false-positive rate elevated, and does
predictive-continuity surprise correctly stay LOW there (would rescue P1) or
does it also spike (matches the global false-positive-rescue finding, but
localizes WHICH kind of same-action motion causes it)?

  - direction_reversal / periodic_repetition driving the failure
    -> supports C2 (predictor tracks low-level motion phase, not action
       identity; need phase-invariant slow latent / motif pooling)
  - regrasp_reposition (fine local hand/finger reconfiguration with no gross
    motion) ALSO driving it, at comparable or higher rate
    -> weighs against "crop out the hand and it'll be fine" as a complete
       fix for C3, since local crops would show the same fine reconfiguration
  - camera_or_scene_noise dominant -> supports C3 (global pooled features
    picking up egomotion unrelated to the interaction)
  - no clear concentration -> neither hypothesis is cleanly supported by this
    slice; do not over-read a 37-event breakdown into a firm architecture call

Usage:
    python -m src.boundary.same_action_subtype_diag \
        --events_dump /workspace/tr1/results/hal/continuity_c1/events_v4.csv \
        --subtype data/gold/same_action_subtype_v1.csv \
        --out /workspace/tr1/results/hal/continuity_c1/subtype_crosstab.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def _f(v):
    return float(v) if v not in (None, "") else float("nan")


def load_events(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "event_id": r["event_id"], "recording_id": r["recording_id"],
                "y": int(r["y"]), "dev_pair_subtype": r["dev_pair_subtype"],
                "efwd_raw": _f(r["cont_efwd_raw"]), "ebwd_raw": _f(r["cont_ebwd_raw"]),
                "efwd_z": _f(r["cont_efwd_z"]), "ebwd_z": _f(r["cont_ebwd_z"]),
                "emin_z": _f(r["cont_emin_z"]), "emax_z": _f(r["cont_emax_z"]),
                "oof_p1": _f(r["oof_p1"]), "oof_p1c": _f(r["oof_p1_plus_continuity"]),
                "coverage_reason": r["coverage_reason"],
            })
    return rows


def load_subtypes(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["event_id"]] = r["subtype"]
    return out


def _stats(vals):
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events_dump", required=True)
    ap.add_argument("--subtype", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    events = load_events(a.events_dump)
    subtypes = load_subtypes(a.subtype)

    pos = [e for e in events if e["y"] == 1]
    neg = [e for e in events if e["y"] == 0]
    print(f"events: {len(events)} ({len(pos)} positive / {len(neg)} negative)")
    tagged = sum(e["event_id"] in subtypes for e in neg)
    print(f"negatives with a motion subtype tag: {tagged}/{len(neg)}")
    if tagged < len(neg):
        missing = [e["event_id"] for e in neg if e["event_id"] not in subtypes]
        print(f"  !! {len(missing)} untagged: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    # P1 false-positive cutoff: same convention as predictive_continuity.py's
    # built-in rescue check, so results are directly comparable -- the median
    # OOF score among TRUE positives, applied here per-subtype.
    p1_pos_scores = [e["oof_p1"] for e in pos if np.isfinite(e["oof_p1"])]
    if len(p1_pos_scores) < 3:
        raise SystemExit("too few positives with a valid oof_p1 to set a cutoff")
    cutoff = float(np.median(p1_pos_scores))
    print(f"P1 false-positive cutoff (median of true-positive OOF scores): {cutoff:.3f}")

    ref_true_pos = _stats([e["efwd_z"] for e in pos])
    print(f"\nreference -- true positives (sharp_visible_transition), "
          f"forward-z: n={ref_true_pos['n']} mean={ref_true_pos['mean']}")

    by_subtype = defaultdict(list)
    for e in neg:
        by_subtype[subtypes.get(e["event_id"], "UNTAGGED")].append(e)

    report = {"n_events": len(events), "n_positive": len(pos), "n_negative": len(neg),
              "p1_false_positive_cutoff": cutoff,
              "reference_true_positive_forward_z": ref_true_pos,
              "subtypes": {}}

    print(f"\n{'subtype':<24} {'n':>3} {'P1 FP rate':>10} "
          f"{'fwd_z(FP)':>10} {'fwd_z(TN)':>10} {'raw_ef(FP)':>11} {'raw_ef(TN)':>11}")
    for sub, evs in sorted(by_subtype.items(), key=lambda kv: -len(kv[1])):
        scorable = [e for e in evs if np.isfinite(e["oof_p1"])]
        fp = [e for e in scorable if e["oof_p1"] >= cutoff]
        tn = [e for e in scorable if e["oof_p1"] < cutoff]
        fp_rate = len(fp) / len(scorable) if scorable else float("nan")
        s_fp_z = _stats([e["efwd_z"] for e in fp])
        s_tn_z = _stats([e["efwd_z"] for e in tn])
        s_fp_raw = _stats([e["efwd_raw"] for e in fp])
        s_tn_raw = _stats([e["efwd_raw"] for e in tn])
        print(f"{sub:<24} {len(evs):>3} {fp_rate:>10.2f} "
              f"{('%.2f' % s_fp_z['mean']) if s_fp_z['mean'] is not None else '  n/a':>10} "
              f"{('%.2f' % s_tn_z['mean']) if s_tn_z['mean'] is not None else '  n/a':>10} "
              f"{('%.3f' % s_fp_raw['mean']) if s_fp_raw['mean'] is not None else '   n/a':>11} "
              f"{('%.3f' % s_tn_raw['mean']) if s_tn_raw['mean'] is not None else '   n/a':>11}")
        report["subtypes"][sub] = {
            "n": len(evs), "n_scorable": len(scorable), "p1_false_positive_rate": fp_rate,
            "false_positive_forward_z": s_fp_z, "true_negative_forward_z": s_tn_z,
            "false_positive_forward_raw": s_fp_raw, "true_negative_forward_raw": s_tn_raw,
            "event_ids": [e["event_id"] for e in evs],
        }

    # Direct read against the two hypotheses.
    phase_subtypes = {"direction_reversal", "periodic_repetition"}
    scene_subtypes = {"camera_or_scene_noise"}
    local_subtypes = {"regrasp_reposition", "spatial_phase_shift"}

    def pooled_fp_rate(names):
        evs = [e for n in names for e in by_subtype.get(n, [])]
        scorable = [e for e in evs if np.isfinite(e["oof_p1"])]
        if not scorable:
            return None
        return sum(e["oof_p1"] >= cutoff for e in scorable) / len(scorable)

    print("\n--- hypothesis read ---")
    r_phase = pooled_fp_rate(phase_subtypes)
    r_scene = pooled_fp_rate(scene_subtypes)
    r_local = pooled_fp_rate(local_subtypes)
    print(f"phase-change subtypes (direction_reversal+periodic_repetition) "
          f"P1-FP rate: {r_phase}")
    print(f"camera/scene-noise subtype P1-FP rate: {r_scene}")
    print(f"local-manipulation subtypes (regrasp+spatial_phase) P1-FP rate: {r_local}")
    if r_scene is None:
        print("  -> no camera_or_scene_noise events in this dev set at all: the "
              "original human/VLM audit did not attribute ANY of these 37 "
              "negatives to camera motion, which is direct (if indirect) "
              "evidence against 'global-feature egomotion pollution' as the "
              "primary driver of C1's failure on THIS dataset -- it does not "
              "rule out egomotion mattering on batch3's noisier recordings.")
    report["hypothesis_read"] = {
        "phase_change_fp_rate": r_phase, "scene_noise_fp_rate": r_scene,
        "local_manipulation_fp_rate": r_local,
    }

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
