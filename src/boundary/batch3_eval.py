"""Evaluate BOTH frozen verifier artifacts against batch3 blind-review labels.

Pre-registered protocol (see execution plan / mentor step list):
  - The blind reviewer fills `temporal_truth` with a pair-taxonomy subtype,
    never seeing model scores. This script is the FIRST time labels and
    hidden scores meet.
  - Each frozen artifact's threshold is applied EXACTLY ONCE, unchanged.
    No threshold re-selection, no primary/secondary re-ordering, no
    picking-the-better-artifact after seeing results.
  - Subtype -> binary mapping follows pair_taxonomy.py:
        sharp_visible_transition      -> positive (a real boundary: keep OK)
        same_action_internal_motion   -> negative (false keep if selected)
        gradual_phase_transition,
        camera_or_viewpoint_shift,
        visibility_or_offscreen,
        annotation_convention,
        ambiguous                     -> excluded from binary precision, but
                                         still counted in the all-candidate
                                         coverage denominator (they route to
                                         manual review in deployment).
  - Report precision + Wilson 95% CI + coverage over three denominators
    (all candidates / scorable candidates / binary-labelled candidates),
    false keeps listed individually, and per-recording concentration.

Partial-review mode: if only a subset of the 240 rows is labelled, results
are computed on the labelled subset and the report says so explicitly --
numbers are provisional until the full sheet is labelled.

Usage:
    python -m src.boundary.batch3_eval \
        --manifest  ~/Downloads/tr1_audits/batch3/batch3_manifest.jsonl \
        --labels    ~/Downloads/batch3_blind_review_filled_60.csv \
        --out_json  ~/Downloads/tr1_audits/batch3/batch3_eval_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

POSITIVE = {"sharp_visible_transition"}
NEGATIVE = {"same_action_internal_motion"}
EXCLUDED = {
    "gradual_phase_transition", "camera_or_viewpoint_shift",
    "visibility_or_offscreen", "annotation_convention", "ambiguous",
}


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_artifact(rows, name, score_key, scorable_key, keep_key):
    """rows: joined dicts with label + manifest fields. Threshold decisions
    were precomputed at sampling time (provisional_keep) from the frozen
    threshold -- we reuse them verbatim so no threshold is ever re-derived
    here."""
    labelled = [r for r in rows if r["subtype"]]
    binary = [r for r in labelled if r["subtype"] in POSITIVE | NEGATIVE]
    excluded = [r for r in labelled if r["subtype"] in EXCLUDED]
    unscorable = [r for r in labelled if not r[scorable_key]]

    kept = [r for r in binary if r[scorable_key] and r[keep_key]]
    true_keeps = [r for r in kept if r["subtype"] in POSITIVE]
    false_keeps = [r for r in kept if r["subtype"] in NEGATIVE]
    # excluded-subtype events the artifact would auto-keep: in deployment these
    # are boundary calls on gradual/offscreen/camera events -- flag them, they
    # are neither confirmed right nor wrong by the binary protocol.
    kept_excluded = [r for r in excluded if r[scorable_key] and r[keep_key]]

    n_keep = len(kept)
    k = len(true_keeps)
    prec = k / n_keep if n_keep else float("nan")
    lo, hi = wilson_interval(k, n_keep)

    base_pos = sum(r["subtype"] in POSITIVE for r in binary)
    base_rate = base_pos / len(binary) if binary else float("nan")

    per_rec = Counter(r["recording_id"] for r in kept)
    fk_per_rec = Counter(r["recording_id"] for r in false_keeps)

    report = {
        "artifact": name,
        "n_labelled": len(labelled),
        "n_binary": len(binary),
        "n_excluded_subtype": len(excluded),
        "n_unscorable_labelled": len(unscorable),
        "n_selected": n_keep,
        "n_true_keeps": k,
        "n_false_keeps": len(false_keeps),
        "precision": prec,
        "wilson_95ci": [lo, hi],
        "base_positive_rate_binary": base_rate,
        "coverage_selected_over_binary": n_keep / len(binary) if binary else float("nan"),
        "coverage_selected_over_labelled": n_keep / len(labelled) if labelled else float("nan"),
        "n_kept_excluded_subtype": len(kept_excluded),
        "kept_excluded_subtypes": Counter(r["subtype"] for r in kept_excluded),
        "false_keeps": [
            {"event_id": r["event_id"], "recording_id": r["recording_id"],
             "t": r["t"], "score": r[score_key], "candidate_type": r["candidate_type"],
             "subtype": r["subtype"]}
            for r in false_keeps
        ],
        "keeps_per_recording_top": per_rec.most_common(8),
        "false_keeps_per_recording": dict(fk_per_rec),
    }
    return report


def print_report(rep):
    print(f"\n=== {rep['artifact']} ===")
    print(f"labelled events used: {rep['n_labelled']}  "
          f"(binary {rep['n_binary']}, excluded-subtype {rep['n_excluded_subtype']}, "
          f"unscorable {rep['n_unscorable_labelled']})")
    print(f"base positive rate among binary: {rep['base_positive_rate_binary']:.3f}")
    if rep["n_selected"] == 0:
        print("selected: 0 events -- no precision estimate possible")
    else:
        lo, hi = rep["wilson_95ci"]
        print(f"selected (auto-keep): {rep['n_selected']}  "
              f"true {rep['n_true_keeps']} / false {rep['n_false_keeps']}")
        print(f"PRECISION: {rep['precision']:.3f}  Wilson95 [{lo:.3f}, {hi:.3f}]")
    print(f"coverage: {rep['coverage_selected_over_binary']:.3f} of binary-labelled, "
          f"{rep['coverage_selected_over_labelled']:.3f} of all labelled")
    if rep["n_kept_excluded_subtype"]:
        print(f"NOTE: {rep['n_kept_excluded_subtype']} excluded-subtype events would "
              f"also be auto-kept: {dict(rep['kept_excluded_subtypes'])}")
    if rep["false_keeps"]:
        print("false keeps:")
        for fk in rep["false_keeps"]:
            print(f"  {fk['event_id']}  score={fk['score']:.3f}  ({fk['candidate_type']})")
    print(f"keeps per recording (top): {rep['keeps_per_recording_top']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()

    manifest = {}
    with open(os.path.expanduser(a.manifest), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                manifest[r["event_id"]] = r

    rows = []
    with open(os.path.expanduser(a.labels), newline="", encoding="utf-8-sig") as f:
        for lr in csv.DictReader(f):
            m = manifest.get(lr["event_id"])
            if m is None:
                raise SystemExit(f"label row {lr['event_id']} not in manifest -- wrong file pair?")
            rows.append({
                "event_id": lr["event_id"],
                "recording_id": m["recording_id"],
                "t": m["t"],
                "candidate_type": m["candidate_type"],
                "subtype": (lr.get("temporal_truth") or "").strip(),
                "primary_score": m["primary_score"],
                "primary_scorable": bool(m["primary_scorable"]),
                "primary_provisional_keep": bool(m["primary_provisional_keep"]),
                "secondary_score": m["secondary_score"],
                "secondary_scorable": bool(m["secondary_scorable"]),
                "secondary_provisional_keep": bool(m["secondary_provisional_keep"]),
            })

    known = POSITIVE | NEGATIVE | EXCLUDED
    bad = sorted({r["subtype"] for r in rows if r["subtype"] and r["subtype"] not in known})
    if bad:
        raise SystemExit(f"unknown temporal_truth values: {bad}")

    n_labelled = sum(bool(r["subtype"]) for r in rows)
    print(f"manifest events: {len(manifest)}  label rows: {len(rows)}  "
          f"labelled: {n_labelled}")
    if n_labelled < len(rows):
        print(f"*** PARTIAL REVIEW: {n_labelled}/{len(rows)} labelled -- results are "
              f"provisional until the full sheet is done ***")
    print("subtype distribution:", dict(Counter(r["subtype"] for r in rows if r["subtype"])))
    print("candidate_type among labelled:",
          dict(Counter(r["candidate_type"] for r in rows if r["subtype"])))

    reports = []
    for name, sk, ck, kk in [
        ("PRIMARY (no-clip / full, threshold 0.8486)",
         "primary_score", "primary_scorable", "primary_provisional_keep"),
        ("SECONDARY (clipped / visual_only, threshold 0.4902)",
         "secondary_score", "secondary_scorable", "secondary_provisional_keep"),
    ]:
        rep = evaluate_artifact(rows, name, sk, ck, kk)
        print_report(rep)
        reports.append(rep)

    if a.out_json:
        out = os.path.expanduser(a.out_json)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"n_manifest": len(manifest), "n_labelled": n_labelled,
                       "partial_review": n_labelled < len(rows),
                       "reports": reports}, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
