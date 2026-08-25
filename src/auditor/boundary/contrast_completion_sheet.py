"""Fill the missing SIDE of one-sided development recordings.

batch4 was demoted to a development set on 2026-08-25. Under that protocol it
may be used to choose designs, mine pairs and decide what to build; it may not
supply a deployment threshold, a certificate, or a reported population number,
and it may not be re-used as a test set after anything has been tuned on it.

Restricted to detector peaks -- the only candidates a reranker meets at
inference -- batch4 holds 6 recordings with both classes. But it holds 10 with
positives and no negative, and 24 with negatives and no positive. Those 34 are
one audited timestamp away from being usable, so the cheap move is to complete
them rather than to build a new development set: roughly 60 targeted decisions
against the 240 batch4 spent to reach 6.

THE SAMPLING RULE IS MECHANICAL AND NEVER TOUCHES THE MODEL'S SCORE.

    a recording missing NEGATIVES -> peaks in the INTERIOR of a stored
                                     segment, at least `--interior_margin_s`
                                     from any stored boundary
    a recording missing POSITIVES -> peaks within the frozen tolerance of a
                                     stored segment boundary

Selection inside each pool is by time, seeded and deterministic. `detector_
score` is never read, never sorted on, never thresholded. Choosing which
candidates a person looks at by how the model scored them would build a
development set shaped like the model's current opinion, and every later
comparison would be against a target the model helped draw.

STORED GT IS A SAMPLER HERE, NOT A LABEL. It decides which instants are worth
a person's attention, and nothing else -- the verdict columns come back empty
and a human fills them. This distinction matters because stored GT was
measured 44% wrong on batch4: using a 0.556-precision signal to raise the hit
rate of an audit is efficient, and using it to decide right from wrong is the
error that measurement exposed.

The sheet carries its label columns and its data columns in ONE file.
batch4_blind_review_labels.csv shipped with `temporal_truth` empty on all 240
rows because the verdicts lived in a different file, and reading the wrong one
of the pair caused two separate misreadings.

Usage:
    python -m src.auditor.boundary.contrast_completion_sheet \
        --audit data/gold/batch4_joint_audit.csv \
        --manifest results/hal/batch4/batch4_manifest.jsonl \
        --peaks results/.../peaks.jsonl \
        --segments_root /shared/datasets/.../recordings \
        --out data/gold/batch5_contrast_completion_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

TOL_S = 1.0
INTERIOR_MARGIN_S = 5.0
PER_RECORDING = 2
SEED = 20260825

FILL_COLUMNS = ("temporal_event_type", "within_1s_tolerance",
                "interaction_relation", "true_boundary_start_s", "audit_note")

HEADER_NOTE = (
    "batch4 = DEVELOPMENT SET. These rows complete the missing contrast side "
    "of one-sided recordings. Fill the columns listed in FILL: leave nothing "
    "blank, use cannot_determine when the view does not settle it. "
    "`sampled_because` records why this instant was shown and is NOT a hint "
    "about the answer -- a row sampled near a stored boundary is frequently "
    "not a boundary, which is the whole reason it is being checked."
)


def _norm_rid(x):
    m = re.search(r"(\d+)$", str(x).strip())
    return f"recording_{int(m.group(1)):06d}" if m else str(x).strip()


def load_peaks(path):
    """recording_id -> sorted times. Tolerant about the time field name."""
    by = defaultdict(list)
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        t = None
        for k in ("candidate_time", "candidate_time_s", "peak_time",
                  "pred_time", "t", "time_s"):
            if d.get(k) is not None:
                t = float(d[k])
                break
        if t is None:
            continue
        by[_norm_rid(d.get("recording_id") or d.get("video") or "")].append(t)
    return {k: sorted(set(round(x, 1) for x in v)) for k, v in by.items()}


def load_segment_boundaries(root, rid):
    """Interior boundaries from stored segments.json. SAMPLER ONLY.

    The recording's own start and end are dropped: they are
    initial_action_start and terminal_action_end, a separate audit class, and
    a candidate near them answers a different question than the one being
    asked."""
    p = os.path.join(root, rid, "segments.json")
    if not os.path.isfile(p):
        return None, None
    d = json.load(open(p, encoding="utf-8"))
    segs = d.get("segments") or []
    if not segs:
        return None, None
    starts = [float(s["start_s"]) for s in segs if "start_s" in s]
    ends = [float(s["end_s"]) for s in segs if "end_s" in s]
    if not starts:
        return None, None
    lo, hi = min(starts), max(ends or starts)
    bnd = sorted({round(t, 1) for t in starts + ends
                  if lo + 1e-6 < t < hi - 1e-6})
    return bnd, segs


def sides(audit_rows, manifest):
    """Which side each recording already has, on detector peaks only."""
    have = defaultdict(lambda: {"pos": 0, "neg": 0, "seen": set()})
    for r in audit_rows:
        rid = _norm_rid(r.get("recording_id", ""))
        try:
            t = round(float(r.get("candidate_time_s")), 1)
        except (TypeError, ValueError):
            continue
        have[rid]["seen"].add(t)
        if manifest.get((rid, t)) != "raw_change_peak":
            continue
        if (r.get("temporal_event_type") == "task_boundary"
                and r.get("within_1s_tolerance") == "yes"):
            have[rid]["pos"] += 1
        elif (r.get("temporal_event_type") == "no_boundary"
              and r.get("interaction_relation") == "same_instance"):
            have[rid]["neg"] += 1
    return have


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--peaks", required=True,
                    help="detector peaks for the development recordings, "
                         "JSONL with recording_id and a time field")
    ap.add_argument("--segments_root", required=True,
                    help="the dataset's recordings/ directory, holding "
                         "<recording_id>/segments.json. SAMPLER ONLY -- "
                         "stored GT was 44%% wrong on this batch and supplies "
                         "no verdict here.")
    ap.add_argument("--tol_s", type=float, default=TOL_S)
    ap.add_argument("--interior_margin_s", type=float,
                    default=INTERIOR_MARGIN_S)
    ap.add_argument("--per_recording", type=int, default=PER_RECORDING)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.auditor.boundary.ontology_constitution import Constitution
    Constitution().check_dataset_use("batch4_joint_audit", "mine_pairs")

    rows = [{k.lstrip("﻿").strip(): (v or "").strip()
             for k, v in r.items()}
            for r in csv.DictReader(open(a.audit, encoding="utf-8-sig"))]
    mf = {}
    for l in open(a.manifest, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            mf[(_norm_rid(d["recording_id"]), round(float(d["t"]), 1))] = \
                d.get("candidate_type", "")
    peaks = load_peaks(a.peaks)
    have = sides(rows, mf)

    need_pos = sorted(r for r, v in have.items() if v["neg"] and not v["pos"])
    need_neg = sorted(r for r, v in have.items() if v["pos"] and not v["neg"])
    both = sorted(r for r, v in have.items() if v["pos"] and v["neg"])
    print(f"{len(both)} recordings already have both, "
          f"{len(need_pos)} need a positive, {len(need_neg)} need a negative")
    print(f"peaks available for {len(peaks)} recordings")

    rng = np.random.default_rng(a.seed)
    out, misses = [], defaultdict(int)
    for rid, want in [(r, "positive") for r in need_pos] + \
                     [(r, "negative") for r in need_neg]:
        pk = peaks.get(rid)
        if not pk:
            misses["no detector peaks for this recording"] += 1
            continue
        bnd, _ = load_segment_boundaries(a.segments_root, rid)
        if not bnd:
            misses["no stored segments.json to sample against"] += 1
            continue
        b = np.array(bnd, float)
        seen = have[rid]["seen"]
        pool = []
        for t in pk:
            if any(abs(t - s) <= 0.05 for s in seen):
                continue           # already audited in batch4
            d = float(np.abs(b - t).min())
            if want == "positive" and d <= a.tol_s:
                pool.append((t, d))
            elif want == "negative" and d >= a.interior_margin_s:
                pool.append((t, d))
        if not pool:
            misses[f"no eligible peak for a {want}"] += 1
            continue
        # deterministic, and NOT by model score -- the score is not even
        # loaded. Ordering the pool by how the detector rated it would shape
        # the development set like the model's current opinion.
        idx = rng.permutation(len(pool))[:a.per_recording]
        for i in sorted(idx):
            t, d = pool[i]
            out.append({
                "recording_id": rid,
                "candidate_time_s": f"{t:.1f}",
                "candidate_type": "raw_change_peak",
                "needed_side": want,
                "sampled_because": (f"within {a.tol_s}s of a stored boundary "
                                    f"(gap {d:.2f}s)" if want == "positive"
                                    else f"{d:.2f}s from the nearest stored "
                                         f"boundary, segment interior"),
                **{c: "" for c in FILL_COLUMNS},
            })

    print(f"\n{len(out)} rows over "
          f"{len({r['recording_id'] for r in out})} recordings")
    for k, n in sorted(misses.items(), key=lambda x: -x[1]):
        print(f"  {n:>4} recordings skipped: {k}")
    reachable = len(both) + len({r["recording_id"] for r in out})
    print(f"\n  recordings with BOTH if every row comes back as hoped: "
          f"{reachable}")
    print(f"  It will be fewer. A row sampled near a stored boundary is often "
          f"audited as\n  not a boundary -- stored-GT precision was .556 on "
          f"this batch -- and that is\n  the correct outcome, not a failed "
          f"row. Expect roughly half the positive\n  rows to land.")

    if not out:
        raise SystemExit("nothing to write")
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(f"# {HEADER_NOTE}\n")
        f.write(f"# FILL: {', '.join(FILL_COLUMNS)}\n")
        f.write(f"# tolerance {a.tol_s}s, interior margin "
                f"{a.interior_margin_s}s, seed {a.seed}, "
                f"{a.per_recording} rows per recording\n")
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {a.out}")
    print(f"  Label columns and data columns are in THIS file. The blind "
          f"review sheet\n  that split them shipped 240 rows of empty "
          f"temporal_truth.")


if __name__ == "__main__":
    main()
