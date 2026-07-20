"""B6 human-audit CSV template. Reproduces the EXACT same balanced sample
(same seed/config/category caps) that dump_boundary_clips.py already
rendered to PNGs, so `sample_id` here lines up 1:1 with the filenames on
disk (`{pool}_{recording_id}_t{time}.png`) -- open the PNG named by
sample_id, fill in the row.

Columns:
  sample_id, pool (TP/FN missed_same/FN missed_diff/FP false_peak),
  model_peak_offset (signed distance in seconds from the model's own
  prediction to the nearest GT boundary -- 0 for true positives; for missed
  boundaries, distance to the nearest predicted peak REGARDLESS of tolerance,
  i.e. how close the model got even though it didn't count as a hit; for
  false peaks, distance to the nearest GT boundary),
  then EMPTY columns for the human: primary_error_type, secondary_error_type
  (both free text, may hold multiple comma-separated categories -- a single
  clip can be "camera motion + grasp motion", don't force one label),
  GT_boundary_quality {clear/arguable/ambiguous}, visual_evidence_present
  {yes/weak/no}, notes.

Usage (server, must match the dump_boundary_clips.py call that made the PNGs):
    python -m src.boundary.audit_template \
        --logits /tmp/b2_logits.pt --data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/boundary_clips/audit_template.csv \
        --thr 0.45 --min_gap 1.0 --tol 0.5 --n_per_category 15 --per_video_cap 2
"""
import argparse, csv, json, random
from collections import Counter

import torch


def peaks_threshold(prob, times, thr, min_gap):
    cand = [i for i in range(len(prob))
            if prob[i] >= thr
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--n_per_category", type=int, default=15)
    ap.add_argument("--per_video_cap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.logits)
    data = torch.load(a.logits, weights_only=False)
    rng = random.Random(a.seed)

    cands = {"missed_same": [], "missed_diff": [], "false_peak": [], "true_positive": []}
    for v in data:
        segs = sorted(v["segments"], key=lambda s: s[1])
        starts = {round(s[1], 2): s[0] for s in segs}
        ends = {round(s[2], 2): s[0] for s in segs}
        gts, preds = v["gt"], peaks_threshold(v["prob"], v["times"], a.thr, a.min_gap)
        used = set()
        for p in preds:
            best, bj = a.tol + 1, -1
            for j, g in enumerate(gts):
                if j not in used and abs(p - g) < best:
                    best, bj = abs(p - g), j
            if bj >= 0 and best <= a.tol:
                used.add(bj)
        for j, g in enumerate(gts):
            before = ends.get(round(g, 2), "<gap>")
            after = starts.get(round(g, 2), "<gap>")
            nearest_pred = min((abs(p - g) for p in preds), default=None)
            if j in used:
                cands["true_positive"].append((v["recording_id"], g, before, after, 0.0))
            else:
                kind = "missed_same" if before == after else "missed_diff"
                cands[kind].append((v["recording_id"], g, before, after, nearest_pred))
        for p in preds:
            d = min((abs(p - g) for g in gts), default=999)
            if d > a.tol:
                containing = next((s for s in segs if s[1] <= p <= s[2]), None)
                name = containing[0] if containing else "<gap>"
                nearest_gt = min((abs(p - g) for g in gts), default=None)
                cands["false_peak"].append((v["recording_id"], p, name, name, nearest_gt))

    selected = []
    for kind, items in cands.items():
        rng.shuffle(items)
        cap = Counter(); picked = []
        for rid, t, before, after, offset in items:
            if len(picked) >= a.n_per_category:
                break
            if cap[rid] >= a.per_video_cap:
                continue
            cap[rid] += 1
            picked.append((kind, rid, t, before, after, offset))
        selected += picked

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "pool", "recording_id", "time_s", "before_name",
                    "after_name", "model_peak_offset_s",
                    # -- fill these in while looking at the matching PNG --
                    "primary_error_type", "secondary_error_type",
                    "GT_boundary_quality", "visual_evidence_present", "notes"])
        for kind, rid, t, before, after, offset in selected:
            sample_id = f"{kind}_{rid}_t{t:.1f}"
            w.writerow([sample_id, kind, rid, round(t, 2), before, after,
                       round(offset, 2) if offset is not None else "",
                       "", "", "", "", ""])

    print(f"wrote {len(selected)} audit rows -> {a.out}")
    print("primary_error_type / secondary_error_type suggested vocabulary "
          "(free text, comma-separate multiple, don't force exactly one):")
    print("  semantic-only transition, object/state transition, grasp/release, "
          "ordinary within-action motion, camera/viewpoint motion, "
          "pause/resume, occlusion, annotation ambiguity")


if __name__ == "__main__":
    main()
