"""B-final -- boundary error audit + offset distribution + reproducibility
freeze, in one pass over saved logits (train_head_multi.py --save_logits).
This is boundary's counterpart to naming's N9 convergence step: a
quantitative "why does it fail" layer, not just F1@tol.

1. CLASSIFICATION (per GT boundary and per predicted peak):
   GT boundaries:
     exact          : matched, |offset| <= exact_tol (default 0.25s)
     early          : matched, pred_time < gt_time, exact_tol < |offset| <= tol
     late           : matched, pred_time > gt_time, exact_tol < |offset| <= tol
     missed         : no predicted peak within tol
   Predicted peaks:
     matched        : assigned to a GT boundary (counted above)
     duplicate      : an EXTRA peak within tol of an already-matched GT
                      boundary (decode/NMS issue at that specific boundary)
     false_near_edge: unmatched, but close to a segment start/end (localization
                      candidate -- probably the right region, wrong instant)
     false_mid_segment: unmatched, deep inside a segment (spurious motion,
                      not a localization issue)

   Missed GT boundaries are further split by a signal-presence heuristic
   (no new human labels needed, though it complements audit_template.py's
   human categories, doesn't replace them):
     weak_signal            : max probability within +-1s of the GT time is
                               BELOW that video's own median probability --
                               the model produced essentially no local signal
                               here at all (candidates: no strong visual
                               transition, or annotation ambiguity -- confirm
                               visually via the audit_template.py clips)
     signal_present_not_top : local probability WAS locally elevated (above
                               that video's median) but still didn't clear
                               the decode threshold/NMS -- candidates:
                               representation/ranking didn't rank it highly
                               enough relative to other peaks, or decode
                               calibration

2. OFFSET DISTRIBUTION: for all matched (exact+early+late) pairs, reports
   median absolute error and % within {0.25, 0.5, 1.0, 2.0}s -- the number
   that explains "nearGT~=0.50 but strict F1@0.5~=0.30": if most matched
   predictions cluster just outside 0.5s, F1@0.5 punishes near-misses that a
   human would call correct.

3. REPRODUCIBILITY FREEZE: saves full per-recording predictions+classification
   (predictions.jsonl), the exact decode config, git commit (via
   run_manifest), and the list of recording_ids in this split (group
   assignment) -- everything needed to reproduce this exact number without
   re-running training.

Usage (server, after a train_head_multi.py --save_logits run with the
current dump_logits() that includes segments):
    python -m src.boundary.boundary_error_audit \
        --logits /workspace/tr1/results/boundary/b2_logits.pt \
        --out_dir /workspace/tr1/results/boundary/error_audit \
        --thr 0.45 --min_gap 1.0 --tol 0.5 --exact_tol 0.25
"""
import argparse, json, os, statistics
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


def local_max_prob(prob, times, g, window=1.0):
    vals = [prob[i] for i, t in enumerate(times) if abs(t - g) <= window]
    return max(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--exact_tol", type=float, default=0.25)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.logits)
    data = torch.load(a.logits, weights_only=False)
    os.makedirs(a.out_dir, exist_ok=True)

    gt_class = Counter()
    peak_class = Counter()
    offsets = []  # signed, matched only
    weak_signal_missed = 0; signal_present_missed = 0
    per_recording_preds = []

    for v in data:
        prob, times, gts = v["prob"], v["times"], v["gt"]
        segs = sorted(v["segments"], key=lambda s: s[1])
        overall_median = statistics.median(prob)
        preds = peaks_threshold(prob, times, a.thr, a.min_gap)

        # match each GT to the NEAREST pred within tol; track ALL preds
        # within tol of each GT to detect duplicates separately
        used_preds = set()          # nearest matches only -- excluded from unmatched classification
        duplicate_preds = set()     # extra peaks near an already-matched GT -- get their OWN pred_record
        gt_records = []
        for g in gts:
            within = [p for p in preds if abs(p - g) <= a.tol]
            if not within:
                lp = local_max_prob(prob, times, g)
                kind = "weak_signal" if lp < overall_median else "signal_present_not_top"
                if kind == "weak_signal":
                    weak_signal_missed += 1
                else:
                    signal_present_missed += 1
                gt_class["missed"] += 1
                gt_records.append({"gt_time": g, "status": "missed", "signal": kind,
                                   "local_max_prob": round(lp, 3), "video_median_prob": round(overall_median, 3)})
                continue
            nearest = min(within, key=lambda p: abs(p - g))
            offset = nearest - g
            offsets.append(offset)
            used_preds.add(nearest)
            for extra in within:
                if extra != nearest:
                    duplicate_preds.add(extra)
                    peak_class["duplicate"] += 1
            if abs(offset) <= a.exact_tol:
                status = "exact"
            elif offset < 0:
                status = "early"
            else:
                status = "late"
            gt_class[status] += 1
            gt_records.append({"gt_time": g, "status": status, "offset": round(offset, 3),
                               "matched_pred_time": round(nearest, 3)})

        pred_records = []
        for p in preds:
            if p in used_preds:
                continue
            if p in duplicate_preds:
                nearest_g = min(gts, key=lambda g: abs(g - p)) if gts else None
                pred_records.append({"pred_time": round(p, 3), "status": "duplicate",
                                     "near_gt_time": round(nearest_g, 3) if nearest_g is not None else None})
                continue
            containing = next((s for s in segs if s[1] <= p <= s[2]), None)
            if containing:
                frac = (p - containing[1]) / max(containing[2] - containing[1], 1e-6)
                kind = "false_near_edge" if (frac < 0.15 or frac > 0.85) else "false_mid_segment"
            else:
                kind, frac = "false_gap", None
            peak_class[kind] += 1
            pred_records.append({"pred_time": round(p, 3), "status": kind,
                                 "frac_into_segment": round(frac, 3) if frac is not None else None,
                                 "containing_segment": containing[0] if containing else None})

        per_recording_preds.append({"recording_id": v.get("recording_id", ""),
                                    "gt_boundaries": gt_records, "predicted_peaks": pred_records})

    # ---------------- offset distribution ----------------
    abs_offsets = [abs(o) for o in offsets]
    print(f"\n=== boundary classification (thr={a.thr}, min_gap={a.min_gap}, "
          f"tol={a.tol}, exact_tol={a.exact_tol}) ===")
    total_gt = sum(gt_class[k] for k in ("exact", "early", "late", "missed"))
    print(f"GT boundaries (n={total_gt}):")
    for k in ("exact", "early", "late", "missed"):
        print(f"  {k:10s} {gt_class[k]:5d}  {gt_class[k]/max(total_gt,1):.1%}")
    print(f"  missed breakdown: weak_signal={weak_signal_missed} "
          f"({weak_signal_missed/max(gt_class['missed'],1):.1%} of missed) -- "
          f"no local signal at all, likely no strong visual transition or "
          f"annotation ambiguity (verify visually);  "
          f"signal_present_not_top={signal_present_missed} "
          f"({signal_present_missed/max(gt_class['missed'],1):.1%} of missed) -- "
          f"model produced local signal but it wasn't the top peak/didn't "
          f"clear threshold -- ranking/calibration, not blindness")

    total_pred = sum(peak_class[k] for k in ("duplicate", "false_near_edge", "false_mid_segment", "false_gap"))
    print(f"\npredicted peaks not counted as clean matches (n={total_pred}):")
    for k in ("duplicate", "false_near_edge", "false_mid_segment", "false_gap"):
        if peak_class[k]:
            print(f"  {k:18s} {peak_class[k]:5d}  {peak_class[k]/max(total_pred,1):.1%}")

    print(f"\n=== offset distribution (matched pairs only, n={len(offsets)}) ===")
    if abs_offsets:
        print(f"median absolute error: {statistics.median(abs_offsets):.3f}s  "
              f"mean: {statistics.mean(abs_offsets):.3f}s")
        for t in (0.25, 0.5, 1.0, 2.0):
            frac = sum(o <= t for o in abs_offsets) / len(abs_offsets)
            print(f"  within +-{t}s: {frac:.1%}")
        print(f"early/late split (of matched, non-exact): "
              f"early={gt_class['early']}  late={gt_class['late']}")
        print("read: if 'within +-0.5s' is much higher than the GT-boundary "
              "'exact' rate but strict F1@0.5 is still low, the tolerance-vs-"
              "count mismatch (near-misses vs the exact_tol cutoff) explains "
              "part of the F1 gap -- this is a localization/timing issue, not "
              "a detection failure.")

    # ---------------- reproducibility freeze ----------------
    pred_path = os.path.join(a.out_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for rec in per_recording_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    recording_ids = sorted({r["recording_id"] for r in per_recording_preds})
    write_manifest(pred_path, input_paths=[a.logits],
                   extra={"thr": a.thr, "min_gap": a.min_gap, "tol": a.tol, "exact_tol": a.exact_tol,
                          "n_recordings": len(recording_ids), "recording_ids": recording_ids,
                          "gt_classification": dict(gt_class), "peak_classification": dict(peak_class),
                          "median_abs_offset": statistics.median(abs_offsets) if abs_offsets else None})
    print(f"\nwrote per-recording predictions+classification -> {pred_path}")
    print(f"wrote reproducibility manifest -> {pred_path}.manifest.json "
          f"(git commit, decode config, recording_id split, full classification counts)")


if __name__ == "__main__":
    main()
