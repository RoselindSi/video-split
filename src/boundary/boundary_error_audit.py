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


def all_local_maxima(prob, times):
    return [i for i in range(len(prob))
            if (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]


def peaks_threshold(prob, times, thr, min_gap):
    cand = [i for i in all_local_maxima(prob, times) if prob[i] >= thr]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def peaks_threshold_instrumented(prob, times, thr, min_gap):
    """Same greedy NMS as peaks_threshold, but also returns which kept peak
    suppressed each rejected above-threshold candidate. Under this decode
    algorithm (fixed threshold + greedy min_gap NMS, no global proposal
    budget) there are exactly two ways a real above-threshold local maximum
    can fail to survive: it never clears --thr, or it's suppressed by a
    higher-scoring peak within --min_gap. There's no third "outranked by a
    far-away peak" failure mode here (that only exists under a budgeted
    top-K decoder) -- see boundary_candidate_recall.py for that question."""
    cand = [i for i in all_local_maxima(prob, times) if prob[i] >= thr]
    cand.sort(key=lambda i: -prob[i])
    kept = []
    suppressed_by = {}
    for i in cand:
        blocker = next((j for j in kept if abs(times[i] - times[j]) < min_gap), None)
        if blocker is None:
            kept.append(i)
        else:
            suppressed_by[i] = blocker
    return kept, suppressed_by


def local_max_prob(prob, times, g, window=1.0):
    vals = [prob[i] for i, t in enumerate(times) if abs(t - g) <= window]
    return max(vals) if vals else 0.0


def nearest_local_maximum(prob, times, local_maxima_idx, g, window=1.0):
    """Best (highest-score) local maximum within `window` of g, or None."""
    near = [i for i in local_maxima_idx if abs(times[i] - g) <= window]
    return max(near, key=lambda i: prob[i]) if near else None


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
    missed_subtype = Counter()  # fine-grained split of signal_present_not_top
    offsets = []  # signed, matched only
    missed_nearest_dists = []  # unbounded nearest-pred distance for MISSED GTs only
    weak_signal_missed = 0; signal_present_missed = 0
    per_recording_preds = []

    for v in data:
        prob, times, gts = v["prob"], v["times"], v["gt"]
        segs = sorted(v["segments"], key=lambda s: s[1])
        overall_median = statistics.median(prob)
        preds = peaks_threshold(prob, times, a.thr, a.min_gap)
        local_maxima_idx = all_local_maxima(prob, times)
        kept_idx, suppressed_by = peaks_threshold_instrumented(prob, times, a.thr, a.min_gap)

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
                subtype = None
                if kind == "signal_present_not_top":
                    best = nearest_local_maximum(prob, times, local_maxima_idx, g)
                    if best is None:
                        subtype = "not_a_local_maximum"
                    elif prob[best] < a.thr:
                        subtype = "below_threshold"
                    elif best in suppressed_by:
                        subtype = "suppressed_by_min_gap"
                    elif best in kept_idx:
                        # this local max DID survive NMS and become a final
                        # prediction, but its own distance to g exceeds the
                        # strict matching tol (or a nearer GT claimed it
                        # first) -- a tol/assignment interaction, not a
                        # decode failure per se
                        subtype = "kept_but_outside_matching_tol"
                    else:
                        subtype = "unexplained"
                    missed_subtype[subtype] += 1
                if kind == "weak_signal":
                    weak_signal_missed += 1
                else:
                    signal_present_missed += 1
                gt_class["missed"] += 1
                # unbounded nearest-pred distance -- tells us whether a LOOSER
                # tolerance would actually rescue this miss, unlike the
                # matched-pair offset stats below which are capped at `tol` by
                # construction (matched pairs can never be >tol apart, so
                # their own distribution can't show "near misses just outside
                # tol" -- this is the only correct way to measure that).
                nearest_dist = min((abs(p - g) for p in preds), default=None)
                missed_nearest_dists.append(nearest_dist)
                rec = {"gt_time": g, "status": "missed", "signal": kind, "subtype": subtype,
                      "local_max_prob": round(lp, 3), "video_median_prob": round(overall_median, 3),
                      "nearest_pred_dist": round(nearest_dist, 3) if nearest_dist is not None else None}
                if kind == "signal_present_not_top" and subtype in ("below_threshold", "suppressed_by_min_gap",
                                                                     "kept_but_outside_matching_tol"):
                    best = nearest_local_maximum(prob, times, local_maxima_idx, g)
                    rec["local_peak_score"] = round(float(prob[best]), 3)
                    rec["local_peak_time"] = round(float(times[best]), 3)
                    rec["threshold_margin"] = round(float(prob[best]) - a.thr, 3)
                    if subtype == "suppressed_by_min_gap":
                        blocker = suppressed_by[best]
                        rec["suppressing_peak_time"] = round(float(times[blocker]), 3)
                        rec["suppressing_peak_score"] = round(float(prob[blocker]), 3)
                gt_records.append(rec)
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
    print(f"  signal_present_not_top subtype breakdown (n={signal_present_missed}) -- "
          f"the ONLY two decode-level failure modes for an above-threshold "
          f"local max under this fixed-threshold+min_gap-NMS decoder are "
          f"below_threshold and suppressed_by_min_gap (no global proposal "
          f"budget exists here, so there's no third 'outranked by a far-away "
          f"peak' mode -- that only applies to a budgeted top-K decoder, see "
          f"boundary_candidate_recall.py):")
    for st, c in missed_subtype.most_common():
        print(f"    {st:28s} {c:5d}  {c/max(signal_present_missed,1):.1%}")

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
        print("NOTE: 'within +-Xs' above is capped at 100% once X>=tol by "
              "construction (a pair can't be 'matched' at all if it's farther "
              "than tol apart) -- it can only show whether near-misses cluster "
              "just under exact_tol, NOT whether missed GTs have a near-miss "
              "just OUTSIDE tol. See the rescue-rate section below for that.")

    if missed_nearest_dists:
        finite = [d for d in missed_nearest_dists if d is not None]
        print(f"\n=== would a LOOSER tolerance rescue missed GT boundaries? "
              f"(n_missed={len(missed_nearest_dists)}, has a pred at all in "
              f"this video: {len(finite)}) ===")
        print(f"nearest predicted peak distance for MISSED GTs: "
              f"median={statistics.median(finite):.2f}s  mean={statistics.mean(finite):.2f}s")
        for t in (0.75, 1.0, 1.5, 2.0, 3.0):
            rescued = sum(d <= t for d in finite)
            print(f"  would be rescued at tol={t}s: {rescued}/{len(missed_nearest_dists)} "
                  f"= {rescued/len(missed_nearest_dists):.1%}")
        print("read: if rescue rate stays low even at tol=2-3s, missed boundaries "
              "genuinely have NO nearby prediction (consistent with the "
              "signal_present_not_top finding -- the model's peak, if any, is "
              "elsewhere, not just slightly mistimed) -- confirms this is a "
              "ranking/detection problem, not a tolerance-calibration problem, "
              "and tightens the case against further threshold/min_gap tuning.")

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
