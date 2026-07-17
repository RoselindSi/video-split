"""B2 -- is the boundary problem SCORING (ranking) or DECODING (calibration)?

pred/GT~=0.99 but F1~0.30 = right COUNT, wrong POSITION -- ambiguous between
"the model doesn't rank true boundaries highly" (feature/temporal problem) and
"the ranking is fine but threshold/NMS picks the wrong peaks" (decode problem).
This script takes ONE saved set of val logits (from train_head_multi.py
--save_logits) and answers it directly:

  1. threshold sweep (0.20-0.70): full PR curve at the CURRENT min_gap/sigma.
  2. oracle-count top-K: decode with the TRUE number of GT boundaries per
     video (upper bound on ranking quality, independent of any threshold).
  3. min_gap sweep {0.5,1,1.5,2} x smoothing {none,0.25,0.5,1.0}s: does NMS
     width or curve smoothing change things.
  4. per-boundary diagnostics: predicted peaks within +-1s of each GT
     (duplicate-detection), false peaks >1s from any GT, missed GT count,
     peak-to-nearest-GT distance histogram.

Verdict:
  oracle-count F1 >> best threshold-decode F1  -> ranking is fine, problem is
    calibration/decode -- freeze a decoder and move on (B4+).
  oracle-count F1 is ALSO ~0.30 -> the model isn't ranking true boundaries
    above false ones even when told the count -- problem is feature/temporal
    modeling, decode tuning won't fix it (this rules out B2 being the answer).

NOTE: "adaptive-count" decode (from the B2 plan) needs a trained COUNT head,
which train_head_multi.py doesn't have -- not implemented here. If oracle-count
clears threshold-decode by a wide margin, that's the next lever (predict count,
then top-K by that), not more threshold tuning.

Usage (server, after a run_head_multi.py --save_logits pass):
    python -m src.boundary.decode_sweep --logits /tmp/b2_logits.pt
"""
import argparse, statistics
import numpy as np
import torch


def smooth(prob, times, sigma):
    if sigma <= 0:
        return prob
    t = np.asarray(times)
    out = np.zeros_like(prob)
    # small dense sequences (~1000-1500 frames) -- O(T^2) gaussian is fine here
    for i in range(len(t)):
        w = np.exp(-((t - t[i]) ** 2) / (2 * sigma ** 2))
        out[i] = np.sum(w * prob) / np.sum(w)
    return out


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


def peaks_oracle_count(prob, times, k, min_gap):
    cand = [i for i in range(len(prob))
            if (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
        if len(kept) >= k:
            break
        if all(abs(times[i] - times[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def bf1(preds, gts, tol):
    used = set(); tp = 0
    for p in preds:
        best, bj = tol + 1, -1
        for j, g in enumerate(gts):
            if j not in used and abs(p - g) < best:
                best, bj = abs(p - g), j
        if bj >= 0 and best <= tol:
            used.add(bj); tp += 1
    pr = tp / max(len(preds), 1); rc = tp / max(len(gts), 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return f1, pr, rc


def peak_diagnostics(preds, gts, near_window=1.0, tol=0.5):
    near_counts, dists, false_peaks, missed = [], [], 0, 0
    used = set()
    for g in gts:
        c = sum(1 for p in preds if abs(p - g) <= near_window)
        near_counts.append(c)
        if not any(abs(p - g) <= tol for p in preds):
            missed += 1
    for p in preds:
        if gts:
            d = min(abs(p - g) for g in gts)
        else:
            d = float("inf")
        dists.append(d)
        if d > 1.0:
            false_peaks += 1
    return near_counts, dists, false_peaks, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True,
                    help="output of train_head_multi.py --save_logits")
    ap.add_argument("--tol", type=float, default=0.5, help="F1 match tolerance (s)")
    a = ap.parse_args()

    data = torch.load(a.logits, weights_only=False)
    if not data:
        raise SystemExit("empty logits file -- did the run finish an eval epoch "
                          "with --save_logits set on a single variant/seed?")
    print(f"loaded {len(data)} videos")

    # 1) threshold sweep @ default min_gap=1.0, no smoothing
    print("\n=== threshold sweep (min_gap=1.0, no smoothing) ===")
    print(f"{'thr':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'pred/gt':>8s}")
    best_thr_f1, best_thr = -1, None
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        f1s, ps, rs, ratios = [], [], [], []
        for v in data:
            pk = peaks_threshold(v["prob"], v["times"], thr, 1.0)
            f1, pr, rc = bf1(pk, v["gt"], a.tol)
            f1s.append(f1); ps.append(pr); rs.append(rc)
            ratios.append(len(pk) / max(len(v["gt"]), 1))
        mf1 = statistics.mean(f1s)
        print(f"{thr:5.2f} {statistics.mean(ps):6.2f} {statistics.mean(rs):6.2f} "
              f"{mf1:6.3f} {statistics.mean(ratios):8.2f}")
        if mf1 > best_thr_f1:
            best_thr_f1, best_thr = mf1, thr

    # 2) oracle-count top-K (upper bound on ranking quality)
    oracle_f1s = []
    for v in data:
        pk = peaks_oracle_count(v["prob"], v["times"], len(v["gt"]), 1.0)
        oracle_f1s.append(bf1(pk, v["gt"], a.tol)[0])
    oracle_f1 = statistics.mean(oracle_f1s)
    print(f"\noracle-count top-K F1 (min_gap=1.0): {oracle_f1:.3f}  "
          f"(best threshold-decode F1: {best_thr_f1:.3f} @ thr={best_thr})")

    # 3) min_gap x smoothing grid (oracle-count decode, isolates NMS/smoothing
    #    from threshold calibration)
    print("\n=== min_gap x smoothing grid (oracle-count decode) ===")
    print(f"{'min_gap':>8s} {'smooth':>7s} {'F1':>6s}")
    grid_best_f1, grid_best_cfg = -1, None
    for min_gap in (0.5, 1.0, 1.5, 2.0):
        for sig in (0.0, 0.25, 0.5, 1.0):
            f1s = []
            for v in data:
                prob = smooth(np.asarray(v["prob"]), v["times"], sig) if sig > 0 else np.asarray(v["prob"])
                pk = peaks_oracle_count(prob.tolist(), v["times"], len(v["gt"]), min_gap)
                f1s.append(bf1(pk, v["gt"], a.tol)[0])
            mf1 = statistics.mean(f1s)
            print(f"{min_gap:8.1f} {sig:7.2f} {mf1:6.3f}")
            if mf1 > grid_best_f1:
                grid_best_f1, grid_best_cfg = mf1, (min_gap, sig)

    # 4) per-boundary diagnostics at the best threshold config found above
    all_near, all_dists, fp_total, miss_total, gt_total = [], [], 0, 0, 0
    for v in data:
        pk = peaks_threshold(v["prob"], v["times"], best_thr, 1.0)
        near, dists, fp, missed = peak_diagnostics(pk, v["gt"], tol=a.tol)
        all_near += near; all_dists += dists
        fp_total += fp; miss_total += missed; gt_total += len(v["gt"])
    print(f"\n=== per-boundary diagnostics @ thr={best_thr} ===")
    print(f"mean predicted peaks within +-1s of each GT: "
          f"{statistics.mean(all_near) if all_near else 0:.2f} "
          f"(>1.3 => duplicate peaks on same boundary, widen min_gap/narrow sigma_s)")
    print(f"false peaks (>1s from any GT): {fp_total} "
          f"({fp_total / max(sum(len(v['prob']) for v in data), 1) * 1000:.2f} per 1000 frames)")
    print(f"missed GT boundaries (no pred within {a.tol}s): {miss_total}/{gt_total} "
          f"= {miss_total/max(gt_total,1):.1%}")
    if all_dists:
        finite = [d for d in all_dists if d != float('inf')]
        if finite:
            print(f"peak-to-nearest-GT distance: mean={statistics.mean(finite):.2f}s "
                  f"median={statistics.median(finite):.2f}s")

    print(f"\n=== VERDICT ===")
    print(f"oracle-count F1={oracle_f1:.3f} vs best threshold-decode F1={best_thr_f1:.3f} "
          f"(gap={oracle_f1-best_thr_f1:.3f}); best min_gap/smoothing grid F1={grid_best_f1:.3f} "
          f"@ {grid_best_cfg}")
    if oracle_f1 - best_thr_f1 > 0.05:
        print("oracle-count clearly beats threshold decode -> ranking is OK, "
              "ceiling is in calibration/decode. Freeze best decoder config, "
              "move to B4 (regularize) or a trained count head, NOT more "
              "feature engineering.")
    else:
        print("oracle-count is close to threshold decode -> even given the true "
              "count, the model isn't ranking true boundaries above false ones. "
              "Decode tuning has hit its ceiling; the problem is in the "
              "feature/temporal modeling (matches the B3 negative result). "
              "Move to B6 FP/FN audit to see WHICH boundary types are failing.")


if __name__ == "__main__":
    main()
