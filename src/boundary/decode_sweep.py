"""B2 (full) -- is the boundary bottleneck RANKING (does the model put true
boundaries at high probability at all) or DECODING/CALIBRATION (does the
right threshold/NMS/count exist but we're not using it)?

Produces the single table that answers this, comparing 5 decoders on the
SAME saved logits (train_head_multi.py --save_logits, no retraining):

  Fixed threshold (thr=0.5, min_gap=1.0s)  -- the naive baseline
  Best threshold  (grid search @ min_gap=1.0s)
  Best threshold + gap (grid search over threshold AND min_gap jointly)
  Oracle top-K    (told the TRUE boundary count per video -- upper bound on
                    ranking quality, not a deployable decoder)
  Adaptive-K      (K estimated from a boundary-rate lambda fit on train data:
                    K_i = round(lambda * duration_i), never sees the true
                    count -- IS deployable, tests whether a cheap count
                    estimate closes most of the oracle gap)

for each, reports F1@0.25s / F1@0.5s / F1@1.0s / pred:gt ratio, plus whether
it's an oracle (uses ground truth at decode time) or not.

Also reports, once, decoder-independent RANKING-only diagnostics:
  - frame-level PR-AUC (frame is "positive" if within 0.5s of a true
    boundary; this is decode-free -- pure ranking quality)
  - GT-boundary probability percentile rank (for each true boundary, what
    fraction of all frames in that video have LOWER probability -- 100% =
    perfectly ranked at the top, 50% = indistinguishable from a random
    frame)
  - nearest-peak-to-GT distance distribution at the best threshold decode

Read: if Oracle top-K clears Best-threshold-decode by a wide margin, the
model IS ranking true boundaries highly and the ceiling is in
decode/calibration -- worth trying Adaptive-K, NOT worth changing the
backbone. If Oracle top-K is close to Best-threshold-decode (both mediocre),
even the upper bound is bad -- decode tuning is maxed out, the problem is
representation/ranking (temporal head capacity, features, or needs
hand/language signal -- see the B6 audit for which).

Usage (server, after a train_head_multi.py --save_logits run):
    python -m src.boundary.decode_sweep \
        --logits /tmp/b2_logits.pt \
        --train_data /workspace/tr1/data_recseg/recseg_train.json
"""
import argparse, json, statistics
import numpy as np
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


def peaks_topk(prob, times, k, min_gap):
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


def video_boundary_rate(recseg_path):
    """Mean (internal boundary count) / (video duration) across recordings,
    for the adaptive-K decoder's lambda. Uses the same segments format as
    train_head_multi.py (list of [name, start, end])."""
    rates = []
    for r in json.load(open(recseg_path)):
        segs = sorted(r.get("solution", []), key=lambda s: s[1])
        if len(segs) < 2:
            continue
        ts = sorted({round(s[1], 2) for s in segs} | {round(s[2], 2) for s in segs})
        n_internal = max(len(ts) - 2, 0)
        duration = segs[-1][2] - segs[0][1]
        if duration > 0:
            rates.append(n_internal / duration)
    return statistics.mean(rates) if rates else 0.0


def pr_auc(all_probs, all_labels):
    """Manual PR-AUC (no sklearn dependency): sort by score desc, walk the
    precision/recall curve, trapezoidal integration over recall."""
    order = np.argsort(-np.asarray(all_probs))
    labels = np.asarray(all_labels)[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True,
                    help="output of train_head_multi.py --save_logits")
    ap.add_argument("--train_data", default=None,
                    help="recseg_train.json, for the adaptive-K boundary-rate "
                         "lambda. If omitted, adaptive-K is skipped.")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.logits)
    data = torch.load(a.logits, weights_only=False)
    if not data:
        raise SystemExit("empty logits file -- did the run finish an eval "
                          "epoch with --save_logits set on a single "
                          "variant/seed?")
    print(f"loaded {len(data)} videos")

    # ---------------- decoder-independent ranking diagnostics ----------------
    all_probs, all_labels, pctile_ranks, near_dists = [], [], [], []
    for v in data:
        prob = np.asarray(v["prob"]); times = np.asarray(v["times"])
        gts = v["gt"]
        is_pos = np.zeros(len(times), dtype=int)
        for g in gts:
            is_pos |= (np.abs(times - g) <= 0.5)
        all_probs.extend(prob.tolist()); all_labels.extend(is_pos.tolist())
        for g in gts:
            j = int(np.argmin(np.abs(times - g)))
            pctile = float((prob <= prob[j]).mean())
            pctile_ranks.append(pctile)

    auc = pr_auc(all_probs, all_labels)
    print(f"\n=== ranking-only diagnostics (decode-free) ===")
    print(f"frame-level PR-AUC (positive = within 0.5s of a GT boundary): {auc:.3f}")
    print(f"GT-boundary probability percentile rank (mean over all GT "
          f"boundaries; 100% = ranked above every other frame in its video, "
          f"50% = indistinguishable from a random frame): "
          f"{statistics.mean(pctile_ranks):.1%}  (median {statistics.median(pctile_ranks):.1%})")

    # ---------------- threshold sweep (fixes min_gap=1.0) ----------------
    best_thr, best_thr_f1 = 0.5, -1
    thr_results = {}
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        f1s = []
        for v in data:
            pk = peaks_threshold(v["prob"], v["times"], thr, 1.0)
            f1s.append(bf1(pk, v["gt"], 0.5)[0])
        thr_results[thr] = statistics.mean(f1s)
        if thr_results[thr] > best_thr_f1:
            best_thr_f1, best_thr = thr_results[thr], thr
    print(f"\n=== threshold sweep @ min_gap=1.0 (F1@0.5) ===")
    for thr, f1 in thr_results.items():
        marker = "  <-- best" if thr == best_thr else ""
        print(f"  thr={thr:.2f}  F1={f1:.3f}{marker}")
    neighbors = [thr_results.get(round(best_thr + d, 2)) for d in (-0.05, 0.05)]
    neighbors = [x for x in neighbors if x is not None]
    if neighbors and (best_thr_f1 - min(neighbors)) > 0.05:
        print(f"  WARNING: best threshold is a narrow spike (neighbors "
              f"{neighbors} vs best {best_thr_f1:.3f}) -- decoder may be unstable.")

    # ---------------- threshold x min_gap joint grid ----------------
    best_grid_f1, best_grid_cfg = -1, (best_thr, 1.0)
    grid = {}
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for mg in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            f1s = []
            for v in data:
                pk = peaks_threshold(v["prob"], v["times"], thr, mg)
                f1s.append(bf1(pk, v["gt"], 0.5)[0])
            mf1 = statistics.mean(f1s)
            grid[(thr, mg)] = mf1
            if mf1 > best_grid_f1:
                best_grid_f1, best_grid_cfg = mf1, (thr, mg)
    print(f"\n=== threshold x min_gap joint grid: best = thr={best_grid_cfg[0]} "
          f"min_gap={best_grid_cfg[1]} -> F1={best_grid_f1:.3f} ===")

    # ---------------- multi-tolerance F1 for each decoder ----------------
    def eval_decoder(decode_fn, oracle):
        f1s = {0.25: [], 0.5: [], 1.0: []}
        ratios = []
        for v in data:
            pk = decode_fn(v)
            for tol in f1s:
                f1s[tol].append(bf1(pk, v["gt"], tol)[0])
            ratios.append(len(pk) / max(len(v["gt"]), 1))
        return {tol: statistics.mean(vals) for tol, vals in f1s.items()}, \
               statistics.mean(ratios), oracle

    lam = video_boundary_rate(a.train_data) if a.train_data else None
    decoders = {
        "Fixed threshold (0.5, gap=1.0)": (
            lambda v: peaks_threshold(v["prob"], v["times"], 0.5, 1.0), False),
        f"Best threshold ({best_thr}, gap=1.0)": (
            lambda v: peaks_threshold(v["prob"], v["times"], best_thr, 1.0), False),
        f"Best threshold+gap ({best_grid_cfg[0]}, gap={best_grid_cfg[1]})": (
            lambda v: peaks_threshold(v["prob"], v["times"], *best_grid_cfg), False),
        "Oracle top-K (true count)": (
            lambda v: peaks_topk(v["prob"], v["times"], len(v["gt"]), best_grid_cfg[1]), True),
    }
    if lam is not None:
        print(f"\nadaptive-K lambda (boundaries/sec, from train_data): {lam:.4f}")
        def adaptive_decode(v):
            duration = v["times"][-1] - v["times"][0] if len(v["times"]) > 1 else 0
            k = max(1, round(lam * duration))
            return peaks_topk(v["prob"], v["times"], k, best_grid_cfg[1])
        decoders["Adaptive-K (estimated count)"] = (adaptive_decode, False)

    print(f"\n=== FINAL COMPARISON TABLE ===")
    print(f"{'Decoder':38s} {'F1@0.25':>8s} {'F1@0.5':>8s} {'F1@1.0':>8s} "
          f"{'Pred/GT':>8s} {'Oracle':>7s}")
    rows = {}
    for name, (fn, oracle) in decoders.items():
        f1s, ratio, is_oracle = eval_decoder(fn, oracle)
        rows[name] = (f1s, ratio, is_oracle)
        print(f"{name:38s} {f1s[0.25]:8.3f} {f1s[0.5]:8.3f} {f1s[1.0]:8.3f} "
              f"{ratio:8.2f} {'Yes' if is_oracle else 'No':>7s}")

    # ---------------- nearest-peak distance @ best threshold+gap ----------------
    dists = []
    for v in data:
        pk = peaks_threshold(v["prob"], v["times"], *best_grid_cfg)
        for p in pk:
            if v["gt"]:
                dists.append(min(abs(p - g) for g in v["gt"]))
    if dists:
        print(f"\nnearest-peak-to-GT distance @ best threshold+gap: "
              f"mean={statistics.mean(dists):.2f}s median={statistics.median(dists):.2f}s")

    # ---------------- verdict ----------------
    best_deploy_f1 = max(rows[n][0][0.5] for n in rows if not rows[n][2])
    oracle_f1 = max((rows[n][0][0.5] for n in rows if rows[n][2]), default=None)
    print(f"\n=== VERDICT ===")
    if oracle_f1 is not None:
        gap = oracle_f1 - best_deploy_f1
        print(f"best deployable F1@0.5={best_deploy_f1:.3f}  oracle-count F1@0.5={oracle_f1:.3f}  gap={gap:.3f}")
        if gap > 0.05:
            print("Oracle clears deployable decoding by a real margin -> ranking is "
                  "usable, the ceiling is calibration/count estimation. Try "
                  "Adaptive-K seriously (if listed above, check how close it got "
                  "to oracle); do NOT change the backbone yet.")
        else:
            print("Oracle is close to the best deployable decoder -> decode tuning "
                  "is maxed out. The problem is representation/ranking, not "
                  "calibration. Proceed to the B6 human audit to see WHICH "
                  "boundary types are failing before picking TCN vs language vs "
                  "hand-object signal.")
    f1_25, f1_50, f1_10 = (rows[f"Best threshold+gap ({best_grid_cfg[0]}, gap={best_grid_cfg[1]})"][0][t]
                           for t in (0.25, 0.5, 1.0))
    print(f"\nlocalization read: F1@0.25={f1_25:.3f} F1@0.5={f1_50:.3f} F1@1.0={f1_10:.3f}")
    if f1_10 - f1_25 > 0.1:
        print("F1@1.0 >> F1@0.25 -> model knows roughly WHERE but not precisely WHEN "
              "(localization/temporal-head issue).")
    elif f1_10 < 0.35:
        print("all three tolerances low -> model isn't reliably finding semantic "
              "boundaries at all, not just imprecise timing.")
    else:
        print("F1@0.25 close to F1@0.5/1.0 -> localization isn't the main issue; "
              "the gap is missed/false detections, not offset timing.")


if __name__ == "__main__":
    main()
