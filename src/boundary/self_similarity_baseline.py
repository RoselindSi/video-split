"""Training-free self-similarity boundary baseline (P0 deliverable #1).

Uses the ALREADY-CACHED frozen Qwen3-VL features (no re-extraction, no Qwen call).
Tests whether Qwen features carry a boundary signal on their own, via change
scores between consecutive (or windowed) frame features:

    adjacent:  c_t = 1 - cos(h_{t-1}, h_t)
    windowed:  c_t = 1 - cos(mean(h_{t-w:t}), mean(h_{t:t+w}))

then smoothing -> peak detection -> min-distance NMS -> threshold search on val.
Metric is BOUNDARY-level (predicted peak time vs GT boundary time within a
tolerance), not segment IoU -- the right question here is "is the boundary
signal present in the features".

NOTE: the cached features were blur-filtered (th_blur=100 dropped ~33% of frames
on this data), so surviving frames may already be missing boundary frames; this
baseline runs on what survived. The filter_audit.py deliverable quantifies that.

Usage (server):
    python -m src.boundary.self_similarity_baseline --val /workspace/tr1/data_recseg/feat_val.pt
"""
import argparse, statistics
import numpy as np
import torch


def change_scores(feats, w):
    """Change score = L2 distance on per-dim STANDARDIZED features.

    Raw Qwen features carry a huge common component (norm ~4900) that dominates
    cosine (adjacent cos ~1.0, signal washed out). Standardizing (subtract mean,
    divide std -- exactly what train_head feeds the head) removes it; L2 distance
    then reflects the real frame-to-frame change that the supervised head sees.
    """
    f = feats.float().numpy()
    f = (f - f.mean(0)) / (f.std(0) + 1e-5)
    T = f.shape[0]
    adj = np.zeros(T)
    adj[1:] = np.linalg.norm(f[1:] - f[:-1], axis=1)
    win = np.zeros(T)
    for t in range(1, T):
        a = f[max(0, t - w):t].mean(0)
        b = f[t:min(T, t + w)].mean(0)
        win[t] = float(np.linalg.norm(b - a))
    return adj, win


def smooth(x, k=3):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def gt_boundaries(segments):
    ts = set()
    for _, s, e in segments:
        ts.add(round(s, 2)); ts.add(round(e, 2))
    b = sorted(ts)
    return b[1:-1] if len(b) > 2 else b       # drop video-edge endpoints


def peak_times_topk(score, times, k, min_gap_s):
    """Top-k local maxima by score (threshold-free; k from GT count)."""
    cand = [i for i in range(len(score))
            if (i == 0 or score[i] >= score[i - 1])
            and (i == len(score) - 1 or score[i] >= score[i + 1])]
    cand.sort(key=lambda i: -score[i])
    kept = []
    for i in cand:
        if len(kept) >= k:
            break
        if all(abs(times[i] - times[j]) >= min_gap_s for j in kept):
            kept.append(i)
    return sorted(times[i] for i in kept)


def match_f1(preds, gts, tol):
    used = set(); tp = 0
    for p in preds:
        best, bj = tol + 1, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            d = abs(p - g)
            if d < best:
                best, bj = d, j
        if bj >= 0 and best <= tol:
            used.add(bj); tp += 1
    prec = tp / max(len(preds), 1)
    rec = tp / max(len(gts), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--w", type=int, default=4, help="window for windowed change score")
    ap.add_argument("--smooth_k", type=int, default=3)
    ap.add_argument("--min_gap_s", type=float, default=1.0)
    a = ap.parse_args()

    va = torch.load(a.val, weights_only=False)
    va = [x for x in va if x["feats"].dim() == 2 and x["feats"].shape[0] > 2]

    pre = []
    all_adj, all_win = [], []
    for x in va:
        adj, win = change_scores(x["feats"], a.w)
        all_adj.append(adj[1:]); all_win.append(win[1:])
        pre.append({"adj": smooth(adj, a.smooth_k), "win": smooth(win, a.smooth_k),
                    "times": x["times"].numpy(), "gts": gt_boundaries(x["segments"])})

    # reveal the true signal scale (score = L2 on standardized feats, not cosine)
    for name, arr in (("adjacent L2", np.concatenate(all_adj)),
                      ("windowed L2", np.concatenate(all_win))):
        ps = {p: round(float(np.percentile(arr, p)), 4) for p in (50, 90, 99)}
        print(f"signal scale [{name}]: p50 {ps[50]} p90 {ps[90]} p99 {ps[99]}")
    print()

    # threshold-free: take top-(GT count) peaks per video; also a random baseline
    for sig in ("adj", "win"):
        f5s, f10s, r5s = [], [], []
        for pr in pre:
            k = len(pr["gts"])
            pk = peak_times_topk(pr[sig], pr["times"], k, a.min_gap_s)
            f5, _, r5 = match_f1(pk, pr["gts"], 0.5)
            f10, _, _ = match_f1(pk, pr["gts"], 1.0)
            f5s.append(f5); f10s.append(f10); r5s.append(r5)
        print(f"[{sig} top-k] F1@0.5s {statistics.mean(f5s):.3f} | "
              f"F1@1.0s {statistics.mean(f10s):.3f} | R@0.5s {statistics.mean(r5s):.3f}")

    # random-peak control: how much does chance get at these densities?
    rng = np.random.default_rng(0); rf = []
    for pr in pre:
        k = len(pr["gts"]); T = len(pr["times"])
        if T < 2:
            continue
        idx = rng.choice(T, size=min(k, T), replace=False)
        pk = sorted(pr["times"][i] for i in idx)
        f5, _, _ = match_f1(pk, pr["gts"], 0.5); rf.append(f5)
    print(f"[random top-k control] F1@0.5s {statistics.mean(rf):.3f}")
    print("\nNOTE: top-k uses GT boundary count -> this is an ORACLE-COUNT feature "
          "probe (for comparing feature variants), NOT a deployable baseline.")
    print("Standardization here is PER-VIDEO; train_head uses TRAIN-GLOBAL mu/sd -- "
          "keep this in mind when comparing self-sim vs head numbers.")


if __name__ == "__main__":
    main()
