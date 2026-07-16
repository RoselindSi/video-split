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
    f = torch.nn.functional.normalize(feats.float(), dim=-1).numpy()  # [T,D]
    T = f.shape[0]
    adj = np.zeros(T)
    adj[1:] = 1 - np.sum(f[1:] * f[:-1], axis=1)
    win = np.zeros(T)
    for t in range(1, T):
        a = f[max(0, t - w):t].mean(0)
        b = f[t:min(T, t + w)].mean(0)
        an, bn = a / (np.linalg.norm(a) + 1e-8), b / (np.linalg.norm(b) + 1e-8)
        win[t] = 1 - float(np.dot(an, bn))
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


def peak_times(score, times, thr, min_gap_s):
    cand = [i for i in range(len(score))
            if score[i] >= thr
            and (i == 0 or score[i] >= score[i - 1])
            and (i == len(score) - 1 or score[i] >= score[i + 1])]
    cand.sort(key=lambda i: -score[i])
    kept = []
    for i in cand:
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
    for x in va:
        adj, win = change_scores(x["feats"], a.w)
        pre.append({"adj": smooth(adj, a.smooth_k), "win": smooth(win, a.smooth_k),
                    "times": x["times"].numpy(), "gts": gt_boundaries(x["segments"])})

    for sig in ("adj", "win"):
        best = None
        for thr in np.linspace(0.02, 0.5, 25):
            f5s, p5s, r5s, f1s = [], [], [], []
            for pr in pre:
                pk = peak_times(pr[sig], pr["times"], thr, a.min_gap_s)
                f5, p5, r5 = match_f1(pk, pr["gts"], 0.5)
                f10, _, _ = match_f1(pk, pr["gts"], 1.0)
                f5s.append(f5); p5s.append(p5); r5s.append(r5); f1s.append(f10)
            m = statistics.mean(f5s)
            if best is None or m > best[0]:
                best = (m, thr, statistics.mean(p5s), statistics.mean(r5s),
                        statistics.mean(f1s))
        f5, thr, p5, r5, f10 = best
        print(f"[{sig}] best thr {thr:.3f} | F1@0.5s {f5:.3f} "
              f"(P {p5:.3f} R {r5:.3f}) | F1@1.0s {f10:.3f}")
    print("\n(boundary-level detection metric; GT boundaries = internal segment "
          "transitions. Features are blur-filtered -- see filter_audit.py)")


if __name__ == "__main__":
    main()
