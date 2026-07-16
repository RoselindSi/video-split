"""Region attribution probe for multi-region pooled features (nearly free).

The multi cache stores 5760 = 5 x 1152 = concat[global, left, right, center,
spatial_max] per frame. This splits it back and asks, WITHOUT re-extraction:
  - which single region carries the most boundary signal?
  - is a max-over-regions change score (transitions live in ONE small region ->
    summing regions dilutes it) better than the concatenated-L2?
  - do we actually need all 5760 dims, or does global+best-local suffice?

Change score per region r = L2 on per-dim standardized features (same as
self_similarity_baseline). Aggregations compared:
  per-region, mean-over-regions, max-over-regions, global+best-local, concat(all).

Usage (server):
    python -m src.boundary.region_attribution --val /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt
"""
import argparse, statistics
import numpy as np
import torch

REGION_NAMES = ["global", "left", "right", "center", "spatial_max"]


def standardize(f):
    return (f - f.mean(0)) / (f.std(0) + 1e-5)


def adj_change(f):                       # [T,D] -> [T] adjacent L2 on standardized
    z = standardize(f.float().numpy())
    c = np.zeros(z.shape[0])
    c[1:] = np.linalg.norm(z[1:] - z[:-1], axis=1)
    return c


def win_change(f, w=4):
    z = standardize(f.float().numpy()); T = z.shape[0]
    c = np.zeros(T)
    for t in range(1, T):
        a = z[max(0, t - w):t].mean(0); b = z[t:min(T, t + w)].mean(0)
        c[t] = float(np.linalg.norm(b - a))
    return c


def gt_boundaries(segments):
    ts = set()
    for _, s, e in segments:
        ts.add(round(s, 2)); ts.add(round(e, 2))
    b = sorted(ts)
    return b[1:-1] if len(b) > 2 else b


def topk_peaks(score, times, k, min_gap_s):
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


def f1(preds, gts, tol):
    used = set(); tp = 0
    for p in preds:
        best, bj = tol + 1, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            if abs(p - g) < best:
                best, bj = abs(p - g), j
        if bj >= 0 and best <= tol:
            used.add(bj); tp += 1
    prec = tp / max(len(preds), 1); rec = tp / max(len(gts), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def eval_signal(pre, sig_key):
    f5 = [f1(topk_peaks(p[sig_key], p["times"], len(p["gts"]), 1.0), p["gts"], 0.5)
          for p in pre]
    f10 = [f1(topk_peaks(p[sig_key], p["times"], len(p["gts"]), 1.0), p["gts"], 1.0)
           for p in pre]
    return statistics.mean(f5), statistics.mean(f10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--use", choices=["adj", "win"], default="win")
    a = ap.parse_args()

    va = torch.load(a.val, weights_only=False)
    va = [x for x in va if x["feats"].dim() == 2 and x["feats"].shape[0] > 2]
    D5 = va[0]["feats"].shape[-1]
    assert D5 % 5 == 0, f"expected 5x concat, got {D5}"
    D = D5 // 5
    chg = adj_change if a.use == "adj" else (lambda f: win_change(f, a.w))

    pre = []
    for x in va:
        f = x["feats"]                                  # [T, 5D]
        regs = [f[:, r * D:(r + 1) * D] for r in range(5)]
        per = [chg(r) for r in regs]                    # 5 x [T]
        rec = {"times": x["times"].numpy(), "gts": gt_boundaries(x["segments"])}
        for ri, name in enumerate(REGION_NAMES):
            rec[name] = per[ri]
        rec["mean5"] = np.mean(per, axis=0)
        rec["max5"] = np.max(per, axis=0)               # max-over-regions
        rec["concat"] = chg(f)                          # L2 on full 5760
        # global + best-local decided after seeing per-region below
        rec["_per"] = per
        pre.append(rec)

    print(f"=== REGION ATTRIBUTION ({a.use} change, n={len(pre)}) ===")
    print(f"{'signal':16s} {'F1@0.5':>8s} {'F1@1.0':>8s}")
    for key in REGION_NAMES + ["mean5", "max5", "concat"]:
        f5, f10 = eval_signal(pre, key)
        print(f"{key:16s} {f5:8.3f} {f10:8.3f}")

    # global + each single local, and global+max(local regions)
    print("--- combos ---")
    for ri, name in enumerate(REGION_NAMES):
        if name == "global":
            continue
        for p in pre:
            p["_gl"] = np.maximum(p["global"], p[name])
        f5, f10 = eval_signal(pre, "_gl")
        print(f"{'max(global,'+name+')':16s} {f5:8.3f} {f10:8.3f}")
    for p in pre:
        loc = np.max([p["left"], p["right"], p["center"], p["spatial_max"]], axis=0)
        p["_gml"] = np.maximum(p["global"], loc)
    f5, f10 = eval_signal(pre, "_gml")
    print(f"{'max(global,anyloc)':16s} {f5:8.3f} {f10:8.3f}")
    print("\nref: single global 1152 was win F1@0.5 0.195; full concat 0.207; "
          "head oracle 0.331. Pick most compact signal >= concat.")


if __name__ == "__main__":
    main()
