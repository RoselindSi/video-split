"""E2 step 2 — train a small supervised temporal boundary head.

Input: [T, D] frozen-ViT feature caches (extract_features.py) + GT segments.
Target: per-timestep soft boundary label (Gaussian bump at each internal segment
junction). A small dilated 1D-Conv head predicts p(boundary | t); at eval we
peak-pick boundaries, split [0, duration] into segments, and score against GT
with the same metrics as eval_hf (f1@iou, boundary_score, count_acc).

Goal of this v1: test whether DENSE supervision (vs GRPO's sparse scalar reward)
(a) beats S1c on boundaries, and especially (b) fixes the multi-segment
under-segmentation (gt=7-9 videos that S1c scored 0 on).

Usage (server):
    python -m src.boundary.train_head --train /tmp/feat_train.pt --val /tmp/feat_val.pt
"""
import argparse, statistics
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------- labels
def soft_labels(times, segments, sigma_s=1.0):
    t = times.numpy()
    T = len(t)
    lab = np.zeros(T, dtype=np.float32)
    segs = sorted(segments, key=lambda x: x[1])
    junctions = [segs[i][1] for i in range(1, len(segs))]   # internal starts
    for b in junctions:
        lab = np.maximum(lab, np.exp(-((t - b) ** 2) / (2 * sigma_s ** 2)))
    return lab, junctions


# ----------------------------------------------------------------- model
class BoundaryHead(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 8)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(4)])
        self.out = nn.Conv1d(d, 1, 1)

    def forward(self, x):                 # x: [B, T, D_in]
        h = self.proj(x.transpose(1, 2))  # [B, d, T]
        for conv, gn in zip(self.blocks, self.norm):
            h = h + torch.relu(gn(conv(h)))
        return self.out(h).squeeze(1)     # [B, T] logits


# ----------------------------------------------------------------- decode + metrics
def decode(prob, times, dur, thresh, min_gap_s):
    peaks = []
    for i in range(len(prob)):
        if prob[i] < thresh:
            continue
        if (i == 0 or prob[i] >= prob[i - 1]) and (i == len(prob) - 1 or prob[i] >= prob[i + 1]):
            peaks.append(i)
    # NMS by min gap in seconds, keep higher prob
    peaks.sort(key=lambda i: -prob[i])
    kept = []
    for i in peaks:
        if all(abs(times[i] - times[j]) >= min_gap_s for j in kept):
            kept.append(i)
    bnds = sorted(times[i].item() for i in kept)
    pts = [0.0] + bnds + [dur]
    return [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]


def decode_topk(prob, times, dur, k_internal, min_gap_s):
    """Take the top-k_internal boundary peaks by probability (adaptive per video).
    k_internal = (#segments - 1). Bypasses any fixed threshold."""
    cand = [i for i in range(len(prob))
            if (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i])
    kept = []
    for i in cand:
        if len(kept) >= k_internal:
            break
        if all(abs(times[i] - times[j]) >= min_gap_s for j in kept):
            kept.append(i)
    bnds = sorted(times[i].item() for i in kept)
    pts = [0.0] + bnds + [dur]
    return [(pts[m], pts[m + 1]) for m in range(len(pts) - 1)]


def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts):
    pairs = []
    used = set()
    for pi, p in enumerate(preds):
        best, bj = 0.0, -1
        for gj, g in enumerate(gts):
            if gj in used:
                continue
            v = iou((p[0], p[1]), (g[1], g[2]))
            if v > best:
                best, bj = v, gj
        if bj >= 0:
            used.add(bj); pairs.append((pi, bj, best))
    return pairs


def eval_item(preds, item):
    gts = item["segments"]; dur = item["duration"]
    pairs = greedy_match(preds, gts)
    m = {"n_pred": len(preds), "n_gt": len(gts)}
    for t in (0.3, 0.5, 0.7):
        tp = sum(1 for _, _, i in pairs if i >= t)
        m[f"f1@{t}"] = 2 * tp / max(2 * tp + (len(preds) - tp) + (len(gts) - tp), 1e-6)
    comp = 0.0
    for pi, gj, i in pairs:
        ps, pe = preds[pi]; _, gs, ge = gts[gj]
        al = (1 - abs(gs / dur - ps / dur)) * (1 - abs(ge / dur - pe / dur)) if dur else 1.0
        comp += i * max(0.0, al)
    m["boundary_score"] = comp / max(len(gts), 1)
    m["count_acc"] = max(0.0, 1 - abs(len(preds) - len(gts)) / len(gts)) if gts else 0.0
    return m


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma_s", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min_gap_s", type=float, default=1.5)
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--oracle_count", action="store_true",
                    help="decode top-(gt_count-1) peaks using GT segment count "
                         "(diagnostic upper bound: isolates localization from count)")
    a = ap.parse_args()

    tr = torch.load(a.train, weights_only=False)
    va = torch.load(a.val, weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = tr[0]["feats"].shape[-1]

    # standardize features with train stats
    allf = torch.cat([x["feats"] for x in tr], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-5
    def prep(x):
        return ((x["feats"] - mu) / sd).unsqueeze(0).to(dev)   # [1,T,D]

    net = BoundaryHead(D).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor(a.pos_weight, device=dev)

    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for x in tr:
            lab, _ = soft_labels(x["times"], x["segments"], a.sigma_s)
            y = torch.tensor(lab, device=dev).unsqueeze(0)
            logit = net(prep(x))
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 50 == 0 or ep == a.epochs - 1:
            print(f"ep {ep+1} train_loss {tot/len(tr):.4f}")

    # ----- eval on val
    net.eval(); agg = []; zeros = []
    with torch.no_grad():
        for x in va:
            prob = torch.sigmoid(net(prep(x)))[0].cpu().numpy()
            if a.oracle_count:
                preds = decode_topk(prob, x["times"], x["duration"],
                                    max(len(x["segments"]) - 1, 0), a.min_gap_s)
            else:
                preds = decode(prob, x["times"], x["duration"], a.thresh, a.min_gap_s)
            m = eval_item(preds, x)
            agg.append(m)
            tag = "  <-- ZERO" if m["f1@0.5"] == 0 else ""
            print(f"{x['video'].split('/')[-1]:22s} pred {m['n_pred']:2d} gt {m['n_gt']:2d} "
                  f"f1@.5 {m['f1@0.5']:.2f} bound {m['boundary_score']:.2f}{tag}")
            if m["n_gt"] >= 6:
                zeros.append(m["f1@0.5"])
    print("\n==== BOUNDARY HEAD (val n=%d) ====" % len(agg))
    for k in ["f1@0.3", "f1@0.5", "f1@0.7", "boundary_score", "count_acc"]:
        print(k.ljust(16), round(statistics.mean([m[k] for m in agg]), 3))
    if zeros:
        print(f"multi-seg videos (gt>=6): mean f1@.5 {statistics.mean(zeros):.3f} "
              f"(S1c scored these ~0 -> key test)")
    print("\nref S1c: f1@0.5 0.452 / boundary 0.491 / count_acc 0.385")


if __name__ == "__main__":
    main()
