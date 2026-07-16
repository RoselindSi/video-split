"""E2 step 2 — supervised temporal boundary head (end-to-end, count + endpoints learned).

Input: [T, D] frozen-ViT feature caches (extract_features.py) + GT segments.
Three learned outputs (all densely/directly supervised, no RL, no oracle):
  - boundary  p(internal junction | t)  -- soft Gaussian target at segment joins
  - actionness p(inside active region | t) -- 1 between first_start and last_end
  - count      #segments                 -- pooled regression
Decode: active region [lo,hi] from actionness, k from count, top-(k-1) boundary
peaks within [lo,hi]. --oracle_count keeps the diagnostic upper bound.

Goal: test whether dense supervision beats S1c end-to-end AND fixes the
multi-segment under-segmentation that sparse-reward GRPO could not.

Usage (server):
    python -m src.boundary.train_head --train /tmp/feat_train.pt --val /tmp/feat_val.pt
    python -m src.boundary.train_head ... --oracle_count      # upper-bound diagnostic
"""
import argparse, statistics
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------- labels
def soft_boundary(times, segments, sigma_s=1.0):
    t = times.numpy(); lab = np.zeros(len(t), dtype=np.float32)
    segs = sorted(segments, key=lambda x: x[1])
    for b in [segs[i][1] for i in range(1, len(segs))]:      # internal joins
        lab = np.maximum(lab, np.exp(-((t - b) ** 2) / (2 * sigma_s ** 2)))
    return lab


def actionness(times, segments):
    t = times.numpy(); segs = sorted(segments, key=lambda x: x[1])
    lo, hi = segs[0][1], segs[-1][2]
    return ((t >= lo) & (t <= hi)).astype(np.float32)


# ----------------------------------------------------------------- model
class BoundaryHead(nn.Module):
    def __init__(self, d_in, d=256):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(d, d, 5, padding=2 * r, dilation=r) for r in (1, 2, 4, 8)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(4)])
        self.bound = nn.Conv1d(d, 1, 1)
        self.action = nn.Conv1d(d, 1, 1)
        self.count = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):                 # x: [B, T, D_in]
        h = self.proj(x.transpose(1, 2))
        for conv, gn in zip(self.blocks, self.norm):
            h = h + torch.relu(gn(conv(h)))
        b = self.bound(h).squeeze(1)      # [B, T]
        a = self.action(h).squeeze(1)     # [B, T]
        c = self.count(h.mean(-1)).squeeze(-1)   # [B]
        return b, a, c


# ----------------------------------------------------------------- decode + metrics
def peaks_topk(prob, times, lo, hi, k_internal, min_gap_s):
    cand = [i for i in range(len(prob))
            if lo < times[i] < hi
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i])
    kept = []
    for i in cand:
        if len(kept) >= k_internal:
            break
        if all(abs(times[i] - times[j]) >= min_gap_s for j in kept):
            kept.append(i)
    bnds = sorted(times[i].item() for i in kept)
    pts = [lo] + bnds + [hi]
    return [(pts[m], pts[m + 1]) for m in range(len(pts) - 1)]


def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1]); inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts):
    pairs = []; used = set()
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
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma_s", type=float, default=1.0)
    ap.add_argument("--min_gap_s", type=float, default=2.0)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--w_action", type=float, default=1.0)
    ap.add_argument("--w_count", type=float, default=0.1)
    ap.add_argument("--oracle_count", action="store_true",
                    help="upper-bound: use GT count + GT active region instead of learned")
    ap.add_argument("--count_mode", choices=["head", "peaks"], default="head",
                    help="head=pooled count regression; peaks=count boundary peaks "
                         ">count_thresh within active region (data-free)")
    ap.add_argument("--count_thresh", type=float, default=0.5)
    a = ap.parse_args()

    tr = torch.load(a.train, weights_only=False)
    va = torch.load(a.val, weights_only=False)

    def drop_empty(items, name):
        ok = [x for x in items if x["feats"].dim() == 2 and x["feats"].shape[0] > 0]
        bad = [x.get("recording_id", x.get("video")) for x in items if x not in ok]
        if bad:
            print(f"WARNING dropping {len(bad)} empty-feature {name} items: {bad}")
        return ok

    tr = drop_empty(tr, "train")
    va = drop_empty(va, "val")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = tr[0]["feats"].shape[-1]
    allf = torch.cat([x["feats"] for x in tr], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-5
    prep = lambda x: ((x["feats"] - mu) / sd).unsqueeze(0).to(dev)

    net = BoundaryHead(D).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor(a.pos_weight, device=dev)
    bce = nn.functional.binary_cross_entropy_with_logits

    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for x in tr:
            bl = torch.tensor(soft_boundary(x["times"], x["segments"], a.sigma_s), device=dev).unsqueeze(0)
            al = torch.tensor(actionness(x["times"], x["segments"]), device=dev).unsqueeze(0)
            cl = torch.tensor([float(len(x["segments"]))], device=dev)
            b, ac, c = net(prep(x))
            loss = (bce(b, bl, pos_weight=pw) + a.w_action * bce(ac, al)
                    + a.w_count * ((c - cl) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 100 == 0 or ep == a.epochs - 1:
            print(f"ep {ep+1} loss {tot/len(tr):.4f}")

    net.eval(); agg = []; multi = []; cnt_err = []
    with torch.no_grad():
        for x in va:
            b, ac, c = net(prep(x))
            bp = torch.sigmoid(b)[0].cpu().numpy()
            ap_ = torch.sigmoid(ac)[0].cpu().numpy()
            times = x["times"]; dur = x["duration"]
            if a.oracle_count:
                segs = sorted(x["segments"], key=lambda s: s[1])
                lo, hi = segs[0][1], segs[-1][2]
                k = len(segs)
            else:
                act = [i for i in range(len(ap_)) if ap_[i] > 0.5]
                lo = times[act[0]].item() if act else 0.0
                hi = times[act[-1]].item() if act else dur
                if a.count_mode == "peaks":
                    npk = sum(1 for i in range(len(bp))
                              if lo < times[i] < hi and bp[i] > a.count_thresh
                              and (i == 0 or bp[i] >= bp[i - 1])
                              and (i == len(bp) - 1 or bp[i] >= bp[i + 1]))
                    k = npk + 1
                else:
                    k = max(int(round(c.item())), 1)
            preds = peaks_topk(bp, times, lo, hi, max(k - 1, 0), a.min_gap_s)
            m = eval_item(preds, x)
            agg.append(m)
            cnt_err.append(abs(len(preds) - m["n_gt"]))
            tag = "  <-- ZERO" if m["f1@0.5"] == 0 else ""
            print(f"{x['video'].split('/')[-1]:22s} pred {m['n_pred']:2d} gt {m['n_gt']:2d} "
                  f"predk {k if not a.oracle_count else m['n_gt']:2d} "
                  f"f1@.5 {m['f1@0.5']:.2f} bound {m['boundary_score']:.2f}{tag}")
            if m["n_gt"] >= 6:
                multi.append(m["f1@0.5"])
    mode = "ORACLE count+endpoints" if a.oracle_count else "END-TO-END (learned count+endpoints)"
    print(f"\n==== BOUNDARY HEAD [{mode}] (val n=%d) ====" % len(agg))
    for k in ["f1@0.3", "f1@0.5", "f1@0.7", "boundary_score", "count_acc"]:
        print(k.ljust(16), round(statistics.mean([m[k] for m in agg]), 3))
    if not a.oracle_count:
        print("count |pred-gt| mean", round(statistics.mean(cnt_err), 2))
    if multi:
        print(f"multi-seg (gt>=6) mean f1@.5 {statistics.mean(multi):.3f}")
    print("\nref S1c: f1@0.5 0.452 / boundary 0.491 / count_acc 0.385 (multi-seg ~0)")


if __name__ == "__main__":
    main()
