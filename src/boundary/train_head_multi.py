"""Parameter-controlled supervised head comparison over multi-region features.

Loads multi-region caches (per frame = concat[global,left,right,center,
spatial_max], 5x1152), derives all H0-H4 inputs OFFLINE from the same file
(no re-extraction), and trains the SAME temporal head under matched capacity so
that any gain is attributable to spatial information, not model size.

  H0 global        : region[0]                       (1152)
  H1 left          : region[1]                        (1152)
  H2 spatial_max   : region[4]  (strongest compact)   (1152)
  H3 mean5         : mean over 5 regions              (1152)
  H4 region_attn   : shared W: z_{t,r}=W h_{t,r}; a=softmax_r(q.z); z=sum a*z
                     (learned per-frame region weighting)  (-> proj_dim)

Capacity control: every variant projects its per-frame input to the SAME
proj_dim before the SAME temporal head. H0-H3 use a single Linear(1152->proj);
H4 shares that Linear across regions + a tiny attention vector. So the temporal
head and its input width are identical across H0-H4.

Metric = boundary-level F1 (oracle GT count, top-k peaks) so it lines up with the
self-sim/region-attribution probes and the head oracle (0.331 on global).

Usage (server, after feat_train/val multi are extracted):
    python -m src.boundary.train_head_multi \
        --train /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --val   /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --variant spatial_max --seeds 0 1 2
"""
import argparse, statistics
import numpy as np
import torch
import torch.nn as nn

REGION = {"global": 0, "left": 1, "right": 2, "center": 3, "spatial_max": 4}


# ---------------- input derivation (offline, from the 5x1152 cache) ----------
def to_regions(feats, D=1152):
    T = feats.shape[0]
    return feats.float().view(T, 5, D)              # [T,5,D]


# ---------------- labels ----------------
def soft_boundary(times, segments, sigma_s=1.0):
    t = times.numpy(); lab = np.zeros(len(t), np.float32)
    segs = sorted(segments, key=lambda x: x[1])
    for b in [segs[i][1] for i in range(1, len(segs))]:
        lab = np.maximum(lab, np.exp(-((t - b) ** 2) / (2 * sigma_s ** 2)))
    return lab


def actionness(times, segments):
    t = times.numpy(); segs = sorted(segments, key=lambda x: x[1])
    return ((t >= segs[0][1]) & (t <= segs[-1][2])).astype(np.float32)


# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, variant, D=1152, proj=256, d=256):
        super().__init__()
        self.variant = variant
        if variant == "region_attn":
            self.proj = nn.Linear(D, proj)          # shared across regions
            self.q = nn.Parameter(torch.randn(proj) * 0.02)
        elif variant == "concat":
            self.proj = nn.Linear(5 * D, proj)      # NOT capacity-matched (ceiling ref)
        else:
            self.proj = nn.Linear(D, proj)
        self.tin = nn.Conv1d(proj, d, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * r, dilation=r)
                                     for r in (1, 2, 4, 8)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(4)])
        self.bound = nn.Conv1d(d, 1, 1)

    def forward(self, regions):                     # regions: [T,5,D]
        if self.variant == "region_attn":
            z = self.proj(regions)                  # [T,5,proj]
            a = torch.softmax(z @ self.q, dim=1)    # [T,5]
            x = (a.unsqueeze(-1) * z).sum(1)        # [T,proj]
        elif self.variant == "concat":
            x = self.proj(regions.reshape(regions.shape[0], -1))   # [T,proj] from 5760
        elif self.variant == "mean5":
            x = self.proj(regions.mean(1))          # [T,proj] mean over 5 regions
        else:
            x = self.proj(regions[:, REGION[self.variant]])   # single region [T,proj]
        h = self.tin(x.transpose(0, 1).unsqueeze(0))   # [1,d,T]
        for conv, gn in zip(self.blocks, self.norm):
            h = h + torch.relu(gn(conv(h)))
        return self.bound(h).squeeze(0).squeeze(0)     # [T] logits


# ---------------- decode + boundary-F1 (oracle count) ----------------
def topk_peaks(prob, times, k, min_gap):
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


def gt_bounds(segs):
    ts = sorted({round(s[1], 2) for s in segs} | {round(s[2], 2) for s in segs})
    return ts[1:-1] if len(ts) > 2 else ts


def bf1(preds, gts, tol):
    used = set(); tp = 0
    for p in preds:
        best, bj = tol + 1, -1
        for j, g in enumerate(gts):
            if j not in used and abs(p - g) < best:
                best, bj = abs(p - g), j
        if bj >= 0 and best <= tol:
            used.add(bj); tp += 1
    pr, rc = tp / max(len(preds), 1), tp / max(len(gts), 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


def run_seed(tr, va, variant, mu, sd, dev, a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Head(variant, proj=a.proj).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor(a.pos_weight, device=dev)
    bce = nn.functional.binary_cross_entropy_with_logits

    def prep(x):
        r = to_regions(x["feats"]).to(dev)          # move to device BEFORE using
        return (r - mu) / sd                        # mu/sd already on dev; standardize

    best_f5, best_train_f5 = -1.0, 0.0
    for ep in range(a.epochs):
        net.train()
        for x in tr:
            y = torch.tensor(soft_boundary(x["times"], x["segments"], a.sigma_s), device=dev)
            loss = bce(net(prep(x)), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            net.eval()
            def eval_split(split):
                f5 = []
                for x in split:
                    prob = torch.sigmoid(net(prep(x))).cpu().numpy()
                    gts = gt_bounds(x["segments"])
                    pk = topk_peaks(prob, x["times"].numpy(), len(gts), a.min_gap_s)
                    f5.append(bf1(pk, gts, 0.5))
                return statistics.mean(f5)
            with torch.no_grad():
                m = eval_split(va)
                if m > best_f5:
                    best_f5 = m
                    best_train_f5 = eval_split(tr[:len(va)])   # train F1 at val-best
    return best_f5, best_train_f5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--variant", default="all",
                    choices=["all", "global", "left", "spatial_max",
                             "mean5", "region_attn", "concat"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--proj", type=int, default=256)
    ap.add_argument("--sigma_s", type=float, default=1.0)
    ap.add_argument("--min_gap_s", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    a = ap.parse_args()

    tr = torch.load(a.train, weights_only=False)
    va = torch.load(a.val, weights_only=False)
    tr = [x for x in tr if x["feats"].dim() == 2 and x["feats"].shape[0] > 4]
    va = [x for x in va if x["feats"].dim() == 2 and x["feats"].shape[0] > 4]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # train-global standardization stats over [.,5,1152]
    allr = torch.cat([to_regions(x["feats"]) for x in tr], 0)   # [N,5,D]
    mu, sd = allr.mean(0, keepdim=True), allr.std(0, keepdim=True) + 1e-5
    mu, sd = mu.to(dev), sd.to(dev)

    # class-balance diagnostic: with no-blur (long dense sequences ~1200-1400
    # frames), positive boundary frames are a tiny fraction -> pos_weight/sigma
    # tuned for the old short (~60-frame) sequences may no longer fit.
    pos_frac, seq_len = [], []
    for x in tr:
        lab = soft_boundary(x["times"], x["segments"], a.sigma_s)
        pos_frac.append(float((lab > 0.5).mean())); seq_len.append(len(lab))
    print(f"[diag] seq_len mean {statistics.mean(seq_len):.0f} | "
          f"positive-frame frac (soft>0.5) mean {statistics.mean(pos_frac):.4f} "
          f"-> ~1:{1/max(statistics.mean(pos_frac),1e-6):.0f} pos:neg "
          f"(current pos_weight={a.pos_weight}, sigma_s={a.sigma_s})\n")

    variants = (["global", "left", "spatial_max", "mean5", "region_attn", "concat"]
                if a.variant == "all" else [a.variant])
    print(f"{'variant':14s} {'val F1@0.5 (seeds)':26s} {'val_mean':>8s} "
          f"{'std':>6s} {'train_f5':>9s}")
    for v in variants:
        res = [run_seed(tr, va, v, mu, sd, dev, a, s) for s in a.seeds]
        val_rs = [r[0] for r in res]; tr_rs = [r[1] for r in res]
        print(f"{v:14s} {str([round(r,3) for r in val_rs]):26s} "
              f"{statistics.mean(val_rs):8.3f} {statistics.pstdev(val_rs):6.3f} "
              f"{statistics.mean(tr_rs):9.3f}")
    print("\ntrain_f5 >> val_f5 -> overfit/too little data; both low -> underfit/"
          "config mismatch. NOTE: 0.331 was OLD data (short blur seqs) -- NOT "
          "directly comparable; rerun old features through THIS script to compare.")


if __name__ == "__main__":
    main()
