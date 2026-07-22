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


# ---------------- temporal-change features (B3) ----------------
DELTA_MULT = {"none": 1, "delta1": 2, "delta_sym": 3, "window": 2}


def add_delta(src, mode, w=2):
    """src [T,C] -> [T, C*DELTA_MULT[mode]]. Boundaries need 'what CHANGED', not
    'what is here'. Parameter-fair: the WIDER input is projected back to the
    same proj dim before the (identical) temporal head -- see Head.__init__."""
    if mode == "none":
        return src
    T = src.shape[0]
    back = src.clone(); back[1:] = src[1:] - src[:-1]; back[0] = 0
    if mode == "delta1":
        return torch.cat([src, back], -1)
    if mode == "delta_sym":
        fwd = src.clone(); fwd[:-1] = src[1:] - src[:-1]; fwd[-1] = 0
        return torch.cat([src, fwd, back], -1)
    if mode == "window":                            # mean(t:t+w) - mean(t-w:t)
        x = src.transpose(0, 1).unsqueeze(0)        # [1,C,T]
        pad = torch.nn.functional.pad(x, (w, w), mode="replicate")
        pool = torch.nn.functional.avg_pool1d(pad, kernel_size=w, stride=1)
        fut = pool[..., w + 1:w + 1 + T]; past = pool[..., :T]
        wdiff = (fut - past).squeeze(0).transpose(0, 1)   # [T,C]
        return torch.cat([src, wdiff], -1)
    raise ValueError(mode)


# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, variant, D=1152, proj=256, d=256, delta_mode="none"):
        super().__init__()
        self.variant = variant
        self.delta_mode = delta_mode
        mult = DELTA_MULT[delta_mode]
        if variant == "region_attn":
            assert delta_mode == "none", "delta only for region-select variants"
            self.proj = nn.Linear(D, proj)          # shared across regions
            self.q = nn.Parameter(torch.randn(proj) * 0.02)
        elif variant == "concat":
            assert delta_mode == "none", "delta only for region-select variants"
            self.proj = nn.Linear(5 * D, proj)      # NOT capacity-matched (ceiling ref)
        else:
            self.proj = nn.Linear(D * mult, proj)   # wider input, SAME proj/head
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
        else:
            src = regions.mean(1) if self.variant == "mean5" else regions[:, REGION[self.variant]]
            x = self.proj(add_delta(src, self.delta_mode))         # [T,proj]
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


def thresh_peaks(prob, times, thr, min_gap):
    """Threshold decode: FREE number of peaks (>thr). Unlike oracle top-k, the
    predicted count varies -> reveals over/under-prediction (the pos_weight
    symptom: too-high probs -> many peaks -> low precision)."""
    cand = [i for i in range(len(prob))
            if prob[i] >= thr
            and (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i]); kept = []
    for i in cand:
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
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return f1, pr, rc, len(preds), len(gts)


def run_seed(tr, va, variant, mu, sd, dev, a, seed, extra=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Head(variant, proj=a.proj, delta_mode=a.delta_mode).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor(a.pos_weight, device=dev)
    bce = nn.functional.binary_cross_entropy_with_logits

    def prep(x):
        # x["R"] is the standardized [T,5,D] tensor PRE-MOVED to GPU once in main
        # (avoids re-uploading ~29MB and re-standardizing every forward pass --
        # that repeated host->device transfer was the training bottleneck).
        return x["R"]

    def oracle_f5(split):
        out = []
        for x in split:
            prob = torch.sigmoid(net(prep(x))).cpu().numpy()
            gts = gt_bounds(x["segments"])
            pk = topk_peaks(prob, x["times"].numpy(), len(gts), a.min_gap_s)
            out.append(bf1(pk, gts, 0.5)[0])
        return statistics.mean(out)

    def thresh_diag(split, window=1.0):
        """Free-count decode diagnostics, plus near_gt_peaks: mean # predicted
        peaks within +-window seconds of EACH gt boundary. This distinguishes
        two different over-prediction failure modes:
          near_gt_peaks >> 1  -> duplicate peaks clustered around the same true
                                  boundary (decoding/NMS issue, e.g. min_gap_s
                                  or sigma_s too narrow for the peak's width)
          near_gt_peaks ~= 1 but pred/gt >> 1 -> extra peaks scattered AWAY
                                  from any true boundary (genuine false positives)
        """
        prec, rec, ratio, mprob, near = [], [], [], [], []
        for x in split:
            prob = torch.sigmoid(net(prep(x))).cpu().numpy()
            gts = gt_bounds(x["segments"])
            pk = thresh_peaks(prob, x["times"].numpy(), a.thr, a.min_gap_s)
            _, pr, rc, npd, ngt = bf1(pk, gts, 0.5)
            prec.append(pr); rec.append(rc); mprob.append(float(prob.mean()))
            ratio.append(npd / max(ngt, 1))
            if gts:
                near.append(statistics.mean(
                    sum(1 for p in pk if abs(p - g) <= window) for g in gts))
        return {"thr_prec": statistics.mean(prec), "thr_rec": statistics.mean(rec),
                "pred_ratio": statistics.mean(ratio), "mean_prob": statistics.mean(mprob),
                "near_gt_peaks": statistics.mean(near) if near else 0.0}

    def dump_logits(split):
        """Per-video (raw logits, sigmoid prob, soft target, valid_mask, times,
        gt boundary times, named segments) at the CURRENT weights, for offline
        decode-sweep (B2) AND FP/FN audit (B6) -- decoupled from training so
        both can run without retraining. segments (raw [name,start,end]
        triples) let the audit label each GT boundary with what changed (or
        didn't) across it, and place false peaks inside a named segment, with
        no need to re-load the original recseg json.

        valid_mask is all-True for every frame: run_seed calls net(prep(x))
        on ONE full video at a time (see the `for x in tr` training loop
        above) -- there is no cross-video batching or padding anywhere in
        this script, so there is no padding-contaminates-the-stats failure
        mode to guard against. Saved anyway so downstream scripts don't have
        to assume it and so this is auditable rather than asserted."""
        out = []
        for x in split:
            logits = net(prep(x))
            prob = torch.sigmoid(logits).cpu().numpy().tolist()
            y = soft_boundary(x["times"], x["segments"], a.sigma_s).tolist()
            out.append({"recording_id": x.get("recording_id", ""),
                        "logits": logits.detach().cpu().numpy().tolist(),
                        "prob": prob, "y_soft_target": y,
                        "valid_mask": [True] * len(prob),
                        "times": x["times"].numpy().tolist(),
                        "gt": gt_bounds(x["segments"]),
                        "segments": [[s[0], float(s[1]), float(s[2])] for s in x["segments"]]})
        return out

    best = {"val_f5": -1.0}
    for ep in range(a.epochs):
        net.train()
        for x in tr:
            y = torch.tensor(soft_boundary(x["times"], x["segments"], a.sigma_s), device=dev)
            loss = bce(net(prep(x)), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                m = oracle_f5(va)
                if m > best["val_f5"]:
                    diag = thresh_diag(va)
                    best = {"val_f5": m, "train_f5": oracle_f5(tr[:len(va)]), **diag}
                    if a.save_logits:
                        best["_logits"] = dump_logits(va)
                    if a.infer_extra and extra is not None:
                        # part2/unseen-generalization logits, dumped at the SAME
                        # best-val epoch as the val logits (so they reflect the
                        # same model state), with the head trained ONLY on part1
                        # train -- never sees `extra`.
                        best["_extra_logits"] = dump_logits(extra)
    return best


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
    ap.add_argument("--delta_mode", default="none",
                    choices=["none", "delta1", "delta_sym", "window"],
                    help="B3 temporal-change features: delta1=[h, h_t-h_t-1]; "
                         "delta_sym=[h, fwd-diff, back-diff]; window=[h, "
                         "mean(t:t+w)-mean(t-w:t)]. Only for region-select variants.")
    ap.add_argument("--sigma_s", type=float, default=1.0)
    ap.add_argument("--min_gap_s", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--thr", type=float, default=0.5,
                    help="threshold for the free-count decode diagnostic")
    ap.add_argument("--save_logits", default=None,
                    help="path to save per-video val (prob,times,gt) at the "
                         "best epoch for offline B2 decode-sweep diagnostics. "
                         "Only meaningful with a single --variant and a single "
                         "seed (the first seed's best-epoch logits are saved).")
    ap.add_argument("--infer_extra", default=None,
                    help="path to an EXTRA feature .pt (e.g. unseen part_02) to "
                         "run the trained head on for generalization testing. "
                         "Standardized with the SAME part1-train mu/sd; the head "
                         "is trained only on --train and never sees this. Logits "
                         "dumped at the best-val epoch. Requires a single "
                         "--variant and single seed.")
    ap.add_argument("--infer_extra_out", default=None,
                    help="where to save the --infer_extra logits (same format as "
                         "--save_logits, consumable by all boundary diagnostics).")
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
    del allr

    # PRE-MOVE standardized features to GPU once (was the training bottleneck:
    # re-uploading + re-standardizing 29MB/video every forward pass).
    for x in tr + va:
        x["R"] = ((to_regions(x["feats"]) - mu.cpu()) / sd.cpu()).to(dev)

    extra = None
    if a.infer_extra:
        extra = torch.load(a.infer_extra, weights_only=False)
        extra = [x for x in extra if x["feats"].dim() == 2 and x["feats"].shape[0] > 4]
        for x in extra:
            # SAME part1-train mu/sd -- an unseen-data feature must be
            # standardized by the training statistics, not its own, or the
            # generalization test conflates distribution shift with recentering.
            x["R"] = ((to_regions(x["feats"]) - mu.cpu()) / sd.cpu()).to(dev)
        print(f"[infer_extra] loaded {len(extra)} unseen recordings from {a.infer_extra}")

    # class-balance diagnostic (HARD positive fraction; the soft-mass ratio was a
    # bug -> millions). pos_frac = mean frac of frames with soft label > 0.5;
    # neg:pos ~= (1-frac)/frac, which is the pos_weight that balances BCE.
    seq_len, fracs = [], []
    for x in tr:
        lab = soft_boundary(x["times"], x["segments"], a.sigma_s)
        seq_len.append(len(lab)); fracs.append(float((lab > 0.5).mean()))
    frac = statistics.mean(fracs)
    print(f"[diag] seq_len mean {statistics.mean(seq_len):.0f} | pos-frame frac "
          f"(soft>0.5) {frac:.3f} -> neg:pos ~= {(1-frac)/max(frac,1e-6):.1f} "
          f"(pos_weight balancing BCE ~= this; current {a.pos_weight}, "
          f"sigma_s={a.sigma_s})\n")

    variants = (["global", "left", "spatial_max", "mean5", "region_attn", "concat"]
                if a.variant == "all" else [a.variant])
    print(f"[cfg] sigma_s={a.sigma_s} pos_weight={a.pos_weight} thr={a.thr} "
          f"min_gap_s={a.min_gap_s} delta_mode={a.delta_mode}")
    print(f"{'variant':12s} {'val_f5':>7s} {'std':>5s} {'train_f5':>8s} "
          f"{'thr_P':>6s} {'thr_R':>6s} {'pred/gt':>7s} {'meanp':>6s} "
          f"{'nearGT':>7s} {'verdict':>16s}")
    if a.save_logits and (a.variant == "all" or len(a.seeds) > 1):
        print("WARNING: --save_logits intended for a single --variant and a "
              "single seed; only the first variant/seed's logits will be saved.")
    for vi, v in enumerate(variants):
        res = [run_seed(tr, va, v, mu, sd, dev, a, s, extra=extra) for s in a.seeds]
        if a.infer_extra and a.infer_extra_out and vi == 0:
            torch.save(res[0].pop("_extra_logits", []), a.infer_extra_out)
            print(f"saved UNSEEN-generalization logits ({v}, seed {a.seeds[0]}) "
                  f"-> {a.infer_extra_out}")
            from src.eval.run_manifest import write_manifest
            write_manifest(a.infer_extra_out, input_paths=[a.train, a.infer_extra],
                           extra={"variant": v, "seed": a.seeds[0], "sigma_s": a.sigma_s,
                                  "pos_weight": a.pos_weight, "note": "head trained on "
                                  "part1 --train, inferred on unseen --infer_extra"})
        else:
            if vi == 0:
                res[0].pop("_extra_logits", None)
        if a.save_logits and vi == 0:
            torch.save(res[0].pop("_logits", []), a.save_logits)
            print(f"saved val logits ({v}, seed {a.seeds[0]}) -> {a.save_logits}")
            from src.eval.run_manifest import write_manifest
            write_manifest(a.save_logits, input_paths=[a.train, a.val],
                           extra={"variant": v, "seed": a.seeds[0], "sigma_s": a.sigma_s,
                                  "pos_weight": a.pos_weight, "delta_mode": a.delta_mode,
                                  "val_f5": res[0]["val_f5"], "train_f5": res[0]["train_f5"]})
        else:
            res[0].pop("_logits", None)
        vf = [r["val_f5"] for r in res]
        agg = lambda k: statistics.mean([r[k] for r in res])
        ratio, prec, near = agg("pred_ratio"), agg("thr_prec"), agg("near_gt_peaks")
        if ratio > 1.2:
            verdict = "dup-near-boundary" if near > 1.3 else "over-pred(scattered)"
        elif ratio < 0.8:
            verdict = "under-prediction"
        else:
            verdict = "count-balanced"
        print(f"{v:12s} {statistics.mean(vf):7.3f} {statistics.pstdev(vf):5.3f} "
              f"{agg('train_f5'):8.3f} {prec:6.2f} {agg('thr_rec'):6.2f} "
              f"{ratio:7.2f} {agg('mean_prob'):6.3f} {near:7.2f} {verdict:>16s}")
    print("\ntrain_f5>>val_f5 = overfit. nearGT = mean predicted peaks within "
          "+-1s of each GT boundary at threshold decode: >1.3 means duplicate "
          "peaks clustered on the SAME true boundary (fix: widen min_gap_s or "
          "narrow sigma_s), ~1 with pred_ratio>>1 means genuine false positives "
          "scattered away from boundaries (fix: pos_weight/thr). "
          "0.331 was OLD short-blur data, NOT directly comparable -- see the "
          "fair-comparison run using this same script on old-data features.")


if __name__ == "__main__":
    main()
