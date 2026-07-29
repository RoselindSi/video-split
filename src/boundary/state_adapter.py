"""Track B: small audited-pair contrastive adapter over the frozen Qwen
features, producing an action-STATE embedding — plus nested grouped
evaluation of HAL-v2 features computed on it.

Why this exists (diagnosis it responds to): the held-out rejection of the
frozen HAL verifier decomposed into two causes. (1) DEV->TEST feature shift
(context_change mean z +0.78 under the DEV scaler) — addressable with
recording-relative normalization. (2) The deeper one: false keeps were
driven by huge, honest `context_change`/`change_persistence` values — the
pre/post representations really do stay different, yet the human says the
ACTION didn't change. Persistent visual difference != new action state, and
no amount of re-weighting hand-made difference features on the RAW frozen
embedding fixes that. So instead of stacking more difference features, use
the 175 human-audited decisive events as supervision to learn a small
embedding in which "same action despite motion" is close and "real
transition" is separated — exactly the event-gated HAL idea (no global
slow-latent smoothness; fast real boundaries may jump).

Supervision (from data/gold/audit_188_gold_v2.jsonl, human labels ONLY — no
HAL pseudo-labels):
    boundary_contrastive_role == positive             -> push d(c_L, c_R) > m,
                                                          pull within-side crops
    boundary_contrastive_role == motion_hard_negative -> pull d(c_L, c_R) -> 0
    exclude / ambiguous / unresolved                  -> masked out entirely

Architecture (deliberately tiny; ~175 events cannot feed anything bigger):
    frozen feats [T, 5760] -> Linear 256 -> GELU
      -> Conv1d(k=3) -> GELU -> Conv1d(k=3, dilation=2)   (temporal context)
      -> Linear 128 -> L2-normalize                        (state space c_t)
  ~1.8M trainable params. Windows are pooled from the per-frame c_t.

HAL-v2 features on an embedding (raw or learned), per candidate time t —
each is one of the structures the failure analysis said v1 lacks:
    sep@{0.5,1,2,4}s : d(mu_L, mu_R) / (W_L + W_R + eps)  (BETWEEN-state
                        separation relative to WITHIN-state noise, not raw
                        difference)
    return_gap       : d(pre, post-near) - d(pre, post-far). Positive =>
                        the state REVERTS (repetitive motion / regrasp /
                        camera excursion), a real transition shouldn't.
    dir_consistency  : mean pairwise cos of (mu_post_k - mu_pre) for three
                        consecutive post windows — real transitions leave in
                        one direction; oscillation doesn't.
    scale_agreement  : sep@2 / (sep@0.5 + eps) — does the short-scale change
                        survive at the longer scale?
  Plus z_local (median/MAD over a whole-recording time grid, clipped to
  ±5) for the sep features — the recording-relative normalization that the
  shift analysis demanded, with clipping so no single z=10 dominates.

Evaluation protocol — the part that keeps this honest: `eval` runs grouped
K-fold (by recording) where BOTH the adapter AND the logistic head are
trained inside each fold; held-out fold events are scored by an adapter
that never saw their recordings. Reported per arm (v1 raw / v2 raw /
v2 adapter): OOF AUROC, precision-coverage curve, and coverage at 0.90
precision. NOTE: reading coverage@0.90 off the pooled OOF curve is still a
DEVELOPMENT number (the operating point is chosen post hoc on the same OOF
predictions); the deployable claim must wait for a frozen
model+features+threshold evaluated once on a new batch3 of unseen
recordings. All 188 events (both former splits) are development data now —
batch2 stopped being a test set the moment we analysed its failures.

Usage (server):
  python -m src.boundary.state_adapter train \
      --feat_cache ...part1_train.pt --feat_cache ...part1_val.pt \
      --feat_cache ...part2.pt \
      --out_dir /workspace/tr1/results/hal/state_adapter
  python -m src.boundary.state_adapter eval \
      --feat_cache ... (same three) \
      --out_dir /workspace/tr1/results/hal/state_adapter --folds 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at
from src.boundary.hal_vlm_fusion import HAL_FEATURE_NAMES, fit_logreg, _sigmoid

SNIPPET_HALF_S = 10.0     # context handed to the encoder around each event
SCALES = [0.5, 1.0, 2.0, 4.0]
EPS = 1e-6


# ---------------------------------------------------------------- data ----

def build_events(gold, ctx, by_rid):
    """One entry per decisive event: role (+1 push / 0 pull), center time,
    recording ref. Ambiguous/exclude rows are masked out entirely."""
    events = []
    for g in gold:
        role = g.get("boundary_contrastive_role")
        if role not in ("positive", "motion_hard_negative"):
            continue
        c = ctx.get(g["event_id"], {})
        rid = g.get("recording_id") or c.get("recording_id")
        rec = by_rid.get(rid)
        t = c.get("pred_time")
        if t is None:
            t = c.get("gt_time")
        if rec is None or t is None:
            continue
        events.append({"event_id": g["event_id"], "recording_id": rid,
                       "t": float(t), "y": 1 if role == "positive" else 0, "rec": rec})
    return events


def snippet(rec, t, half=SNIPPET_HALF_S):
    """(feats [T,D] float32, times [T]) within t±half."""
    times = rec["times"]
    mask = (times >= t - half) & (times <= t + half)
    if int(mask.sum()) < 6:
        return None, None
    return rec["feats"][mask].float(), times[mask]


def pool_window(emb, times, lo, hi):
    """Mean state embedding in [lo, hi); None if <1 frame. emb is already
    L2-normalized per frame; the pooled mean is re-normalized."""
    m = (times >= lo) & (times < hi)
    if int(m.sum()) < 1:
        return None
    v = emb[m].mean(0)
    return F.normalize(v, dim=-1)


def cosd(a, b):
    return 1.0 - float(F.cosine_similarity(a[None], b[None]))


# --------------------------------------------------------------- model ----

class StateAdapter(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):  # x [T, D]
        h = F.gelu(self.proj(x)).T[None]          # [1, H, T]
        h = F.gelu(self.c1(h))
        h = F.gelu(self.c2(h)).squeeze(0).T       # [T, H]
        return F.normalize(self.head(h), dim=-1)  # [T, out]


def event_loss(emb, times, t, y, margin=0.5, w=2.0):
    """Push/pull loss on one event. Windows: L=[t-w,t), R=[t,t+w); within-
    side crops are the two halves of each side (positives only)."""
    cl = pool_window(emb, times, t - w, t)
    cr = pool_window(emb, times, t, t + w)
    if cl is None or cr is None:
        return None
    d_lr = 1.0 - F.cosine_similarity(cl[None], cr[None])
    if y == 0:                                   # same action: pull together
        return d_lr.pow(2).squeeze()
    loss = F.relu(margin - d_lr).pow(2).squeeze()   # real boundary: push apart
    for lo, hi in [(t - w, t), (t, t + w)]:      # each side internally stable
        a = pool_window(emb, times, lo, (lo + hi) / 2)
        b = pool_window(emb, times, (lo + hi) / 2, hi)
        if a is not None and b is not None:
            loss = loss + 0.5 * (1.0 - F.cosine_similarity(a[None], b[None])).pow(2).squeeze()
    return loss


def train_adapter(events, in_dim, device, epochs=40, lr=1e-3, seed=0,
                  val_recordings=None, verbose=True):
    """val_recordings: held out for early stopping only (never gradients)."""
    torch.manual_seed(seed)
    random.Random(seed).shuffle(events)
    val_recordings = val_recordings or set()
    tr = [e for e in events if e["recording_id"] not in val_recordings]
    va = [e for e in events if e["recording_id"] in val_recordings]
    model = StateAdapter(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, float("inf")
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for e in tr:
            fx, tx = snippet(e["rec"], e["t"])
            if fx is None:
                continue
            loss = event_loss(model(fx.to(device)), tx.to(device), e["t"], e["y"])
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item(); n += 1
        vtot = vn = 0
        if va:
            model.eval()
            with torch.no_grad():
                for e in va:
                    fx, tx = snippet(e["rec"], e["t"])
                    if fx is None:
                        continue
                    l = event_loss(model(fx.to(device)), tx.to(device), e["t"], e["y"])
                    if l is not None:
                        vtot += l.detach().item(); vn += 1
            vloss = vtot / max(vn, 1)
            if vloss < best_val:
                best_val, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:>3}  train={tot / max(n, 1):.4f}"
                  + (f"  val={vtot / max(vn, 1):.4f}" if va else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


# ---------------------------------------------------- HAL-v2 features -----

V2_NAMES = ([f"sep_{s}s" for s in SCALES] + [f"sep_{s}s_zlocal" for s in SCALES]
            + ["return_gap", "dir_consistency", "scale_agreement"])


def _win_stats(emb, times, lo, hi):
    m = (times >= lo) & (times < hi)
    if int(m.sum()) < 1:
        return None, None
    sub = emb[m]
    mu = F.normalize(sub.mean(0), dim=-1)
    spread = float((1.0 - F.cosine_similarity(sub, mu[None].expand_as(sub))).mean()) if len(sub) > 1 else 0.0
    return mu, spread


def v2_features_at(emb, times, t):
    out = {}
    for s in SCALES:
        L = _win_stats(emb, times, t - s, t)
        R = _win_stats(emb, times, t, t + s)
        out[f"sep_{s}s"] = (cosd(L[0], R[0]) / (L[1] + R[1] + EPS)) \
            if L[0] is not None and R[0] is not None else None
    pre, _ = _win_stats(emb, times, t - 2.0, t)
    near, _ = _win_stats(emb, times, t, t + 2.0)
    far, _ = _win_stats(emb, times, t + 4.0, t + 8.0)
    out["return_gap"] = (cosd(pre, near) - cosd(pre, far)) \
        if pre is not None and near is not None and far is not None else None
    deltas = []
    if pre is not None:
        for k in range(3):
            mu, _ = _win_stats(emb, times, t + k, t + k + 1.0)
            if mu is not None:
                deltas.append(mu - pre)
    if len(deltas) >= 2:
        cs = [float(F.cosine_similarity(deltas[i][None], deltas[j][None]))
              for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
        out["dir_consistency"] = sum(cs) / len(cs)
    else:
        out["dir_consistency"] = None
    s05, s2 = out.get("sep_0.5s"), out.get("sep_2.0s")
    out["scale_agreement"] = (s2 / (s05 + EPS)) if s05 is not None and s2 is not None else None
    return out


def featurize_events(events, embed_fn, stride_s=4.0, max_grid=200):
    """v2 features + clipped z_local per event. embed_fn(rec) -> (emb, times)
    cached per recording (the adapter runs once per recording, not per
    event). z_local baseline = whole-recording grid, NOT just audited
    events."""
    cache = {}
    baselines = {}
    for e in events:
        rid = e["recording_id"]
        if rid not in cache:
            cache[rid] = embed_fn(e["rec"])
            emb, tt = cache[rid]
            t0, t1 = float(tt[0]), float(tt[-1])
            grid = np.arange(t0 + 4.0, max(t0 + 4.0, t1 - 8.0), stride_s)
            if len(grid) > max_grid:
                grid = grid[np.linspace(0, len(grid) - 1, max_grid).astype(int)]
            acc = defaultdict(list)
            for gt_ in grid:
                f = v2_features_at(emb, tt, float(gt_))
                for s in SCALES:
                    v = f.get(f"sep_{s}s")
                    if v is not None:
                        acc[f"sep_{s}s"].append(v)
            baselines[rid] = {k: (float(np.median(v)), float(np.median(np.abs(np.array(v) - np.median(v)))))
                              for k, v in acc.items()}
    rows = []
    for e in events:
        emb, tt = cache[e["recording_id"]]
        f = v2_features_at(emb, tt, e["t"])
        base = baselines.get(e["recording_id"], {})
        for s in SCALES:
            v, (med, mad) = f.get(f"sep_{s}s"), base.get(f"sep_{s}s", (None, None))
            f[f"sep_{s}s_zlocal"] = float(np.clip((v - med) / (mad + EPS), -5, 5)) \
                if v is not None and med is not None else None
        rows.append([f.get(k) if f.get(k) is not None else np.nan for k in V2_NAMES])
    return np.array(rows, dtype=float)


# ------------------------------------------------------------ eval CV -----

def precision_coverage(y, p, target=0.90):
    order = np.argsort(-p)
    ys = y[order]
    best_cov = 0.0
    curve = []
    for k in range(1, len(ys) + 1):
        prec = ys[:k].mean()
        curve.append((k, float(prec)))
        if prec >= target:
            best_cov = k / len(ys)
    return best_cov, curve


def grouped_folds(recordings, k, seed=0):
    recs = sorted(set(recordings))
    random.Random(seed).shuffle(recs)
    return [set(recs[i::k]) for i in range(k)]


def loro_fit_predict(X, y, groups, l2=1.0):
    preds = np.full(len(y), np.nan)
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        te, tr = groups == g, groups != g
        if len(set(y[tr].tolist())) < 2:
            continue
        cm = np.where(np.isnan(np.nanmean(X[tr], 0)), 0, np.nanmean(X[tr], 0))
        Xtr = np.where(np.isnan(X[tr]), cm, X[tr]); Xte = np.where(np.isnan(X[te]), cm, X[te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        w, b = fit_logreg((Xtr - mu) / sd, y[tr], l2=l2)
        preds[te] = _sigmoid((Xte - mu) / sd @ w + b)
    return preds


def _auroc(y, p):
    """Rank-based AUROC; nan if either class is empty."""
    m = ~np.isnan(p)
    y_, p_ = np.asarray(y)[m], np.asarray(p)[m]
    if len(set(y_.tolist())) < 2:
        return float("nan")
    order = np.argsort(p_)
    ranks = np.empty(len(p_)); ranks[order] = np.arange(len(p_))
    pos, neg = ranks[y_ == 1], ranks[y_ == 0]
    return float((pos[:, None] > neg[None, :]).mean())


def report_arm(name, y, p, target=0.90):
    m = ~np.isnan(p)
    y_, p_ = y[m], p[m]
    order = np.argsort(p_)
    ranks = np.empty(len(p_)); ranks[order] = np.arange(len(p_))
    pos, neg = ranks[y_ == 1], ranks[y_ == 0]
    auroc = ((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    cov, _ = precision_coverage(y_, p_, target)
    print(f"  {name:<24} n={len(y_):<4} AUROC={auroc:.3f}  coverage@{target:.0%}prec={cov:.3f}")
    return {"n": int(len(y_)), "auroc": float(auroc), f"coverage_at_{target}": float(cov)}


# -------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    y = np.array([e["y"] for e in events], dtype=float)
    groups = [e["recording_id"] for e in events]
    in_dim = events[0]["rec"]["feats"].shape[1]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"events: {len(events)} ({int(y.sum())} positive / {int(len(y) - y.sum())} "
          f"motion-hard-negative) across {len(set(groups))} recordings; feat dim {in_dim}")
    print("NOTE: all 188 audited events are DEVELOPMENT data now -- batch2 ceased to "
          "be a held-out test when its failures were analysed. Deployable claims wait for batch3.")

    if a.mode == "train":
        folds = grouped_folds(groups, 5, a.seed)
        model = train_adapter(events, in_dim, a.device, a.epochs, a.lr, a.seed,
                              val_recordings=folds[0])
        ck = os.path.join(a.out_dir, "state_adapter.pt")
        torch.save({"state_dict": model.state_dict(), "in_dim": in_dim,
                    "val_recordings": sorted(folds[0])}, ck)
        print(f"saved {ck}  (early-stopped on {len(folds[0])} held-out recordings)")
        try:
            from src.eval.run_manifest import write_manifest
            write_manifest(ck, input_paths=[a.gold, a.context] + a.feat_cache,
                           extra={"epochs": a.epochs, "lr": a.lr, "seed": a.seed})
        except Exception as e:
            print(f"[manifest] skipped ({e})")
        return

    # ---- eval: nested grouped CV -- adapter AND head trained per fold ----
    print(f"\nnested grouped {a.folds}-fold CV (adapter + logistic head both "
          f"trained inside each fold; held-out recordings never seen by either)")
    raw_fn = lambda rec: (F.normalize(rec["feats"].float(), dim=-1), rec["times"])

    # arm 1: v1 raw features (the rejected family, as the baseline to beat)
    X_v1 = np.array([[hal_features_at(e["rec"]["feats"], e["rec"]["times"], e["t"]).get(k) or np.nan
                      for k in HAL_FEATURE_NAMES] for e in events], dtype=float)
    p_v1 = loro_fit_predict(X_v1, y, groups)

    # arm 2: v2 features on the RAW frozen embedding (new structure, no learning)
    X_v2raw = featurize_events(events, raw_fn)
    p_v2raw = loro_fit_predict(X_v2raw, y, groups)

    # arm 3: v2 features on the ADAPTER embedding, nested per fold
    p_v2ad = np.full(len(events), np.nan)
    fold_diag = []
    folds = grouped_folds(groups, a.folds, a.seed)
    for fi, held in enumerate(folds):
        tr_ev = [e for e in events if e["recording_id"] not in held]
        te_ev = [e for e in events if e["recording_id"] in held]
        if not te_ev or len({e["y"] for e in tr_ev}) < 2:
            continue
        print(f"  fold {fi + 1}/{a.folds}: train {len(tr_ev)} ev / test {len(te_ev)} ev "
              f"({len(held)} recordings held out)")
        inner_val = grouped_folds([e["recording_id"] for e in tr_ev], 5, a.seed + fi)[0]
        model = train_adapter(list(tr_ev), in_dim, a.device, a.epochs, a.lr,
                              a.seed + fi, val_recordings=inner_val, verbose=False)
        def ad_fn(rec, _m=model, _d=a.device):
            with torch.no_grad():
                return _m(rec["feats"].float().to(_d)).cpu(), rec["times"]
        Xtr = featurize_events(tr_ev, ad_fn)
        Xte = featurize_events(te_ev, ad_fn)
        ytr = np.array([e["y"] for e in tr_ev], dtype=float)
        cm = np.where(np.isnan(np.nanmean(Xtr, 0)), 0, np.nanmean(Xtr, 0))
        Xtr_i = np.where(np.isnan(Xtr), cm, Xtr); Xte_i = np.where(np.isnan(Xte), cm, Xte)
        mu, sd = Xtr_i.mean(0), Xtr_i.std(0) + 1e-8
        w, b = fit_logreg((Xtr_i - mu) / sd, ytr)
        pte = _sigmoid((Xte_i - mu) / sd @ w + b)
        # per-fold TRAIN vs TEST AUROC: the only way to tell "the adapter
        # learned nothing" (train also low) from "it overfits across
        # recordings" (train high, test low) -- a pooled OOF number alone
        # cannot separate those two, and they need opposite fixes.
        ptr = _sigmoid((Xtr_i - mu) / sd @ w + b)
        yte_arr = np.array([e["y"] for e in te_ev], dtype=float)
        tr_auc, te_auc = _auroc(ytr, ptr), _auroc(yte_arr, pte)
        fold_diag.append({"fold": fi + 1, "n_train": len(tr_ev), "n_test": len(te_ev),
                          "n_recordings_held": len(held),
                          "train_auroc": tr_auc, "test_auroc": te_auc,
                          "test_pos_frac": float(yte_arr.mean()) if len(yte_arr) else None})
        print(f"      train AUROC={tr_auc:.3f}  test AUROC={te_auc:.3f}  "
              f"test positive frac={yte_arr.mean():.2f}")
        idx = {e["event_id"]: i for i, e in enumerate(events)}
        for e, p in zip(te_ev, pte):
            p_v2ad[idx[e["event_id"]]] = p

    print("\n=== OOF comparison (development numbers -- operating point NOT frozen here) ===")
    res = {
        "v1_raw_rejected_baseline": report_arm("v1 raw (rejected)", y, p_v1),
        "v2_raw_frozen_embedding": report_arm("v2 raw embedding", y, p_v2raw),
        "v2_adapter_embedding": report_arm("v2 adapter embedding", y, p_v2ad),
        "adapter_per_fold": fold_diag,
    }
    out = os.path.join(a.out_dir, "nested_cv_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")
    if fold_diag:
        tr_m = float(np.nanmean([d["train_auroc"] for d in fold_diag]))
        te_m = float(np.nanmean([d["test_auroc"] for d in fold_diag]))
        print(f"\nadapter per-fold mean: train AUROC={tr_m:.3f}  test AUROC={te_m:.3f}")
        if tr_m > 0.85 and te_m < 0.65:
            print("  -> train high / test low: the adapter OVERFITS across recordings "
                  "(needs stronger regularization or more recordings, not a new objective)")
        elif tr_m < 0.70:
            print("  -> train ALSO low: the contrastive objective is not fitting even the "
                  "training pairs -- supervision/implementation issue, not generalization")
        spread = [d["test_auroc"] for d in fold_diag if not np.isnan(d["test_auroc"])]
        if spread and (max(spread) - min(spread)) > 0.25:
            print(f"  -> test AUROC varies {min(spread):.2f}-{max(spread):.2f} across folds: "
                  f"recording-level domain shift dominates the average")
    print("read: if v2-adapter does not beat v2-raw, the learned state space adds "
          "nothing yet (more data or different supervision needed); if v2-raw already "
          "beats v1, the structural features alone were the missing piece.")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(out, input_paths=[a.gold, a.context] + a.feat_cache,
                       extra={"folds": a.folds, "epochs": a.epochs, "seed": a.seed})
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
