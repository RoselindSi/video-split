"""C2: slow-semantic-latent verifier, trained as supervised metric learning
directly on the sharp/same-action labels -- unlike C1 (self-supervised
future prediction on an unlabelled pool), C2 trains end-to-end on the
labelled clean events, per outer CV fold, exactly like P1.

Motivation (from the C1 subtype cross-tab, 145/145 final result): C1 failed
because its predictor tracks low-level MOTION PHASE (direction, velocity,
regrasp timing), not high-level ACTION IDENTITY. P1's same-action true
negatives that involve regrasp_reposition or direction_reversal already
showed continuity-surprise close to true-boundary levels (19-event and
6-event means of 2.01/2.10 vs true-positive mean 2.474) -- the phenomenon
is broad (71% of P1-correct negatives), not a 2-event edge case.

Architecture:

    left 4s window  -> frozen-PCA embed -> small bidirectional Transformer
                        -> attention pool -> L2-normalize -> s_left
    right 4s window -> (same encoder, shared weights)                -> s_right

    left window also split into two 2s halves -> s_left1, s_left2
    right window split the same way            -> s_right1, s_right2

Loss (per event):

    same-action (y=0):  pull s_left, s_right together
                         (minimize 1 - cos(s_left, s_right))
    sharp        (y=1):  push apart beyond a margin
                         (hinge: max(0, margin - (1 - cos(s_left, s_right))))
    ALL events:          s_left1 ~= s_left2, s_right1 ~= s_right2
                         (intra-window stability -- this is the term that
                         forces the encoder to discard phase/velocity/
                         regrasp-timing information: if direction reversal
                         happened INSIDE the 4s window, s_left1 and s_left2
                         still have to agree despite very different local
                         motion, so the encoder cannot use raw velocity as
                         its representation)

Collapse is not free: if the encoder maps everything to one point, sharp
events pay the full hinge margin (1-cos=0 gives loss=margin>0), so the
sharp-vs-same-action term alone prevents total collapse as long as sharp
examples exist in training, which they always do (108/145). Verified below
on synthetic data anyway, since "the loss discourages collapse" and "SGD
actually avoids it in practice" are different claims.

CRITICAL synthetic test (run before trusting this on real features): can
this architecture separate "same action with an internal phase reversal"
from "true identity change", on a HELD-OUT recording, when a single
distance/prediction-error probe (C1's failure mode) cannot? See
scripts/verify_slow_latent_synthetic() equivalent in the test invoked from
this module's __main__ guard is not included here -- verification was run
ad hoc against this file during development; the important synthetic
result is recorded in memory (continuity-c1-plan.md) and the module
docstring is not a substitute for rerunning it if this file changes.

Usage (server, mirrors predictive_continuity.py's event loading):
    python -m src.boundary.slow_latent_c2 \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --same_action_subtype data/gold/same_action_subtype_v1.csv \
        --epochs 60 --device cuda \
        --out /workspace/tr1/results/hal/slow_latent_c2/report.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    stratified_grouped_folds, build_matrices, REL_NAMES,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid

WINDOW_S = 4.0     # each side's full window
HALF_S = 2.0       # each side split into two halves for the stability term
EMB = 64


def infer_n_frames(recs, window_s, cap=64):
    """Same robust median-across-all-recordings logic as C1's infer_n_past
    (that function's docstring explains why reading a single recording is
    wrong)."""
    dts = []
    for rec in recs.values():
        t = rec["times"]
        if len(t) >= 3:
            dts.append(float(np.median(np.diff(t.numpy()))))
    dt = float(np.median(dts))
    return min(cap, max(2, int(round(window_s / dt))))


class SlowLatentEncoder(torch.nn.Module):
    """Frozen-PCA embed -> small bidirectional Transformer -> attention pool
    -> L2-normalized slow-state vector. No causal mask: this summarizes a
    fixed window, it does not predict anything sequentially."""

    def __init__(self, d_in, n_frames, d=EMB, nhead=4, nlayers=2):
        super().__init__()
        self.n_frames = n_frames
        self.register_buffer("pca_mu", torch.zeros(d_in))
        self.register_buffer("pca_W", torch.zeros(d_in, d))
        self.pos = torch.nn.Parameter(torch.randn(n_frames, d) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=nlayers)
        self.pool_query = torch.nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = torch.nn.MultiheadAttention(d, nhead, batch_first=True)
        self.head = torch.nn.Linear(d, d)

    def fit_pca(self, frames):
        mu = frames.mean(0)
        X = (frames - mu).numpy()
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        d = self.pca_W.shape[1]
        W = Vt[:d].T
        if W.shape[1] < d:
            W = np.pad(W, ((0, 0), (0, d - W.shape[1])))
        self.pca_mu.copy_(mu)
        self.pca_W.copy_(torch.tensor(W, dtype=torch.float32))

    def embed(self, x):
        return (x - self.pca_mu) @ self.pca_W

    def forward(self, window):          # window: [B, T, d_in] (T can be < n_frames)
        h = self.embed(window) + self.pos[:window.shape[1]]
        h = self.enc(h)                 # no mask -- full bidirectional context
        q = self.pool_query.expand(h.shape[0], -1, -1)
        pooled, _ = self.pool(q, h, h)  # [B, 1, d]
        s = self.head(pooled.squeeze(1))
        return F.normalize(s, dim=-1)


def _slice_stack(feats, times, lo, hi, n_frames):
    """Frames in [lo, hi), resampled/padded to exactly n_frames. Returns
    None if fewer than 60% of n_frames are actually available (too close
    to a recording edge or a clipped neighbor boundary)."""
    m = (times >= lo) & (times < hi)
    idx = torch.nonzero(m).flatten()
    if idx.numel() < max(2, int(0.6 * n_frames)):
        return None
    x = feats[idx]
    if x.shape[0] >= n_frames:
        # evenly subsample to n_frames so windows near a clip boundary
        # (fewer raw frames than the nominal count) still line up with the
        # positional embedding shape used at training time
        pick = torch.linspace(0, x.shape[0] - 1, n_frames).round().long()
        return x[pick]
    pad = x[:1].repeat(n_frames - x.shape[0], 1)
    return torch.cat([pad, x], 0)


def build_windows(rec, t, n_frames, clip_neighbors=False):
    """Returns dict with L, R, L1, L2, R1, R2 (each [n_frames or n_frames/2,
    D] or None), using neighbor-clipped bounds exactly like pairwise_verifier's
    side_vectors when clip_neighbors=True (imported lazily to avoid a hard
    dependency loop)."""
    from src.boundary.pairwise_verifier import neighbor_bounds
    times, feats = rec["times"], rec["feats"]
    if clip_neighbors:
        lo, hi = neighbor_bounds(rec, t)
    else:
        lo, hi = float(times[0]), float(times[-1])
    Llo, Lhi = max(lo, t - WINDOW_S), t
    Rlo, Rhi = t, min(hi, t + WINDOW_S)
    nh = max(2, n_frames // 2)
    return {
        "L": _slice_stack(feats, times, Llo, Lhi, n_frames),
        "R": _slice_stack(feats, times, Rlo, Rhi, n_frames),
        "L1": _slice_stack(feats, times, Llo, (Llo + Lhi) / 2, nh),
        "L2": _slice_stack(feats, times, (Llo + Lhi) / 2, Lhi, nh),
        "R1": _slice_stack(feats, times, Rlo, (Rlo + Rhi) / 2, nh),
        "R2": _slice_stack(feats, times, (Rlo + Rhi) / 2, Rhi, nh),
    }


def slow_latent_loss(model, batch, margin=0.5, lambda_stability=1.0, device="cpu"):
    """batch: list of (windows_dict, y). Returns (loss, distance_per_event)."""
    keys = ["L", "R", "L1", "L2", "R1", "R2"]
    stacked = {k: [] for k in keys}
    ys, usable = [], []
    for w, y in batch:
        if any(w[k] is None for k in keys):
            usable.append(False)
            continue
        usable.append(True)
        for k in keys:
            stacked[k].append(w[k])
        ys.append(y)
    if not ys:
        return None, None
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    enc = {}
    for k in keys:
        x = torch.stack(stacked[k]).float().to(device)
        enc[k] = model(x)
    d_lr = 1 - F.cosine_similarity(enc["L"], enc["R"])
    d_l = 1 - F.cosine_similarity(enc["L1"], enc["L2"])
    d_r = 1 - F.cosine_similarity(enc["R1"], enc["R2"])

    pull = (d_lr * (1 - y)).sum() / max(1, int((1 - y).sum()))
    push = (F.relu(margin - d_lr) * y).sum() / max(1, int(y.sum()))
    stability = (d_l + d_r).mean()
    loss = pull + push + lambda_stability * stability
    return loss, d_lr.detach().cpu().numpy()


def score_events(model, events, n_frames, device, clip_neighbors=False, return_geometry=False):
    """Returns score = distance(s_left, s_right) (higher = more separated =
    more "sharp-like"). If return_geometry, also returns d_l, d_r (the two
    intra-window stability distances) so a postmortem can check whether the
    encoder is actually behaving "slow" (low d_l/d_r everywhere) rather than
    just checking the final d_lr-based classification number -- section 5's
    point that AUROC alone cannot tell you whether phase-invariance was
    actually learned or whether the classifier head found an unrelated
    shortcut."""
    scores = np.full(len(events), np.nan)
    d_ls = np.full(len(events), np.nan)
    d_rs = np.full(len(events), np.nan)
    model.eval()
    with torch.no_grad():
        for i, e in enumerate(events):
            w = build_windows(e["rec"], e["t"], n_frames, clip_neighbors)
            if w["L"] is None or w["R"] is None:
                continue
            sl = model(w["L"][None].float().to(device))
            sr = model(w["R"][None].float().to(device))
            scores[i] = float((1 - F.cosine_similarity(sl, sr)).item())
            if return_geometry and w["L1"] is not None and w["L2"] is not None:
                sl1 = model(w["L1"][None].float().to(device))
                sl2 = model(w["L2"][None].float().to(device))
                d_ls[i] = float((1 - F.cosine_similarity(sl1, sl2)).item())
            if return_geometry and w["R1"] is not None and w["R2"] is not None:
                sr1 = model(w["R1"][None].float().to(device))
                sr2 = model(w["R2"][None].float().to(device))
                d_rs[i] = float((1 - F.cosine_similarity(sr1, sr2)).item())
    if return_geometry:
        return scores, d_ls, d_rs
    return scores


def train_fold(events_tr, d_in, n_frames, epochs, device, seed=0, lr=1e-3,
               batch_size=16, clip_neighbors=False):
    torch.manual_seed(seed)
    model = SlowLatentEncoder(d_in, n_frames).to(device)
    sub = torch.cat([e["rec"]["feats"][torch.randperm(len(e["rec"]["feats"]))[:100]]
                     for e in events_tr[:40]], 0).float()
    model.fit_pca(sub.cpu())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    windows = [(build_windows(e["rec"], e["t"], n_frames, clip_neighbors), e["y"])
              for e in events_tr]
    windows = [(w, y) for w, y in windows if all(w[k] is not None for k in w)]
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(windows))
        tot, n = 0.0, 0
        for i in range(0, len(idx), batch_size):
            batch = [windows[j] for j in idx[i:i + batch_size]]
            loss, _ = slow_latent_loss(model, batch, device=device)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); n += 1
    return model


def p1_plus_c2_fold_eval(Ls, Rs, X_rel, y, groups, folds, c2_events, n_frames,
                         epochs, device, pca_dim=64, l2=5.0, clip_neighbors=False):
    """Grouped CV where, per fold, BOTH P1's PCA/logreg AND the C2 encoder
    are fit on the training split only, then both score the held-out fold.
    Mirrors predictive_continuity.p1_fold_eval's no-leakage structure."""
    oof_p1 = np.full(len(y), np.nan)
    oof_c2 = np.full(len(y), np.nan)
    oof_fused = np.full(len(y), np.nan)
    oof_dl = np.full(len(y), np.nan)
    oof_dr = np.full(len(y), np.nan)
    per_fold_p1, per_fold_fused = [], []
    for fi, f in enumerate(folds):
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 4 or len(set(y[tr].tolist())) < 2:
            continue
        pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), pca_dim)
        st_rel = _impute_scale_fit(X_rel[tr])
        def blk(m):
            return np.concatenate([pair_block(pca_apply(pca, Ls[m]), pca_apply(pca, Rs[m])),
                                   _impute_scale_apply(st_rel, X_rel[m])], 1)
        Ptr = blk(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), y[tr], l2=l2)
        p1_te = _sigmoid(_impute_scale_apply(stP, blk(te)) @ w + b)
        oof_p1[te] = p1_te

        d_in = c2_events[0]["rec"]["feats"].shape[1]
        events_tr = [c2_events[i] for i in np.nonzero(tr)[0]]
        model = train_fold(events_tr, d_in, n_frames, epochs, device, seed=fi,
                           clip_neighbors=clip_neighbors)
        c2_te, dl_te, dr_te = score_events(
            model, [c2_events[i] for i in np.nonzero(te)[0]],
            n_frames, device, clip_neighbors, return_geometry=True)
        oof_c2[np.nonzero(te)[0]] = c2_te
        oof_dl[np.nonzero(te)[0]] = dl_te
        oof_dr[np.nonzero(te)[0]] = dr_te

        # simple late fusion: average of two independently-calibrated [0,1]
        # scores (P1's sigmoid output and C2's distance rescaled by the
        # TRAIN split's own min/max, never touching test labels)
        c2_tr = score_events(model, events_tr, n_frames, device, clip_neighbors)
        finite = c2_tr[np.isfinite(c2_tr)]
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        c2_te_norm = np.clip((c2_te - lo) / max(hi - lo, 1e-6), 0, 1)
        fused = np.where(np.isfinite(c2_te_norm), 0.5 * p1_te + 0.5 * c2_te_norm, p1_te)
        oof_fused[np.nonzero(te)[0]] = fused

    m1 = np.isfinite(oof_p1)
    m2 = np.isfinite(oof_fused)
    au_p1 = _auroc(y[m1], oof_p1[m1]) if len(set(y[m1].tolist())) == 2 else float("nan")
    au_fused = _auroc(y[m2], oof_fused[m2]) if len(set(y[m2].tolist())) == 2 else float("nan")
    au_c2_only = (_auroc(y[np.isfinite(oof_c2)], oof_c2[np.isfinite(oof_c2)])
                 if len(set(y[np.isfinite(oof_c2)].tolist())) == 2 else float("nan"))
    for f in folds:
        te = np.array([g in f for g in groups]) & m1
        if te.sum() >= 2 and len(set(y[te].tolist())) == 2:
            per_fold_p1.append(_auroc(y[te], oof_p1[te]))
        te2 = np.array([g in f for g in groups]) & m2
        if te2.sum() >= 2 and len(set(y[te2].tolist())) == 2:
            per_fold_fused.append(_auroc(y[te2], oof_fused[te2]))
    return {"au_p1": au_p1, "au_c2_only": au_c2_only, "au_fused": au_fused,
            "per_fold_p1": per_fold_p1, "per_fold_fused": per_fold_fused,
            "oof_p1": oof_p1, "oof_c2": oof_c2, "oof_fused": oof_fused,
            "oof_dl": oof_dl, "oof_dr": oof_dr}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--same_action_subtype", default="data/gold/same_action_subtype_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="2 fps caches -- used for P1's own baseline features "
                         "(build_matrices), matching every other P1 baseline "
                         "number reported throughout this project.")
    ap.add_argument("--c2_feat_cache", action="append", default=None,
                    help="caches C2's slow-latent encoder trains/scores on "
                         "(e.g. 10 fps). Defaults to --feat_cache. Kept SEPARATE "
                         "from P1's features for the same reason predictive_"
                         "continuity.py separates --feat_cache/--cont_feat_cache: "
                         "P1's baseline AUROC must stay reproducible across every "
                         "report, and fine motion (regrasp, reversal) is exactly "
                         "what 2 fps under-samples (Nyquist ~1 Hz).")
    ap.add_argument("--clip_windows_at_neighbors", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_config", default="configs/slow_latent_gate_c2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_events",
                    help="optional CSV path: one row per clean-145 event with "
                         "event_id, recording_id, y, dev_pair_subtype, "
                         "same_action_subtype (joined from --same_action_subtype "
                         "for negatives), OOF d_lr/d_l/d_r geometry, and OOF "
                         "P1/C2/fused scores -- the input to c2_postmortem.py.")
    a = ap.parse_args()

    by_rid = load_feature_caches(a.feat_cache)
    c2_by_rid = load_feature_caches(a.c2_feat_cache) if a.c2_feat_cache else by_rid
    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    events = build_events(gold, ctx, by_rid)
    labels = T.load_pair_labels(a.pair_labels)
    events = T.apply_to_events(events, labels)
    print(f"clean events: {len(events)}  "
          f"subtypes={dict(Counter(e.get('temporal_pair_subtype') for e in events))}")

    X_v1, Ls, Rs, X_rel, keep, crops = build_matrices(events, a.clip_windows_at_neighbors)
    events = [events[i] for i in keep]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = [e["recording_id"] for e in events]

    n_frames = infer_n_frames(c2_by_rid, WINDOW_S)
    print(f"window frames: {n_frames} per {WINDOW_S}s side window "
          f"(from cache's actual frame rate)")

    # C2 windows come from c2_by_rid (may be a different-fps cache); build a
    # PARALLEL event list carrying that cache's rec, keyed to the same
    # event_id/recording_id/t/y so P1's grouped folds and C2's windows always
    # refer to the same underlying event.
    n_missing_c2 = sum(e["recording_id"] not in c2_by_rid for e in events)
    if n_missing_c2:
        print(f"  !! {n_missing_c2} events lack a C2 cache for their recording "
              f"-- those events will be unscorable for C2 (P1 alone still works)")
    c2_events = [dict(e, rec=c2_by_rid.get(e["recording_id"])) for e in events]

    folds = stratified_grouped_folds(groups, y, 5, seed=0)
    res = p1_plus_c2_fold_eval(Ls, Rs, X_rel, y, groups, folds, c2_events, n_frames,
                               a.epochs, a.device, clip_neighbors=a.clip_windows_at_neighbors)

    au_p1, au_c2, au_fused = res["au_p1"], res["au_c2_only"], res["au_fused"]
    gain = au_fused - au_p1
    pf_p1, pf_f = res["per_fold_p1"], res["per_fold_fused"]
    folds_improved = sum(f > p for f, p in zip(pf_f, pf_p1))
    worst_drop = (min(pf_f) - min(pf_p1)) if pf_f and pf_p1 else float("nan")
    print(f"\nP1 alone:        {au_p1:.3f}  per-fold {[round(x,3) for x in pf_p1]}")
    print(f"C2 alone:        {au_c2:.3f}")
    print(f"P1 + C2 (fused): {au_fused:.3f}  per-fold {[round(x,3) for x in pf_f]}")
    print(f"gain {gain:+.3f}  folds improved {folds_improved}/{len(pf_p1)}  "
          f"worst-fold change {worst_drop:+.3f}")

    # subtype-specific false-positive check (mentor's point 8.3): does C2
    # INCREASE the false-positive rate on the two subtypes that broke C1?
    import csv
    diag = {}
    sub_map = {}
    if os.path.exists(a.same_action_subtype):
        with open(a.same_action_subtype, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sub_map[r["event_id"]] = r["subtype"]
        watch = {"regrasp_reposition", "direction_reversal"}
        cutoff_p1 = float(np.nanmedian(res["oof_p1"][y == 1]))
        cutoff_fused = float(np.nanmedian(res["oof_fused"][y == 1]))
        for name, oof, cutoff in [("p1", res["oof_p1"], cutoff_p1),
                                  ("fused", res["oof_fused"], cutoff_fused)]:
            neg_watch = [i for i, e in enumerate(events)
                        if y[i] == 0 and sub_map.get(e["event_id"]) in watch
                        and np.isfinite(oof[i])]
            fp = sum(oof[i] >= cutoff for i in neg_watch)
            rate = fp / len(neg_watch) if neg_watch else float("nan")
            print(f"  {name}: regrasp+direction_reversal FP rate = {rate:.3f} "
                  f"({fp}/{len(neg_watch)})")
            diag[f"{name}_watch_subtype_fp_rate"] = rate
    else:
        print(f"  (no subtype file at {a.same_action_subtype} -- skipping "
              f"regrasp/direction_reversal FP check)")

    if a.dump_events:
        with open(os.path.expanduser(a.dump_events), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "y", "dev_pair_subtype",
                       "same_action_subtype", "oof_d_lr", "oof_d_l", "oof_d_r",
                       "oof_p1", "oof_c2", "oof_fused"])
            def fmt(v):
                return "" if v is None or not np.isfinite(v) else f"{v:.6f}"
            for i, e in enumerate(events):
                w.writerow([e["event_id"], e["recording_id"], int(y[i]),
                           e.get("temporal_pair_subtype") or "",
                           sub_map.get(e["event_id"], ""),
                           fmt(res["oof_c2"][i]), fmt(res["oof_dl"][i]), fmt(res["oof_dr"][i]),
                           fmt(res["oof_p1"][i]), fmt(res["oof_c2"][i]), fmt(res["oof_fused"][i])])
        print(f"\nwrote per-event dump ({len(events)} rows) -> {a.dump_events}")

    verdict = "NOT EVALUATED (gate config missing)"
    if os.path.exists(a.gate_config):
        gate = json.load(open(a.gate_config, encoding="utf-8"))
        watch_ok = True
        if "p1_watch_subtype_fp_rate" in diag and "fused_watch_subtype_fp_rate" in diag:
            watch_ok = (diag["fused_watch_subtype_fp_rate"]
                       <= diag["p1_watch_subtype_fp_rate"] + gate["max_watch_subtype_fp_increase"])
        ok = (gain >= gate["min_auroc_gain"]
              and (folds_improved / len(pf_p1) if pf_p1 else 0) >= gate["min_folds_improved_frac"]
              and (worst_drop >= -gate["max_worst_fold_drop"] if pf_p1 else False)
              and watch_ok)
        verdict = "ADOPT C2" if ok else "DO NOT ADOPT (gate failed)"
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
        json.dump({
            "n_events": len(events), "window_frames": n_frames, "epochs": a.epochs,
            "p1_alone": {"pooled": au_p1, "per_fold": pf_p1},
            "c2_alone": au_c2,
            "p1_plus_c2_fused": {"pooled": au_fused, "per_fold": pf_f},
            "gain": gain, "folds_improved": folds_improved,
            "worst_fold_change": worst_drop, "diagnostics": diag, "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
