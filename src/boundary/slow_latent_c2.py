"""C2-v2: slow-semantic-latent verifier, rewritten after a mentor review of
v1's report_v1/postmortem found two structural bugs that made v1's result
uninterpretable as a test of the slow-latent hypothesis (not a sign bug or
CV leak -- a representation-path and scoring-path mismatch):

  BUG 1 (v1): L (40 frames) and its halves L1/L2 (20 frames each) were
  encoded as three SEPARATE forward passes through the shared encoder.
  Different sequence lengths mean different position-embedding ranges and
  different attention-pooling distributions -- nothing forces the 20-frame
  path's output to be the SAME representation that the 40-frame path uses
  for classification. The "intra-window stability" term could be satisfied
  by a computational shortcut specific to the 20-frame path while the
  40-frame path kept encoding raw motion/phase, which defeats the entire
  point of the constraint.
  FIX (v2): encode each FULL side window exactly once, keep the Transformer's
  hidden-state sequence, and pool THREE different slices of that SAME
  sequence (full, first half, second half) for s_L/s_L1/s_L2. Stability now
  constrains the actual representation used for d_lr.

  BUG 2 (v1): inference used only d_lr = distance(s_L, s_R). d_l, d_r were
  computed during training (for the stability loss) but discarded at
  scoring time -- exactly the information that should distinguish "these
  two sides differ because the ACTION changed" from "these two sides differ
  because side is internally unstable" (v1's own geometry diagnostic found
  direction_reversal's d_l = 0.206, ~9x every other subtype -- the reversal
  was happening WITHIN one side, invisible to a d_lr-only score).
  FIX (v2): score q = d_lr - alpha * 0.5*(d_l + d_r), alpha=0.5 fixed before
  seeing any v2 result. d_lr alone is still computed and reported for
  comparison, never silently dropped.

Also fixed per the same review, none of them hyperparameter tuning:
  - main discriminative window WINDOW_S 4.0 -> 2.0s (v1's own dev-165
    windowing diagnostics found ~66% of events get their 4s side window
    truncated by a neighboring annotation boundary under clip_neighbors,
    median usable half-window ~2s -- a 4s "this side is one stable action"
    assumption was optimistic for roughly two-thirds of events)
  - padding is now tracked with an explicit mask (src_key_padding_mask in
    the Transformer, key_padding_mask in attention pooling) instead of
    silently duplicating the edge frame and letting the model treat it as
    an independent real observation
  - a VICReg-style variance floor (anti-collapse: v1's C2-alone=0.563 could
    not distinguish "learned a modest real signal" from "near-collapsed,
    low effective rank, OOF ranking still weakly informative by accident")
  - cross-scale consistency: s_L ~= normalize(s_L1 + s_L2), guards against
    the pooling operator finding a shortcut even with shared hidden states
  - 3 random seeds per fold, scores averaged -- a single-seed +0.003 on 145
    events proves nothing about generalization

Fixed weights, NOT tuned against any result (mentor's explicit instruction:
"第一版不要做大型 grid search"):
    lambda_stability = 0.5, lambda_scale = 0.25, lambda_var = 0.1
    tau (stability robust-clip cap) = 0.2, gamma (variance floor) = 0.05
    alpha (intra-window discount in the score) = 0.5

Usage (server):
    python -m src.boundary.slow_latent_c2 \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --c2_feat_cache /workspace/tr1/data_recseg/feat_10fps_full_noblur_multi.pt \
        --c2_feat_cache /workspace/tr1/data_recseg/feat_10fps_missing_multi.pt \
        --c2_feat_cache /workspace/tr1/data_recseg/feat_10fps_refill_multi.pt \
        --c2_feat_cache /workspace/tr1/data_recseg/feat_10fps_eval_clean_multi.pt \
        --same_action_subtype data/gold/same_action_subtype_v1.csv \
        --epochs 60 --n_seeds 3 --device cuda \
        --out /workspace/tr1/results/hal/slow_latent_c2/report_v2.json \
        --dump_events /workspace/tr1/results/hal/slow_latent_c2/events_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    stratified_grouped_folds, build_matrices,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid

WINDOW_S = 2.0          # each side's PRIMARY discriminative window (was 4.0)
EMB = 64
TAU_STABILITY = 0.2     # robust clip cap on d_l/d_r inside the stability loss
GAMMA_VAR = 0.05        # per-dimension batch-std floor
LAMBDA_STABILITY = 0.5
LAMBDA_SCALE = 0.25
LAMBDA_VAR = 0.1
ALPHA_INTRA = 0.5       # score = d_lr - ALPHA_INTRA * 0.5*(d_l + d_r)
N_SEEDS = 3


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
    return min(cap, max(4, int(round(window_s / dt))))


class SlowLatentEncoder(torch.nn.Module):
    """Frozen-PCA embed -> small bidirectional Transformer -> attention pool.
    Exposes encode_hidden/pool_hidden SEPARATELY (v1 only exposed a single
    forward() that did both) so a caller can encode a window's hidden-state
    sequence ONCE and pool different slices of it for the full/half
    representations -- the core v1 fix."""

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

    def encode_hidden(self, window, pad_mask):
        """window: [B,T,d_in], pad_mask: [B,T] bool (True = PADDING, ignore).
        Returns H: [B,T,d] -- the Transformer's per-position hidden states,
        NOT pooled yet."""
        h = self.embed(window) + self.pos[:window.shape[1]]
        return self.enc(h, src_key_padding_mask=pad_mask)

    def pool_hidden(self, H, pad_mask):
        """H: [B,T',d] (any contiguous slice of a hidden-state sequence),
        pad_mask: [B,T'] bool. Every row's mask MUST have >=1 False (real)
        entry -- callers with a possibly-fully-padded slice must handle that
        themselves (see encode_sides_batch's `valid` flags)."""
        q = self.pool_query.expand(H.shape[0], -1, -1)
        pooled, _ = self.pool(q, H, H, key_padding_mask=pad_mask)
        return F.normalize(self.head(pooled.squeeze(1)), dim=-1)


def _slice_stack_masked(feats, times, lo, hi, n_frames):
    """Frames in [lo, hi), resampled/left-padded to exactly n_frames.
    Returns (frames[n_frames,D], pad_mask[n_frames] bool True=PAD), or None
    if fewer than 60% of n_frames are actually available. Unlike v1, padded
    positions are marked so the model can ignore them (src_key_padding_mask)
    instead of treating a duplicated edge frame as an independent real
    observation."""
    m = (times >= lo) & (times < hi)
    idx = torch.nonzero(m).flatten()
    if idx.numel() < max(2, int(0.6 * n_frames)):
        return None
    x = feats[idx]
    if x.shape[0] >= n_frames:
        pick = torch.linspace(0, x.shape[0] - 1, n_frames).round().long()
        return x[pick], torch.zeros(n_frames, dtype=torch.bool)
    n_pad = n_frames - x.shape[0]
    pad = x[:1].repeat(n_pad, 1)
    frames = torch.cat([pad, x], 0)
    mask = torch.cat([torch.ones(n_pad, dtype=torch.bool),
                      torch.zeros(x.shape[0], dtype=torch.bool)])
    return frames, mask


def build_windows(rec, t, n_frames, clip_neighbors=False):
    """Returns {"L": (frames,mask) or None, "R": (frames,mask) or None}.
    Only the FULL side window is extracted -- halves come from slicing the
    encoder's hidden states, not from separate extraction (see module
    docstring, BUG 1)."""
    from src.boundary.pairwise_verifier import neighbor_bounds
    times, feats = rec["times"], rec["feats"]
    if clip_neighbors:
        lo, hi = neighbor_bounds(rec, t)
    else:
        lo, hi = float(times[0]), float(times[-1])
    Llo, Lhi = max(lo, t - WINDOW_S), t
    Rlo, Rhi = t, min(hi, t + WINDOW_S)
    return {
        "L": _slice_stack_masked(feats, times, Llo, Lhi, n_frames),
        "R": _slice_stack_masked(feats, times, Rlo, Rhi, n_frames),
    }


def encode_sides_batch(model, frames, pad_mask, device):
    """frames: [B,T,d_in], pad_mask: [B,T] bool. Encodes ONCE, pools full +
    both halves from the SAME hidden-state sequence. Returns
    (s_full, s1, valid1, s2, valid2) where valid1/valid2 are [B] bool flags
    -- a half can be fully padding (e.g. the window is mostly missing near a
    clipped neighbor boundary); such rows get a safe dummy pooled vector
    (never used downstream, guarded by the valid flag) instead of NaN from a
    softmax over an all -inf row."""
    frames = frames.to(device); pad_mask = pad_mask.to(device)
    H = model.encode_hidden(frames, pad_mask)
    s_full = model.pool_hidden(H, pad_mask)
    T_ = frames.shape[1]
    half = T_ // 2
    m1, m2 = pad_mask[:, :half], pad_mask[:, half:]
    valid1, valid2 = ~m1.all(dim=1), ~m2.all(dim=1)
    m1_safe = m1.clone(); m1_safe[~valid1, 0] = False
    m2_safe = m2.clone(); m2_safe[~valid2, 0] = False
    s1 = model.pool_hidden(H[:, :half], m1_safe)
    s2 = model.pool_hidden(H[:, half:], m2_safe)
    return s_full, s1, valid1, s2, valid2


def slow_latent_loss(model, batch, device="cpu", margin=0.5,
                     tau=TAU_STABILITY, lam_stability=LAMBDA_STABILITY,
                     lam_scale=LAMBDA_SCALE, lam_var=LAMBDA_VAR, gamma_var=GAMMA_VAR):
    """batch: list of (windows_dict, y). Returns (loss, component_dict) or
    (None, None) if nothing in the batch was scorable."""
    Lf, Lm, Rf, Rm, ys = [], [], [], [], []
    for w, y in batch:
        if w["L"] is None or w["R"] is None:
            continue
        lf, lm = w["L"]; rf, rm = w["R"]
        Lf.append(lf); Lm.append(lm); Rf.append(rf); Rm.append(rm); ys.append(y)
    if not ys:
        return None, None
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    Lf, Lm = torch.stack(Lf).float(), torch.stack(Lm)
    Rf, Rm = torch.stack(Rf).float(), torch.stack(Rm)

    s_L, s_L1, v_L1, s_L2, v_L2 = encode_sides_batch(model, Lf, Lm, device)
    s_R, s_R1, v_R1, s_R2, v_R2 = encode_sides_batch(model, Rf, Rm, device)

    d_lr = 1 - F.cosine_similarity(s_L, s_R)
    pull = (d_lr * (1 - y)).sum() / max(1, int((1 - y).sum()))
    push = (F.relu(margin - d_lr) * y).sum() / max(1, int(y.sum()))

    valid_l, valid_r = v_L1 & v_L2, v_R1 & v_R2
    d_l = 1 - F.cosine_similarity(s_L1, s_L2)
    d_r = 1 - F.cosine_similarity(s_R1, s_R2)
    d_l_clamped = torch.clamp(d_l, max=tau) * valid_l.float()
    d_r_clamped = torch.clamp(d_r, max=tau) * valid_r.float()
    n_valid = max(1, int(valid_l.float().sum() + valid_r.float().sum()))
    stability = (d_l_clamped.sum() + d_r_clamped.sum()) / n_valid

    combo_L = F.normalize(s_L1 + s_L2, dim=-1)
    combo_R = F.normalize(s_R1 + s_R2, dim=-1)
    scale_L = (1 - F.cosine_similarity(s_L, combo_L)) * valid_l.float()
    scale_R = (1 - F.cosine_similarity(s_R, combo_R)) * valid_r.float()
    scale = (scale_L.sum() + scale_R.sum()) / n_valid

    allvecs = torch.cat([s_L, s_R], 0)
    var_loss = F.relu(gamma_var - allvecs.std(dim=0)).mean()

    loss = pull + push + lam_stability * stability + lam_scale * scale + lam_var * var_loss
    comp = {"pull": float(pull.detach()), "push": float(push.detach()),
           "stability": float(stability.detach()), "scale": float(scale.detach()),
           "var": float(var_loss.detach())}
    return loss, comp


def score_events(model, events, n_frames, device, clip_neighbors=False):
    """Returns d_lr, d_l, d_r, q (each len(events), np.nan where
    unscorable). q = d_lr - ALPHA_INTRA * 0.5*(d_l+d_r) when both d_l,d_r
    are available, else falls back to raw d_lr (flagged via q==d_lr, not a
    silent default)."""
    d_lr = np.full(len(events), np.nan)
    d_l = np.full(len(events), np.nan)
    d_r = np.full(len(events), np.nan)
    model.eval()
    with torch.no_grad():
        for i, e in enumerate(events):
            w = build_windows(e["rec"], e["t"], n_frames, clip_neighbors)
            if w["L"] is None or w["R"] is None:
                continue
            lf, lm = w["L"]; rf, rm = w["R"]
            s_L, s_L1, v_L1, s_L2, v_L2 = encode_sides_batch(
                model, lf[None].float(), lm[None], device)
            s_R, s_R1, v_R1, s_R2, v_R2 = encode_sides_batch(
                model, rf[None].float(), rm[None], device)
            d_lr[i] = float((1 - F.cosine_similarity(s_L, s_R)).item())
            if bool(v_L1[0]) and bool(v_L2[0]):
                d_l[i] = float((1 - F.cosine_similarity(s_L1, s_L2)).item())
            if bool(v_R1[0]) and bool(v_R2[0]):
                d_r[i] = float((1 - F.cosine_similarity(s_R1, s_R2)).item())
    has_intra = np.isfinite(d_l) & np.isfinite(d_r)
    d_intra = np.where(has_intra, 0.5 * (np.nan_to_num(d_l) + np.nan_to_num(d_r)), 0.0)
    q = np.where(has_intra, d_lr - ALPHA_INTRA * d_intra, d_lr)
    return d_lr, d_l, d_r, q


def train_fold(events_tr, d_in, n_frames, epochs, device, seed=0, lr=1e-3, batch_size=16):
    torch.manual_seed(seed)
    model = SlowLatentEncoder(d_in, n_frames).to(device)
    sub = torch.cat([e["rec"]["feats"][torch.randperm(len(e["rec"]["feats"]))[:100]]
                     for e in events_tr[:40]], 0).float()
    model.fit_pca(sub.cpu())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    windows = [(build_windows(e["rec"], e["t"], n_frames), e["y"]) for e in events_tr]
    windows = [(w, y) for w, y in windows if w["L"] is not None and w["R"] is not None]
    rng = np.random.RandomState(seed)
    last_comp = {}
    for ep in range(epochs):
        idx = rng.permutation(len(windows))
        agg = Counter()
        n = 0
        for i in range(0, len(idx), batch_size):
            batch = [windows[j] for j in idx[i:i + batch_size]]
            loss, comp = slow_latent_loss(model, batch, device=device)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in comp.items():
                agg[k] += v
            n += 1
        last_comp = {k: v / max(1, n) for k, v in agg.items()}
    return model, last_comp


def train_fold_multiseed(events_tr, d_in, n_frames, epochs, device, base_seed, n_seeds=N_SEEDS):
    """Trains n_seeds independent models on the SAME training split. Returns
    the list of (model, final_loss_components) -- averaging happens at
    SCORE time (score_multiseed), not by averaging weights."""
    out = []
    for s in range(n_seeds):
        model, comp = train_fold(events_tr, d_in, n_frames, epochs, device,
                                 seed=base_seed * 100 + s)
        out.append((model, comp))
    return out


def score_multiseed(models, events, n_frames, device):
    """Averages d_lr/d_l/d_r/q across seeds (per event, ignoring NaN from
    any seed that found an event unscorable -- that should not happen since
    scorability depends only on the data, not the seed, but guarded anyway).

    Returns the PER-SEED score matrices as well, not just the mean and the
    per-event std: a mean per-event std of 0.185 tells you individual scores
    move a lot, but not whether each seed's model is individually a decent
    ranker (three good models disagreeing on hard events) or individually a
    poor one (three noisy models). Those two cases have different meanings
    for whether the approach is data-starved, so the archival report needs
    each seed's own pooled OOF AUROC."""
    all_q, all_dlr = [], []
    for model, _ in models:
        d_lr, d_l, d_r, q = score_events(model, events, n_frames, device)
        all_q.append(q); all_dlr.append(d_lr)
    all_q = np.array(all_q); all_dlr = np.array(all_dlr)
    return {"dlr": np.nanmean(all_dlr, axis=0), "q": np.nanmean(all_q, axis=0),
            "q_std": np.nanstd(all_q, axis=0),
            "per_seed_q": all_q, "per_seed_dlr": all_dlr}


def p1_plus_c2_fold_eval(Ls, Rs, X_rel, y, groups, folds, c2_events, n_frames,
                         epochs, device, n_seeds=N_SEEDS, pca_dim=64, l2=5.0,
                         nested_fusion=True):
    """Grouped CV: P1's PCA/logreg and C2's encoder(s) are both fit on the
    training split only, per outer fold (no leakage). C2 trains n_seeds
    models per fold and scores are averaged across seeds.

    Reports THREE ways of combining P1 and C2, per mentor's point 5/7, none
    presented as "the" result in isolation:
      fixed_fusion:  0.5*P1_sigmoid + 0.5*minmax(C2 q-score)  (comparable to
                     v1's reported number)
      nested_fusion: a small L2-regularized logistic regression on
                     [P1_logit, d_lr, d_l, d_r, q], fit on an INNER grouped
                     split of the outer-fold's training data (so the LR
                     itself never sees in-sample C2 scores), applied to the
                     outer test fold. This is an approximation of a full
                     nested design (single seed for the inner C2 model, to
                     bound cost) and is reported as a diagnostic, not a
                     replacement for fixed_fusion.
      c2_alone / d_lr_alone: for comparison against v1's raw-d_lr number.
    """
    oof_p1 = np.full(len(y), np.nan)
    oof_dlr = np.full(len(y), np.nan)
    oof_q = np.full(len(y), np.nan)
    oof_q_std = np.full(len(y), np.nan)
    oof_dl = np.full(len(y), np.nan)
    oof_dr = np.full(len(y), np.nan)
    oof_fixed = np.full(len(y), np.nan)
    oof_fixed_dlr = np.full(len(y), np.nan)
    oof_nested = np.full(len(y), np.nan)
    oof_q_per_seed = np.full((n_seeds, len(y)), np.nan)
    oof_dlr_per_seed = np.full((n_seeds, len(y)), np.nan)
    per_fold_p1, per_fold_fixed, per_fold_nested = [], [], []
    fold_diagnostics = []

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
        p1_tr_prob = _sigmoid(_impute_scale_apply(stP, Ptr) @ w + b)
        p1_te = _sigmoid(_impute_scale_apply(stP, blk(te)) @ w + b)
        oof_p1[te] = p1_te

        d_in = c2_events[0]["rec"]["feats"].shape[1]
        events_tr = [c2_events[i] for i in np.nonzero(tr)[0]]
        events_te = [c2_events[i] for i in np.nonzero(te)[0]]

        models = train_fold_multiseed(events_tr, d_in, n_frames, epochs, device,
                                      base_seed=fi, n_seeds=n_seeds)
        sc_te = score_multiseed(models, events_te, n_frames, device)
        sc_tr = score_multiseed(models, events_tr, n_frames, device)
        dlr_te, q_te, q_std_te = sc_te["dlr"], sc_te["q"], sc_te["q_std"]
        dlr_tr, q_tr = sc_tr["dlr"], sc_tr["q"]
        te_idx = np.nonzero(te)[0]
        for s in range(min(n_seeds, sc_te["per_seed_q"].shape[0])):
            oof_q_per_seed[s, te_idx] = sc_te["per_seed_q"][s]
            oof_dlr_per_seed[s, te_idx] = sc_te["per_seed_dlr"][s]
        # d_l/d_r from the first seed only (diagnostic geometry; averaging
        # cosine-derived distances across seeds is fine for q/d_lr since we
        # average the FINAL scores, not intermediate vectors, but for the
        # standalone d_l/d_r geometry table one representative seed is
        # sufficient and cheaper)
        _, dl_te, dr_te, _ = score_events(models[0][0], events_te, n_frames, device)

        oof_dlr[np.nonzero(te)[0]] = dlr_te
        oof_q[np.nonzero(te)[0]] = q_te
        oof_q_std[np.nonzero(te)[0]] = q_std_te
        oof_dl[np.nonzero(te)[0]] = dl_te
        oof_dr[np.nonzero(te)[0]] = dr_te

        def _fuse(tr_scores, te_scores):
            """0.5/0.5 blend of P1's sigmoid and a train-fold min-max
            normalization of a C2 score (normalizer fit on the TRAINING
            split only -- using the test fold's own min/max would leak the
            test distribution into its own scores)."""
            fin = tr_scores[np.isfinite(tr_scores)]
            lo, hi = (float(fin.min()), float(fin.max())) if len(fin) else (0.0, 1.0)
            norm = np.clip((te_scores - lo) / max(hi - lo, 1e-6), 0, 1)
            return np.where(np.isfinite(norm), 0.5 * p1_te + 0.5 * norm, p1_te)

        oof_fixed[te_idx] = _fuse(q_tr, q_te)
        # POST-HOC DESCRIPTIVE ONLY -- the same fixed fusion applied to raw
        # d_lr instead of q. Reported because q and d_lr have materially
        # different standalone AUROCs, so silently reporting only the q
        # fusion would itself be selective; NOT a gate result and explicitly
        # not grounds for reviving C2 (choosing whichever score formula
        # scores best on this OOF split is precisely the dev-set selection
        # the review prohibited).
        oof_fixed_dlr[te_idx] = _fuse(dlr_tr, dlr_te)

        if nested_fusion:
            inner_folds = stratified_grouped_folds(
                [groups[i] for i in np.nonzero(tr)[0]], y[tr], min(3, max(2, int(tr.sum() // 15))),
                seed=fi)
            inner_rows = []
            for gi in inner_folds:
                inner_te_mask = np.array([groups[i] in gi for i in np.nonzero(tr)[0]])
                if inner_te_mask.sum() < 2 or (~inner_te_mask).sum() < 4:
                    continue
                inner_tr_idx = np.nonzero(tr)[0][~inner_te_mask]
                inner_te_idx = np.nonzero(tr)[0][inner_te_mask]
                inner_events_tr = [c2_events[i] for i in inner_tr_idx]
                inner_model, _ = train_fold(inner_events_tr, d_in, n_frames, epochs, device,
                                            seed=fi * 1000 + 1)
                inner_dlr, inner_dl, inner_dr, inner_q = score_events(
                    inner_model, [c2_events[i] for i in inner_te_idx], n_frames, device)
                inner_p1_prob = p1_tr_prob[np.searchsorted(np.nonzero(tr)[0], inner_te_idx)]
                for j, gidx in enumerate(inner_te_idx):
                    inner_rows.append((inner_p1_prob[j], inner_dlr[j], inner_dl[j],
                                       inner_dr[j], inner_q[j], y[gidx]))
            if len(inner_rows) >= 15:
                arr = np.array(inner_rows, dtype=float)
                Xf, yf = arr[:, :5], arr[:, 5]
                m = np.isfinite(Xf).all(1)
                if m.sum() >= 10 and len(set(yf[m].tolist())) == 2:
                    st_nest = _impute_scale_fit(Xf[m])
                    wn, bn = fit_logreg(_impute_scale_apply(st_nest, Xf[m]), yf[m], l2=10.0)
                    Xte = np.stack([p1_te, dlr_te, dl_te, dr_te, q_te], axis=1)
                    Xte_i = np.where(np.isfinite(Xte), Xte, 0.0)
                    nested = _sigmoid(_impute_scale_apply(st_nest, Xte_i) @ wn + bn)
                    oof_nested[np.nonzero(te)[0]] = nested

        # train-vs-OOF on the SAME quantity (seed 0's d_lr, both sides) --
        # comparing seed-0 train against a 3-seed-averaged OOF would confound
        # overfitting with the variance reduction from averaging.
        tr_dlr_s0, te_dlr_s0 = sc_tr["per_seed_dlr"][0], sc_te["per_seed_dlr"][0]
        def _au(yy, ss):
            m = np.isfinite(ss)
            return (_auroc(yy[m], ss[m])
                    if m.sum() >= 2 and len(set(yy[m].tolist())) == 2 else None)
        fold_diagnostics.append({"fold": fi,
                                 "loss_components_last_epoch_seed0": models[0][1],
                                 "train_auroc_seed0": _au(y[tr], tr_dlr_s0),
                                 "oof_auroc_seed0": _au(y[te], te_dlr_s0),
                                 "n_train": int(tr.sum()), "n_test": int(te.sum())})

    m1 = np.isfinite(oof_p1)
    m2 = np.isfinite(oof_fixed)
    m3 = np.isfinite(oof_nested)
    au_p1 = _auroc(y[m1], oof_p1[m1]) if len(set(y[m1].tolist())) == 2 else float("nan")
    au_fixed = _auroc(y[m2], oof_fixed[m2]) if len(set(y[m2].tolist())) == 2 else float("nan")
    au_nested = (_auroc(y[m3], oof_nested[m3])
                if m3.sum() > 0 and len(set(y[m3].tolist())) == 2 else float("nan"))
    au_dlr = (_auroc(y[np.isfinite(oof_dlr)], oof_dlr[np.isfinite(oof_dlr)])
             if len(set(y[np.isfinite(oof_dlr)].tolist())) == 2 else float("nan"))
    au_q = (_auroc(y[np.isfinite(oof_q)], oof_q[np.isfinite(oof_q)])
           if len(set(y[np.isfinite(oof_q)].tolist())) == 2 else float("nan"))
    # Per-fold AUROCs are collected as ALIGNED (p1, fixed) PAIRS, not as two
    # independently-filtered lists: the per-fold delta below is only
    # meaningful if entry i of each list is the same fold, which independent
    # `if` guards do not guarantee if a fold ever qualifies for one score
    # and not the other.
    per_fold_delta = []
    for f in folds:
        base = np.array([g in f for g in groups])
        te, te2, te3 = base & m1, base & m2, base & m3
        ok1 = te.sum() >= 2 and len(set(y[te].tolist())) == 2
        ok2 = te2.sum() >= 2 and len(set(y[te2].tolist())) == 2
        if ok1 and ok2:
            a1, a2 = _auroc(y[te], oof_p1[te]), _auroc(y[te2], oof_fixed[te2])
            per_fold_p1.append(a1)
            per_fold_fixed.append(a2)
            per_fold_delta.append(a2 - a1)
        if te3.sum() >= 2 and len(set(y[te3].tolist())) == 2:
            per_fold_nested.append(_auroc(y[te3], oof_nested[te3]))

    m4 = np.isfinite(oof_fixed_dlr)
    au_fixed_dlr = (_auroc(y[m4], oof_fixed_dlr[m4])
                   if m4.sum() > 0 and len(set(y[m4].tolist())) == 2 else float("nan"))
    per_seed_au_q, per_seed_au_dlr = [], []
    for s in range(n_seeds):
        for arr, dest in ((oof_q_per_seed[s], per_seed_au_q),
                          (oof_dlr_per_seed[s], per_seed_au_dlr)):
            ms = np.isfinite(arr)
            dest.append(_auroc(y[ms], arr[ms])
                        if ms.sum() >= 2 and len(set(y[ms].tolist())) == 2 else float("nan"))

    return {"au_p1": au_p1, "au_dlr": au_dlr, "au_q": au_q,
            "au_fixed": au_fixed, "au_nested": au_nested, "au_fixed_dlr": au_fixed_dlr,
            "per_fold_p1": per_fold_p1, "per_fold_fixed": per_fold_fixed,
            "per_fold_nested": per_fold_nested, "per_fold_delta": per_fold_delta,
            "per_seed_au_q": per_seed_au_q, "per_seed_au_dlr": per_seed_au_dlr,
            "oof_p1": oof_p1, "oof_dlr": oof_dlr, "oof_q": oof_q, "oof_q_std": oof_q_std,
            "oof_dl": oof_dl, "oof_dr": oof_dr,
            "oof_fixed": oof_fixed, "oof_fixed_dlr": oof_fixed_dlr, "oof_nested": oof_nested,
            "fold_diagnostics": fold_diagnostics}


def geometry_report(events, y, oof_dlr, oof_dl, oof_dr, sub_map):
    """Per-subtype d_lr/d_l/d_r means (mentor's section on checking whether
    phase-invariance was actually learned, not just whether AUROC looks
    acceptable), plus embedding-collapse indicators are left to
    c2_postmortem.py (which already has effective-rank/variance tooling
    patterns established for HAL/state_adapter -- not duplicated here)."""
    groups = {}
    for i, e in enumerate(events):
        key = "sharp_visible_transition" if y[i] == 1 else (sub_map.get(e["event_id"]) or "untagged")
        groups.setdefault(key, []).append(i)
    out = {}
    for key, idx in groups.items():
        idx = np.array(idx)
        def m(arr):
            v = arr[idx]; v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else None
        out[key] = {"n": len(idx), "d_lr_mean": m(oof_dlr), "d_l_mean": m(oof_dl), "d_r_mean": m(oof_dr)}
    return out


def _provenance(a):
    """Everything needed to reproduce this run, embedded in the report so an
    archived report_v*.json is self-contained: the exact commit, the exact
    argv, and the fixed hyperparameters (which live as module constants and
    would otherwise be recoverable only by checking out the right commit)."""
    def _git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "args": vars(a),
        "fixed_hyperparameters": {
            "WINDOW_S": WINDOW_S, "EMB": EMB, "TAU_STABILITY": TAU_STABILITY,
            "GAMMA_VAR": GAMMA_VAR, "LAMBDA_STABILITY": LAMBDA_STABILITY,
            "LAMBDA_SCALE": LAMBDA_SCALE, "LAMBDA_VAR": LAMBDA_VAR,
            "ALPHA_INTRA": ALPHA_INTRA, "N_SEEDS": N_SEEDS,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--same_action_subtype", default="data/gold/same_action_subtype_v1.csv")
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="2 fps caches -- used for P1's own baseline features.")
    ap.add_argument("--c2_feat_cache", action="append", default=None,
                    help="caches C2 trains/scores on (e.g. 10 fps). Defaults to --feat_cache.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--no_nested_fusion", action="store_true",
                    help="skip the nested-fusion diagnostic (roughly 3x slower per fold)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_config", default="configs/slow_latent_gate_c2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_events")
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

    X_v1, Ls, Rs, X_rel, keep, crops = build_matrices(events, False)
    events = [events[i] for i in keep]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = [e["recording_id"] for e in events]

    n_frames = infer_n_frames(c2_by_rid, WINDOW_S)
    print(f"window frames: {n_frames} per {WINDOW_S}s side window "
          f"(v2: 2s primary window, was 4s in v1)")

    n_missing_c2 = sum(e["recording_id"] not in c2_by_rid for e in events)
    if n_missing_c2:
        print(f"  !! {n_missing_c2} events lack a C2 cache for their recording")
    c2_events = [dict(e, rec=c2_by_rid.get(e["recording_id"])) for e in events]

    sub_map = {}
    if os.path.exists(a.same_action_subtype):
        with open(a.same_action_subtype, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sub_map[r["event_id"]] = r["subtype"]

    folds = stratified_grouped_folds(groups, y, 5, seed=0)
    print(f"\ntraining {a.n_seeds} seeds x 5 folds "
          f"(+ inner nested-fusion folds unless --no_nested_fusion) ...")
    res = p1_plus_c2_fold_eval(Ls, Rs, X_rel, y, groups, folds, c2_events, n_frames,
                               a.epochs, a.device, n_seeds=a.n_seeds,
                               nested_fusion=not a.no_nested_fusion)

    au_p1 = res["au_p1"]
    print(f"\nP1 alone:              {au_p1:.3f}  per-fold {[round(x,3) for x in res['per_fold_p1']]}")
    print(f"C2 d_lr alone (v1-comparable): {res['au_dlr']:.3f}")
    print(f"C2 q alone (v2 relative score): {res['au_q']:.3f}")
    print(f"P1 + C2 fixed fusion [fuses q, NOT d_lr]:  {res['au_fixed']:.3f}  "
          f"per-fold {[round(x,3) for x in res['per_fold_fixed']]}")
    gain_fixed = res["au_fixed"] - au_p1
    pf_p1, pf_f, pf_d = res["per_fold_p1"], res["per_fold_fixed"], res["per_fold_delta"]
    folds_improved = sum(f > p for f, p in zip(pf_f, pf_p1))
    # Two DIFFERENT quantities that v1/v2 both printed as one misleading
    # "worst-fold change". min_fold_auroc_change compares the two score
    # curves' MINIMA, which need not come from the same fold; the gate's
    # max_worst_fold_drop was always meant to bound how far any INDIVIDUAL
    # fold regresses, which is worst_per_fold_delta. The gate below now uses
    # the latter. This does not retroactively flip v1's or v2's verdict --
    # both already failed on gain (+0.003 and +0.007 vs a required +0.02) --
    # but the old quantity could have masked a real per-fold regression in
    # some other run, so it is fixed rather than merely relabelled.
    min_fold_auroc_change = (min(pf_f) - min(pf_p1)) if pf_f and pf_p1 else float("nan")
    worst_per_fold_delta = min(pf_d) if pf_d else float("nan")
    print(f"  fixed-fusion gain {gain_fixed:+.3f}  folds improved {folds_improved}/{len(pf_p1)}")
    print(f"  per-fold delta {[round(x, 3) for x in pf_d]}")
    print(f"  worst per-fold delta {worst_per_fold_delta:+.3f}  (this is what the gate bounds)")
    print(f"  min fold AUROC {min(pf_p1):.3f} -> {min(pf_f):.3f} "
          f"({min_fold_auroc_change:+.3f})  (a DIFFERENT quantity: the two minima "
          f"need not be the same fold)")
    if res["per_fold_nested"]:
        gain_nested = res["au_nested"] - au_p1
        print(f"P1 + C2 nested fusion: {res['au_nested']:.3f}  "
              f"per-fold {[round(x,3) for x in res['per_fold_nested']]}  "
              f"gain {gain_nested:+.3f}  (DIAGNOSTIC ONLY, not the pre-registered result)")

    print(f"\nP1 + C2 fixed fusion on RAW d_lr: {res['au_fixed_dlr']:.3f}  "
          f"gain {res['au_fixed_dlr'] - au_p1:+.3f}")
    print("  ^ POST-HOC DESCRIPTIVE ONLY. The pre-registered gate result is the "
          "q fusion above. Reported so the archive is not selective (q and d_lr "
          "have different standalone AUROCs), NOT as an alternative result -- "
          "picking whichever score formula wins on this same dev OOF split is "
          "exactly the selection the review ruled out.")

    print(f"\nseed-to-seed OOF q-score std (mean over events): "
          f"{np.nanmean(res['oof_q_std']):.4f}  "
          f"(high relative to the sharp/same-action score gap means single-seed "
          f"numbers like v1's are not trustworthy)")
    print(f"per-seed pooled OOF AUROC  q: {[round(x, 3) for x in res['per_seed_au_q']]}  "
          f"d_lr: {[round(x, 3) for x in res['per_seed_au_dlr']]}")
    print("  ^ distinguishes 'each seed is a decent ranker, they disagree on hard "
          "events' from 'each seed is individually noisy' -- the per-event std "
          "alone cannot tell these apart")

    print("\n=== train vs OOF AUROC (seed 0, d_lr both sides) ===")
    print(f"{'fold':>4} {'n_tr':>5} {'n_te':>5} {'train':>8} {'OOF':>8} {'gap':>8}")
    for fd in res["fold_diagnostics"]:
        tr_a, te_a = fd.get("train_auroc_seed0"), fd.get("oof_auroc_seed0")
        gap = f"{tr_a - te_a:+.3f}" if (tr_a is not None and te_a is not None) else "n/a"
        def f3(v): return f"{v:.3f}" if v is not None else "n/a"
        print(f"{fd['fold']:>4} {fd.get('n_train', 0):>5} {fd.get('n_test', 0):>5} "
              f"{f3(tr_a):>8} {f3(te_a):>8} {gap:>8}")
    print("  a large positive gap = the encoder fits its training identities but "
          "does not transfer (the failure mode synthetic testing could not resolve "
          "at 5 events/fold); a small gap with low OOF = it never learned much")

    print("\n=== geometry by subtype (v2 windows, 2s) ===")
    geo = geometry_report(events, y, res["oof_dlr"], res["oof_dl"], res["oof_dr"], sub_map)
    print(f"{'subtype':<28} {'n':>4} {'d_lr':>8} {'d_l':>8} {'d_r':>8}")
    for key in sorted(geo, key=lambda k: -geo[k]["n"]):
        g = geo[key]
        def fmt(v): return f"{v:.4f}" if v is not None else "n/a"
        print(f"{key:<28} {g['n']:>4} {fmt(g['d_lr_mean']):>8} {fmt(g['d_l_mean']):>8} {fmt(g['d_r_mean']):>8}")

    watch = {"regrasp_reposition", "direction_reversal"}
    diag = {}
    cutoff_p1 = float(np.nanmedian(res["oof_p1"][y == 1]))
    cutoff_fixed = float(np.nanmedian(res["oof_fixed"][y == 1]))
    for name, oof, cutoff in [("p1", res["oof_p1"], cutoff_p1),
                              ("fixed_fusion", res["oof_fixed"], cutoff_fixed)]:
        neg_watch = [i for i, e in enumerate(events)
                    if y[i] == 0 and sub_map.get(e["event_id"]) in watch
                    and np.isfinite(oof[i])]
        fp = sum(oof[i] >= cutoff for i in neg_watch)
        rate = fp / len(neg_watch) if neg_watch else float("nan")
        print(f"  {name}: regrasp+direction_reversal FP rate = {rate:.3f} ({fp}/{len(neg_watch)})")
        diag[f"{name}_watch_subtype_fp_rate"] = rate

    if a.dump_events:
        with open(os.path.expanduser(a.dump_events), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "y", "dev_pair_subtype", "same_action_subtype",
                       "oof_d_lr", "oof_d_l", "oof_d_r", "oof_q", "oof_q_std",
                       "oof_p1", "oof_fixed", "oof_nested", "oof_fixed_dlr"])
            def fmt(v):
                return "" if v is None or not np.isfinite(v) else f"{v:.6f}"
            for i, e in enumerate(events):
                w.writerow([e["event_id"], e["recording_id"], int(y[i]),
                           e.get("temporal_pair_subtype") or "", sub_map.get(e["event_id"], ""),
                           fmt(res["oof_dlr"][i]), fmt(res["oof_dl"][i]), fmt(res["oof_dr"][i]),
                           fmt(res["oof_q"][i]), fmt(res["oof_q_std"][i]),
                           fmt(res["oof_p1"][i]), fmt(res["oof_fixed"][i]),
                           fmt(res["oof_nested"][i]), fmt(res["oof_fixed_dlr"][i])])
        print(f"\nwrote per-event dump ({len(events)} rows) -> {a.dump_events}")

    verdict = "NOT EVALUATED (gate config missing)"
    if os.path.exists(a.gate_config):
        gate = json.load(open(a.gate_config, encoding="utf-8"))
        watch_ok = (diag["fixed_fusion_watch_subtype_fp_rate"]
                   <= diag["p1_watch_subtype_fp_rate"] + gate["max_watch_subtype_fp_increase"])
        ok = (gain_fixed >= gate["min_auroc_gain"]
              and (folds_improved / len(pf_p1) if pf_p1 else 0) >= gate["min_folds_improved_frac"]
              and (worst_per_fold_delta >= -gate["max_worst_fold_drop"] if pf_d else False)
              and watch_ok)
        verdict = "ADOPT C2" if ok else "DO NOT ADOPT (gate failed)"
    print(f"\nVERDICT (fixed fusion, pre-registered gate): {verdict}")

    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
        json.dump({
            "version": "v2", "n_events": len(events), "window_s": WINDOW_S,
            "window_frames": n_frames, "epochs": a.epochs, "n_seeds": a.n_seeds,
            "p1_alone": {"pooled": au_p1, "per_fold": pf_p1},
            "c2_dlr_alone": res["au_dlr"], "c2_q_alone": res["au_q"],
            "p1_plus_c2_fixed": {"pooled": res["au_fixed"], "per_fold": pf_f},
            "p1_plus_c2_nested_diagnostic": {"pooled": res["au_nested"],
                                             "per_fold": res["per_fold_nested"]},
            "p1_plus_c2_fixed_on_raw_dlr_POSTHOC": res["au_fixed_dlr"],
            "gain_fixed": gain_fixed, "folds_improved": folds_improved,
            "per_fold_delta": res["per_fold_delta"],
            "worst_per_fold_delta": worst_per_fold_delta,
            "min_fold_auroc_change": min_fold_auroc_change,
            "seed_std_mean": float(np.nanmean(res["oof_q_std"])),
            "per_seed_auroc_q": res["per_seed_au_q"],
            "per_seed_auroc_dlr": res["per_seed_au_dlr"],
            "geometry_by_subtype": geo, "watch_subtype_diag": diag,
            "fold_diagnostics": res["fold_diagnostics"], "verdict": verdict,
            "provenance": _provenance(a),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
