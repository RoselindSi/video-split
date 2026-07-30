"""C1: predictive-continuity head (ESTimator/SBD-style) on frozen Qwen features.

Motivation (batch3 failure attribution): P1 answers "are the left and right
states different?" -- it cannot answer "is this difference explained by the
CURRENT action's own dynamics?". Repetitive tasks (folding, coiling, bottle
pushing, sweeping) produce large but PREDICTABLE visual change, so a
next-state predictor should have LOW error there and HIGH error at genuine
action switches. This script trains such a predictor self-supervised and
tests whether its error scores add anything to P1 on the frozen 145-pair
clean development set.

Data discipline:
  - Predictor trains ONLY on recordings that are (a) not among the 48
    development recordings (they carry the eval events) and (b) not among
    the batch3-sampled recordings (batch3 stays pristine for one-shot
    confirmation later). Self-supervision uses no labels, but excluding
    both pools keeps every later comparison clean.
  - Evaluation uses the SAME frozen 145 clean pairs and the same stratified
    grouped folds as pairwise_verifier.py. batch3 labels are used only for
    an optional frozen ranking diagnostic (no threshold is derived).

Anti-collapse guards (the frozen features have effective rank ~3/5760, so a
trivial "predict the global mean" solution is a real risk):
  - The embedding is a FROZEN PCA fitted once on training frames -- the
    predictor cannot reshape the space it is scored in, so representation
    collapse is impossible by construction.
  - A shuffled-future control is reported: if error(true future) is not
    clearly below error(random other window), the predictor learned nothing
    and every downstream number should be ignored.
  - Architecture is a small causal transformer, not a fixed linear filter:
    extrapolating a periodic motion requires coefficients that depend on
    the observed frequency (x_{t+1} = 2cos(w) x_t - x_{t-1}), i.e. in-context
    adaptivity. A synthetic check with novel oscillation planes verified a
    fixed-filter TCN fails exactly this test while the transformer passes.

KNOWN PHYSICAL LIMIT (found during synthetic verification): at 2 fps the
Nyquist rate is 1 Hz, so repetitive sub-motions with ~1 s period (sweeping
strokes, fast fold presses) are ALIASED in the feature sequence and their
periodicity is unrecoverable at any model capacity -- synthetic oscillators
at 0.5-0.8 Hz were unpredictable on held-out recordings while 0.1-0.3 Hz
ones gave perfect boundary/mid separation (AUROC 1.0). What C1 can exploit
at 2 fps is the slower envelope (scene composition, hand-position
distribution, object layout drift), not fast periodic motion itself. If the
real-data control passes but standalone AUROC is mediocre, the fix is
higher-fps local features near candidates (the C3 direction), not a bigger
predictor. Also note: single direction REVERSALS (bottle push left->right)
break constant-velocity prediction at any frame rate, so some strong_align
negatives will legitimately score high -- the gate decides whether the net
effect still helps.

Pre-registered adoption gate: configs/continuity_gate_c1.json (written
before any result was seen).

Usage (server):
    python -m src.boundary.predictive_continuity \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --exclude_manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --epochs 8 --device cuda \
        --out /workspace/tr1/results/hal/continuity_c1/report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T
from src.boundary.hal_features import load_feature_caches
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.state_adapter import build_events, _auroc
from src.boundary.pairwise_verifier import (
    REL_NAMES, stratified_grouped_folds, build_matrices,
    _impute_scale_fit, _impute_scale_apply, pca_fit, pca_apply, pair_block,
)

PAST_S = 4.0          # context the predictor sees (seconds)
FUT_LO, FUT_HI = 0.25, 1.25   # predicted (pooled) future window


def infer_n_past(recs, cap=64):
    """Frames per PAST_S window, from the cache's actual frame rate."""
    rec = next(iter(recs.values()))
    times = rec["times"]
    dt = float(np.median(np.diff(times.numpy()[:200])))
    return min(cap, max(4, int(round(PAST_S / dt))))
EMB = 128

CONT_NAMES = ["cont_efwd_z", "cont_ebwd_z", "cont_emin_z", "cont_emax_z"]


class ContinuityPredictor(torch.nn.Module):
    """frozen-PCA embed -> small causal transformer -> predict pooled
    projected future. The PCA is fitted once (fit_pca) and registered as
    buffers; only the transformer + head train."""

    def __init__(self, d_in, n_past, d=EMB, nhead=4, nlayers=2):
        super().__init__()
        self.n_past = n_past
        self.register_buffer("pca_mu", torch.zeros(d_in))
        self.register_buffer("pca_W", torch.zeros(d_in, d))
        self.pos = torch.nn.Parameter(torch.randn(n_past, d) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = torch.nn.Linear(d, d)

    def fit_pca(self, frames):               # frames: [N, d_in] tensor
        mu = frames.mean(0)
        X = (frames - mu).numpy()
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        d = self.pca_W.shape[1]
        W = Vt[:d].T
        if W.shape[1] < d:                   # rank-deficient input
            W = np.pad(W, ((0, 0), (0, d - W.shape[1])))
        self.pca_mu.copy_(mu)
        self.pca_W.copy_(torch.tensor(W, dtype=torch.float32))

    def embed(self, x):                      # [..., d_in] -> [..., d] (frozen)
        return (x - self.pca_mu) @ self.pca_W

    def forward(self, past):                 # past: [B, T, d_in]
        h = self.embed(past) + self.pos
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            h.shape[1], device=h.device)
        h = self.enc(h, mask=mask)
        return self.head(h[:, -1])           # [B, d]


def _past_stack(feats, times, t, n_past, reverse=False):
    """Last n_past frames strictly before t (after t, mirrored, when
    reverse=True). Returns [n_past, D] or None if too few usable frames."""
    if reverse:
        m = (times > t) & (times <= t + PAST_S)
        # ascending-time order puts t+eps (nearest t) first; flip so the
        # nearest-to-t frame is LAST, mirroring the forward branch below
        # (verified: feats[i]=times[i] probe confirms both branches end on
        # the frame closest to t, farthest-first / oldest-first otherwise).
        idx = torch.nonzero(m).flatten().flip(0)
    else:
        m = (times >= t - PAST_S) & (times < t)
        idx = torch.nonzero(m).flatten()
    if idx.numel() < max(4, n_past * 3 // 4):
        return None
    x = feats[idx][-n_past:]
    if x.shape[0] < n_past:
        x = torch.cat([x[:1].repeat(n_past - x.shape[0], 1), x], 0)
    return x


def _future_pool(feats, times, t, reverse=False):
    if reverse:
        m = (times >= t - FUT_HI) & (times <= t - FUT_LO)
    else:
        m = (times >= t + FUT_LO) & (times <= t + FUT_HI)
    if int(m.sum()) < 1:
        return None
    return feats[m].mean(0)


def _sample_pairs(recs, stride=1.0, edge=PAST_S + FUT_HI):
    out = []
    for rid, rec in recs.items():
        times = rec["times"]
        t0, t1 = float(times[0]) + edge, float(times[-1]) - edge
        t = t0
        while t < t1:
            out.append((rid, round(t, 2)))
            t += stride
    return out


def train_predictor(recs, epochs, device, batch=256, lr=1e-3, seed=0):
    d_in = next(iter(recs.values()))["feats"].shape[1]
    n_past = infer_n_past(recs)
    model = ContinuityPredictor(d_in, n_past)
    # frozen PCA embedding, fitted on a frame subsample across recordings
    rng = np.random.RandomState(seed)
    sub = []
    for rec in recs.values():
        idx = rng.choice(len(rec["feats"]), min(200, len(rec["feats"])), replace=False)
        sub.append(rec["feats"][idx].float())
    model.fit_pca(torch.cat(sub, 0))
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    pairs = _sample_pairs(recs)
    print(f"training pairs: {len(pairs)} from {len(recs)} recordings  d_in={d_in}  "
          f"n_past={n_past} frames per {PAST_S}s window")
    for ep in range(epochs):
        rng.shuffle(pairs)
        tot, n = 0.0, 0
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            past, fut = [], []
            for rid, t in chunk:
                rec = recs[rid]
                rev = bool(rng.randint(2))               # train both directions
                p = _past_stack(rec["feats"], rec["times"], t, n_past, reverse=rev)
                f_ = _future_pool(rec["feats"], rec["times"], t, reverse=rev)
                if p is None or f_ is None:
                    continue
                past.append(p); fut.append(f_)
            if len(past) < 8:
                continue
            past = torch.stack(past).float().to(device)
            fut = torch.stack(fut).float().to(device)
            pred = model(past)
            tgt = model.embed(fut)                       # frozen space: no grad path
            loss = 1 - F.cosine_similarity(pred, tgt).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(past); n += len(past)
        print(f"  epoch {ep}: loss {tot / max(n, 1):.4f}")
    model.eval()
    return model


@torch.no_grad()
def pred_error(model, rec, t, device, reverse=False):
    p = _past_stack(rec["feats"], rec["times"], t, model.n_past, reverse=reverse)
    f_ = _future_pool(rec["feats"], rec["times"], t, reverse=reverse)
    if p is None or f_ is None:
        return None
    pred = model(p[None].float().to(device))
    tgt = model.embed(f_[None].float().to(device))
    return float(1 - F.cosine_similarity(pred, tgt).item())


@torch.no_grad()
def shuffled_control(model, recs, device, n=300, seed=0):
    """error(true future) vs error(random other window's future)."""
    rng = np.random.RandomState(seed)
    pairs = _sample_pairs(recs)
    rng.shuffle(pairs)
    true_e, shuf_e = [], []
    for (rid, t) in pairs[:n]:
        rec = recs[rid]
        e = pred_error(model, rec, t, device)
        rid2, t2 = pairs[rng.randint(len(pairs))]
        p = _past_stack(rec["feats"], rec["times"], t, model.n_past)
        f2 = _future_pool(recs[rid2]["feats"], recs[rid2]["times"], t2)
        if e is None or p is None or f2 is None:
            continue
        pred = model(p[None].float().to(device))
        tgt = model.embed(f2[None].float().to(device))
        true_e.append(e)
        shuf_e.append(float(1 - F.cosine_similarity(pred, tgt).item()))
    return float(np.mean(true_e)), float(np.mean(shuf_e))


@torch.no_grad()
def recording_error_baseline(model, rec, device, stride=2.0):
    es = []
    times = rec["times"]
    t0 = float(times[0]) + PAST_S + FUT_HI
    t1 = float(times[-1]) - PAST_S - FUT_HI
    t = t0
    while t < t1:
        e = pred_error(model, rec, t, device)
        if e is not None:
            es.append(e)
        t += stride
    if len(es) < 5:
        return None, None
    es = np.array(es)
    med = float(np.median(es))
    mad = float(np.median(np.abs(es - med))) * 1.4826 + 1e-6
    return med, mad


def continuity_features(model, events, device, cont_by_rid=None):
    """CONT_NAMES columns per event; NaN where unscorable. When
    cont_by_rid is given (e.g. a 10 fps cache), continuity scores are
    computed from THAT cache's frames for the same recording_id, while the
    events' own rec (2 fps) is untouched for P1."""
    base = {}
    X = np.full((len(events), len(CONT_NAMES)), np.nan)
    for i, e in enumerate(events):
        rid = e["recording_id"]
        rec = cont_by_rid.get(rid) if cont_by_rid is not None else e["rec"]
        if rec is None:
            continue
        if rid not in base:
            base[rid] = recording_error_baseline(model, rec, device)
        med, mad = base[rid]
        if med is None:
            continue
        ef = pred_error(model, rec, e["t"], device)
        eb = pred_error(model, rec, e["t"], device, reverse=True)
        zf = (ef - med) / mad if ef is not None else np.nan
        zb = (eb - med) / mad if eb is not None else np.nan
        X[i] = [zf, zb, np.nanmin([zf, zb]), np.nanmax([zf, zb])]
    return X


def p1_fold_eval(Ls, Rs, X_rel, y, groups, folds, pca_dim=64, l2=5.0):
    """P1 no-clip pipeline, pooled OOF predictions."""
    oof = np.full(len(y), np.nan)
    for f in folds:
        te = np.array([g in f for g in groups])
        tr = ~te
        pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), pca_dim)
        st_rel = _impute_scale_fit(X_rel[tr])
        def blk(m):
            return np.concatenate([pair_block(pca_apply(pca, Ls[m]), pca_apply(pca, Rs[m])),
                                   _impute_scale_apply(st_rel, X_rel[m])], 1)
        Ptr = blk(tr)
        stP = _impute_scale_fit(Ptr)
        w, b = fit_logreg(_impute_scale_apply(stP, Ptr), y[tr], l2=l2)
        oof[te] = _sigmoid(_impute_scale_apply(stP, blk(te)) @ w + b)
    per_fold = []
    for f in folds:
        te = np.array([g in f for g in groups])
        if len(set(y[te].tolist())) == 2:
            per_fold.append(_auroc(y[te], oof[te]))
    return _auroc(y, oof), per_fold, oof


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True,
                    help="2 fps caches -- used for the P1 baseline features")
    ap.add_argument("--cont_feat_cache", action="append", default=None,
                    help="caches for the continuity predictor (e.g. 10 fps). "
                         "Defaults to --feat_cache. 2 fps aliases ~1s-period "
                         "repetition (see module docstring); 10 fps is the "
                         "recommended setting.")
    ap.add_argument("--max_train_recordings", type=int, default=0,
                    help="subsample the self-supervised training pool (0 = all)")
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--exclude_manifest", action="append", default=[],
                    help="batch3 manifest(s); their recordings are excluded from "
                         "self-supervised training")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_config", default="configs/continuity_gate_c1.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    by_rid = load_feature_caches(a.feat_cache)
    cont_by_rid = (load_feature_caches(a.cont_feat_cache)
                   if a.cont_feat_cache else by_rid)
    gold = S.load_gold(a.gold)
    dev_recs = {g.get("recording_id") for g in gold if g.get("recording_id")}
    b3_recs = set()
    for mp in a.exclude_manifest:
        with open(os.path.expanduser(mp), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    b3_recs.add(json.loads(line)["recording_id"])
    train_ids = sorted(set(cont_by_rid) - dev_recs - b3_recs)
    if a.max_train_recordings and len(train_ids) > a.max_train_recordings:
        rng0 = np.random.RandomState(a.seed)
        train_ids = sorted(rng0.choice(train_ids, a.max_train_recordings,
                                       replace=False).tolist())
    print(f"cont caches: {len(cont_by_rid)} recordings; "
          f"excluded dev={len(dev_recs & set(cont_by_rid))} "
          f"batch3={len(b3_recs & set(cont_by_rid))}; TRAIN pool={len(train_ids)}")
    if len(train_ids) < 20:
        raise SystemExit("training pool too small -- check cache paths / exclusions")
    train_recs = {r: cont_by_rid[r] for r in train_ids}

    device = a.device
    model = train_predictor(train_recs, a.epochs, device, seed=a.seed)

    te_true, te_shuf = shuffled_control(model, train_recs, device, seed=a.seed)
    print(f"\nshuffled-future control: err(true)={te_true:.4f}  err(shuffled)={te_shuf:.4f}")
    control_ok = te_true < 0.8 * te_shuf
    if not control_ok:
        print("  !! predictor failed the control -- it is not using the past. "
              "Downstream numbers are reported but MUST NOT be trusted.")

    # ---- frozen 145-pair development evaluation (no-clip, deployment-honest)
    ctx = S.load_context(a.context)
    events = build_events(gold, ctx, by_rid)
    labels = T.load_pair_labels(a.pair_labels)
    events = T.apply_to_events(events, labels)
    print(f"\nclean events: {len(events)}  "
          f"subtypes={dict(Counter(e.get('temporal_pair_subtype') for e in events))}")

    X_v1, Ls, Rs, X_rel, keep, crops = build_matrices(events, clip_neighbors=False)
    events = [events[i] for i in keep]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = [e["recording_id"] for e in events]

    Xc = continuity_features(model, events, device, cont_by_rid=cont_by_rid)
    n_missing = sum(e["recording_id"] not in cont_by_rid for e in events)
    if n_missing:
        print(f"  !! {n_missing} eval events lack a continuity cache for their "
              f"recording -- extract those recordings too or scores stay NaN")
    print(f"continuity features: {np.isfinite(Xc).all(1).sum()}/{len(Xc)} fully scorable")

    standalone = {}
    for j, nm in enumerate(CONT_NAMES):
        m = np.isfinite(Xc[:, j])
        standalone[nm] = _auroc(y[m], Xc[m, j]) if len(set(y[m].tolist())) == 2 else float("nan")
    print("standalone AUROC:", {k: round(v, 3) for k, v in standalone.items()})

    folds = stratified_grouped_folds(groups, y, 5, seed=0)
    au_p1, pf_p1, _ = p1_fold_eval(Ls, Rs, X_rel, y, groups, folds)
    X_rel_c = np.concatenate([X_rel, Xc], 1)
    au_c, pf_c, _ = p1_fold_eval(Ls, Rs, X_rel_c, y, groups, folds)
    gain = au_c - au_p1
    folds_improved = sum(c > p for c, p in zip(pf_c, pf_p1))
    worst_drop = min(pf_c) - min(pf_p1)
    print(f"\nP1 baseline (no-clip): pooled {au_p1:.3f}  per-fold "
          f"{[round(x, 3) for x in pf_p1]}")
    print(f"P1 + continuity:       pooled {au_c:.3f}  per-fold "
          f"{[round(x, 3) for x in pf_c]}")
    print(f"gain {gain:+.3f}  folds improved {folds_improved}/{len(pf_p1)}  "
          f"worst-fold change {worst_drop:+.3f}")

    verdict = "NOT EVALUATED (gate config missing)"
    if os.path.exists(a.gate_config):
        gate = json.load(open(a.gate_config, encoding="utf-8"))
        ok = (control_ok
              and gain >= gate["min_auroc_gain"]
              and folds_improved / len(pf_p1) >= gate["min_folds_improved_frac"]
              and worst_drop >= -gate["max_worst_fold_drop"])
        verdict = "ADOPT continuity features" if ok else "DO NOT ADOPT (gate failed)"
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
        json.dump({
            "train_recordings": len(train_ids), "epochs": a.epochs,
            "control": {"err_true": te_true, "err_shuffled": te_shuf, "ok": control_ok},
            "standalone_auroc": standalone,
            "p1_baseline": {"pooled": au_p1, "per_fold": pf_p1},
            "p1_plus_continuity": {"pooled": au_c, "per_fold": pf_c},
            "gain": gain, "folds_improved": folds_improved,
            "worst_fold_change": worst_drop, "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    print(f"wrote {a.out}")

    torch.save({"state_dict": model.state_dict(), "n_past": model.n_past},
               os.path.expanduser(a.out) + ".predictor.pt")
    print(f"saved predictor -> {a.out}.predictor.pt")


if __name__ == "__main__":
    main()
