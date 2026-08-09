"""Boundary v1 training: recording-grouped 5-fold OOF, per-head masked loss.

WHAT THIS TESTS. One thing: whether POINT / INTERVAL / NO_TRANSITION /
UNOBSERVABLE is a better-posed target than sharp-versus-same_action. The
comparison is built in -- the same folds, the same features, the same encoder,
with the old binary read off the morphology head's own predictions -- because
a new number on a new target next to an old number on an old target compares
nothing.

ONE CONFIGURATION. No architecture sweep, no hidden-size search, no dilation
grid. Trying several and reporting the best is how every earlier operating
point in this project came to breach under nested selection, and this file
exists to test a reformulation, not to find a winner. The defaults are fixed
here and changing them is a new run with a new name.

EVERYTHING FITTED IS FITTED INSIDE THE FOLD. PCA and the feature scaler see
only the training recordings. The held-out recordings of a fold contribute
nothing to any statistic used to score them.

PER-HEAD MASKS, NOT A CLEAN SUBSET. An event with no morphology target may
still carry a relation or nuisance target and stays in the batch for those.
Dropping the whole event -- the old CLEAN_BINARY filter -- is how gradual,
camera, offscreen and annotation left training entirely and then arrived at
inference as most of the wrong auto-keeps.

WHAT IT DOES NOT DO. No thresholds, no operating point, no actions. The policy
reads the ontology and this file never sees it.

Usage:
    python -m src.auditor.boundary.train \
        --labels data/gold/boundary_v1_labels.json \
        --feat_cache .../recseg_train_feats.pt --feat_cache .../val_feats.pt \
        --local_cache .../local_train.pt --local_cache .../local_val.pt \
        --out /workspace/tr1/results/auditor/boundary_v1_oof.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.common.feature_loader import load_caches, build_events, stack
from src.auditor.common.temporal_encoder import n_params
from src.auditor.boundary.model import (BoundaryModel, build_input,
                                        MORPHOLOGY, RELATION)
from src.boundary.pairwise_verifier import stratified_grouped_folds


def pca_fit(X, dim):
    """X [N,D] -> (mean, components). Fitted on training frames only."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # economy SVD on the smaller side; N frames is far larger than we need
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu.astype(np.float32), Vt[:dim].T.astype(np.float32)


def pca_apply(p, X):
    mu, W = p
    return (X - mu) @ W


def project_seq(p, seq):
    """seq [N,T,D] -> [N,T,dim] without materialising N*T*D twice."""
    n, t, d = seq.shape
    return pca_apply(p, seq.reshape(-1, d)).reshape(n, t, -1)


def masked_ce(logits, y, mask, weight=None):
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], y[mask], weight=weight)


def masked_l1(pred, y, mask):
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], y[mask])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_relation", type=float, default=0.3)
    ap.add_argument("--lambda_offset", type=float, default=0.1)
    ap.add_argument("--out")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    lab = json.load(open(a.labels, encoding="utf-8"))
    print(f"{len(lab['events'])} labelled events from "
          f"{os.path.basename(a.labels)}")
    gc = load_caches(a.feat_cache)
    lc = load_caches(a.local_cache)
    print(f"caches: {len(gc)} global recordings, {len(lc)} local")
    ev = build_events(lab["events"], gc, lc, a.half_s, a.n_frames)
    if not ev:
        raise SystemExit("no event has sequences; check the cache paths")

    groups = [e["recording_id"] for e in ev]
    m_idx = {k: i for i, k in enumerate(MORPHOLOGY)}
    r_idx = {k: i for i, k in enumerate(RELATION)}
    y_m = np.array([m_idx.get(e["morphology"], -1) for e in ev])
    y_r = np.array([r_idx.get(e["candidate_relation"], -1) for e in ev])
    y_o = np.array([e["offset_s"] if e["offset_s"] is not None else 0.0
                    for e in ev], np.float32)
    mask_m = y_m >= 0
    mask_r = y_r >= 0
    mask_o = np.array([e["offset_s"] is not None for e in ev])
    print(f"  morphology supervised on {int(mask_m.sum())}, relation on "
          f"{int(mask_r.sum())}, offset on {int(mask_o.sum())}")
    print(f"  morphology classes: "
          f"{dict(Counter(e['morphology'] for e in ev if e['morphology']))}")
    rc = Counter(e['candidate_relation'] for e in ev if e['candidate_relation'] in r_idx)
    print(f"  relation classes:   {dict(rc)}")
    thin = [k for k in RELATION if rc.get(k, 0) < a.n_folds]
    if thin:
        print(f"  !! {thin} have fewer events than folds. They are trained on "
              f"but NOT reported: a per-class number from\n     fewer than one "
              f"event per fold is decided by which fold the events landed in.")

    G = stack(ev, "g")
    L = stack(ev, "l")
    VG = torch.from_numpy(stack(ev, "valid_g"))
    VL = torch.from_numpy(stack(ev, "valid_l"))

    # stratify the folds on morphology so a fold is not handed one class
    strat = np.where(mask_m, y_m, len(MORPHOLOGY)).astype(float)
    folds = stratified_grouped_folds(groups, strat, a.n_folds, seed=a.seed)

    oof_m = np.full((len(ev), len(MORPHOLOGY)), np.nan)
    oof_r = np.full((len(ev), len(RELATION)), np.nan)
    oof_o = np.full(len(ev), np.nan)
    oof_w = np.full(len(ev), np.nan)

    for fi, f in enumerate(folds):
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 20:
            print(f"  fold {fi}: too small, skipped")
            continue
        # PCA on TRAINING frames only, and only the frames that are real
        pg = pca_fit(G[tr][stack([e for i, e in enumerate(ev) if tr[i]],
                                 "valid_g")], a.pca_dim)
        pl = pca_fit(L[tr][stack([e for i, e in enumerate(ev) if tr[i]],
                                 "valid_l")], a.pca_dim)
        Pg = torch.from_numpy(project_seq(pg, G)).float()
        Pl = torch.from_numpy(project_seq(pl, L)).float()
        # scale on training events only, per projected channel
        s = Pg[torch.from_numpy(tr)].reshape(-1, Pg.shape[-1]).std(0).clamp(min=1e-6)
        Pg = Pg / s
        s = Pl[torch.from_numpy(tr)].reshape(-1, Pl.shape[-1]).std(0).clamp(min=1e-6)
        Pl = Pl / s
        X, M = build_input(Pg, Pl, VG, VL)

        model = BoundaryModel(X.shape[-1], hidden=a.hidden, dropout=a.dropout)
        if fi == 0:
            print(f"\n  encoder input {X.shape[-1]} dims, "
                  f"{n_params(model)} parameters against "
                  f"{int(tr.sum())} training events")
            if n_params(model) > 20 * int(tr.sum()):
                print("  !! more than 20 parameters per training event. The "
                      "regularisers are load-bearing here, and an in-sample\n"
                      "     number from this model means nothing -- only the "
                      "out-of-fold columns below do.")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay)
        tr_t = torch.from_numpy(tr)
        ym = torch.from_numpy(np.where(mask_m, y_m, 0)).long()
        yr = torch.from_numpy(np.where(mask_r, y_r, 0)).long()
        yo = torch.from_numpy(y_o)
        mm = torch.from_numpy(mask_m) & tr_t
        mr = torch.from_numpy(mask_r) & tr_t
        mo = torch.from_numpy(mask_o) & tr_t
        # inverse-frequency weights so UNOBSERVABLE (19 events) is not simply
        # never predicted; computed on the training fold only
        cnt = np.bincount(y_m[mask_m & tr], minlength=len(MORPHOLOGY)) + 1
        wm = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
                          dtype=torch.float32)

        model.train()
        for ep in range(a.epochs):
            opt.zero_grad()
            out = model(X, M)
            loss = masked_ce(out["morphology"], ym, mm, wm) \
                + a.lambda_relation * masked_ce(out["relation"], yr, mr) \
                + a.lambda_offset * masked_l1(out["offset"], yo, mo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(X, M)
            te_t = torch.from_numpy(te)
            oof_m[te] = F.softmax(out["morphology"][te_t], -1).numpy()
            oof_r[te] = F.softmax(out["relation"][te_t], -1).numpy()
            oof_o[te] = out["offset"][te_t].numpy()
            oof_w[te] = out["log_width"][te_t].exp().numpy()
        print(f"  fold {fi}: {int(tr.sum())} train / {int(te.sum())} test, "
              f"final loss {loss.item():.4f}")

    rows = []
    for i, e in enumerate(ev):
        rows.append({
            "event_id": e["event_id"], "recording_id": e["recording_id"],
            "subtype": e["subtype"], "candidate_time": e["candidate_time"],
            "morphology_true": e["morphology"],
            "relation_true": e["candidate_relation"],
            "offset_true": e["offset_s"],
            "morphology": {k: float(oof_m[i, j])
                           for j, k in enumerate(MORPHOLOGY)},
            "relation": {k: float(oof_r[i, j]) for j, k in enumerate(RELATION)},
            "offset": float(oof_o[i]) if np.isfinite(oof_o[i]) else None,
            "width_s": float(oof_w[i]) if np.isfinite(oof_w[i]) else None,
            "coverage_g": e["coverage_g"], "coverage_l": e["coverage_l"],
        })
    blob = {"config": {k: v for k, v in vars(a).items() if k != "out"},
            "morphology_classes": MORPHOLOGY, "relation_classes": RELATION,
            "n_events": len(rows), "events": rows}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump(blob, open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")
    print("\nNo metric is printed here on purpose. Scoring lives in "
          "src.auditor.boundary.evaluate, so a training run cannot report\n"
          "the number it was tuned against.")


if __name__ == "__main__":
    main()
