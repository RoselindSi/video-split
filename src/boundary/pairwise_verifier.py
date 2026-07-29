"""Pairwise boundary verifier: P0/P1/P2/P3 ablation under grouped nested CV.

Replaces the abandoned "learn a global action-state manifold first" approach.
Evidence for the change (all development numbers, 175 events / 47
recordings): v1 raw features AUROC 0.702 -> v2 structural features 0.608 ->
v2 on a contrastively-trained adapter 0.576. Making the contrastive objective
shape the whole embedding made things monotonically worse, so the task is
demoted to what it actually is -- a LEFT/RIGHT WINDOW RELATION
CLASSIFICATION, trained directly on the end objective (BCE), with contrastive
supervision reintroduced only as a small, ablatable auxiliary term.

Arms:
  P0  v1 raw HAL features + logistic regression        (the 0.702 baseline)
  P1  pairwise projected classifier: shared projection of multi-scale pooled
      left/right features, then [zL, zR, |zL-zR|, zL*zR] + relative features
      -> logistic head. Projection is PCA fit INSIDE each training fold.
  P2  P1 + same-side crop consistency as an auxiliary loss
  P3  P1 + small-weight audited contrastive loss (push valid apart, pull
      motion-hard-negatives together)

P2/P3 use a tiny torch head so the auxiliary losses have something to act on;
P0/P1 stay closed-form logistic fits. The claim "contrastive supervision
helps" is only earned if P2 or P3 beats P1 *stably across folds*, which is
why per-fold numbers are printed, not just the pooled mean.

Relative features (beyond raw change magnitude) implement the two mechanisms
the false-keep analysis pointed at, and are the reason P1 can beat P0 without
any learned embedding:
  return_gap            d(pre,post_near) - d(pre,post_far). The audited
                        false keeps included cases where the state reverts --
                        a real transition should not come back.
  dir_consistency       mean pairwise cos of successive (post_k - pre) --
                        oscillation vs a consistent departure.
  contamination_flag /  distance to the nearest OTHER audited event in the
  nearest_other_event_s same recording. Group 1 of the manual watch list is a
                        false keep 1.0s from a true boundary, whose +-3s
                        windows overlap it by ~83% -- the feature literally
                        cannot separate them, so the model should at least be
                        able to see that the window is contaminated.
  recording z_local     clipped +-5 recording-relative normalization of the
                        scale separations (the DEV->TEST shift fix).

Protocol: grouped K-fold by recording, everything (PCA, scaler, head) fit
inside the fold. Reports per-fold and pooled AUROC + coverage@0.90-precision.
All 188 audited events are development data; a deployable claim needs a
frozen config evaluated once on a fresh batch3 of unseen recordings.

Usage (server):
    python -m src.boundary.pairwise_verifier \
        --feat_cache /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg/feat_val_full_noblur_multi.pt \
        --feat_cache /workspace/tr1/data_recseg_part2/feat_part2_full_noblur_multi.pt \
        --out /workspace/tr1/results/hal/pairwise_verifier/report.json --folds 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor import gold_schema as S
from src.boundary.hal_features import load_feature_caches, hal_features_at
from src.boundary.hal_vlm_fusion import HAL_FEATURE_NAMES, fit_logreg, _sigmoid
from src.boundary.state_adapter import (
    build_events, grouped_folds, precision_coverage, _auroc, EPS,
)
from src.boundary import pair_taxonomy as T

SCALES = [0.5, 1.0, 2.0, 4.0]


def stratified_grouped_folds(groups, y, k, seed=0):
    """Grouped folds that also balance the classes across folds.

    Plain random recording assignment produced folds with as few as 3
    negatives out of 25 events (positive fraction 0.88), whose fold-level
    AUROC (0.242) is essentially noise yet still drags the mean. Recordings
    are therefore sorted by their positive fraction and dealt round-robin,
    which keeps folds grouped by recording (no leakage) while making the
    per-fold class mix comparable."""
    import numpy as _np
    groups = _np.asarray(groups); y = _np.asarray(y)
    stats = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        stats.append((g, float(y[m].mean()), int(m.sum())))
    rng = _np.random.RandomState(seed)
    rng.shuffle(stats)
    stats.sort(key=lambda s_: s_[1])
    folds = [set() for _ in range(k)]
    sizes = [0] * k
    for i, (g, _f, n) in enumerate(stats):
        j = (i % k) if i // k % 2 == 0 else (k - 1 - i % k)   # serpentine
        folds[j].add(g); sizes[j] += n
    return folds


# ------------------------------------------------------- feature building --

def neighbor_bounds(rec, t):
    """(lower, upper) time limits for feature windows around candidate `t`,
    set by the NEAREST annotated segment boundaries on each side.

    Manual review of the 26 contradictory pairs found `window_spans_two_
    actions` TRUE in 21 of them (81%), while `only_annotation_convention`
    was FALSE in 25/26 and `unlabelled_subaction_present` FALSE in 23/26 --
    i.e. the gold labels were mostly right and the geometry disagreed with
    them because the +-2s/+-3s windows reached across OTHER actions, so the
    "left state" and "right state" were not clean. Clipping each window at
    the neighbouring boundaries makes the pooled sides represent the two
    states actually being compared.

    This uses annotation context, which is legitimate for a VERIFIER (the
    question is "given the surrounding segmentation, is THIS candidate a
    real boundary"), but it would be circular for a from-scratch detector --
    do not reuse this in a decoder that is supposed to find boundaries."""
    segs = rec.get("segments") or []
    lo, hi = -1e9, 1e9
    for _name, s0, s1 in segs:
        for edge in (float(s0), float(s1)):
            if edge < t - 1e-6:
                lo = max(lo, edge)
            elif edge > t + 1e-6:
                hi = min(hi, edge)
    return lo, hi


def _pool(feats, times, lo, hi):
    m = (times >= lo) & (times < hi)
    if int(m.sum()) < 1:
        return None
    return F.normalize(feats[m].float().mean(0), dim=-1)


def _cosd(a, b):
    return float(1.0 - F.cosine_similarity(a[None], b[None]))


def side_vectors(rec, t, scales=SCALES, clip_neighbors=False):
    """Multi-scale pooled left/right raw vectors, concatenated."""
    feats, times = rec["feats"], rec["times"]
    blo, bhi = neighbor_bounds(rec, t) if clip_neighbors else (-1e9, 1e9)
    L, R = [], []
    for s in scales:
        l = _pool(feats, times, max(t - s, blo), t)
        r = _pool(feats, times, t, min(t + s, bhi))
        if l is None or r is None:
            return None, None
        L.append(l); R.append(r)
    return torch.cat(L), torch.cat(R)


def relative_features(rec, t, other_event_times, clip_neighbors=False):
    """The mechanism-specific scalars, computed on RAW features."""
    feats, times = rec["feats"], rec["times"]
    blo, bhi = neighbor_bounds(rec, t) if clip_neighbors else (-1e9, 1e9)
    out = {}
    for s in SCALES:
        lo_s, hi_s = max(t - s, blo), min(t + s, bhi)
        l, r = _pool(feats, times, lo_s, t), _pool(feats, times, t, hi_s)
        if l is None or r is None:
            out[f"sep_{s}"] = np.nan
            continue
        ml = (times >= lo_s) & (times < t)
        mr = (times >= t) & (times < hi_s)
        sl = feats[ml].float(); sr = feats[mr].float()
        wl = float((1 - F.cosine_similarity(F.normalize(sl, dim=-1), l[None].expand_as(sl))).mean()) if len(sl) > 1 else 0.0
        wr = float((1 - F.cosine_similarity(F.normalize(sr, dim=-1), r[None].expand_as(sr))).mean()) if len(sr) > 1 else 0.0
        out[f"sep_{s}"] = _cosd(l, r) / (wl + wr + EPS)
    pre = _pool(feats, times, max(t - 2, blo), t)
    near = _pool(feats, times, t, min(t + 2, bhi))
    far = _pool(feats, times, min(t + 4, bhi), min(t + 8, bhi))
    out["return_gap"] = (_cosd(pre, near) - _cosd(pre, far)) if None not in (pre, near, far) else np.nan
    deltas = []
    if pre is not None:
        for k in range(3):
            mu = _pool(feats, times, t + k, t + k + 1)
            if mu is not None:
                deltas.append(mu - pre)
    if len(deltas) >= 2:
        cs = [float(F.cosine_similarity(deltas[i][None], deltas[j][None]))
              for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
        out["dir_consistency"] = sum(cs) / len(cs)
    else:
        out["dir_consistency"] = np.nan
    out["scale_agreement"] = out.get("sep_2.0", np.nan) / (out.get("sep_0.5", np.nan) + EPS)
    # window contamination: how close is the nearest OTHER audited event?
    others = [abs(t - o) for o in other_event_times if abs(t - o) > 1e-6]
    d_other = min(others) if others else 999.0
    out["nearest_other_event_s"] = d_other
    out["contamination_flag"] = 1.0 if d_other < 3.0 else 0.0
    # how much the widest window had to be truncated by a neighbouring
    # boundary -- 0 means the +-4s window was already inside one segment
    out["left_room_s"] = min(4.0, t - blo) if blo > -1e8 else 4.0
    out["right_room_s"] = min(4.0, bhi - t) if bhi < 1e8 else 4.0
    return out


REL_NAMES = ([f"sep_{s}" for s in SCALES] + ["left_room_s", "right_room_s",
             "return_gap", "dir_consistency",
             "scale_agreement", "nearest_other_event_s", "contamination_flag"])


def build_matrices(events, clip_neighbors=False):
    """Returns X_v1 (5 raw HAL feats), side vectors L/R, and relative feats."""
    by_rec = {}
    for e in events:
        by_rec.setdefault(e["recording_id"], []).append(e["t"])
    X_v1, Ls, Rs, X_rel, keep = [], [], [], [], []
    for i, e in enumerate(events):
        l, r = side_vectors(e["rec"], e["t"], clip_neighbors=clip_neighbors)
        if l is None:
            continue
        v1 = hal_features_at(e["rec"]["feats"], e["rec"]["times"], e["t"])
        X_v1.append([v1.get(k) if v1.get(k) is not None else np.nan for k in HAL_FEATURE_NAMES])
        rel = relative_features(e["rec"], e["t"], by_rec[e["recording_id"]],
                                clip_neighbors=clip_neighbors)
        X_rel.append([rel.get(k, np.nan) for k in REL_NAMES])
        Ls.append(l.numpy()); Rs.append(r.numpy()); keep.append(i)
    return (np.array(X_v1, dtype=float), np.array(Ls), np.array(Rs),
            np.array(X_rel, dtype=float), keep)


# ------------------------------------------------------------ fold utils --

def _impute_scale_fit(X):
    cm = np.nanmean(X, 0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    Xi = np.where(np.isnan(X), cm, X)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
    return {"cm": cm, "mu": mu, "sd": sd}


def _impute_scale_apply(st, X):
    Xi = np.where(np.isnan(X), st["cm"], X)
    return (Xi - st["mu"]) / st["sd"]


def pca_fit(X, k):
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return {"mu": mu, "W": Vt[:k].T}


def pca_apply(p, X):
    return (X - p["mu"]) @ p["W"]


def pair_block(zl, zr):
    return np.concatenate([zl, zr, np.abs(zl - zr), zl * zr], axis=1)


# ------------------------------------------------------------ torch head --

class PairHead(torch.nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_head(Xtr, ytr, zl_tr, zr_tr, aux, epochs=300, lr=1e-3,
                     lam=0.1, seed=0, device="cpu"):
    """aux: None | 'crop' | 'contrastive'. The auxiliary term is deliberately
    small (lam) -- the review's point is that a contrastive objective must
    not dominate the end task, which is what made the adapter worse."""
    torch.manual_seed(seed)
    X = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yv = torch.tensor(ytr, dtype=torch.float32, device=device)
    zl = torch.tensor(zl_tr, dtype=torch.float32, device=device)
    zr = torch.tensor(zr_tr, dtype=torch.float32, device=device)
    head = PairHead(X.shape[1]).to(device)
    proj = torch.nn.Linear(zl.shape[1], 32).to(device) if aux else None
    params = list(head.parameters()) + (list(proj.parameters()) if proj else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(X), yv)
        if aux == "crop":
            # same-side halves should agree: a cheap stability regularizer
            # that uses NO label information at all
            a, b = F.normalize(proj(zl), dim=-1), F.normalize(proj(zr), dim=-1)
            loss = loss + lam * ((1 - F.cosine_similarity(a, a.detach())).pow(2).mean()
                                 + (1 - F.cosine_similarity(b, b.detach())).pow(2).mean())
        elif aux == "contrastive":
            a, b = F.normalize(proj(zl), dim=-1), F.normalize(proj(zr), dim=-1)
            d = 1 - F.cosine_similarity(a, b)
            pos = F.relu(0.5 - d).pow(2)[yv == 1].mean() if (yv == 1).any() else 0.0
            neg = d.pow(2)[yv == 0].mean() if (yv == 0).any() else 0.0
            loss = loss + lam * (pos + neg)
        loss.backward()
        opt.step()
    head.eval()
    return head


def predict_torch(head, X, device="cpu"):
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()


# ----------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.1, help="auxiliary-loss weight for P2/P3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pair_labels",
                    help="relabelled sheet (build_pair_relabel_sheet.py output, filled in). "
                         "With it, the experiment runs on the CLEAN subset "
                         "(strong_separate vs strong_align) and P0 is recomputed on that "
                         "SAME subset -- comparing a new model on a clean subset against "
                         "the old 0.702 on the old mixed labels would not be a fair test.")
    ap.add_argument("--gate_config",
                    help="JSON of pre-registered adoption thresholds, written BEFORE "
                         "seeing results. Keys: min_auroc_gain, min_folds_improved_frac, "
                         "min_coverage_at_90, max_worst_fold_drop")
    ap.add_argument("--stratified_folds", action="store_true", default=True)
    ap.add_argument("--clip_windows_at_neighbors", action="store_true",
                    help="bound each feature window at the neighbouring annotated segment "
                         "edges. Manual review found 81%% of contradictory pairs had windows "
                         "spanning two actions; this is the direct fix.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gold = S.load_gold(a.gold)
    ctx = S.load_context(a.context)
    by_rid = load_feature_caches(a.feat_cache)
    events = build_events(gold, ctx, by_rid)
    clean_mode = bool(a.pair_labels)
    if clean_mode:
        labels = T.load_pair_labels(a.pair_labels)
        events = T.apply_to_events(events, labels)
        if len(events) < 20:
            raise SystemExit(f"only {len(events)} events survive the clean filter -- "
                             f"relabel more rows before running this comparison")
        from collections import Counter
        print("  subtypes in clean subset:",
              dict(Counter(e.get("temporal_pair_subtype") for e in events)))
    else:
        print("  NOTE: running on the OLD binary labels (no --pair_labels). The adapter "
              "experiment showed this supervision is internally contradictory; treat "
              "these numbers as the legacy baseline only.")
    X_v1, Ls, Rs, X_rel, keep = build_matrices(events, a.clip_windows_at_neighbors)
    if a.clip_windows_at_neighbors:
        li, ri = REL_NAMES.index("left_room_s"), REL_NAMES.index("right_room_s")
        room = np.minimum(X_rel[:, li], X_rel[:, ri])
        print(f"  window clipping active: {(room < 4.0).mean():.1%} of events had a "
              f"+-4s window truncated by a neighbouring boundary "
              f"(median usable half-window {np.median(room):.2f}s)")
    events = [events[i] for i in keep]
    y = np.array([e["y"] for e in events], dtype=float)
    groups = np.array([e["recording_id"] for e in events])
    print(f"events usable: {len(events)} ({int(y.sum())} positive / "
          f"{int(len(y) - y.sum())} motion-HN), {len(set(groups))} recordings")
    print("all 188 audited events are DEVELOPMENT data; deployable claims need batch3.")

    folds = (stratified_grouped_folds(groups, y, a.folds, a.seed)
             if a.stratified_folds else grouped_folds(list(groups), a.folds, a.seed))
    arms = ["P0_v1_logreg", "P1_pairwise_proj", "P2_plus_crop", "P3_plus_contrastive"]
    preds = {k: np.full(len(events), np.nan) for k in arms}
    per_fold = {k: [] for k in arms}

    for fi, held in enumerate(folds):
        te = np.array([g in held for g in groups])
        tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        ytr, yte = y[tr], y[te]

        st1 = _impute_scale_fit(X_v1[tr])
        w, b = fit_logreg(_impute_scale_apply(st1, X_v1[tr]), ytr)
        preds["P0_v1_logreg"][te] = _sigmoid(_impute_scale_apply(st1, X_v1[te]) @ w + b)

        pca = pca_fit(np.concatenate([Ls[tr], Rs[tr]], 0), a.pca_dim)
        zl_tr, zr_tr = pca_apply(pca, Ls[tr]), pca_apply(pca, Rs[tr])
        zl_te, zr_te = pca_apply(pca, Ls[te]), pca_apply(pca, Rs[te])
        st_rel = _impute_scale_fit(X_rel[tr])
        Ptr = np.concatenate([pair_block(zl_tr, zr_tr), _impute_scale_apply(st_rel, X_rel[tr])], 1)
        Pte = np.concatenate([pair_block(zl_te, zr_te), _impute_scale_apply(st_rel, X_rel[te])], 1)
        stP = _impute_scale_fit(Ptr)
        Ptr_s, Pte_s = _impute_scale_apply(stP, Ptr), _impute_scale_apply(stP, Pte)

        w1, b1 = fit_logreg(Ptr_s, ytr, l2=5.0)
        preds["P1_pairwise_proj"][te] = _sigmoid(Pte_s @ w1 + b1)

        for arm, aux in [("P2_plus_crop", "crop"), ("P3_plus_contrastive", "contrastive")]:
            head = train_torch_head(Ptr_s, ytr, zl_tr, zr_tr, aux, lam=a.lam,
                                    seed=a.seed + fi, device=a.device)
            preds[arm][te] = predict_torch(head, Pte_s, a.device)

        line = (f"  fold {fi+1}/{a.folds} (n_test={int(te.sum())} "
                f"[{int(yte.sum())}+/{int(len(yte)-yte.sum())}-], {len(held)} recs):")
        for k in arms:
            auc = _auroc(yte, preds[k][te])
            per_fold[k].append(auc)
            line += f"  {k.split('_')[0]}={auc:.3f}"
        print(line)

    print("\n=== pooled OOF (development numbers) ===")
    res = {}
    for k in arms:
        m = ~np.isnan(preds[k])
        auc = _auroc(y[m], preds[k][m])
        cov, _ = precision_coverage(y[m], preds[k][m], 0.90)
        folds_ok = [v for v in per_fold[k] if not np.isnan(v)]
        res[k] = {"pooled_auroc": float(auc), "coverage_at_0.9_precision": float(cov),
                  "per_fold_auroc": [float(v) for v in per_fold[k]],
                  "fold_mean": float(np.mean(folds_ok)) if folds_ok else None,
                  "fold_min": float(np.min(folds_ok)) if folds_ok else None}
        print(f"  {k:<22} AUROC={auc:.3f}  coverage@90%={cov:.3f}  "
              f"fold mean={res[k]['fold_mean']:.3f} min={res[k]['fold_min']:.3f}")

    p1, p0 = res["P1_pairwise_proj"], res["P0_v1_logreg"]
    print(f"\n  P1 vs P0 (the architecture question): {p1['pooled_auroc']:.3f} vs "
          f"{p0['pooled_auroc']:.3f}")
    for k in ("P2_plus_crop", "P3_plus_contrastive"):
        beats = sum(1 for x, z in zip(per_fold[k], per_fold["P1_pairwise_proj"])
                    if not np.isnan(x) and not np.isnan(z) and x > z)
        n = sum(1 for x, z in zip(per_fold[k], per_fold["P1_pairwise_proj"])
                if not np.isnan(x) and not np.isnan(z))
        print(f"  {k} beats P1 in {beats}/{n} folds "
              f"({'STABLE -- auxiliary supervision earns its place' if beats >= n - 1 and n else 'not stable -- do not adopt'})")
    print("\n  reminder: coverage@0.90 here is chosen post hoc on these same OOF "
          "predictions, so it is an upper bound, not a deployable operating point.")

    # --- pre-registered adoption gate ------------------------------------
    if a.gate_config:
        with open(a.gate_config, encoding="utf-8") as f:
            gate = json.load(f)
        base_key = gate.get("baseline_arm", "P0_v1_logreg")
        cand_key = gate.get("candidate_arm", "P1_pairwise_proj")
        base, cand = res[base_key], res[cand_key]
        bf = [v for v in per_fold[base_key] if not np.isnan(v)]
        cf = [v for v in per_fold[cand_key] if not np.isnan(v)]
        paired = [(x, z) for x, z in zip(per_fold[cand_key], per_fold[base_key])
                  if not np.isnan(x) and not np.isnan(z)]
        improved_frac = (sum(1 for x, z in paired if x > z) / len(paired)) if paired else 0.0
        gain = cand["pooled_auroc"] - base["pooled_auroc"]
        worst_drop = (min(bf) - min(cf)) if bf and cf else float("inf")
        checks = {
            "auroc_gain": (gain, gate.get("min_auroc_gain", 0.03), gain >= gate.get("min_auroc_gain", 0.03)),
            "folds_improved_frac": (improved_frac, gate.get("min_folds_improved_frac", 0.8),
                                    improved_frac >= gate.get("min_folds_improved_frac", 0.8)),
            "coverage_at_90": (cand["coverage_at_0.9_precision"], gate.get("min_coverage_at_90", 0.10),
                               cand["coverage_at_0.9_precision"] >= gate.get("min_coverage_at_90", 0.10)),
            "worst_fold_not_degraded": (worst_drop, gate.get("max_worst_fold_drop", 0.05),
                                        worst_drop <= gate.get("max_worst_fold_drop", 0.05)),
        }
        print(f"\n=== PRE-REGISTERED GATE ({cand_key} vs {base_key}) ===")
        print(f"  thresholds fixed in {a.gate_config} "
              f"(written before results -- do not edit them now)")
        for name, (val, thr, ok) in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<26} {val:+.3f}  (threshold {thr})")
        passed = all(ok for _, _, ok in checks.values())
        print(f"  -> {'GATE PASSED: freeze this config and collect batch3' if passed else 'GATE NOT PASSED: do NOT collect batch3 or restore auto-keep'}")
        res["gate"] = {"config": gate, "passed": bool(passed),
                       "checks": {k: {"value": float(v), "threshold": float(t), "pass": bool(o)}
                                  for k, (v, t, o) in checks.items()}}
    else:
        print("\n  no --gate_config given: adoption thresholds were NOT pre-registered, "
              "so any 'improvement' read off this table is exploratory only.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}")
    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(a.out, input_paths=[a.gold, a.context] + a.feat_cache,
                       extra={"folds": a.folds, "pca_dim": a.pca_dim, "lam": a.lam})
    except Exception as e:
        print(f"[manifest] skipped ({e})")


if __name__ == "__main__":
    main()
