"""AUTO_ACCEPT_LABEL: `correct` against everything else. 67 vs 121, 48 recordings.

This is the only semantic target the gold can carry -- the other five statuses
are one judgement recorded in three collinear columns -- and it is also the
decision the product makes, so the two coincide for once.

A VIDEO-ONLY HEAD CANNOT VERIFY A LABEL, and this file will not pretend
otherwise. Asking "is this label correct" requires comparing the label to the
video, and the frozen Qwen features are visual. A head fitted on video alone
can only learn WHICH VIDEO REGIONS TEND TO CARRY BAD LABELS -- a prior over
scenes, not a verification. That number is worth having, because it bounds how
much of the status is predictable without reading the label at all, and
anything the full model gains above it is what reading the label bought. It is
labelled `video prior` everywhere and is not a product head.

THE VERIFICATION ARM REUSES THE NAMING PIPELINE, which already compares a
generated name to a stored one. `src/eval/score_names.py` computes verb-cluster
match, object F1, genericity and embedding similarity between two names; the
naming model's own prediction for a segment is video-grounded, so those
comparisons against the stored label ARE a video-to-text signal. They enter
here as features rather than as a score, and their coverage against the 188 is
printed before anything is fitted -- a naming jsonl that covers 40 of them
would make the arm a different experiment on a different population.

WHICH SEGMENT. The gold is per candidate EVENT and the label being judged is
the segment's. The window is centred on the candidate rather than on the
segment, so a long segment is only partly seen. That is a real limitation of
v1 and the alternative -- features spanning the segment -- is a change to the
extraction rather than to this file.

Usage:
    python -m src.auditor.semantic.accept_experiment \
        --labels data/gold/semantic_v1_labels.json \
        --feat_cache ... --local_cache ... \
        --naming_jsonl /workspace/tr1/results/naming/naming_qwen3.jsonl \
        --out /workspace/tr1/results/auditor/semantic_accept.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.common.feature_loader import load_caches, build_events, stack
from src.auditor.common.temporal_encoder import n_params
from src.auditor.boundary.model import build_input
from src.auditor.boundary.relation_experiment import (
    RelationHead, pca_fit, proj, paired_delta, boot)
from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
from src.boundary.state_adapter import _auroc


def support_features(row):
    """How much the video supports the STORED verb against its alternatives.

    n7_scored.jsonl carries a log-score per candidate verb and marks which
    letters are ground truth, so the margin between the stored verb and the
    best alternative is a direct, video-grounded measure of support -- which
    is what the original plan called P(primary verb supported), and much
    closer to it than comparing two generated names would have been.

    Returned as features, not as a verdict. A margin is not a probability and
    the scores are not calibrated across segments."""
    sc = row.get("scores") or row.get("contrastive_scores") or {}
    gt = row.get("gt_letters") or []
    if not sc or not gt:
        return None
    vals = {k: float(v) for k, v in sc.items()}
    gt_vals = [vals[g] for g in gt if g in vals]
    others = [v for k, v in vals.items() if k not in gt]
    if not gt_vals or not others:
        return None
    best_gt, best_other = max(gt_vals), max(others)
    import math
    mx = max(vals.values())
    z = [math.exp(v - mx) for v in vals.values()]
    tot = sum(z) or 1.0
    ent = -sum((q / tot) * math.log((q / tot) + 1e-12) for q in z)
    rank = 1 + sum(1 for v in others if v > best_gt)
    return [best_gt - best_other, best_gt, best_other, float(rank), ent,
            float(len(vals))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True,
                    help="output of src.auditor.semantic.labels --out")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--naming_jsonl", action="append", default=[],
                    help="naming-eval jsonl with a predicted name per "
                         "segment; the verification arm needs it")
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    blob = json.load(open(a.labels, encoding="utf-8"))
    lab = blob["events"]
    by_id = {e["event_id"]: e for e in lab}
    print(f"{len(lab)} semantic-labelled events over "
          f"{len({e['recording_id'] for e in lab})} recordings")
    print(f"  status: {dict(Counter(e['status'] for e in lab).most_common())}")

    # ------------------------------------------------------- naming coverage
    # The naming jsonls are keyed by (recording_id, segment_idx) and the audit
    # by event, so the join is by TIME CONTAINMENT and only n7-style rows
    # carry start/end. That is the whole join, and it is reported before use.
    segs = defaultdict(list)
    n_rows = 0
    for p in a.naming_jsonl:
        if not os.path.exists(p):
            print(f"  !! {p} not found")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_rows += 1
                if r.get("start") is None or r.get("end") is None:
                    continue
                segs[r["recording_id"]].append(r)
    n_timed = sum(len(v) for v in segs.values())
    print(f"  naming rows {n_rows}; {n_timed} carry start/end and can be "
          f"joined by time, over {len(segs)} recordings")
    if n_rows and not n_timed:
        print(f"  !! none of them carry segment bounds. Only n7-style rows do; "
              f"the others are keyed by segment_idx alone and cannot be\n"
              f"     matched to a candidate time without the segment table.")

    def seg_for(eid, rid, t):
        for r in segs.get(rid, []):
            if float(r["start"]) <= t <= float(r["end"]):
                return r
        return None

    pred_name = {}  # kept for the output schema; unused by the support arm

    # --------------------------------------------------------------- features
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # not `gc`: that shadows the stdlib module, and this file calls
    # gc.collect() between seeds
    gcache, lcache = load_caches(a.feat_cache), load_caches(a.local_cache)
    src = [{"event_id": e["event_id"], "recording_id": e["recording_id"],
            "candidate_time": _t(e["event_id"])} for e in lab]
    src = [s for s in src if s["candidate_time"] is not None]
    ev = build_events(src, gcache, lcache, a.half_s, a.n_frames)
    for e in ev:
        e["_y"] = float(by_id[e["event_id"]]["auto_accept_target"])
    y = np.array([e["_y"] for e in ev], float)
    groups = [e["recording_id"] for e in ev]
    print(f"  {len(ev)} with sequences: {int(y.sum())} accept / "
          f"{int((1 - y).sum())} review over {len(set(groups))} recordings")
    if len(ev) < 40 or len(set(groups)) < a.n_folds:
        raise SystemExit("too few events or recordings for a grouped split")

    G, L = stack(ev, "g"), stack(ev, "l")
    VG = torch.from_numpy(stack(ev, "valid_g"))
    VL = torch.from_numpy(stack(ev, "valid_l"))
    vg_np, vl_np = stack(ev, "valid_g"), stack(ev, "valid_l")
    folds = stratified_grouped_folds(groups, y, a.n_folds, seed=a.fold_seed)
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip()]

    def train(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        out = np.full(len(ev), np.nan)
        for fi, f in enumerate(folds):
            te = np.array([g in f for g in groups])
            tr = ~te
            if te.sum() < 2 or tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
                continue
            pg = pca_fit(G[tr][vg_np[tr]], a.pca_dim)
            pl = pca_fit(L[tr][vl_np[tr]], a.pca_dim)
            Pg = torch.from_numpy(proj(pg, G)).float()
            Pl = torch.from_numpy(proj(pl, L)).float()
            for P in (Pg, Pl):
                sd = P[torch.from_numpy(tr)].reshape(-1, P.shape[-1]).std(0)
                P /= sd.clamp(min=1e-6)
            X, M = build_input(Pg, Pl, VG, VL)
            model = RelationHead(X.shape[-1], a.hidden, a.dropout)
            if fi == 0 and seed == seeds[0]:
                print(f"\n  {n_params(model)} parameters against "
                      f"{int(tr.sum())} training events")
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                    weight_decay=a.weight_decay)
            yt = torch.from_numpy(y).long()
            trt = torch.from_numpy(tr)
            c = np.bincount(y[tr].astype(int), minlength=2) + 1
            w = torch.tensor((c.sum() / c) / ((c.sum() / c).mean()),
                             dtype=torch.float32)
            model.train()
            for _ in range(a.epochs):
                opt.zero_grad()
                loss = F.cross_entropy(model(X, M)[trt], yt[trt], weight=w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                out[te] = F.softmax(model(X, M)[torch.from_numpy(te)],
                                    -1)[:, 1].numpy()
        return out

    per = [train(sd) for sd in seeds]
    au = [_auroc(y[np.isfinite(p)], p[np.isfinite(p)]) for p in per]
    p_vid = np.nanmean(per, axis=0)
    print(f"\n  per-seed AUROC {[f'{x:.3f}' for x in au]}; "
          f"mean {np.mean(au):.3f} +/- {np.std(au, ddof=1) if len(au) > 1 else 0:.3f}")

    # ------------------------------------------------- the naming arm, nested
    nf = {}
    for e in ev:
        row = seg_for(e["event_id"], e["recording_id"],
                      float(_t(e["event_id"]) or -1))
        if row is None:
            continue
        f_ = support_features(row)
        if f_ is not None:
            nf[e["event_id"]] = f_
    print(f"  verb-support features computable on {len(nf)}/{len(ev)} events "
          f"over {len({by_id[k]['recording_id'] for k in nf})} recordings")
    if 0 < len(nf) < 40:
        print(f"  !! {len(nf)} is below the floor this file will fit on. The "
              f"naming pipeline covers 84 segments in total and the audit\n"
              f"     covers 188 events over 48 recordings, so the overlap is "
              f"small by construction -- the verification arm needs naming\n"
              f"     run over the audited segments, not a bigger model.")
    p_name = np.full(len(ev), np.nan)
    if len(nf) >= 40:
        Z = np.array([nf.get(e["event_id"], [np.nan] * 6) for e in ev], float)
        have = np.isfinite(Z).all(1)
        for f in folds:
            te = np.array([g in f for g in groups]) & have
            tr = (~np.array([g in f for g in groups])) & have
            if te.sum() < 2 or tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
                continue
            w, b = fit_logreg(Z[tr], y[tr], l2=1.0)
            p_name[te] = _sigmoid(Z[te] @ w + b)

    print(f"\n{'=' * 82}\nAUTO_ACCEPT: `correct` against everything else"
          f"\n{'=' * 82}")
    rows = [("video prior (no label read)", p_vid)]
    if np.isfinite(p_name).sum() >= 40:
        rows.append(("naming comparison only", p_name))
        both = np.isfinite(p_vid) & np.isfinite(p_name)
        if both.sum() >= 40:
            comb = np.full(len(ev), np.nan)
            for f in folds:
                te = np.array([g in f for g in groups]) & both
                tr = (~np.array([g in f for g in groups])) & both
                if te.sum() < 2 or tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
                    continue
                Zc = np.stack([p_vid, p_name], 1)
                w, b = fit_logreg(Zc[tr], y[tr], l2=1.0)
                comb[te] = _sigmoid(Zc[te] @ w + b)
            rows.append(("video + naming, nested", comb))
    for name, p in rows:
        m = np.isfinite(p)
        lo, hi = boot(y[m], p[m], [groups[i] for i in np.where(m)[0]],
                      a.n_boot, a.seed)
        print(f"  {name:<32} {_auroc(y[m], p[m]):.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"n={int(m.sum())}")
    if len(rows) > 1:
        print(f"\n  paired, recording-grouped:")
        for name, p in rows[1:]:
            m = np.isfinite(p) & np.isfinite(p_vid)
            d, lo, hi = paired_delta(y[m], p[m], p_vid[m],
                                     [groups[i] for i in np.where(m)[0]],
                                     a.n_boot, a.seed)
            v = ("no detectable difference" if lo <= 0 <= hi
                 else "reading the label helps" if lo > 0 else "worse")
            print(f"  {name + ' minus video prior':<44} {d:+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]   {v}")
        print(f"\n  The delta is the point. The video prior alone is not a "
              f"label verifier, however well it scores -- it can only learn\n"
              f"  which scenes tend to carry bad labels. What the full model "
              f"gains above it is what reading the label bought.")
    else:
        print(f"\n  Only the video prior ran. With no naming predictions "
              f"there is nothing here that reads the label, so this number\n"
              f"  bounds how much of the status is a property of the scene "
              f"and says nothing about verification.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump({"n": len(ev), "per_seed_auroc": au,
                   "events": [{"event_id": e["event_id"],
                               "recording_id": e["recording_id"],
                               "y": e["_y"],
                               "p_video": None if not np.isfinite(p_vid[i]) else float(p_vid[i]),
                               "p_naming": None if not np.isfinite(p_name[i]) else float(p_name[i])}
                              for i, e in enumerate(ev)]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


def _t(eid):
    try:
        return float(eid.rsplit("_t", 1)[1])
    except (IndexError, ValueError):
        return None


if __name__ == "__main__":
    main()
