"""Train morphology ONCE on recordings the detector never scored, infer on the ones it did.

This is not `train.py`. That file answers whether the four-class target is
better posed than the old binary, and does it with five-fold OOF over its own
events. This one answers a different question and needs a different split:

    what does a REAL learned morphology veto buy on the detector's own
    candidate pool, at a fixed review budget?

The split that makes it answerable is already there. Morphology supervision
covers 160 recordings; the detector's out-of-fold candidates come from 36. The
288 supervised events outside those 36 train the head, the 36 receive the
predictions, and no fold machinery is needed because the two sets are
disjoint by construction. That disjointness is asserted, not assumed.

THE CANDIDATE POOL IS READ, NEVER RE-DERIVED. `detector_calibration
--emit_candidates` is the only place peaks are picked and matched. A second
implementation here would differ by a candidate or two and the comparison
between arms would be measuring that instead of the veto.

ONLY THE MORPHOLOGY HEAD IS TRAINED AND ONLY ITS OUTPUT IS EMITTED. The model
carries relation, offset, width and two visibility heads because the
architecture is the deliverable, but relation has 10 usable events and
observability has none. Their outputs on this run are near-random and shipping
them as evidence would automate on a head with nothing behind it.

AND WITHIN MORPHOLOGY, ONLY NO_TRANSITION IS EXPECTED TO WORK. It has 101
training events outside the evaluation recordings and it is the class that
answers the largest error family -- a peak in the middle of a segment, which
was 56.5% of the detector's false positives. INTERVAL has 25 and UNOBSERVABLE
18; both are emitted and neither should be asked to license anything.

Usage:
    python -m src.auditor.boundary.morphology_external \
        --labels data/gold/boundary_v1_labels.json \
        --detector_candidates results/auditor/oof_candidates.jsonl \
        --feat_cache ... --local_cache ... \
        --out results/auditor/morphology_external_oof.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.boundary.model import (MORPHOLOGY, BoundaryModel, build_input)
from src.auditor.boundary.train import masked_ce, pca_fit, project_seq
from src.auditor.common.feature_loader import build_events, load_caches, stack
from src.auditor.common.temporal_encoder import n_params


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/gold/boundary_v1_labels.json")
    ap.add_argument("--detector_candidates", required=True)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cands = [json.loads(l) for l in open(a.detector_candidates,
                                         encoding="utf-8") if l.strip()]
    eval_rids = {c["recording_id"] for c in cands}
    print(f"candidate manifest: {len(cands)} candidates over "
          f"{len(eval_rids)} recordings")

    blob = json.load(open(a.labels, encoding="utf-8"))
    gold = blob.get("events", blob if isinstance(blob, list) else [])
    sup = [e for e in gold if e.get("morphology") in MORPHOLOGY]
    print(f"\nmorphology supervision: {len(sup)} events over "
          f"{len({e['recording_id'] for e in sup})} recordings")
    print(f"  {dict(Counter(e['morphology'] for e in sup))}")

    train_gold = [e for e in sup if e["recording_id"] not in eval_rids]
    train_rids = {e["recording_id"] for e in train_gold}
    overlap = train_rids & eval_rids
    if overlap:
        raise SystemExit(
            f"TRAIN ∩ EVAL RECORDINGS = {len(overlap)}: {sorted(overlap)[:5]}"
            f"\n  A head that trained on a recording it is about to score "
            f"carries information the\n  detector's own out-of-fold split was "
            f"built to exclude, and the veto would\n  measure that instead of "
            f"morphology.")
    print(f"\nTRAIN ∩ EVAL RECORDINGS = 0")
    print(f"  training on {len(train_gold)} events over {len(train_rids)} "
          f"recordings")
    print(f"  {dict(Counter(e['morphology'] for e in train_gold))}")
    thin = [k for k, v in Counter(e["morphology"]
                                  for e in train_gold).items() if v < 40]
    if thin:
        print(f"  !! {thin} have under 40 training events. They are emitted "
              f"and must not license\n     any action on this run.")

    gc = load_caches(a.feat_cache)
    lc = load_caches(a.local_cache)
    print(f"\ncaches: {len(gc)} global recordings, {len(lc)} local")

    print(f"\ntraining events:")
    train_ev = build_events(train_gold, gc, lc, a.half_s, a.n_frames)
    if len(train_ev) < len(train_gold):
        print(f"  {len(train_gold)} eligible labels -> {len(train_ev)} with "
              f"usable sequences. The shortfall is\n  reported rather than "
              f"padded: substituting a neighbouring frame for a missing one "
              f"is\n  a morphology answer.")
    if not train_ev:
        raise SystemExit("no training event has sequences; check cache paths")

    # PCA AND SCALE ON TRAINING FRAMES ONLY. The candidates contribute to
    # nothing fitted -- not the projection, not the scale, not the class
    # weights. A projection fitted on the evaluation recordings would leak
    # them into every prediction made about them.
    G, L = stack(train_ev, "g"), stack(train_ev, "l")
    VG = torch.from_numpy(stack(train_ev, "valid_g"))
    VL = torch.from_numpy(stack(train_ev, "valid_l"))
    pg = pca_fit(G[stack(train_ev, "valid_g")], a.pca_dim)
    pl = pca_fit(L[stack(train_ev, "valid_l")], a.pca_dim)
    Pg = torch.from_numpy(project_seq(pg, G)).float()
    Pl = torch.from_numpy(project_seq(pl, L)).float()
    sg = Pg.reshape(-1, Pg.shape[-1]).std(0).clamp(min=1e-6)
    sl = Pl.reshape(-1, Pl.shape[-1]).std(0).clamp(min=1e-6)
    X, M = build_input(Pg / sg, Pl / sl, VG, VL)

    m_idx = {k: i for i, k in enumerate(MORPHOLOGY)}
    y = torch.from_numpy(np.array([m_idx[e["morphology"]]
                                   for e in train_ev])).long()
    mask = torch.ones(len(train_ev), dtype=torch.bool)
    cnt = np.bincount(y.numpy(), minlength=len(MORPHOLOGY)) + 1
    w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
                     dtype=torch.float32)

    model = BoundaryModel(X.shape[-1], hidden=a.hidden, dropout=a.dropout)
    print(f"\nencoder input {X.shape[-1]} dims, {n_params(model)} parameters "
          f"against {len(train_ev)} training events")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    model.train()
    for ep in range(a.epochs):
        opt.zero_grad()
        # ONLY morphology. relation/offset/width send no gradient here, the
        # same as in train.py, and their outputs are not written out.
        loss = masked_ce(model(X, M)["morphology"], y, mask, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"  final training loss {loss.item():.4f}  (in-sample, and it means "
          f"nothing about the\n  predictions below -- those are on recordings "
          f"this model never saw)")

    print(f"\ncandidate events:")
    stubs = [{"event_id": c["candidate_id"],
              "recording_id": c["recording_id"],
              "candidate_time": c["candidate_time"]} for c in cands]
    cand_ev = build_events(stubs, gc, lc, a.half_s, a.n_frames)
    have = {e["event_id"] for e in cand_ev}

    Gc, Lc = stack(cand_ev, "g"), stack(cand_ev, "l")
    VGc = torch.from_numpy(stack(cand_ev, "valid_g"))
    VLc = torch.from_numpy(stack(cand_ev, "valid_l"))
    Pgc = torch.from_numpy(project_seq(pg, Gc)).float() / sg
    Plc = torch.from_numpy(project_seq(pl, Lc)).float() / sl
    Xc, Mc = build_input(Pgc, Plc, VGc, VLc)
    model.eval()
    with torch.no_grad():
        P = F.softmax(model(Xc, Mc)["morphology"], -1).numpy()

    by = {c["candidate_id"]: c for c in cands}
    with open(a.out, "w", encoding="utf-8") as f:
        for i, e in enumerate(cand_ev):
            c = by[e["event_id"]]
            f.write(json.dumps({
                "candidate_id": e["event_id"],
                "recording_id": e["recording_id"],
                "candidate_time": c["candidate_time"],
                "detector_score": c["detector_score"],
                "p_point": float(P[i, m_idx["POINT_TRANSITION"]]),
                "p_interval": float(P[i, m_idx["INTERVAL_TRANSITION"]]),
                "p_no_transition": float(P[i, m_idx["NO_TRANSITION"]]),
                "p_unobservable": float(P[i, m_idx["UNOBSERVABLE"]]),
                "coverage_g": e["coverage_g"], "coverage_l": e["coverage_l"],
            }) + "\n")

    missing = [c["candidate_id"] for c in cands if c["candidate_id"] not in have]
    print(f"\n  candidate manifest       {len(cands)}")
    print(f"  morphology predictions   {len(cand_ev)}")
    print(f"  missing sequence         {len(missing)}")
    if missing:
        print(f"    e.g. {missing[:3]}")
        print(f"    Those are never vetoed by the calibration step, so they "
              f"stay in review rather\n    than disappearing from the "
              f"denominator.")
    pnt = P[:, m_idx["NO_TRANSITION"]]
    print(f"\n  P(NO_TRANSITION) over the candidates: median {np.median(pnt):.3f}"
          f", >=0.5 on {int((pnt >= 0.5).sum())}, >=0.9 on "
          f"{int((pnt >= 0.9).sum())}")
    print(f"\nwrote {a.out}")
    print(f"  Next: detector_calibration --morphology_predictions {a.out} "
          f"--veto morphology_only")


if __name__ == "__main__":
    main()
