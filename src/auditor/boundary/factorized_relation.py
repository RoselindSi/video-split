"""One encoder, two heads: action change and instance reset, composed by rule.

The combined BOUNDARY target is now known to hurt. Training `new_action` and
`same_action_new_instance` as one positive class cost the new_action side
0.120 AUROC against a head trained on that task alone, paired and
recording-grouped, CI [+0.015, +0.222]. That interval excludes zero, which
makes it the first architecture claim in this project with a statistical test
behind it rather than a plausible story.

What it is NOT is a division of labour between representations. Both probes
beat every frozen arm on their own task -- the temporal student carries action
identity AND instance reset, and the earlier reading that the semantic arms
owned new_action came from comparing a head trained on the joint target
against arms that were not. So the encoder stays shared. Only the decision
targets separate.

    Head A   new_action  against  same_action_new_instance + same_instance
    Head B   same_action_new_instance  against  same_instance,
             its loss MASKED on new_action events

Head B is asked a question that only exists inside one action, so training it
on new_action events would be asking "did this instance reset" about an event
where the action itself changed. The mask is the point, not an optimisation.

COMPOSED BY RULE, WITH NO FITTED WEIGHT:

    P(boundary) = pA + (1 - pA) * pB

which reads as "either the action changed, or it did not and the instance
reset". A fitted fusion weight would be a third thing to select on 207 events
and would turn a pre-registered test into a search.

NO THREE-CLASS SOFTMAX. Two independent Bernoullis, because a single softmax
would put the two judgements back into one geometry to compete over -- the
same collapse this file exists to undo -- and because pA and pB are separately
readable while a three-way posterior is not.

EVERYTHING IS TRAINED IN THIS RUN, including the combined baseline and the
three independent probes, so every comparison shares one set of folds and one
seed. Reading a baseline off an earlier file would leave fold alignment as an
assumption.

THE PRE-REGISTERED TEST, stated before the numbers exist:

  headline   composite against the combined binary student on the same 207
             events, paired and grouped by recording. The claim is the delta,
             not a threshold on the absolute value.
  sanity     the shared heads against the INDEPENDENT probes on A, B and C.
             If sharing the encoder pulls them back down -- A well below
             0.852, B below 0.876 -- then the interference is in the encoder's
             gradients too, not only in the merged label, and the next step is
             separate lightweight encoders. That is a different finding and it
             is checked here rather than assumed away.

Usage:
    python -m src.auditor.boundary.factorized_relation \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --labels data/gold/boundary_v1_labels_ontology_v2.json \
        --feat_cache ... --local_cache ... \
        --out /workspace/tr1/results/auditor/factorized_relation.json
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
from src.auditor.common.temporal_encoder import TemporalEncoder, n_params
from src.auditor.boundary.model import build_input
from src.auditor.boundary.relation_experiment import (
    RelationHead, pca_fit, proj, paired_delta, boot)
from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.state_adapter import _auroc

NEW, SANI, SAME = "new_action", "same_action_new_instance", "same_instance"


class Factorized(torch.nn.Module):
    def __init__(self, in_dim, hidden=96, dropout=0.3):
        super().__init__()
        self.enc = TemporalEncoder(in_dim, hidden=hidden, dropout=dropout)
        d = self.enc.out_dim
        self.action = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                          torch.nn.Linear(d, 2))
        self.reset = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                         torch.nn.Linear(d, 2))

    def forward(self, x, m):
        z = self.enc(x, m)
        return self.action(z), self.reset(z)


def class_weight(y, mask):
    c = np.bincount(y[mask].astype(int), minlength=2) + 1
    w = (c.sum() / c) / ((c.sum() / c).mean())
    return torch.tensor(w, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrated", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the reset loss. Fixed at 1.0 on purpose: "
                         "tuning it on 207 events would make this a search")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    import csv
    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        rel = {r["event_id"]: r["instance_relation"] for r in csv.DictReader(f)}
    lab = {e["event_id"]: e
           for e in json.load(open(a.labels, encoding="utf-8"))["events"]}
    use = [e for e in lab if rel.get(e) in (NEW, SANI, SAME)]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gc, lc = load_caches(a.feat_cache), load_caches(a.local_cache)
    ev = build_events([lab[e] for e in use], gc, lc, a.half_s, a.n_frames)
    for e in ev:
        e["_rel"] = rel[e["event_id"]]
    groups = [e["recording_id"] for e in ev]
    r = np.array([e["_rel"] for e in ev])
    y_bnd = (r != SAME).astype(float)
    y_act = (r == NEW).astype(float)
    y_res = (r == SANI).astype(float)
    same_action = (r != NEW)
    print(f"{len(ev)} events over {len(set(groups))} recordings: "
          f"{dict(Counter(r.tolist()))}")
    print(f"  head A positive {int(y_act.sum())}; head B positive "
          f"{int(y_res.sum())} of {int(same_action.sum())} same-action events")

    G, L = stack(ev, "g"), stack(ev, "l")
    VG = torch.from_numpy(stack(ev, "valid_g"))
    VL = torch.from_numpy(stack(ev, "valid_l"))
    vg_np, vl_np = stack(ev, "valid_g"), stack(ev, "valid_l")
    # ONE fold assignment, shared by every model below, so no comparison rests
    # on the folds happening to line up
    folds = stratified_grouped_folds(groups, y_bnd, a.n_folds, seed=a.seed)

    def fold_inputs(tr):
        pg = pca_fit(G[tr][vg_np[tr]], a.pca_dim)
        pl = pca_fit(L[tr][vl_np[tr]], a.pca_dim)
        Pg = torch.from_numpy(proj(pg, G)).float()
        Pl = torch.from_numpy(proj(pl, L)).float()
        for P in (Pg, Pl):
            sd = P[torch.from_numpy(tr)].reshape(-1, P.shape[-1]).std(0)
            P /= sd.clamp(min=1e-6)
        return build_input(Pg, Pl, VG, VL)

    def train_single(y, sub=None, tag=""):
        """One binary head on one target; `sub` restricts the events."""
        keep = np.ones(len(ev), bool) if sub is None else sub
        out = np.full(len(ev), np.nan)
        for f in folds:
            te = np.array([g in f for g in groups]) & keep
            tr = (~np.array([g in f for g in groups])) & keep
            if te.sum() < 2 or tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
                continue
            X, M = fold_inputs(tr)
            model = RelationHead(X.shape[-1], a.hidden, a.dropout)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                    weight_decay=a.weight_decay)
            yt = torch.from_numpy(y).long()
            w = class_weight(y, tr)
            model.train()
            for _ in range(a.epochs):
                opt.zero_grad()
                loss = F.cross_entropy(model(X, M)[torch.from_numpy(tr)],
                                       yt[torch.from_numpy(tr)], weight=w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                out[te] = F.softmax(model(X, M)[torch.from_numpy(te)],
                                    -1)[:, 1].numpy()
        return out

    def train_factorized():
        pa = np.full(len(ev), np.nan)
        pb = np.full(len(ev), np.nan)
        for fi, f in enumerate(folds):
            te = np.array([g in f for g in groups])
            tr = ~te
            if te.sum() < 2 or tr.sum() < 20:
                continue
            X, M = fold_inputs(tr)
            model = Factorized(X.shape[-1], a.hidden, a.dropout)
            if fi == 0:
                print(f"\n  factorized: {n_params(model)} parameters against "
                      f"{int(tr.sum())} training events")
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                    weight_decay=a.weight_decay)
            ya = torch.from_numpy(y_act).long()
            yr = torch.from_numpy(y_res).long()
            m_a = torch.from_numpy(tr)
            # head B never sees a new_action event: "did this instance reset"
            # is not a question about an event where the action changed
            m_b = torch.from_numpy(tr & same_action)
            wa, wb = class_weight(y_act, tr), class_weight(y_res,
                                                           tr & same_action)
            model.train()
            for _ in range(a.epochs):
                opt.zero_grad()
                la, lb = model(X, M)
                loss = F.cross_entropy(la[m_a], ya[m_a], weight=wa)
                if m_b.sum() > 1:
                    loss = loss + a.lam * F.cross_entropy(lb[m_b], yr[m_b],
                                                          weight=wb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                la, lb = model(X, M)
                tt = torch.from_numpy(te)
                pa[te] = F.softmax(la[tt], -1)[:, 1].numpy()
                pb[te] = F.softmax(lb[tt], -1)[:, 1].numpy()
        return pa, pb

    print("\ntraining, all on the same folds:")
    p_comb = train_single(y_bnd, None, "combined")
    print("  combined binary student            done")
    pA_ind = train_single(y_act, None, "probe A")
    print("  independent probe A                done")
    pB_ind = train_single(y_res, same_action, "probe B")
    print("  independent probe B                done")
    sub_c = (r == NEW) | (r == SAME)
    pC_ind = train_single(y_act, sub_c, "probe C")
    print("  independent probe C                done")
    pa, pb = train_factorized()
    print("  shared factorized model            done")

    # the composition is a rule, not a fit
    comp = pa + (1.0 - pa) * pb

    def score(name, y, p, keep):
        m = keep & np.isfinite(p)
        if m.sum() < 8 or len(set(y[m].tolist())) < 2:
            print(f"    {name:<34} too few, withheld")
            return None
        gg = [groups[i] for i in np.where(m)[0]]
        lo, hi = boot(y[m], p[m], gg, a.n_boot, a.seed)
        print(f"    {name:<34} {_auroc(y[m], p[m]):.3f}  [{lo:.3f}, {hi:.3f}]")
        return m

    print(f"\n{'=' * 82}\nHEADLINE: composite against the combined binary "
          f"student\n{'=' * 82}")
    keep = np.isfinite(comp) & np.isfinite(p_comb)
    score("factorized composite", y_bnd, comp, keep)
    score("combined binary student", y_bnd, p_comb, keep)
    gg = [groups[i] for i in np.where(keep)[0]]
    d, lo, hi = paired_delta(y_bnd[keep], comp[keep], p_comb[keep], gg,
                             a.n_boot, a.seed)
    v = ("no detectable difference" if lo <= 0 <= hi else
         "factorising HELPS" if lo > 0 else "factorising HURTS")
    print(f"    composite minus combined           {d:+.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]   {v}")
    print("    The claim is the delta. An absolute value above 0.815 with an "
          "interval spanning zero would not be one.")

    print(f"\n{'=' * 82}\nSANITY: do the shared heads hold up against the "
          f"independent probes?\n{'=' * 82}")
    for name, y, p_sh, p_ind, keep_ in (
            ("A  new_action vs rest", y_act, pa, pA_ind,
             np.ones(len(ev), bool)),
            ("B  reset vs continuous", y_res, pb, pB_ind, same_action),
            ("C  new_action vs same_instance", y_act, pa, pC_ind, sub_c)):
        print(f"\n  {name}")
        m = keep_ & np.isfinite(p_sh) & np.isfinite(p_ind)
        score("shared head", y, p_sh, m)
        score("independent probe", y, p_ind, m)
        gg2 = [groups[i] for i in np.where(m)[0]]
        d2, lo2, hi2 = paired_delta(y[m], p_sh[m], p_ind[m], gg2, a.n_boot,
                                    a.seed)
        v2 = ("no detectable difference" if lo2 <= 0 <= hi2 else
              "sharing helps" if lo2 > 0 else "SHARING HURTS")
        print(f"    shared minus independent           {d2:+.3f}  "
              f"[{lo2:+.3f}, {hi2:+.3f}]   {v2}")
    print("\n  A shared head clearly below its independent probe would mean "
          "the interference lives in the encoder's gradients\n  and not only "
          "in the merged label, and the answer to that is separate "
          "lightweight encoders rather than this model.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump({"lam": a.lam, "n": len(ev),
                   "events": [{"event_id": e["event_id"],
                               "recording_id": e["recording_id"],
                               "relation": e["_rel"],
                               "p_action": None if not np.isfinite(pa[i]) else float(pa[i]),
                               "p_reset": None if not np.isfinite(pb[i]) else float(pb[i]),
                               "p_boundary": None if not np.isfinite(comp[i]) else float(comp[i]),
                               "p_combined": None if not np.isfinite(p_comb[i]) else float(p_comb[i])}
                              for i, e in enumerate(ev)]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
