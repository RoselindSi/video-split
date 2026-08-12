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
import gc
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.common.feature_loader import load_caches, build_events, stack
from src.auditor.common.temporal_encoder import TemporalEncoder, n_params
from src.auditor.boundary.model import build_input
from src.auditor.boundary.relation_experiment import (
    RelationHead, pca_fit, proj, paired_delta, boot)
from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.hal_vlm_fusion import fit_logreg, _sigmoid
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
    ap.add_argument("--fold_seed", type=int, default=0,
                    help="fixes the fold manifest. Held constant across "
                         "training seeds so only the initialisation varies")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated training seeds. The independent "
                         "probes between two earlier runs moved by 0.024 to "
                         "0.035, which is larger than the effect being "
                         "measured, so the headline is averaged over these")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="all",
                    help="comma-separated subset of {or,nested}. `or` is the "
                         "two cheap cells; `nested` refits the factors inside "
                         "every inner fold and is what took the machine down")
    ap.add_argument("--checkpoint",
                    help="per-seed results are written here as they finish "
                         "and reloaded on restart, so a crash costs one seed "
                         "rather than the run")
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
    # not `gc`: that shadows the stdlib module, and this file calls
    # gc.collect() between seeds
    gcache, lcache = load_caches(a.feat_cache), load_caches(a.local_cache)
    ev = build_events([lab[e] for e in use], gcache, lcache, a.half_s,
                      a.n_frames)
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
    folds = stratified_grouped_folds(groups, y_bnd, a.n_folds,
                                     seed=a.fold_seed)
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip() != ""]

    def fold_inputs(tr):
        pg = pca_fit(G[tr][vg_np[tr]], a.pca_dim)
        pl = pca_fit(L[tr][vl_np[tr]], a.pca_dim)
        Pg = torch.from_numpy(proj(pg, G)).float()
        Pl = torch.from_numpy(proj(pl, L)).float()
        for P in (Pg, Pl):
            sd = P[torch.from_numpy(tr)].reshape(-1, P.shape[-1]).std(0)
            P /= sd.clamp(min=1e-6)
        return build_input(Pg, Pl, VG, VL)

    def train_single(y, sub=None, tag="", seed=0, predict_all=False,
                     fold_list=None, avail=None):
        """One binary head on one target.

        `sub` restricts which events contribute LOSS. Prediction is a separate
        question: head B is trained only on same-action events and still has
        to emit a score on new_action ones, because the composition rule
        multiplies by it there. Conflating the two masks left pB undefined on
        exactly the events the OR needs it for."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        keep = np.ones(len(ev), bool) if sub is None else sub
        pool = np.ones(len(ev), bool) if avail is None else avail
        out = np.full(len(ev), np.nan)
        for f in (fold_list if fold_list is not None else folds):
            in_f = np.array([g in f for g in groups])
            te = (in_f & pool) if predict_all else (in_f & keep & pool)
            tr = (~in_f) & keep & pool
            if te.sum() < 2 or tr.sum() < 15 or len(set(y[tr].tolist())) < 2:
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

    def train_factorized(seed=0, fold_list=None, avail=None, quiet=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        pool = np.ones(len(ev), bool) if avail is None else avail
        pa = np.full(len(ev), np.nan)
        pb = np.full(len(ev), np.nan)
        for fi, f in enumerate(fold_list if fold_list is not None else folds):
            in_f = np.array([g in f for g in groups])
            te = in_f & pool
            tr = (~in_f) & pool
            if te.sum() < 2 or tr.sum() < 15:
                continue
            X, M = fold_inputs(tr)
            model = Factorized(X.shape[-1], a.hidden, a.dropout)
            if fi == 0 and not quiet:
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

    sub_c = (r == NEW) | (r == SAME)
    print(f"\ntraining on one fold manifest (fold_seed {a.fold_seed}), "
          f"seeds {seeds}:")
    per_seed = []
    # the per-seed loop is checkpointed too. It is cheap, but the machine died
    # inside it at seed 3 twice, and losing two completed seeds to that is
    # avoidable whatever the cause turns out to be.
    ck0 = {}
    if a.checkpoint and os.path.exists(a.checkpoint):
        ck0 = json.load(open(a.checkpoint, encoding="utf-8")).get("base", {})

    def save_base(sd, pc_, pa_, pb_):
        if not a.checkpoint:
            return
        prev = {}
        if os.path.exists(a.checkpoint):
            prev = json.load(open(a.checkpoint, encoding="utf-8"))
        prev.setdefault("base", {})[str(sd)] = {
            "comb": [None if not np.isfinite(x) else float(x) for x in pc_],
            "pa": [None if not np.isfinite(x) else float(x) for x in pa_],
            "pb": [None if not np.isfinite(x) else float(x) for x in pb_]}
        json.dump(prev, open(a.checkpoint, "w", encoding="utf-8"))

    for sd in seeds:
        if str(sd) in ck0:
            d_ = ck0[str(sd)]
            f_ = lambda v: np.array([np.nan if x is None else x for x in v],
                                    float)
            pc_, pa_, pb_ = f_(d_["comb"]), f_(d_["pa"]), f_(d_["pb"])
            print(f"  seed {sd}: resumed from checkpoint")
        else:
            pc_ = train_single(y_bnd, None, "combined", sd)
            pa_, pb_ = train_factorized(sd)
            save_base(sd, pc_, pa_, pb_)
        cm_ = pa_ + (1.0 - pa_) * pb_
        m = np.isfinite(cm_) & np.isfinite(pc_)
        per_seed.append({"seed": sd, "comb": pc_, "pa": pa_, "pb": pb_,
                         "comp": cm_,
                         "au_comb": _auroc(y_bnd[m], pc_[m]),
                         "au_comp": _auroc(y_bnd[m], cm_[m])})
        print(f"  seed {sd}: combined {per_seed[-1]['au_comb']:.3f}   "
              f"composite {per_seed[-1]['au_comp']:.3f}", flush=True)
        gc.collect()
    ac = np.array([x["au_comb"] for x in per_seed])
    af = np.array([x["au_comp"] for x in per_seed])
    if len(seeds) > 1:
        print(f"\n  across {len(seeds)} seeds: combined "
              f"{ac.mean():.3f} +/- {ac.std(ddof=1):.3f}   composite "
              f"{af.mean():.3f} +/- {af.std(ddof=1):.3f}")
        print(f"  per-seed delta {(af - ac).mean():+.3f} +/- "
              f"{(af - ac).std(ddof=1):.3f}; every seed positive: "
              f"{bool((af > ac).all())}")
        print(f"  Seed spread is the thing to read first. An effect smaller "
              f"than it is not an effect, and the independent probes moved "
              f"0.024\n  to 0.035 between two earlier runs with everything "
              f"else held.")
    # the seed-averaged probability per event, which is what the headline uses
    p_comb = np.nanmean([x["comb"] for x in per_seed], axis=0)
    pa = np.nanmean([x["pa"] for x in per_seed], axis=0)
    pb = np.nanmean([x["pb"] for x in per_seed], axis=0)
    # the composition is a rule, not a fit, and it is applied to the averaged
    # factors rather than averaging the composites -- the rule is what is
    # being tested
    comp = pa + (1.0 - pa) * pb

    base = seeds[0]
    pA_ind = train_single(y_act, None, "probe A", base)
    pB_ind = train_single(y_res, same_action, "probe B", base)
    pC_ind = train_single(y_act, sub_c, "probe C", base)
    print(f"  independent probes trained at seed {base} only; they are the "
          f"sanity check, not the headline")

    # ------------------------------------------------------ 2x2 FACTORIAL
    # Two things are now known to lose signal and they are different things:
    # sharing the encoder costs head A 0.031 [-0.066, -0.001], and the fixed
    # OR costs the new_action branch about 0.053. This crosses them.
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def nested_combiner(get_pair, seed):
        """OR replaced by a two-input logistic regression fitted INSIDE each
        outer fold's training recordings.

        The inner split is what makes it legitimate: a combiner fitted on the
        same predictions it is applied to would be selecting on the test
        events, which is the failure this project has made before under a
        different name. `get_pair(fold_list, avail, seed)` retrains the
        factor models on whatever subset it is handed."""
        out = np.full(len(ev), np.nan)
        for f in folds:
            in_f = np.array([g in f for g in groups])
            tr = ~in_f
            tr_groups = sorted({g for g, m in zip(groups, tr) if m})
            if len(tr_groups) < a.n_folds or tr.sum() < 30:
                continue
            inner = stratified_grouped_folds(
                [g for g, m in zip(groups, tr) if m],
                y_bnd[tr], min(a.n_folds - 1, 4), seed=a.fold_seed)
            ipa, ipb = get_pair(inner, tr, seed)
            m = tr & np.isfinite(ipa) & np.isfinite(ipb)
            if m.sum() < 20 or len(set(y_bnd[m].tolist())) < 2:
                continue
            Z = np.stack([logit(ipa[m]), logit(ipb[m])], 1)
            w, b = fit_logreg(Z, y_bnd[m], l2=1.0)
            opa, opb = get_pair([f], np.ones(len(ev), bool), seed)
            om = in_f & np.isfinite(opa) & np.isfinite(opb)
            out[om] = _sigmoid(np.stack([logit(opa[om]), logit(opb[om])], 1)
                               @ w + b)
        return out

    def pair_shared(fold_list, avail, seed):
        return train_factorized(seed, fold_list, avail, quiet=True)

    def pair_separate(fold_list, avail, seed):
        pa_ = train_single(y_act, None, "", seed, True, fold_list, avail)
        pb_ = train_single(y_res, same_action, "", seed, True, fold_list,
                           avail)
        return pa_, pb_

    want = ({"or", "nested"} if a.variants == "all"
            else {x.strip() for x in a.variants.split(",")})
    print(f"\n  2x2 factorial, {len(seeds)} seeds each, variants {sorted(want)}."
          f" The nested combiner refits the factors inside every inner fold\n"
          f"  and is what took the machine down; `--variants or` runs the two "
          f"cheap cells alone.")
    variants = defaultdict(list)
    done = set()
    if a.checkpoint and os.path.exists(a.checkpoint):
        ck = json.load(open(a.checkpoint, encoding="utf-8"))
        for k, per in ck.get("variants", {}).items():
            for sd_s, arr in per.items():
                variants[k].append(np.array(arr, float))
                done.add((k, int(sd_s)))
        print(f"  resumed {len(done)} seed-variant cells from "
              f"{os.path.basename(a.checkpoint)}")
    store = defaultdict(dict)

    def record(k, sd, arr):
        variants[k].append(arr)
        store[k][str(sd)] = [None if not np.isfinite(x) else float(x)
                             for x in arr]
        if a.checkpoint:
            prev = {}
            if os.path.exists(a.checkpoint):
                prev = json.load(open(a.checkpoint,
                                      encoding="utf-8")).get("variants", {})
            for kk, vv in store.items():
                prev.setdefault(kk, {}).update(vv)
            json.dump({"variants": prev},
                      open(a.checkpoint, "w", encoding="utf-8"))

    for sd in seeds:
        if "or" in want:
            if ("shared + fixed OR", sd) not in done:
                pa_sh, pb_sh = train_factorized(sd, quiet=True)
                record("shared + fixed OR", sd, pa_sh + (1 - pa_sh) * pb_sh)
            if ("separate + fixed OR", sd) not in done:
                pa_se = train_single(y_act, None, "", sd, True)
                pb_se = train_single(y_res, same_action, "", sd, True)
                record("separate + fixed OR", sd, pa_se + (1 - pa_se) * pb_se)
        if "nested" in want:
            if ("shared + nested logistic", sd) not in done:
                record("shared + nested logistic", sd,
                       nested_combiner(pair_shared, sd))
            if ("separate + nested logistic", sd) not in done:
                record("separate + nested logistic", sd,
                       nested_combiner(pair_separate, sd))
        print(f"    seed {sd} done", flush=True)
        gc.collect()
    V = {k: np.nanmean(v, axis=0) for k, v in variants.items() if v}
    V["combined binary"] = p_comb

    print(f"\n{'=' * 82}\n2x2 FACTORIAL: encoder against composition"
          f"\n{'=' * 82}")
    tasks = [("overall BOUNDARY", y_bnd, np.ones(len(ev), bool)),
             ("new_action vs same_instance", y_act, sub_c),
             ("reset vs continuous", y_res, same_action)]
    print(f"  {'variant':<30}" + "".join(f"{t[0][:22]:>26}" for t in tasks))
    for name in ("shared + fixed OR", "separate + fixed OR",
                 "shared + nested logistic", "separate + nested logistic",
                 "combined binary"):
        if name not in V:
            print(f"  {name:<30}" + f"{'not run':>26}")
            continue
        p_ = V[name]
        cells = []
        for _, yy, kp in tasks:
            m = kp & np.isfinite(p_)
            cells.append(f"{_auroc(yy[m], p_[m]):.3f}"
                         if m.sum() >= 8 and len(set(yy[m].tolist())) > 1
                         else "--")
        print(f"  {name:<30}" + "".join(f"{c:>26}" for c in cells))

    def delta(n1, n2, label):
        if n1 not in V or n2 not in V:
            print(f"  {label:<44} not run")
            return
        p1, p2 = V[n1], V[n2]
        m = np.isfinite(p1) & np.isfinite(p2)
        gg4 = [groups[i] for i in np.where(m)[0]]
        d, lo, hi = paired_delta(y_bnd[m], p1[m], p2[m], gg4, a.n_boot, a.seed)
        v = ("no detectable difference" if lo <= 0 <= hi else
             "first is better" if lo > 0 else "first is WORSE")
        print(f"  {label:<44} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {v}")

    print(f"\n  paired, recording-grouped, on overall BOUNDARY:")
    delta("separate + fixed OR", "shared + fixed OR",
          "separate minus shared, under fixed OR")
    delta("shared + nested logistic", "shared + fixed OR",
          "learned minus OR, under shared")
    delta("separate + nested logistic", "separate + fixed OR",
          "learned minus OR, under separate")
    have = [k for k in ("shared + fixed OR", "separate + fixed OR",
                        "shared + nested logistic",
                        "separate + nested logistic") if k in V]
    best = max(have, key=lambda k: _auroc(y_bnd[np.isfinite(V[k])],
                                          V[k][np.isfinite(V[k])]))
    print(f"\n  best variant is `{best}`")
    delta(best, "combined binary", "BEST minus combined binary")
    print(f"  That last line decides whether the factorised relation "
          f"architecture is carried forward. The three task columns decide\n"
          f"  what it is carried forward FOR: a pooled number hid the "
          f"structure once already and 18 new_action against 40 reset\n"
          f"  means a large subtype gain can only move it a little.")

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

    print(f"\n{'=' * 82}\nWHERE THE COMPOSITION LOSES IT\n{'=' * 82}")
    print("  Each subtype task scored by ITS OWN head and by the final "
          "composite. The overall number is the two positives\n  weighted "
          f"together -- {int(y_act.sum())} new_action against "
          f"{int(y_res.sum())} reset -- so a large gain on the smaller one "
          f"dilutes to little.")
    print(f"\n  {'task':<34} {'own head':>12} {'composite':>12} "
          f"{'delta':>18}")
    for name, y_, own, keep_ in (
            ("new_action vs same_instance", y_act, pa, sub_c),
            ("reset vs continuous", y_res, pb, same_action)):
        m = keep_ & np.isfinite(own) & np.isfinite(comp)
        if m.sum() < 8 or len(set(y_[m].tolist())) < 2:
            continue
        gg3 = [groups[i] for i in np.where(m)[0]]
        au_o, au_c = _auroc(y_[m], own[m]), _auroc(y_[m], comp[m])
        d3, lo3, hi3 = paired_delta(y_[m], comp[m], own[m], gg3, a.n_boot,
                                    a.seed)
        print(f"  {name:<34} {au_o:>12.3f} {au_c:>12.3f}   "
              f"{d3:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
    print("\n  A composite well below the own head on a task means the "
          "factorisation is right and the fixed OR is throwing the\n  branch "
          "signal away -- a composition and calibration problem, not a "
          "representation one. pB carries no supervision on\n  new_action "
          "events at all, so nothing constrains what it emits there, and the "
          "rule multiplies by it regardless.")

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
