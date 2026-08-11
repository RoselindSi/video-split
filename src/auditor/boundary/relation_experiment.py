"""Can the frozen features tell an ontology boundary from a continuation?

The target is no longer POINT/INTERVAL/NO. That mixed two axes and the mixing
is now demonstrated rather than suspected: `same_action_new_instance + gap` is
a BOUNDARY and is not a POINT, and `new_action + gradual` is a BOUNDARY and is
not a compact point either. Boundary existence is decided by
instance_relation, and instance_relation alone -- read from
configs/auditor/instance_relation_policy_v2.yaml rather than restated here.

    positive   new_action, same_action_new_instance
    negative   same_instance
    excluded   cannot_determine, initial_action_start, terminal_action_end,
               UNKNOWN

The excluded classes are excluded because their product behaviour is undefined
or their evidence is missing, not because they are hard. Training on them
would be fitting a decision nobody has made.

THE HEADLINE IS NOT AN OVERALL AUROC, IT IS THREE TABLES.

  A  the same events scored by the new student, by the OLD P(POINT), and by
     each frozen arm. This is what says whether the earlier weakness came from
     the target or from the features -- the old score has never been asked
     this question and its number here is the answer to "was the target
     wrong".

  B  the two positives split apart. new_action against same_instance is the
     uncontested contrast; same_action_new_instance against same_instance is
     the one this project has actually been failing at all along, because it
     asks whether the person LET GO and re-engaged rather than whether the
     scene changed. A strong first row beside a weak second one localises the
     gap to interaction reset -- release, idle, workspace exit, recontact --
     and that is a specific feature to build, not a bigger encoder.

  C  dev against batch3, still separated. But the relation-labelled events
     were hand-picked from the UNKNOWN pool, so every number here is a
     LEARNABILITY diagnostic and none of them is a deployment estimate. The
     54 came from a stratified draw over unknowns and the rest were derived
     from legacy labels; the mixture is not any population.

ONE CONFIGURATION, and the same fold discipline as everything else: PCA and
scaling fitted inside the training recordings of each fold, recordings never
split across a boundary.

Usage:
    python -m src.auditor.boundary.relation_experiment \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --labels data/gold/boundary_v1_labels_ontology_v2.json \
        --feat_cache ... --local_cache ... \
        --compare_oof .../boundary_v1_oof_ontology_v2.json \
        --decisions .../policy_decisions_v4...csv \
        --out /workspace/tr1/results/auditor/relation_oof.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.common.feature_loader import load_caches, build_events, stack
from src.auditor.common.temporal_encoder import TemporalEncoder, n_params
from src.auditor.boundary.model import build_input
from src.boundary.pairwise_verifier import stratified_grouped_folds
from src.boundary.state_adapter import _auroc

OLD_ARMS = ["P1 (global) alone", "local alone", "P1 + local, feature-level"]
POINT = "POINT_TRANSITION"


class RelationHead(torch.nn.Module):
    def __init__(self, in_dim, hidden=96, dropout=0.3):
        super().__init__()
        self.enc = TemporalEncoder(in_dim, hidden=hidden, dropout=dropout)
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                        torch.nn.Linear(self.enc.out_dim, 2))

    def forward(self, x, m):
        return self.head(self.enc(x, m))


def pca_fit(X, dim):
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu.astype(np.float32), Vt[:dim].T.astype(np.float32)


def proj(p, seq):
    mu, W = p
    n, t, d = seq.shape
    return ((seq.reshape(-1, d) - mu) @ W).reshape(n, t, -1)


def boot(y, p, groups, n_boot, seed):
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = [i for k in rng.integers(0, len(keys), len(keys))
               for i in by[keys[k]]]
        yy = np.asarray([y[i] for i in idx], float)
        if len(set(yy.tolist())) < 2:
            continue
        v = _auroc(yy, np.asarray([p[i] for i in idx], float))
        if np.isfinite(v):
            out.append(v)
    if len(out) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrated", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--policy",
                    default="configs/auditor/instance_relation_policy_v2.yaml")
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--compare_oof", help="the POINT/INTERVAL/NO run, for A")
    ap.add_argument("--decisions", help="frozen arms, for A")
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    import yaml
    tt = yaml.safe_load(open(a.policy, encoding="utf-8"))["training_target"]
    pos_set, neg_set = set(tt["positive"]), set(tt["negative"])

    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        rel = {r["event_id"]: r for r in csv.DictReader(f)}
    lab = {e["event_id"]: e
           for e in json.load(open(a.labels, encoding="utf-8"))["events"]}
    use = [e for e in lab
           if rel.get(e, {}).get("instance_relation") in pos_set | neg_set]
    print(f"{len(rel)} migrated rows, {len(lab)} labelled events, "
          f"{len(use)} usable")
    print(f"  relations: "
          f"{dict(Counter(rel[e]['instance_relation'] for e in use))}")
    print(f"  excluded by policy: {tt['excluded']}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gc, lc = load_caches(a.feat_cache), load_caches(a.local_cache)
    ev = build_events([lab[e] for e in use], gc, lc, a.half_s, a.n_frames)
    if not ev:
        raise SystemExit("no event has sequences; check the cache paths")
    for e in ev:
        e["_rel"] = rel[e["event_id"]]["instance_relation"]
        e["_y"] = 1.0 if e["_rel"] in pos_set else 0.0
    y = np.array([e["_y"] for e in ev], float)
    groups = [e["recording_id"] for e in ev]
    print(f"  {len(ev)} with sequences: {int(y.sum())} BOUNDARY / "
          f"{int((1 - y).sum())} NO_BOUNDARY over {len(set(groups))} "
          f"recordings")

    G, L = stack(ev, "g"), stack(ev, "l")
    VG = torch.from_numpy(stack(ev, "valid_g"))
    VL = torch.from_numpy(stack(ev, "valid_l"))
    oof = np.full(len(ev), np.nan)
    for fi, f in enumerate(stratified_grouped_folds(groups, y, a.n_folds,
                                                    seed=a.seed)):
        te = np.array([g in f for g in groups])
        tr = ~te
        if te.sum() < 2 or tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
            print(f"  fold {fi}: too small or single-class, skipped")
            continue
        pg = pca_fit(G[tr][stack([e for i, e in enumerate(ev) if tr[i]],
                                 "valid_g")], a.pca_dim)
        pl = pca_fit(L[tr][stack([e for i, e in enumerate(ev) if tr[i]],
                                 "valid_l")], a.pca_dim)
        Pg = torch.from_numpy(proj(pg, G)).float()
        Pl = torch.from_numpy(proj(pl, L)).float()
        for P in (Pg, Pl):
            s = P[torch.from_numpy(tr)].reshape(-1, P.shape[-1]).std(0).clamp(min=1e-6)
            P /= s
        X, M = build_input(Pg, Pl, VG, VL)
        model = RelationHead(X.shape[-1], a.hidden, a.dropout)
        if fi == 0:
            print(f"\n  encoder input {X.shape[-1]}, {n_params(model)} "
                  f"parameters against {int(tr.sum())} training events")
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay)
        yt = torch.from_numpy(y).long()
        trt = torch.from_numpy(tr)
        cnt = np.bincount(y[tr].astype(int), minlength=2) + 1
        w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
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
            oof[te] = F.softmax(model(X, M)[torch.from_numpy(te)], -1)[:, 1].numpy()
        print(f"  fold {fi}: {int(tr.sum())} train / {int(te.sum())} test, "
              f"loss {loss.item():.4f}")

    # -------------------------------------------------------- comparison arms
    old = {}
    if a.compare_oof and os.path.exists(a.compare_oof):
        for r in json.load(open(a.compare_oof, encoding="utf-8"))["events"]:
            if r.get("morphology"):
                old[r["event_id"]] = r["morphology"][POINT]
    arms = defaultdict(dict)
    if a.decisions and os.path.exists(a.decisions):
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for c in OLD_ARMS:
                    try:
                        arms[r["event_id"]][c] = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        pass

    def table(sel, title):
        if len(sel) < 8:
            print(f"\n  {title}: {len(sel)} events, too few to score")
            return
        yy = np.array([e["_y"] for e in sel], float)
        if len(set(yy.tolist())) < 2:
            print(f"\n  {title}: only one class")
            return
        gg = [e["recording_id"] for e in sel]
        idx = {e["event_id"]: i for i, e in enumerate(ev)}
        cols = [("relation student", [oof[idx[e["event_id"]]] for e in sel])]
        if old:
            have = [e for e in sel if e["event_id"] in old]
            if len(have) == len(sel):
                cols.append(("old P(POINT)", [old[e["event_id"]] for e in sel]))
        for c in OLD_ARMS:
            if all(c in arms.get(e["event_id"], {}) for e in sel):
                cols.append((c, [arms[e["event_id"]][c] for e in sel]))
        print(f"\n  {title}: {len(sel)} events, {int(yy.sum())} positive, "
              f"{len(set(gg))} recordings")
        for name, p in cols:
            p = np.array(p, float)
            if not np.isfinite(p).all():
                print(f"    {name:<30} incomplete, withheld")
                continue
            lo, hi = boot(yy, p, gg, a.n_boot, a.seed)
            print(f"    {name:<30} {_auroc(yy, p):.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\n{'=' * 84}\nA  BOUNDARY vs SAME_INSTANCE, every scorer, same "
          f"events\n{'=' * 84}")
    table(ev, "all relation-labelled")
    print("    The old P(POINT) row is the one that answers 'was the target "
          "wrong'. It has never been asked this question,\n    and if it "
          "reads well here then the earlier weakness was the target rather "
          "than the features.")

    print(f"\n{'=' * 84}\nB  THE TWO POSITIVES, SPLIT\n{'=' * 84}")
    table([e for e in ev if e["_rel"] in ("new_action", "same_instance")],
          "new_action vs same_instance")
    table([e for e in ev
           if e["_rel"] in ("same_action_new_instance", "same_instance")],
          "same_action_new_instance vs same_instance")
    print("\n    The second row is the contrast this project has been failing "
          "at without having a name for it: whether the\n    person LET GO and "
          "re-engaged, not whether the scene changed. Strong first row and "
          "weak second localises the\n    gap to interaction reset -- release, "
          "idle, workspace exit, recontact -- which is a specific feature to "
          "build\n    rather than a bigger encoder.")

    print(f"\n{'=' * 84}\nC  dev AGAINST batch3\n{'=' * 84}")
    table([e for e in ev if "_batch3_" not in e["event_id"]], "dev")
    table([e for e in ev if "_batch3_" in e["event_id"]], "batch3")
    print("\n    LEARNABILITY DIAGNOSTIC, NOT A DEPLOYMENT ESTIMATE. The "
          "relation labels were hand-picked from the UNKNOWN pool and\n    "
          "the rest derived from legacy subtypes; the mixture is not any "
          "population, and a coverage or accuracy read off it\n    would "
          "describe a sample that was constructed rather than drawn.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        json.dump({"config": {k: v for k, v in vars(a).items() if k != "out"},
                   "events": [{"event_id": e["event_id"],
                               "recording_id": e["recording_id"],
                               "relation": e["_rel"], "y": e["_y"],
                               "p_boundary": (float(oof[i])
                                              if np.isfinite(oof[i]) else None)}
                              for i, e in enumerate(ev)]},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
