"""G1: does the interaction graph carry predictive information a visual likelihood lacks?

NO PROMPT, NO LANGUAGE MODEL, NO TEXT AT ALL. E2 ended up measuring context
length and dictionary formatting, so this probe removes every one of those
surfaces. What is left is a frozen embedding per segment and an arithmetic
prior read off the oracle graph.

THE TASK IS RE-IDENTIFICATION, NOT THE SAME/DIFFERENT RELATION E2 ASKED
ABOUT. Whether two adjacent segments are the same state is, in principle,
settled by the footage on either side of the boundary; a history of `A -> B
-> C` is not needed to see that rinsing differs from replacing. The question
a transition graph is actually shaped for is: which of the states already
seen is this one, or is it new. That is where long-range history has
something local evidence does not.

    for each occurrence t with at least one prior occurrence:
        candidates = {nodes seen so far} + {NEW}
        gold       = its node, or NEW on that node's first appearance

THREE SCORERS, THE THIRD BEING THE ONE THAT MATTERS:

    V0  visual only          cos(x_t, mu_n), NEW when the best falls below tau
    V1  visual + true prior  + lambda * log P(n | n_{t-1})
    V2  visual + shuffled    the same lambda over a permuted edge set

V1 - V0 without V1 - V2 would reward the prior for merely existing: adding
any extra term shifts the NEW/REVISIT balance and can move accuracy on its
own. V2 holds the shape of the term fixed and destroys only its content, so
V1 - V2 is the part attributable to the real topology.

TAU AND LAMBDA ARE FITTED BY RECORDING-GROUPED CROSS-VALIDATION, never on the
evaluation fold, because both control the same NEW/REVISIT tradeoff the
metric reads.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

from src.auditor.boundary.interaction_graph_structure_census import load_special

RELEASE = "results/auditor/interaction_graph_release_v1"
NODEMAP = "results/auditor/interaction_graph_schema_v2/v1_to_v2_node_map.json"
SEED = 20260928


def task_rows(release, nodemap):
    """-> per-occurrence decisions with the prefix state available at that time."""
    special = load_special(release, nodemap)
    out = []
    for bd in sorted(glob.glob(os.path.join(release, "frozen_annotations", "*"))):
        batch = os.path.basename(bd)
        for p in sorted(glob.glob(os.path.join(bd, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            vid = doc["video_id"]
            segs = doc["segments"]
            seen, prev_node, hist_edges = [], None, collections.Counter()
            for i, s in enumerate(segs):
                sid = "%s#%02d" % (vid, i + 1)
                if (batch, vid, s["event_id"]) in special:
                    prev_node = None          # 不跨特殊状态连边
                    continue
                if seen:                       # 至少有一次历史才构成决策
                    out.append({
                        "seg_id": sid, "video_id": vid, "batch": batch,
                        "gold": s["event_id"] if s["event_id"] in seen else "NEW",
                        "gold_node": s["event_id"],
                        "candidates": list(seen),
                        "prev_node": prev_node,
                        "edges": dict(hist_edges),
                        "n_seen": len(seen),
                        "prefix_ids": ["%s#%02d" % (vid, j + 1) for j in range(i)],
                        "node_of": {},
                    })
                    # 记下前缀里每个 occurrence 属于哪个节点，供原型计算
                    for j in range(i):
                        k = (batch, vid, segs[j]["event_id"])
                        if k not in special:
                            out[-1]["node_of"]["%s#%02d" % (vid, j + 1)] = \
                                segs[j]["event_id"]
                if prev_node is not None:
                    hist_edges[(prev_node, s["event_id"])] += 1
                if s["event_id"] not in seen:
                    seen.append(s["event_id"])
                prev_node = s["event_id"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    import numpy as np
    z = np.load(a.emb, allow_pickle=True)
    ids = list(z["ids"])
    X = z["vecs"].astype("float64")
    # CENTRED BEFORE NORMALISING. Mean-pooling the tower's patch states leaves
    # a very large component every clip shares -- element std ~190, while the
    # gap between two unrelated clips' means is std ~11 -- so raw cosine
    # similarity came out 0.999976 for every pair and the visual arm was
    # measuring nothing. Removing the corpus mean is what makes the remaining
    # direction the part that distinguishes clips.
    X = X - X.mean(0, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    V = {i: v for i, v in zip(ids, X)}

    rows = [r for r in task_rows(a.release, a.nodemap) if r["seg_id"] in V]
    vids = sorted({r["video_id"] for r in rows})
    print("G1 决策 %d 个 / %d 条视频" % (len(rows), len(vids)))
    g = collections.Counter(("NEW" if r["gold"] == "NEW" else "REVISIT") for r in rows)
    print("  gold  %s   多数类 %.1f%%"
          % (dict(g), 100 * max(g.values()) / len(rows)))
    print("  候选数 中位 %d" % sorted(r["n_seen"] for r in rows)[len(rows) // 2])

    def proto(r, n):
        vs = [V[s] for s in r["prefix_ids"]
              if s in V and r["node_of"].get(s) == n]
        if not vs:
            return None
        m = np.mean(vs, axis=0)
        return m / max(np.linalg.norm(m), 1e-6)

    def edge_logp(r, n, edges):
        if r["prev_node"] is None:
            return 0.0
        tot = sum(v for (s, _), v in edges.items() if s == r["prev_node"])
        c = edges.get((r["prev_node"], n), 0)
        # 加一平滑，候选集含 NEW，故分母 +len(candidates)+1
        return float(np.log((c + 1.0) / (tot + len(r["candidates"]) + 1.0)))

    rng = random.Random(SEED)
    shuffled = {}
    for r in rows:
        ks = list(r["edges"])
        vs = [r["edges"][k] for k in ks]
        rng.shuffle(vs)
        shuffled[r["seg_id"]] = dict(zip(ks, vs))

    def predict(r, lam, tau, edges=None):
        best, bs = "NEW", tau
        for n in r["candidates"]:
            mu = proto(r, n)
            if mu is None:
                continue
            s = float(np.dot(V[r["seg_id"]], mu))
            if lam and edges is not None:
                s += lam * edge_logp(r, n, edges)
            if s > bs:
                best, bs = n, s
        return best

    folds = collections.defaultdict(list)
    for i, v in enumerate(vids):
        folds[i % a.folds].append(v)

    GRID_T = [x / 100 for x in range(50, 100, 2)]
    GRID_L = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

    def run(use_edges):
        preds = {}
        for f in range(a.folds):
            te = set(folds[f])
            tr = [r for r in rows if r["video_id"] not in te]
            best, bp = None, -1
            for lam in (GRID_L if use_edges else [0.0]):
                for tau in GRID_T:
                    ed = (lambda r: (shuffled[r["seg_id"]] if use_edges == "shuf"
                                     else r["edges"])) if use_edges else None
                    acc = sum(1 for r in tr
                              if predict(r, lam, tau, ed(r) if ed else None) == r["gold"])
                    if acc > bp:
                        bp, best = acc, (lam, tau)
            lam, tau = best
            for r in rows:
                if r["video_id"] in te:
                    ed = (shuffled[r["seg_id"]] if use_edges == "shuf"
                          else r["edges"]) if use_edges else None
                    preds[r["seg_id"]] = predict(r, lam, tau, ed)
        return preds

    out = {}
    for name, ue in (("V0 纯视觉", None), ("V1 视觉+真图", True), ("V2 视觉+打乱图", "shuf")):
        pr = run(ue)
        out[name] = pr
        acc = sum(1 for r in rows if pr[r["seg_id"]] == r["gold"]) / len(rows)
        rev = [r for r in rows if r["gold"] != "NEW"]
        new = [r for r in rows if r["gold"] == "NEW"]
        print("%-14s 总准确率 %5.1f%%   REVISIT recall %5.1f%% (n=%d)   NEW recall %5.1f%% (n=%d)"
              % (name, 100 * acc,
                 100 * sum(1 for r in rev if pr[r["seg_id"]] == r["gold"]) / max(1, len(rev)), len(rev),
                 100 * sum(1 for r in new if pr[r["seg_id"]] == "NEW") / max(1, len(new)), len(new)))

    units = collections.defaultdict(list)
    for r in rows:
        units[r["video_id"]].append(r)
    keys = sorted(units)
    rng2 = random.Random(SEED)
    print("\n按录像聚类的配对 bootstrap")
    for hi, lo in (("V1 视觉+真图", "V0 纯视觉"), ("V1 视觉+真图", "V2 视觉+打乱图")):
        ds = []
        for _ in range(2000):
            rr = [x for k in [rng2.choice(keys) for _ in keys] for x in units[k]]
            d = (sum(1 for r in rr if out[hi][r["seg_id"]] == r["gold"])
                 - sum(1 for r in rr if out[lo][r["seg_id"]] == r["gold"])) / len(rr)
            ds.append(d)
        ds.sort()
        obs = (sum(1 for r in rows if out[hi][r["seg_id"]] == r["gold"])
               - sum(1 for r in rows if out[lo][r["seg_id"]] == r["gold"])) / len(rows)
        sig = "  ← CI 不跨 0" if not (ds[50] < 0 < ds[1949]) else ""
        print("  %-30s %+6.1f%%   95%% CI [%+.1f%%, %+.1f%%]%s"
              % ("%s − %s" % (hi, lo), 100 * obs, 100 * ds[50], 100 * ds[1949], sig))
    print("\nV1 − V2 才是真实拓扑的贡献；V1 − V0 里混着「多一项就改变 NEW/REVISIT 平衡」。")


if __name__ == "__main__":
    main()
