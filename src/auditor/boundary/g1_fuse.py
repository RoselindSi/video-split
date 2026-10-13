"""Was it the graph that had nothing, or the way the graph was added?

WHAT THE FIRST FUSION DID WRONG. `cos(x, mu_n) + lambda * log P(n | n_prev)`
adds a quantity bounded in [-1, 1] to one that ranges over about [-3, 0]. A
single lambda has to both rescale and weight, and the value cross-validation
picks is the one that keeps the visual term from being swamped -- which means
the prior arrives already crushed. That is a fusion failure, and it is
indistinguishable from "the prior is useless" if only the fused number is
read.

It matters here because the prior-only probe is emphatic: ranking existing
candidates by P(n | n_prev) alone reaches 89.7% top-1 against 47.6% for a
uniform prior, while the fused score moved not at all (V1 - V2 = 0.0%). A
signal that strong disappearing on contact with the visual term is a
statement about the addition, not about the signal.

THREE FUSIONS, ALL SCORING THE SAME DECISIONS:

    F_cal   visual turned into a log-likelihood first -- log softmax(sim / T)
            -- so both terms are log-probabilities and lambda only weights.
    F_rank  scale-free: average the two rankings. Nothing can be swamped
            because no magnitudes survive.
    F_2st   two stages: the prior keeps the top-k candidates, vision picks
            among them. The prior cannot be outvoted, only overruled on what
            it already admitted.

Each is run with the true prior and with the shuffled one, and the reported
quantity is always true-minus-shuffled. If every fusion still gives zero, the
topology genuinely does not survive contact with visual evidence on this
data. If one of them recovers the prior-only gap, the earlier zero was an
artefact of adding unlike quantities.

T, LAMBDA, K AND TAU ARE ALL FITTED BY RECORDING-GROUPED CV, never on the
fold being scored -- they all trade NEW against REVISIT, which is what the
metric reads.
"""
from __future__ import annotations

import argparse
import collections
import math
import random

from src.auditor.boundary.g1_probe import RELEASE, NODEMAP, SEED, task_rows


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
    X = z["vecs"].astype("float64")
    X = X - X.mean(0, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    V = {i: v for i, v in zip(list(z["ids"]), X)}
    rows = [r for r in task_rows(a.release, a.nodemap) if r["seg_id"] in V]

    def proto(r, n):
        vs = [V[s] for s in r["prefix_ids"]
              if s in V and r["node_of"].get(s) == n]
        if not vs:
            return None
        m = np.mean(vs, axis=0)
        return m / max(np.linalg.norm(m), 1e-9)

    for r in rows:
        r["sims"] = {n: float(np.dot(V[r["seg_id"]], mu))
                     for n in r["candidates"]
                     if (mu := proto(r, n)) is not None}

    rng = random.Random(SEED)
    shuf = {}
    for r in rows:
        ks = list(r["edges"])
        vs = [r["edges"][k] for k in ks]
        rng.shuffle(vs)
        shuf[r["seg_id"]] = dict(zip(ks, vs))

    def prior(r, mode):
        """-> {node: log P}, normalised over the candidate set."""
        if not mode:
            return {n: -math.log(len(r["sims"]) or 1) for n in r["sims"]}
        ed = shuf[r["seg_id"]] if mode == "shuf" else r["edges"]
        if r["prev_node"] is None:
            return {n: -math.log(len(r["sims"]) or 1) for n in r["sims"]}
        tot = sum(v for (s, _), v in ed.items() if s == r["prev_node"])
        raw = {n: (ed.get((r["prev_node"], n), 0) + 1.0)
               / (tot + len(r["sims"]) + 1.0) for n in r["sims"]}
        z0 = sum(raw.values()) or 1.0
        return {n: math.log(v / z0) for n, v in raw.items()}

    def vis_logp(r, T):
        if not r["sims"]:
            return {}
        m = max(r["sims"].values())
        e = {n: math.exp((s - m) / T) for n, s in r["sims"].items()}
        z0 = sum(e.values())
        return {n: math.log(v / z0) for n, v in e.items()}

    def pick_cal(r, T, lam, tau, mode):
        vp, pp = vis_logp(r, T), prior(r, mode)
        best, bs = "NEW", tau
        for n in r["sims"]:
            s = vp[n] + lam * pp[n]
            if s > bs:
                best, bs = n, s
        return best

    def pick_rank(r, lam, tau, mode):
        if not r["sims"]:
            return "NEW"
        pp = prior(r, mode)
        rv = {n: i for i, n in enumerate(sorted(r["sims"], key=lambda k: -r["sims"][k]))}
        rp = {n: i for i, n in enumerate(sorted(pp, key=lambda k: -pp[k]))}
        best, bs = None, None
        for n in r["sims"]:
            s = -((1 - lam) * rv[n] + lam * rp[n])
            if bs is None or s > bs:
                best, bs = n, s
        # NEW 由视觉最高分是否过阈值决定，和排序融合解耦
        return best if max(r["sims"].values()) > tau else "NEW"

    def pick_2st(r, k, tau, mode):
        if not r["sims"]:
            return "NEW"
        pp = prior(r, mode)
        short = sorted(pp, key=lambda n: -pp[n])[:max(1, k)]
        best = max(short, key=lambda n: r["sims"][n])
        return best if r["sims"][best] > tau else "NEW"

    vids = sorted({r["video_id"] for r in rows})
    folds = collections.defaultdict(list)
    for i, v in enumerate(vids):
        folds[i % a.folds].append(v)

    T_G = [0.02, 0.05, 0.1, 0.2]
    L_G = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    TAU_C = [-6, -4, -3, -2, -1.5, -1.0, -0.7, -0.4, -0.2]
    TAU_S = [x / 100 for x in range(-20, 95, 5)]
    K_G = [1, 2, 3]

    def run(kind, mode):
        pr = {}
        for f in range(a.folds):
            te = set(folds[f])
            tr = [r for r in rows if r["video_id"] not in te]
            best, bp = None, -1
            if kind == "cal":
                grid = [(T, l, t) for T in T_G for l in L_G for t in TAU_C]
                fn = lambda r, g: pick_cal(r, g[0], g[1], g[2], mode)
            elif kind == "rank":
                grid = [(0, l, t) for l in [0, .25, .5, .75, 1.0] for t in TAU_S]
                fn = lambda r, g: pick_rank(r, g[1], g[2], mode)
            else:
                grid = [(0, k, t) for k in K_G for t in TAU_S]
                fn = lambda r, g: pick_2st(r, g[1], g[2], mode)
            for g in grid:
                n = sum(1 for r in tr if fn(r, g) == r["gold"])
                if n > bp:
                    bp, best = n, g
            for r in rows:
                if r["video_id"] in te:
                    pr[r["seg_id"]] = fn(r, best)
        return pr

    units = collections.defaultdict(list)
    for r in rows:
        units[r["video_id"]].append(r)
    keys = sorted(units)

    print("决策 %d 个 / %d 条视频；REVISIT %d，NEW %d\n"
          % (len(rows), len(keys),
             sum(1 for r in rows if r["gold"] != "NEW"),
             sum(1 for r in rows if r["gold"] == "NEW")))
    print("%-24s %8s %8s %8s %11s" % ("融合方式", "真图", "打乱图", "无先验", "真−打乱"))
    for kind, label in (("cal", "校准相加 log-softmax"),
                        ("rank", "排序平均（无量纲）"),
                        ("2st", "两阶段（先验筛选）")):
        P = {m: run(kind, m) for m in (True, "shuf", None)}
        acc = {m: sum(1 for r in rows if P[m][r["seg_id"]] == r["gold"]) / len(rows)
               for m in P}
        rr = random.Random(SEED)
        ds = []
        for _ in range(2000):
            g = [x for k in [rr.choice(keys) for _ in keys] for x in units[k]]
            ds.append((sum(1 for r in g if P[True][r["seg_id"]] == r["gold"])
                       - sum(1 for r in g if P["shuf"][r["seg_id"]] == r["gold"]))
                      / len(g))
        ds.sort()
        sig = "*" if not (ds[50] < 0 < ds[1949]) else " "
        print("%-24s %7.1f%% %7.1f%% %7.1f%% %+7.1f%%%s [%+.1f, %+.1f]"
              % (label, 100 * acc[True], 100 * acc["shuf"], 100 * acc[None],
                 100 * (acc[True] - acc["shuf"]), sig, 100 * ds[50], 100 * ds[1949]))
    print("\n（* = 95% CI 不跨 0，按录像聚类；「真−打乱」才是拓扑的贡献）")
    print("对照：纯先验排序上 真图 89.7% / 打乱 86.5% / 均匀 47.6%")


if __name__ == "__main__":
    main()
