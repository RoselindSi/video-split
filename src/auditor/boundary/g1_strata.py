"""Where, if anywhere, the transition topology has room to help.

TWO DIAGNOSTICS, THE SECOND BEING THE DECISIVE ONE.

STRATA. The full G1 result -- true graph and shuffled graph within 0.9 points
-- was measured over a population whose median decision has two candidates:
one previously seen node, or NEW. A transition prior has nothing to arbitrate
there, whatever the history looks like. So the same three scorers are read
again inside four strata, defined only from information that exists before
any scorer runs:

    O0  one historical node          essentially no room
    O1  two historical nodes         limited
    O2  three or more                real room
    O3  O2 and visually ambiguous    where a prior should matter most

`NEW` is not a historical node. Ambiguity is the gap between the best and
second-best visual similarity, and O3 is the bottom half of O2 by that gap --
fixed as a rule rather than tuned, and never chosen by looking at which cases
V1 or V2 happened to get right.

PRIOR-ONLY RANKING. The headline +11.2% of V1 over V0 turned out to come from
the prior term shifting the NEW threshold, not from topology: V1 and V2
recovered NEW at exactly the same rate. So the second diagnostic removes both
confounds at once. Only REVISIT cases with at least two historical nodes, no
visual score at all, rank the existing candidates by P(n | n_prev) alone:

    true prior   vs   shuffled prior   vs   uniform

If all three tie, the topology genuinely carries no ranking signal in this
data, and no interface -- retrieval, a graph-native model, another VLM --
can recover what is not there. If true beats shuffled here while V1 tied V2
in the full probe, the information exists and the additive fusion is what
fails to use it. Those two findings point at completely different next steps.

TIES ARE BROKEN AT RANDOM, IN EXPECTATION. An earlier version sorted ties by
node name, and node names run in order of first appearance, so a name-ordered
sort is really the prior "prefer the node seen earliest". On 109 of these 116
decisions the candidate list is already in that order, and the free prior was
worth +8.2 points to the uniform condition and +25.2 to the frequency one --
which is where the reported "frequency 54.0% vs uniform 47.6%" came from.
Scoring `1/|tied top tier|` is what a scorer with no preference actually gets,
and under it the frequency prior is worse than uniform, not better.

THIS ROUND IS EXPLORATORY. The overall G1 result has already been seen, so
even though the strata use no outcome information, they cannot be presented
later as a pre-registered primary analysis. A positive O3 is a regime to go
and collect, freeze and confirm on unseen video -- not a result.
"""
from __future__ import annotations

import argparse
import collections
import math
import random

from src.auditor.boundary.g1_probe import RELEASE, NODEMAP, SEED, task_rows
from src.auditor.boundary.g1b_support import expected_top1, expected_rank


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--o3_frac", type=float, default=0.5)
    a = ap.parse_args()

    import numpy as np
    z = np.load(a.emb, allow_pickle=True)
    X = z["vecs"].astype("float64")
    X = X - X.mean(0, keepdims=True)            # 见 g1_probe 里的说明
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    V = {i: v for i, v in zip(list(z["ids"]), X)}
    rows = [r for r in task_rows(a.release, a.nodemap) if r["seg_id"] in V]

    def proto(r, n):
        vs = [V[s] for s in r["prefix_ids"]
              if s in V and r["node_of"].get(s) == n]
        if not vs:
            return None
        m = np.mean(vs, axis=0)
        return m / max(np.linalg.norm(m), 1e-6)

    # 视觉相似度，以及最高与次高之差（歧义度）——在任何打分器跑之前算好
    for r in rows:
        sims = {}
        for n in r["candidates"]:
            mu = proto(r, n)
            if mu is not None:
                sims[n] = float(np.dot(V[r["seg_id"]], mu))
        r["sims"] = sims
        s = sorted(sims.values(), reverse=True)
        r["gap"] = (s[0] - s[1]) if len(s) >= 2 else float("inf")

    o2 = [r for r in rows if r["n_seen"] >= 3]
    thr = (sorted(r["gap"] for r in o2)[int(a.o3_frac * len(o2)) - 1]
           if o2 else 0.0)
    for r in rows:
        r["stratum"] = ("O0" if r["n_seen"] == 1 else
                        "O1" if r["n_seen"] == 2 else
                        ("O3" if r["gap"] <= thr else "O2"))

    def edge_logp(r, n, edges):
        if r["prev_node"] is None:
            return 0.0
        tot = sum(v for (s, _), v in edges.items() if s == r["prev_node"])
        c = edges.get((r["prev_node"], n), 0)
        return math.log((c + 1.0) / (tot + len(r["candidates"]) + 1.0))

    rng = random.Random(SEED)
    shuf = {}
    for r in rows:
        ks = list(r["edges"])
        vs = [r["edges"][k] for k in ks]
        rng.shuffle(vs)
        shuf[r["seg_id"]] = dict(zip(ks, vs))

    def predict(r, lam, tau, mode):
        best, bs = "NEW", tau
        for n, s in r["sims"].items():
            if mode:
                s += lam * edge_logp(r, n, shuf[r["seg_id"]] if mode == "shuf"
                                     else r["edges"])
            if s > bs:
                best, bs = n, s
        return best

    vids = sorted({r["video_id"] for r in rows})
    folds = collections.defaultdict(list)
    for i, v in enumerate(vids):
        folds[i % a.folds].append(v)
    GRID_T = [x / 100 for x in range(50, 100, 2)]
    GRID_L = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

    def run(mode):
        pr = {}
        for f in range(a.folds):
            te = set(folds[f])
            tr = [r for r in rows if r["video_id"] not in te]
            best, bp = None, -1
            for lam in (GRID_L if mode else [0.0]):
                for tau in GRID_T:
                    n = sum(1 for r in tr if predict(r, lam, tau, mode) == r["gold"])
                    if n > bp:
                        bp, best = n, (lam, tau)
            for r in rows:
                if r["video_id"] in te:
                    pr[r["seg_id"]] = predict(r, best[0], best[1], mode)
        return pr

    P = {k: run(m) for k, m in (("V0", None), ("V1", True), ("V2", "shuf"))}

    print("=== 诊断一：按 Graph 机会分层 ===")
    print("O3 = O2 中视觉最高与次高相似度之差最小的 %.0f%%（阈值 %.4f）\n"
          % (100 * a.o3_frac, thr))
    print("%-4s %5s %-14s %8s %8s %8s %9s %10s"
          % ("层", "n", "已见节点", "V0", "V1", "V2", "V1−V0", "V1−V2"))
    units_all = collections.defaultdict(list)
    for r in rows:
        units_all[r["video_id"]].append(r)
    for st, desc in (("O0", "1 个"), ("O1", "2 个"), ("O2", ">=3 个"),
                     ("O3", ">=3 且歧义")):
        sub = [r for r in rows if r["stratum"] == st]
        if not sub:
            continue
        acc = {k: sum(1 for r in sub if P[k][r["seg_id"]] == r["gold"]) / len(sub)
               for k in P}
        # 配对 bootstrap，仍按录像聚类
        u = collections.defaultdict(list)
        for r in sub:
            u[r["video_id"]].append(r)
        ks = sorted(u)
        ci = {}
        if len(ks) >= 2:
            rr = random.Random(SEED)
            for hi, lo in (("V1", "V0"), ("V1", "V2")):
                ds = []
                for _ in range(2000):
                    g = [x for k in [rr.choice(ks) for _ in ks] for x in u[k]]
                    ds.append((sum(1 for r in g if P[hi][r["seg_id"]] == r["gold"])
                               - sum(1 for r in g if P[lo][r["seg_id"]] == r["gold"]))
                              / len(g))
                ds.sort()
                ci[(hi, lo)] = (ds[50], ds[1949])
        def fmt(hi, lo):
            d = acc[hi] - acc[lo]
            c = ci.get((hi, lo))
            if not c:
                return "%+5.1f%%" % (100 * d)
            star = "*" if not (c[0] < 0 < c[1]) else " "
            return "%+5.1f%%%s" % (100 * d, star)
        print("%-4s %5d %-14s %7.1f%% %7.1f%% %7.1f%% %9s %10s"
              % (st, len(sub), desc, 100 * acc["V0"], 100 * acc["V1"],
                 100 * acc["V2"], fmt("V1", "V0"), fmt("V1", "V2")))
    print("  （* 表示 95% CI 不跨 0，按录像聚类）")

    print("\n=== 诊断二：只用先验排序，不看视觉、不判 NEW ===")
    sub = [r for r in rows if r["gold"] != "NEW" and r["n_seen"] >= 2]
    print("只取 REVISIT 且已见节点 >=2 的决策：n=%d / %d 条视频"
          % (len(sub), len({r["video_id"] for r in sub})))

    def rank_stats(mode):
        ranks, nll, top1s = [], [], []
        for r in sub:
            ed = (shuf[r["seg_id"]] if mode == "shuf" else r["edges"]) if mode and mode not in ("freq", "supp") else None
            sc = {}
            for n in r["candidates"]:
                if mode == "freq":
                    # 只数这个节点在前缀里出现过多少次，完全不看上一个节点
                    c = sum(1 for v in r["node_of"].values() if v == n)
                    sc[n] = math.log((c + 1.0) / (len(r["node_of"])
                                                  + len(r["candidates"]) + 1.0))
                elif mode == "supp":
                    # 只问「这条转移出现过没有」，出现过的一律同权。
                    # 与真图之差 = 频次分布的贡献；与均匀之差 = 可达性的贡献。
                    seen_edge = any(k == (r["prev_node"], n) for k in r["edges"]) \
                        if r["prev_node"] is not None else False
                    sc[n] = 0.0 if seen_edge else -3.0
                else:
                    sc[n] = edge_logp(r, n, ed) if mode else 0.0
            # 平局随机打破的期望值，不是按节点名排序 —— 见文件头
            ranks.append(expected_rank(r["candidates"], sc, r["gold_node"]))
            top1s.append(expected_top1(r["candidates"], sc, r["gold_node"]))
            tot = sum(math.exp(sc[n]) for n in r["candidates"])
            nll.append(-math.log(max(math.exp(sc[r["gold_node"]]) / tot, 1e-12)))
        top1 = sum(top1s) / len(top1s)
        mrr = sum(1 / x for x in ranks) / len(ranks)
        return top1, mrr, sum(ranks) / len(ranks), sum(nll) / len(nll)

    print("%-12s %8s %8s %9s %8s" % ("先验", "top-1", "MRR", "平均排名", "NLL"))
    res = {}
    for name, mode in (("真图（带频次）", True), ("支撑集（等权）", "supp"),
                       ("打乱图", "shuf"), ("频率 P(n)", "freq"), ("均匀", None)):
        t, m, ar, nl = rank_stats(mode)
        res[name] = t
        print("%-12s %7.1f%% %8.3f %9.2f %8.3f" % (name, 100 * t, m, ar, nl))

    u = collections.defaultdict(list)
    for r in sub:
        u[r["video_id"]].append(r)
    ks = sorted(u)
    rr = random.Random(SEED)

    def top1_of(rs, mode):
        # `hit` is not `c`: an earlier version used one name for the hit
        # counter and for the per-node occurrence count, so the inner loop
        # overwrote the counter and every bootstrap interval for the
        # frequency prior came back impossible (+35.7% point estimate inside
        # a [+76.5%, +95.7%] interval).
        hit = 0.0
        for r in rs:
            ed = (shuf[r["seg_id"]] if mode == "shuf" else r["edges"]) if mode and mode not in ("freq", "supp") else None
            sc = {}
            for n in r["candidates"]:
                if mode == "freq":
                    cnt = sum(1 for v in r["node_of"].values() if v == n)
                    sc[n] = math.log((cnt + 1.0) / (len(r["node_of"])
                                                    + len(r["candidates"]) + 1.0))
                elif mode == "supp":
                    seen_edge = any(k == (r["prev_node"], n) for k in r["edges"]) \
                        if r["prev_node"] is not None else False
                    sc[n] = 0.0 if seen_edge else -3.0
                else:
                    sc[n] = edge_logp(r, n, ed) if mode else 0.0
            hit += expected_top1(r["candidates"], sc, r["gold_node"])
        return hit / len(rs)

    print("\n  按录像聚类的配对 bootstrap")
    for hi, lo, hm, lm in (("真图（带频次）", "支撑集（等权）", True, "supp"),
                           ("支撑集（等权）", "频率 P(n)", "supp", "freq"),
                           ("真图（带频次）", "打乱图", True, "shuf"),
                           ("频率 P(n)", "均匀", "freq", None)):
        ds = []
        for _ in range(2000):
            g = [x for k in [rr.choice(ks) for _ in ks] for x in u[k]]
            ds.append(top1_of(g, hm) - top1_of(g, lm))
        ds.sort()
        sig = "  ← CI 不跨 0" if not (ds[50] < 0 < ds[1949]) else ""
        print("    %-14s %+6.1f%%   95%% CI [%+.1f%%, %+.1f%%]%s"
              % ("%s − %s" % (hi, lo), 100 * (res[hi] - res[lo]),
                 100 * ds[50], 100 * ds[1949], sig))


if __name__ == "__main__":
    main()
