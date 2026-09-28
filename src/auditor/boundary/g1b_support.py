"""G1b: is it the legal-successor set, or only the transition frequencies?

WHAT THE FIRST SHUFFLE ACTUALLY TESTED. G1's `shuf` condition permuted the
counts over a fixed set of edge keys. The keys never moved, so
`N+(n_prev)` -- which successors are reachable at all -- came out of the
shuffle unchanged. Every node that was legal stayed legal; only its weight
changed. `true - shuffled = +3.2%` is therefore a statement about transition
frequencies under a fixed successor set, and says nothing about whether the
successor set itself carries information. That is the claim worth testing,
because a reachability constraint is what a graph would contribute to state
tracking; relative frequencies are a refinement of it.

THE NULL HAS TO PRESERVE DEGREE OR IT TESTS NOTHING. Drawing a random
successor set of random size would also destroy out-degree, and a smaller
support set is easier to rank in by itself. So the primary null is a
configuration model on the prefix multigraph: repeatedly pick two edges
`a->b`, `c->d` and rewrite them as `a->d`, `c->b`. Out-degree of every node is
untouched (a and c keep their slots) and in-degree of every node is untouched
(targets are only permuted among themselves). What the swap destroys is the
pairing -- which target belongs to which source -- and nothing else.

    U   uniform over candidates
    F   node frequency in the prefix, ignoring n_prev entirely
    Rt  true reachable set, uniform inside it
    Rs  degree-matched wrong reachable set, uniform inside it

    primary contrast:  Rt - Rs

A SECOND, SOFTER NULL is reported beside it. The configuration model can only
place a node in a shuffled support if that node is a target somewhere in the
prefix, so a node that has only ever been a source can never be drawn. `Rs2`
instead samples |N+(n_prev)| candidates with probability proportional to
in-degree + 1, which can reach every candidate while still keeping frequent
targets frequent. If the two nulls disagree, the effect depends on which
nodes are reachable in principle, and that is worth knowing.

TOP-1 IS THE EXPECTED VALUE UNDER RANDOM TIE-BREAKING, not the result of
sorting ties by node name. Under a uniform prior every candidate ties, and a
name-ordered sort turns that into a fixed, arbitrary answer that can be lucky
on this data; `1/|tied top tier|` is what a scorer with no preference actually
achieves. The same estimator is used for every condition, so the contrasts are
comparable.

R NULL DRAWS, AND THE BOOTSTRAP RESAMPLES THE DRAW TOO. One shuffle is one
sample from the null; reporting it alone would confuse null variance with
signal. Each bootstrap replicate picks recordings and a draw index, so the
interval carries both.

EXPLORATORY, like the strata round: the G1 result has been seen.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random

from src.auditor.boundary.g1_probe import RELEASE, NODEMAP, SEED, task_rows

BOOT = 2000


def edge_list(edges):
    """{(s,t): count} -> [(s,t)] with multiplicity."""
    return [k for k, c in edges.items() for _ in range(int(c))]


def double_swap(E, rng, rounds=20):
    """Configuration model on a directed multigraph: out- and in-degree exact."""
    E = list(E)
    m = len(E)
    if m < 2:
        return E
    for _ in range(rounds * m):
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        (a, b), (c, d) = E[i], E[j]
        if b == d:
            continue                      # 交换后与原来相同
        E[i], E[j] = (a, d), (c, b)
    return E


def support_of(E, prev):
    return {t for (s, t) in E if s == prev}


def expected_top1(cands, sc, gold):
    """P(top-1) for a scorer that breaks its own ties at random."""
    top = max(sc[n] for n in cands)
    tier = [n for n in cands if sc[n] >= top - 1e-12]
    return (1.0 / len(tier)) if gold in tier else 0.0


def expected_rank(cands, sc, gold):
    better = sum(1 for n in cands if sc[n] > sc[gold] + 1e-12)
    tied = sum(1 for n in cands if abs(sc[n] - sc[gold]) <= 1e-12)
    return better + (tied + 1) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=20, help="每条边的交换次数")
    ap.add_argument("--out", help="把逐决策的诊断写成 json，供机会集构造复用")
    a = ap.parse_args()

    rows = task_rows(a.release, a.nodemap)
    # 只看 REVISIT 且已见 >=2 个节点 —— 先验有东西可仲裁的那部分
    sub = [r for r in rows
           if r["gold"] != "NEW" and r["n_seen"] >= 2 and r["prev_node"] is not None]
    print("G1b：REVISIT 且已见节点 >=2 的决策 n=%d / %d 条视频"
          % (len(sub), len({r["video_id"] for r in sub})))

    rng = random.Random(SEED)

    # ---- 先验（都不看视觉）。R- 两条只区分「在/不在可达集」，集内等权。 ----
    IN, OUT = -3.0, 0.0

    def sc_uniform(r):
        return {n: 0.0 for n in r["candidates"]}

    def sc_freq(r):
        tot = len(r["node_of"]) + len(r["candidates"]) + 1.0
        return {n: math.log((sum(1 for v in r["node_of"].values() if v == n) + 1.0)
                            / tot) for n in r["candidates"]}

    def sc_supp(r, supp):
        return {n: (OUT if n in supp else IN) for n in r["candidates"]}

    for r in sub:
        E = edge_list(r["edges"])
        r["S_true"] = support_of(E, r["prev_node"])
        # 入度（全前缀），用于软 null 的抽样权重
        ind = collections.Counter(t for (_, t) in E)
        r["indeg"] = ind
        k = len(r["S_true"])
        r["S_swap"], r["S_soft"] = [], []
        for d in range(a.draws):
            r["S_swap"].append(support_of(double_swap(E, rng, a.rounds), r["prev_node"]))
            pool = list(r["candidates"])
            w = [ind.get(n, 0) + 1.0 for n in pool]
            pick = set()
            while len(pick) < min(k, len(pool)):
                x = rng.choices(pool, weights=w, k=1)[0]
                pick.add(x)
            r["S_soft"].append(pick)

    # ---- degree-match 自检：只有这些数都对上，Rt−Rs 才是「成员身份」的差 ----
    d_out = [len(r["S_true"]) for r in sub]
    d_sw = [len(s) for r in sub for s in r["S_swap"]]
    tin = collections.Counter()
    sin = collections.Counter()
    for r in sub:
        E = edge_list(r["edges"])
        for (_, t) in E:
            tin[t] += 1
    print("\n  degree-match 自检")
    print("    |N+(prev)| 真 中位 %d 均值 %.2f   ｜ 交换后 中位 %d 均值 %.2f"
          % (sorted(d_out)[len(d_out) // 2], sum(d_out) / len(d_out),
             sorted(d_sw)[len(d_sw) // 2], sum(d_sw) / len(d_sw)))
    same = sum(1 for r in sub for s in r["S_swap"] if s == r["S_true"])
    print("    交换后与真可达集完全相同的抽样：%d / %d (%.0f%%) —— 小图上无法避免"
          % (same, len(d_sw), 100 * same / len(d_sw)))
    gt = sum(1 for r in sub if r["gold_node"] in r["S_true"]) / len(sub)
    gs = sum(1 for r in sub for s in r["S_swap"]
             if r["gold_node"] in s) / len(d_sw)
    gf = sum(1 for r in sub for s in r["S_soft"]
             if r["gold_node"] in s) / len(d_sw)
    print("    gold 落在可达集里：真 %.1f%%  交换 %.1f%%  软抽样 %.1f%%"
          % (100 * gt, 100 * gs, 100 * gf))

    # 可达集是否真的缩小候选空间
    disc = [r for r in sub if 0 < len(r["S_true"]) < len(r["candidates"])]
    print("    0 < |N+| < |V_<t| 的决策：%d / %d (%.0f%%)"
          % (len(disc), len(sub), 100 * len(disc) / len(sub)))

    # ---- 逐决策预先算好命中，bootstrap 只做平均 ----
    def score_row(r, cond, draw=0):
        if cond == "U":
            return sc_uniform(r)
        if cond == "F":
            return sc_freq(r)
        if cond == "Rt":
            return sc_supp(r, r["S_true"])
        if cond == "Rs":
            return sc_supp(r, r["S_swap"][draw])
        if cond == "Rs2":
            return sc_supp(r, r["S_soft"][draw])
        raise ValueError(cond)

    CONDS = ["U", "F", "Rt", "Rs", "Rs2"]
    LABEL = {"U": "U  均匀", "F": "F  节点频率（不看 prev）",
             "Rt": "Rt 真可达集（集内等权）",
             "Rs": "Rs 度匹配打乱（双边交换）",
             "Rs2": "Rs2 度匹配打乱（入度加权抽样）"}
    hit = {c: {} for c in CONDS}
    rk = {c: {} for c in CONDS}
    for r in sub:
        for c in CONDS:
            nd = a.draws if c in ("Rs", "Rs2") else 1
            hs, rs = [], []
            for d in range(nd):
                sc = score_row(r, c, d)
                hs.append(expected_top1(r["candidates"], sc, r["gold_node"]))
                rs.append(expected_rank(r["candidates"], sc, r["gold_node"]))
            hit[c][r["seg_id"]] = hs
            rk[c][r["seg_id"]] = rs

    def agg(rs, c, draw=0):
        d = draw if c in ("Rs", "Rs2") else 0
        return (sum(hit[c][r["seg_id"]][d] for r in rs) / len(rs),
                sum(rk[c][r["seg_id"]][d] for r in rs) / len(rs))

    print("\n=== 只用先验排序（无视觉、无语言模型、不判 NEW）===")
    print("top-1 是随机打破平局下的期望值，不是按节点名排序的结果\n")
    print("%-34s %8s %10s" % ("先验", "top-1", "期望排名"))
    pt = {}
    for c in CONDS:
        nd = a.draws if c in ("Rs", "Rs2") else 1
        t = sum(agg(sub, c, d)[0] for d in range(nd)) / nd
        er = sum(agg(sub, c, d)[1] for d in range(nd)) / nd
        pt[c] = t
        print("%-34s %7.1f%% %10.2f" % (LABEL[c], 100 * t, er))

    units = collections.defaultdict(list)
    for r in sub:
        units[r["video_id"]].append(r)
    keys = sorted(units)
    rr = random.Random(SEED + 1)

    def boot(hi, lo):
        ds = []
        for _ in range(BOOT):
            g = [x for k in [rr.choice(keys) for _ in keys] for x in units[k]]
            d = rr.randrange(a.draws)
            ds.append(agg(g, hi, d)[0] - agg(g, lo, d)[0])
        ds.sort()
        return ds[int(.025 * BOOT)], ds[int(.975 * BOOT)]

    print("\n  按录像聚类的配对 bootstrap（Rs 每个 replicate 重抽一次 null）")
    for hi, lo, why in (("Rt", "Rs", "★ 可达集成员身份本身（度已匹配）"),
                        ("Rt", "Rs2", "同上，软 null"),
                        ("Rt", "F", "可达性 vs 只看节点频率"),
                        ("Rs", "U", "打乱的可达集还剩多少（应接近 0）"),
                        ("F", "U", "节点频率本身")):
        lo_, hi_ = boot(hi, lo)
        sig = "  ← CI 不跨 0" if not (lo_ < 0 < hi_) else ""
        print("    %-10s %+6.1f%%   95%% CI [%+.1f%%, %+.1f%%]%s"
              % ("%s − %s" % (hi, lo), 100 * (pt[hi] - pt[lo]),
                 100 * lo_, 100 * hi_, sig))
        print("        （%s）" % why)

    # ---- 只在可达集真正有区分力的子集上重读一遍 ----
    if disc and len({r["video_id"] for r in disc}) >= 2:
        print("\n  只取 0 < |N+| < |V_<t| 的 %d 个决策" % len(disc))
        units2 = collections.defaultdict(list)
        for r in disc:
            units2[r["video_id"]].append(r)
        k2 = sorted(units2)
        rr2 = random.Random(SEED + 2)
        for c in CONDS:
            nd = a.draws if c in ("Rs", "Rs2") else 1
            t = sum(agg(disc, c, d)[0] for d in range(nd)) / nd
            print("    %-34s %7.1f%%" % (LABEL[c], 100 * t))
        for hi, lo in (("Rt", "Rs"), ("Rt", "F")):
            ds = []
            for _ in range(BOOT):
                g = [x for k in [rr2.choice(k2) for _ in k2] for x in units2[k]]
                d = rr2.randrange(a.draws)
                ds.append(agg(g, hi, d)[0] - agg(g, lo, d)[0])
            ds.sort()
            l, h = ds[int(.025 * BOOT)], ds[int(.975 * BOOT)]
            nd = a.draws if hi in ("Rs", "Rs2") else 1
            th = sum(agg(disc, hi, d)[0] for d in range(nd)) / nd
            nd = a.draws if lo in ("Rs", "Rs2") else 1
            tl = sum(agg(disc, lo, d)[0] for d in range(nd)) / nd
            sig = "  ← CI 不跨 0" if not (l < 0 < h) else ""
            print("    %-10s %+6.1f%%   95%% CI [%+.1f%%, %+.1f%%]%s"
                  % ("%s − %s" % (hi, lo), 100 * (th - tl), 100 * l, 100 * h, sig))

    if a.out:
        json.dump([{"seg_id": r["seg_id"], "video_id": r["video_id"],
                    "n_seen": r["n_seen"], "prev_node": r["prev_node"],
                    "gold_node": r["gold_node"],
                    "n_support": len(r["S_true"]),
                    "n_cand": len(r["candidates"]),
                    "gold_in_support": r["gold_node"] in r["S_true"]}
                   for r in sub], open(a.out, "w"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
