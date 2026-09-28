"""The Graph-Opportunity Set: decisions where a transition graph could matter.

WHY THIS AND NOT MORE PROBING. G1b settled that the legal-successor set is
what carries the signal -- degree-matched shuffling removes all of it
(`Rs - U = +2.2%`, interval across zero) while the true set is worth `+36.8%`.
But the size distribution says how it does that: of 116 decisions, 79 have a
`n_prev` with exactly one recorded successor, and in 98.7% of those that one
successor is the answer. The rule being rewarded is "go where you went last
time from here". Where the graph has an actual choice to arbitrate,
`|N+| >= 2`, there are eleven cases and the prior is at chance on them.

So the measurement is not wrong, it is starved. This module defines -- from
information available before any answer is read -- the subpopulation in which
a graph prior and a visual likelihood can disagree, so that collection can be
aimed at it instead of at more of the same.

FOUR ADMISSION CRITERIA, ALL OUTCOME-BLIND:

    C1  at least 3 candidate nodes already exist
    C2  visually ambiguous: s1 - s2 < delta
    C3  0 < |N+(n_prev)| < |V_<t|, so the support set actually excludes
        something and is not vacuous
    C4  the set keeps its NEW cases. A pool of nothing but revisits could be
        scored by a system that never answers NEW, and a graph that only ever
        narrows would look good for the wrong reason.

DELTA IS FIXED AS A QUANTILE OF THE GAP DISTRIBUTION over every decision with
3+ candidates, computed before any condition is scored, and it is a property
of the embedding corpus rather than of who got what right. Choosing it by
looking at where the prior wins would manufacture the regime it claims to
find.

THREE TYPES, ASSIGNED WITHOUT THE GOLD LABEL:

    A  visually decisive (gap >= delta). Negative control: a graph-guided
       tracker must not lose these. Reported as a cost, not a benefit.
    B  visually ambiguous and the support set discriminates. This is where a
       graph should pay.
    C  the visual argmax lies outside N+(n_prev). Vision and graph name
       different states; one of them is wrong and the case is diagnostic
       whichever way it resolves.

WHAT THIS IS FOR. It reframes the question from "does the graph beat vision"
to graph-constrained visual state tracking: vision proposes, the reachable set
constrains, and the only honest place to measure that is where the constraint
is non-trivial and the proposal is uncertain. Type A is in the set to keep the
constraint from being free.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from src.auditor.boundary.g1_probe import RELEASE, NODEMAP, task_rows
from src.auditor.boundary.g1b_support import edge_list, support_of, expected_top1

MIN_CAND = 3
DELTA_Q = 0.5          # δ = 3+ 候选决策上 gap 的中位数，先验地定、不看对错


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--min_cand", type=int, default=MIN_CAND)
    ap.add_argument("--delta_q", type=float, default=DELTA_Q)
    ap.add_argument("--out")
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
                     for n in r["candidates"] if (mu := proto(r, n)) is not None}
        s = sorted(r["sims"].values(), reverse=True)
        r["gap"] = (s[0] - s[1]) if len(s) >= 2 else float("inf")
        r["vis_top"] = (max(r["sims"], key=lambda n: r["sims"][n])
                        if r["sims"] else None)
        r["S"] = (support_of(edge_list(r["edges"]), r["prev_node"])
                  if r["prev_node"] is not None else set())

    # ---- δ：先冻结，再分类。只用 gap 的分布，不碰任何 gold ----
    pool = [r for r in rows if len(r["candidates"]) >= a.min_cand
            and r["gap"] < float("inf")]
    if not pool:
        print("没有满足 C1 的决策")
        return
    gaps = sorted(r["gap"] for r in pool)
    delta = gaps[max(0, int(a.delta_q * len(gaps)) - 1)]
    print("总决策 %d 个；C1（候选 >=%d）后剩 %d 个"
          % (len(rows), a.min_cand, len(pool)))
    print("δ = 3+ 候选决策上 gap 的第 %.0f 分位 = %.4f（先冻结，不看对错）\n"
          % (100 * a.delta_q, delta))

    def kind(r):
        if r["gap"] >= delta:
            return "A"                          # 视觉已经说清楚了
        if r["vis_top"] is not None and r["S"] and r["vis_top"] not in r["S"]:
            return "C"                          # 视觉与图指向不同
        return "B"

    for r in pool:
        r["discrim"] = 0 < len(r["S"]) < len(r["candidates"])
        r["kind"] = kind(r)

    # C3 只对 B/C 强制；A 是负对照，本来就不要求支撑集有区分力
    opp = [r for r in pool if r["kind"] == "A" or r["discrim"]]

    print("=== 机会集构成 ===")
    print("%-4s %-26s %5s %7s %9s %9s"
          % ("类", "定义", "n", "录像", "NEW 占比", "|N+| 中位"))
    for k, desc in (("A", "视觉决定性（负对照）"),
                    ("B", "视觉模糊 + 图有区分力"),
                    ("C", "视觉与图冲突")):
        g = [r for r in opp if r["kind"] == k]
        if not g:
            print("%-4s %-26s %5d" % (k, desc, 0))
            continue
        ns = sorted(len(r["S"]) for r in g)
        print("%-4s %-26s %5d %7d %8.0f%% %9d"
              % (k, desc, len(g), len({r["video_id"] for r in g}),
                 100 * sum(1 for r in g if r["gold"] == "NEW") / len(g),
                 ns[len(ns) // 2]))
    print("%-4s %-26s %5d %7d %8.0f%%"
          % ("合计", "", len(opp), len({r["video_id"] for r in opp}),
             100 * sum(1 for r in opp if r["gold"] == "NEW") / max(1, len(opp))))

    # ---- 各类上两个打分器分别值多少。这一步才用 gold ----
    def vis_top1(g):
        return sum(1 for r in g
                   if (r["vis_top"] if r["sims"] else None) == r["gold_node"]) / len(g)

    def graph_top1(g):
        return sum(expected_top1(r["candidates"],
                                 {n: (0.0 if n in r["S"] else -3.0)
                                  for n in r["candidates"]}, r["gold_node"])
                   for r in g) / len(g)

    print("\n=== 各类上，纯视觉 vs 纯可达集（都不判 NEW，只在候选里排序）===")
    print("%-4s %5s %10s %12s %10s" % ("类", "n", "视觉 top-1", "可达集 top-1", "gold∈N+"))
    for k in ("A", "B", "C"):
        g = [r for r in opp if r["kind"] == k]
        if not g:
            continue
        print("%-4s %5d %9.1f%% %11.1f%% %9.1f%%"
              % (k, len(g), 100 * vis_top1(g), 100 * graph_top1(g),
                 100 * sum(1 for r in g if r["gold_node"] in r["S"]) / len(g)))

    # ---- 要收多少才够。按当前产出率外推 ----
    n_vid = len({r["video_id"] for r in rows})
    print("\n=== 采集目标 ===")
    print("当前 %d 条录像产出 B 型 %d 个、C 型 %d 个"
          % (n_vid, sum(1 for r in opp if r["kind"] == "B"),
             sum(1 for r in opp if r["kind"] == "C")))
    for k in ("B", "C"):
        c = sum(1 for r in opp if r["kind"] == k)
        rate = c / n_vid
        need = 60
        print("  %s 型产出率 %.2f 个/录像 → 要凑到 %d 个需约 %d 条同类录像"
              % (k, rate, need, int(need / rate) if rate else -1))
    print("\n注意：|N+(prev)|>=2 的决策全库只有 11 个。产出率低不是抽样问题，")
    print("      是这批录像的转移图近乎链式——每个节点几乎只有一个观察到的后继。")
    print("      要让 B/C 型变多，得挑本身会在同几个状态间来回切换的录像。")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"criteria": {"min_cand": a.min_cand, "delta_q": a.delta_q,
                                "delta": delta,
                                "note": "δ 与分型均在读取 gold 之前确定"},
                   "items": [{"seg_id": r["seg_id"], "video_id": r["video_id"],
                              "kind": r["kind"], "gap": r["gap"],
                              "n_cand": len(r["candidates"]),
                              "n_support": len(r["S"]),
                              "prev_node": r["prev_node"],
                              "vis_top": r["vis_top"],
                              "gold": r["gold"], "gold_node": r["gold_node"]}
                             for r in sorted(opp, key=lambda x: (x["kind"], x["seg_id"]))]},
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
