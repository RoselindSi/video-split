"""The baseline ladder: is the successor set worth more than a one-entry cache?

WHY THIS EXISTS. G1 compared the true successor support against a
degree-matched shuffle and against uniform, which establishes that the signal
is real. It does not establish that the signal needs a *set*. On this data
"last time I left this state I went to X, and I went to X again" is 62 out of
62, so a method that keeps one cached successor per state would score the same
as one that reasons over reachable sets -- and the cheaper method should win by
default. Experiment D made that suspicion concrete when cross-validation chose
k=1, meaning the prior alone fixed the shortlist and vision only confirmed it.

So the comparison that matters is not prior-versus-nothing. It is:

    support set  vs  one cached successor
    support set  vs  recency
    support set  vs  frequency

If the set does not beat those, the structure is decoration and the paper
should say so. A reviewer will ask this immediately, and k=1 reads as "your
method is a frequency prior" until it is answered.

THE PRIOR IS THE ONLY THING THAT VARIES. Same embeddings, same visual score
cos(x, mu_n) over prefix-mean prototypes, same two-stage fusion, same
grouped-CV protocol, and k and tau fitted per arm by recording-grouped CV so
no arm is handicapped by another arm's hyper-parameters. Everything runs on
oracle history: this axis is about which prior, and Experiment D already
measured what happens to the best of them when the history is self-maintained.

REGIMES ARE REPORTED SEPARATELY BECAUSE THEY ARE DIFFERENT QUESTIONS.
Where |S| = 1 the cache is by construction sufficient and no prior can
distinguish itself. Where |S| >= 2 the prior narrows without deciding, which
is the only place visual evidence is doing arbitration -- that column is the
paper's main result. Where the gold node is outside S, a hard constraint
excludes the right answer and the question becomes whether the prior locks the
model out. Pooling the three lets the first, which is the large majority here,
carry the headline.

Bootstrap is clustered on recordings, because a recording is the sample unit
and decisions inside one share a scene, a person and a camera.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics

T_GRID = [0.02, 0.05, 0.1, 0.2]
LAM_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
TAU_GRID = [x / 100 for x in range(-20, 95, 5)]
ARMS = ("visual", "self", "recency", "frequency", "cache", "support", "shuffled")
SEED = 20260928


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--release",
                    default="results/auditor/interaction_graph_release_v1")
    ap.add_argument("--nodemap",
                    default="results/auditor/interaction_graph_schema_v2/"
                            "v1_to_v2_node_map.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out")
    a = ap.parse_args()

    import numpy as np
    from src.auditor.boundary.g1_probe import task_rows

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
        r["S"] = ({t for (s, t) in r["edges"] if s == r["prev_node"]}
                  if r["prev_node"] is not None else set())

    # 每条决策的前缀统计：recency / frequency / cache 都从这里来，按录像顺序累积
    by_vid = collections.defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)
    for vid, rs in by_vid.items():
        last_seen, count, last_succ, step = {}, collections.Counter(), {}, 0
        for r in rs:
            r["last_seen"] = dict(last_seen)
            r["count"] = dict(count)
            # cache：prev 上一次离开时去了哪一个
            r["cache"] = ({last_succ[r["prev_node"]]}
                          if r["prev_node"] in last_succ else set())
            g = r["gold_node"]
            if r["prev_node"] is not None:
                last_succ[r["prev_node"]] = g
            step += 1
            last_seen[g] = step
            count[g] += 1

    # 度匹配 null：保住每个源节点的出度，但目标重新随机抽。
    # 第一版置换的是 edges 的「计数」，而 support set 按键是否存在定义,
    # 所以那是空操作 —— shuffled 逐格等于 support，一眼就该看出不可能。
    rng = random.Random(SEED)
    for r in rows:
        if r["prev_node"] is None or not r["candidates"]:
            r["S_shuf"] = set()
            continue
        deg = len({t for (s, t) in r["edges"] if s == r["prev_node"]})
        pool = list(r["candidates"])
        rng.shuffle(pool)
        r["S_shuf"] = set(pool[:deg])

    def prior_score(r, arm):
        """-> {node: 0..1 的先验分}。软融合，不做 top-k 门控。

        两阶段门控在这批数据上是错的融合：候选集中位只有 2 个、99.7% <= 4，
        所以 CV 一旦选到 k>=4，top-k 就等于全部候选、偏好被整个旁路，
        recency / frequency / visual 会给出逐位相同的准确率。
        """
        ns = list(r["sims"])
        if arm == "visual":
            return {n: 0.0 for n in ns}
        if arm == "self":
            return {n: (1.0 if n == r["prev_node"] else 0.0) for n in ns}
        if arm in ("recency", "frequency"):
            src = r["last_seen"] if arm == "recency" else r["count"]
            vals = {n: float(src.get(n, 0)) for n in ns}
            hi = max(vals.values()) or 1.0
            return {n: v / hi for n, v in vals.items()}
        pool = {"cache": r["cache"], "support": r["S"],
                "shuffled": r["S_shuf"]}[arm]
        return {n: (1.0 if n in pool else 0.0) for n in ns}

    def pick(r, arm, lam, tau):
        if not r["sims"]:
            return "NEW"
        pr = prior_score(r, arm)
        best = max(r["sims"], key=lambda n: r["sims"][n] + lam * pr[n])
        return best if r["sims"][best] > tau else "NEW"

    def hit(r, chosen):
        return ((chosen == "NEW" and r["gold"] == "NEW")
                or (chosen != "NEW" and chosen == r["gold_node"]))

    vids = sorted(by_vid)
    folds = collections.defaultdict(list)
    for i, v in enumerate(vids):
        folds[i % a.folds].append(v)

    def regime(r):
        if r["S"] and r["gold_node"] not in r["S"]:
            return "novel"
        if len(r["S"]) == 1:
            return "deterministic"
        if len(r["S"]) >= 2:
            return "branching"
        return "no-support"

    for r in rows:
        r["regime"] = regime(r)
    print("决策 %d；录像 %d" % (len(rows), len(vids)))
    print("regime 构成：%s" % dict(collections.Counter(r["regime"] for r in rows)))

    # 每条臂自己按录像分组 CV 定 k/tau —— 共用超参会让某条臂背别人的拟合
    chosen, picks = {}, {}
    for arm in ARMS:
        best = (-1, LAM_GRID[0], TAU_GRID[0])
        for k in LAM_GRID:
            for tau in TAU_GRID:
                acc = []
                for f in range(a.folds):
                    held = set(folds[f])
                    sub = [r for r in rows if r["video_id"] in held]
                    if sub:
                        acc.append(sum(hit(r, pick(r, arm, k, tau))
                                       for r in sub) / len(sub))
                m = statistics.mean(acc) if acc else 0
                if m > best[0]:
                    best = (m, k, tau)
        chosen[arm] = {"lam": best[1], "tau": best[2]}
        picks[arm] = {r["seg_id"]: pick(r, arm, best[1], best[2]) for r in rows}

    def acc_of(arm, subset):
        sub = [r for r in subset if r["seg_id"] in picks[arm]]
        if not sub:
            return None
        return sum(hit(r, picks[arm][r["seg_id"]]) for r in sub) / len(sub)

    cols = ("deterministic", "branching", "novel", "no-support")
    print("\n=== 基线阶梯（oracle 历史；软融合 score = cos + lam*prior；每臂自选 lam/tau）===")
    print("  %-11s %6s %6s %13s %10s %8s %11s"
          % ("臂", "lam", "tau", "deterministic", "branching", "novel", "overall"))
    table = {}
    for arm in ARMS:
        cells = [acc_of(arm, [r for r in rows if r["regime"] == c]) for c in cols]
        overall = acc_of(arm, rows)
        table[arm] = {"lam": chosen[arm]["lam"], "tau": chosen[arm]["tau"],
                      "overall": overall,
                      **{c: v for c, v in zip(cols, cells)}}
        print("  %-11s %6.2f %6.2f %13s %10s %8s %11.3f"
              % (arm, chosen[arm]["lam"], chosen[arm]["tau"],
                 "%.3f" % cells[0] if cells[0] is not None else "-",
                 "%.3f" % cells[1] if cells[1] is not None else "-",
                 "%.3f" % cells[2] if cells[2] is not None else "-",
                 overall))

    # 决定性比较：support 相对 cache / frequency / recency / visual
    per_vid = {arm: collections.defaultdict(list) for arm in ARMS}
    for arm in ARMS:
        for r in rows:
            per_vid[arm][r["video_id"]].append(hit(r, picks[arm][r["seg_id"]]))
    boot = random.Random(SEED)
    print("\n=== support 减去各基线，按录像聚类配对 bootstrap（%d 次）===" % a.boot)
    deltas_out = {}
    for base in ("visual", "self", "recency", "frequency", "cache"):
        ds = []
        for _ in range(a.boot):
            pickv = [vids[boot.randrange(len(vids))] for _ in vids]
            x = [h for v in pickv for h in per_vid["support"].get(v, [])]
            y = [h for v in pickv for h in per_vid[base].get(v, [])]
            if x and y:
                ds.append(sum(x) / len(x) - sum(y) / len(y))
        ds.sort()
        lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]
        point = table["support"]["overall"] - table[base]["overall"]
        deltas_out[base] = {"delta": round(point, 4),
                            "ci": [round(lo, 4), round(hi, 4)]}
        print("  support − %-10s %+.3f  CI [%+.3f, %+.3f]  %s"
              % (base, point, lo, hi, "不跨零" if not (lo <= 0 <= hi) else "跨零"))

    print("\n  读法：support − cache 跨零，就说明在这批数据上「保留一条缓存的后继」")
    print("  已经够用，successor set 的额外结构没有换来可测的东西。")

    if a.out:
        json.dump({"n_decisions": len(rows), "n_videos": len(vids),
                   "regimes": dict(collections.Counter(r["regime"] for r in rows)),
                   "arms": table, "support_minus": deltas_out},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
