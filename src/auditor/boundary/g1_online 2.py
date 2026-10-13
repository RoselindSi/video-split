"""Experiment D: how much of the successor-support gain survives self-maintained state.

THE QUESTION. Every result so far assumed the system knows which earlier
segment was which: `prior()` reads the true `prev_node`, the edge counts come
from true transitions, and the node prototypes are means over segments grouped
by their true labels. A deployed tracker has none of that. It must run

    n_hat_{t-1}  ->  S(n_hat_{t-1})  ->  n_hat_t

and an early identity error then follows it forward, because the successor set
it consults belongs to the node it wrongly believes it is at. If the oracle
gain is +10 and the self-maintained gain is +1, the method is not a method.

WHY THIS DOES NOT NEED BRANCHING DATA, which is the other open gap. Error
propagation is cleanest to measure exactly where the data already is: with
`|S| = 1` the correct successor is unique, so a wrong answer cannot be excused
as an ambiguous choice, and drift is unambiguous. Deterministic data is the
right place for this, not an obstacle to it.

THERE ARE THREE ORACLE DEPENDENCIES, NOT ONE, and swapping only the first
would flatter the result:

    oracle   prev_node, edge counts and prototypes all from true labels
             -- reproduces the existing G1/G1b numbers
    prev     only prev_node predicted; edges and prototypes still true
    full     prev_node, edges and prototypes all from the model's own history,
             including minting a new node whenever it answers NEW

Reporting all three localises the loss instead of just measuring it.

THE DECISION RULE IS HELD FIXED ACROSS MODES. Two-stage fusion -- the prior
keeps the top-k candidates, vision picks among them -- with k and tau fitted
once by recording-grouped CV under `oracle` and then reused unchanged. If each
mode refitted its own hyper-parameters, part of the degradation would be
absorbed by the fit and the comparison would no longer be about the history.

SCORING UNDER `full` IS GENEROUS, DELIBERATELY AND VISIBLY. The model's node
ids are its own, so each predicted node is mapped to a gold node by the
majority gold identity of the segments it assigned there. Majority mapping
forgives a model that split one state into two and then used both consistently,
so `full` numbers are an upper bound on identity quality.

Pair agreement is reported as two numbers, not one. Pooling them lets the
"different state" pairs, which are the large majority, carry the figure: the
first version of this scored `prev` at 0.899 against `oracle` at 0.888 and
looked like a stricter metric while being a looser one. `same_state_linked` --
two occurrences of one state given one id -- is the half that identity is
actually about.

A VISUAL-ONLY ARM IS INCLUDED BECAUSE THE CRITERION IS A GAIN, NOT A LEVEL.
"oracle +10 but predicted +1" is a statement about the margin over vision
alone, so without that floor the absolute accuracies cannot answer it.

`prior_override` counts decisions where unconstrained vision had the right
answer and the prior's shortlist removed it. That is the cost side of the soft
prior, and it is the quantity that says whether a hard constraint would lock
the tracker out of a correct answer.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

K_GRID = [1, 2, 3]
LAM_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
TAU_GRID = [x / 100 for x in range(-20, 95, 5)]


def majority_map(pred_node_of, gold_of):
    """-> {pred_node: gold_node}，按该簇里 gold 身份的多数。"""
    buckets = collections.defaultdict(collections.Counter)
    for sid, pnode in pred_node_of.items():
        g = gold_of.get(sid)
        if g is not None:
            buckets[pnode][g] += 1
    return {p: c.most_common(1)[0][0] for p, c in buckets.items()}


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
    ap.add_argument("--fusion", choices=("twostage", "soft"), default="twostage",
                    help="twostage 是 D 原本的 top-k 门控；soft 是 cos + lam*prior，"
                         "和 g1_ladder 同一族 —— 想把两个实验的数字串成一条阶梯"
                         "就必须用同一族，否则 k=1 的门控是 prior 最强的形态，"
                         "和阶梯里 lam=0.10 不是同一件事")
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
    if not rows:
        raise SystemExit("没有既在 release 又有嵌入的决策行")

    # gold 身份表：供 full 模式的多数映射用
    gold_of = {}
    for r in rows:
        gold_of.update(r["node_of"])
        gold_of[r["seg_id"]] = r["gold_node"]

    by_vid = collections.defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)
    vids = sorted(by_vid)
    print("决策 %d；录像 %d 条；有嵌入的段 %d" % (len(rows), len(vids), len(V)))

    def proto(ids, node_of, node):
        vs = [V[s] for s in ids if s in V and node_of.get(s) == node]
        if not vs:
            return None
        m = np.mean(vs, axis=0)
        return m / max(np.linalg.norm(m), 1e-9)

    def sims_for(seg_id, ids, node_of, cands):
        out = {}
        for n in cands:
            mu = proto(ids, node_of, n)
            if mu is not None:
                out[n] = float(np.dot(V[seg_id], mu))
        return out

    def support(edges, prev):
        if prev is None:
            return set()
        return {t for (s, t), c in edges.items() if s == prev and c > 0}

    def decide(sims, S, k, tau):
        """k 在 twostage 下是 top-k，在 soft 下当 lambda 用。"""
        if not sims:
            return "NEW", None
        vis_best = max(sims, key=lambda n: sims[n])
        if a.fusion == "soft":
            best = max(sims, key=lambda n: sims[n] + k * (1.0 if n in S else 0.0))
        else:
            ranked = sorted(sims, key=lambda n: (n not in S, -sims[n]))
            best = max(ranked[:max(1, int(k))], key=lambda n: sims[n])
        return (best if sims[best] > tau else "NEW"), vis_best

    def run(mode, k, tau, only=None):
        """-> 逐决策结果列表。mode ∈ oracle / prev / full。"""
        out = []
        for vid in vids:
            if only is not None and vid not in only:
                continue
            pred_prev, pred_edges = None, collections.Counter()
            pred_node_of, pred_seen, minted = {}, [], 0
            for r in by_vid[vid]:
                if mode == "full":
                    node_of, ids, cands = pred_node_of, list(pred_node_of), list(pred_seen)
                    edges, prev = pred_edges, pred_prev
                else:
                    node_of, ids, cands = (r["node_of"], r["prefix_ids"],
                                           r["candidates"])
                    edges = collections.Counter(r["edges"])
                    prev = pred_prev if mode == "prev" else r["prev_node"]
                sims = sims_for(r["seg_id"], ids, node_of, cands)
                S = set() if mode == "visual" else support(edges, prev)
                pick, vis_best = decide(sims, S,
                                        len(sims) if mode == "visual" else k,
                                        tau)
                out.append({"video_id": vid, "seg_id": r["seg_id"], "pick": pick,
                            "gold": r["gold"], "gold_node": r["gold_node"],
                            "vis_best": vis_best, "S": sorted(S),
                            "n_sims": len(sims), "mode": mode})
                # 维护模型自己的历史
                if pick == "NEW":
                    minted += 1
                    node = "P%d" % minted if mode == "full" else r["gold_node"]
                    if mode == "full" and node not in pred_seen:
                        pred_seen.append(node)
                else:
                    node = pick
                if mode == "full":
                    if node not in pred_seen:
                        pred_seen.append(node)
                    pred_node_of[r["seg_id"]] = node
                if pred_prev is not None:
                    pred_edges[(pred_prev, node)] += 1
                pred_prev = node
        return out

    def score(res, mode):
        if mode == "full":
            pn = {r["seg_id"]: (r["pick"] if r["pick"] != "NEW" else None)
                  for r in res}
            mp = majority_map({k: v for k, v in pn.items() if v}, gold_of)
        n_id = n_new = 0
        for r in res:
            if mode == "full":
                mapped = mp.get(r["pick"]) if r["pick"] != "NEW" else "NEW"
            else:
                mapped = r["pick"]
            hit = (mapped == "NEW" and r["gold"] == "NEW") or \
                  (mapped != "NEW" and mapped == r["gold_node"])
            r["hit"] = bool(hit)
            n_id += hit
            n_new += ((r["pick"] == "NEW") == (r["gold"] == "NEW"))
        # 配对一致性，正负分开报。合在一起算会被「不同节点」那一大堆负例
        # 抬高 —— 第一版那么算的时候 prev 的 0.899 高过了 oracle 的 0.888，
        # 看着像更严其实更松。同 gold 的那一栏才是「同一个状态有没有被认成
        # 同一个」，也就是 identity 真正要的东西。
        same_n = same_ok = diff_n = diff_ok = 0
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                if res[i]["video_id"] != res[j]["video_id"]:
                    continue
                same_gold = res[i]["gold_node"] == res[j]["gold_node"]
                same_pred = (res[i]["pick"] == res[j]["pick"]
                             and res[i]["pick"] != "NEW")
                if same_gold:
                    same_n += 1
                    same_ok += same_pred
                else:
                    diff_n += 1
                    diff_ok += not same_pred
        # drift：错误的连续长度与恢复率
        runs, cur, recov, errs = [], 0, 0, 0
        prev_bad = False
        for r in res:
            if not r["hit"]:
                cur += 1
                errs += 1
                prev_bad = True
            else:
                if prev_bad:
                    recov += 1
                if cur:
                    runs.append(cur)
                cur, prev_bad = 0, False
        if cur:
            runs.append(cur)
        override = sum(1 for r in res
                       if r["vis_best"] == r["gold_node"]
                       and r["pick"] != r["gold_node"]
                       and r["S"] and r["gold_node"] not in r["S"])
        return {"n": len(res),
                "identity_top1": n_id / max(1, len(res)),
                "new_vs_revisit": n_new / max(1, len(res)),
                "same_state_linked": same_ok / max(1, same_n),
                "diff_state_separated": diff_ok / max(1, diff_n),
                "error_run_median": statistics.median(runs) if runs else 0,
                "error_run_max": max(runs) if runs else 0,
                "recovery_rate": recov / max(1, errs),
                "prior_override": override}

    # k/tau 只在 oracle 下按录像分组 CV 定一次，其余模式沿用 —— 每个模式各自
    # 调参会把退化吸收进拟合里，比较就不再是关于历史来源的了。
    folds = collections.defaultdict(list)
    for i, v in enumerate(vids):
        folds[i % a.folds].append(v)
    grid = LAM_GRID if a.fusion == "soft" else K_GRID
    best, bk, bt = -1, grid[0], TAU_GRID[0]
    for k in grid:
        for tau in TAU_GRID:
            acc = []
            for f in range(a.folds):
                held = set(folds[f])
                res = run("oracle", k, tau, only=held)
                if res:
                    acc.append(score(res, "oracle")["identity_top1"])
            m = statistics.mean(acc) if acc else 0
            if m > best:
                best, bk, bt = m, k, tau
    print("融合 %s；oracle 下 CV 选出 %s=%.2f tau=%.2f（identity top-1 %.3f）"
          % (a.fusion, "lam" if a.fusion == "soft" else "k", bk, bt, best))

    print("\n=== Experiment D：oracle → 自维护状态 ===")
    print("  %-8s %10s %10s %10s %10s %8s %8s"
          % ("模式", "identity", "NEW判别", "同态连上", "异态分开",
             "最长错串", "恢复率"))
    report = {}
    for mode in ("visual", "oracle", "prev", "full"):
        res = run(mode, bk, bt)
        s = score(res, mode)
        report[mode] = s
        print("  %-8s %10.3f %10.3f %10.3f %10.3f %8d %8.3f"
              % (mode, s["identity_top1"], s["new_vs_revisit"],
                 s["same_state_linked"], s["diff_state_separated"],
                 s["error_run_max"], s["recovery_rate"]))

    vis, o, p, f = (report[m]["identity_top1"]
                    for m in ("visual", "oracle", "prev", "full"))
    print("\n  相对视觉-only 的增益（这才是判据的口径）：")
    print("    oracle %+.3f / 只换前驱 %+.3f / 自维护 %+.3f"
          % (o - vis, p - vis, f - vis))
    if o - vis > 0:
        print("    自维护保留了 oracle 增益的 %.0f%%"
              % (100 * (f - vis) / (o - vis)))
    print("  绝对值 oracle − full = %+.3f" % (f - o))
    print("  只换前驱身份的那一步损失 %+.3f；再换掉 edges 和原型又损失 %+.3f"
          % (p - o, f - p))
    print("  prior 把视觉本来对的答案排除掉：oracle %d / prev %d / full %d 次"
          % tuple(report[m]["prior_override"] for m in ("oracle", "prev", "full")))
    print("\n  「同态连上」才是 identity 要的那一半；「异态分开」会被大量"
          "负例抬高，不能当更严的口径读。")

    # 按录像聚类的配对 bootstrap。增益只有几个百分点而样本单位是录像不是
    # 决策，所以不给 CI 就等于没给结论 —— 可能连「有增益可丢」都不成立。
    import random as _rnd
    print("\n=== 按录像聚类配对 bootstrap（4000 次，录像为单位）===")
    per_vid = {}
    for mode in ("visual", "oracle", "prev", "full"):
        res = run(mode, bk, bt)
        d = collections.defaultdict(list)
        # 整批算一次来填 hit。曾经为了方便对每行单独调 score()，而 full 模式
        # 下那会用单行去建多数映射，映射对那一行必然正确，把命中率虚抬到让
        # CI 与点估计符号相反。
        score(res, mode)
        for r in res:
            d[r["video_id"]].append(r["hit"])
        per_vid[mode] = d
    rng = _rnd.Random(20260928)
    for mode in ("oracle", "prev", "full"):
        deltas = []
        for _ in range(4000):
            pick = [vids[rng.randrange(len(vids))] for _ in vids]
            a_hit = [h for v in pick for h in per_vid[mode].get(v, [])]
            b_hit = [h for v in pick for h in per_vid["visual"].get(v, [])]
            if a_hit and b_hit:
                deltas.append(sum(a_hit) / len(a_hit) - sum(b_hit) / len(b_hit))
        deltas.sort()
        lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
        crosses = lo <= 0 <= hi
        report[mode]["gain_ci"] = [round(lo, 4), round(hi, 4)]
        print("  %-8s − visual  %+.3f  CI [%+.3f, %+.3f]  %s"
              % (mode, report[mode]["identity_top1"]
                 - report["visual"]["identity_top1"], lo, hi,
                 "跨零" if crosses else "不跨零"))

    if a.out:
        json.dump({"k": bk, "tau": bt, "n_decisions": len(rows),
                   "n_videos": len(vids), "modes": report},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
