"""How much branching structure exists, and whether it exists in time to be tested.

THE DISTINCTION THIS MODULE IS FOR. `|N+(prev)| >= 2` is a statement about the
prefix, not about the recording. A recording where `A` eventually leads to both
`B` and `C` still yields no testable branching decision if the second branch
only appears after the last visit to `A`. Calling that recording "branching"
would be the same mistake as letting the current edge into the prior: the
structure has to be there *before* the decision, or it is not available to
anything making that decision.

So every node is counted twice: out-degree in the full recording graph, and
out-degree in the prefix at each decision. The gap between them is the part
that exists but arrives too late, and it says whether the fix is longer
recordings or different ones.

THE THREE TYPES ARE THE USER'S, taken literally:

    G-simple  |N+(A)| = 1         deterministic transition memory
    G-branch  |N+(A)| >= 2 and the support still excludes some candidate,
              with at least 3 historical nodes -- the graph constrains but
              does not answer
    G-novel   gold not in N+(A), with N+(A) non-empty -- the transition is
              new, and a hard reachability mask would be wrong here

G-branch and G-novel overlap by construction and the cross-tabulation is
reported rather than resolved, because a case that is both is the sharpest
one: the graph offers two options and the answer is a third.

PER-RECORDING TOPOLOGY IS REPORTED so that collecting more is a measurement
rather than a guess about task semantics. What predicts yield is revisits to a
node that then goes somewhere else -- the `A -> B -> A -> C` shape -- and that
is countable in an existing annotation without watching anything.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

from src.auditor.boundary.g1_probe import RELEASE, NODEMAP, task_rows
from src.auditor.boundary.g1b_support import edge_list, support_of
from src.auditor.boundary.interaction_graph_structure_census import load_special

MIN_HIST = 3


def classify(r, min_hist=MIN_HIST):
    """-> set of types. Uses only the prefix and the gold node, never the future."""
    S, cand = r["S"], r["candidates"]
    out = set()
    if len(S) == 1:
        out.add("G-simple")
    if len(S) >= 2 and len(S) < len(cand) and r["n_seen"] >= min_hist:
        out.add("G-branch")
    if S and r["gold_node"] not in S:
        out.add("G-novel")
    return out


def full_graph(doc, batch, special):
    """Out-degree over the whole recording -- the ceiling, not what is testable."""
    E, prev = collections.Counter(), None
    for s in doc["segments"]:
        if (batch, doc["video_id"], s["event_id"]) in special:
            prev = None
            continue
        if prev is not None:
            E[(prev, s["event_id"])] += 1
        prev = s["event_id"]
    od = collections.defaultdict(set)
    for (a, b) in E:
        od[a].add(b)
    return od


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--min_hist", type=int, default=MIN_HIST)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = task_rows(a.release, a.nodemap)
    for r in rows:
        r["S"] = (support_of(edge_list(r["edges"]), r["prev_node"])
                  if r["prev_node"] is not None else set())
        r["types"] = classify(r, a.min_hist)

    vids = {r["video_id"] for r in rows}
    print("全量标注上的决策 %d 个 / %d 条录像（不要求有 embedding）\n"
          % (len(rows), len(vids)))

    # ---- 三类产量 ----
    print("=== 三类决策的产量（严格前缀）===")
    print("%-10s %6s %7s %-38s" % ("类型", "n", "录像", "定义"))
    for t, desc in (("G-simple", "|N+(A)|=1，确定性转移记忆"),
                    ("G-branch", "|N+(A)|>=2 且未穷尽候选，历史节点>=%d" % a.min_hist),
                    ("G-novel", "gold 不在 N+(A)，且 N+(A) 非空")):
        g = [r for r in rows if t in r["types"]]
        print("%-10s %6d %7d %-38s"
              % (t, len(g), len({r["video_id"] for r in g}), desc))
    nb = [r for r in rows if not r["types"]]
    print("%-10s %6d %7d %-38s"
          % ("（无）", len(nb), len({r["video_id"] for r in nb}),
             "N+(A) 为空，或支撑集穷尽了候选"))

    print("\n  G-branch × G-novel 交叉")
    for b in (True, False):
        for n in (True, False):
            g = [r for r in rows
                 if (("G-branch" in r["types"]) == b)
                 and (("G-novel" in r["types"]) == n)]
            if g:
                print("    branch=%-5s novel=%-5s  %4d" % (b, n, len(g)))

    # ---- branching 是不存在，还是来得太晚 ----
    special = load_special(a.release, a.nodemap)
    ceil_nodes, ceil_vids = 0, set()
    for bd in sorted(glob.glob(os.path.join(a.release, "frozen_annotations", "*"))):
        batch = os.path.basename(bd)
        for p in sorted(glob.glob(os.path.join(bd, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            od = full_graph(doc, batch, special)
            k = sum(1 for n, t in od.items() if len(t) >= 2)
            ceil_nodes += k
            if k:
                ceil_vids.add(doc["video_id"])
    tested = {(r["video_id"], r["prev_node"]) for r in rows if len(r["S"]) >= 2}
    print("\n=== branching 是不存在，还是来得太晚 ===")
    print("  整条录像上出度 >=2 的节点：%d 个，分布在 %d / %d 条录像"
          % (ceil_nodes, len(ceil_vids), len(vids)))
    print("  在某个决策点**前缀里**就已经出度 >=2 的 (录像,节点)：%d 个"
          % len(tested))
    print("  → 差额 %d 个是「分支存在，但第二条在测试点之后才出现」"
          % (ceil_nodes - len(tested)))

    # ---- 每条录像的拓扑，用来挑下一批该拍什么 ----
    per = collections.defaultdict(lambda: {"dec": 0, "branch": 0, "novel": 0,
                                           "simple": 0, "nodes": set(),
                                           "revisit": 0})
    for r in rows:
        d = per[r["video_id"]]
        d["dec"] += 1
        d["nodes"].update(r["candidates"])
        d["revisit"] += r["gold"] != "NEW"
        for t, k in (("G-branch", "branch"), ("G-novel", "novel"),
                     ("G-simple", "simple")):
            if t in r["types"]:
                d[k] += 1
    rich = sorted(per.items(), key=lambda kv: -(kv[1]["branch"] + kv[1]["novel"]))
    print("\n=== 产出最高的 10 条录像（下一批的选片judgement 依据）===")
    print("%-24s %5s %6s %7s %7s %8s %8s"
          % ("录像", "决策", "节点", "revisit", "branch", "novel", "simple"))
    for v, d in rich[:10]:
        print("%-24s %5d %6d %7d %7d %8d %8d"
              % (v, d["dec"], len(d["nodes"]), d["revisit"],
                 d["branch"], d["novel"], d["simple"]))
    zero = sum(1 for _, d in per.items() if d["branch"] + d["novel"] == 0)
    print("  branch+novel 为 0 的录像：%d / %d" % (zero, len(per)))

    # 一个不看内容就能算的选片判据：revisit 次数
    print("\n=== 能否只用 revisit 数预测产量（不看视频内容）===")
    print("%-14s %7s %10s %10s" % ("revisit 分箱", "录像数", "branch/录像", "novel/录像"))
    for lo, hi, lab in ((0, 1, "0-1"), (2, 3, "2-3"), (4, 6, "4-6"),
                        (7, 99, ">=7")):
        g = [d for d in per.values() if lo <= d["revisit"] <= hi]
        if not g:
            continue
        print("%-14s %7d %10.2f %10.2f"
              % (lab, len(g), sum(d["branch"] for d in g) / len(g),
                 sum(d["novel"] for d in g) / len(g)))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump([{"seg_id": r["seg_id"], "video_id": r["video_id"],
                    "types": sorted(r["types"]),
                    "prev_node": r["prev_node"], "gold_node": r["gold_node"],
                    "gold": r["gold"], "support": sorted(r["S"]),
                    "candidates": list(r["candidates"]), "n_seen": r["n_seen"]}
                   for r in rows if r["types"]],
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
