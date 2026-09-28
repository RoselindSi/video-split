"""What the first branching batch actually produced, against what it was selected for.

THE BATCH WAS SELECTED ON A PROXY and this is the first time that proxy meets
a real v2.1 annotation. Windows were ranked by how often the *old* dataset's
free-text labels changed object family, on the argument that verbs drift but
objects do not. Whether that predicted anything is answerable now: the
annotator's own node sequence is the ground truth the proxy was guessing at,
and the two are put side by side rather than only the new numbers reported.

THE YIELD IS THE POINT. Decisions are classified exactly as `branch_mine`
does, from the prefix only:

    G-simple  |N+(A)| = 1        deterministic successor memory
    G-branch  |N+(A)| >= 2, support still excludes a candidate, >=3 nodes seen
    G-novel   the answer is not in N+(A), which a hard mask would forbid

G-branch is what the next experiment is short of -- two decisions in the
entire earlier corpus -- so a batch that yields none is a result about the
domain, not a packaging problem, and needs to be reported as one.

CONTRACT USAGE IS CHECKED IN THE SAME PASS, because E0 established that a
field nobody uses is a hole in the contract rather than a fact about the
video: `uncertain` and `waiting` went unused across eleven trial videos and
that is what forced the v2.1 rewrite. Two things are worth reading here. The
confidence and special-state fields either got used or they did not. And
`evidence_timestamp` earlier than the first segment carrying that node is
consistent, while a timestamp later than it means the node was assigned before
its evidence existed -- which is the definition of retrospective, and should
have `retrospective_confirmation` set.
"""
from __future__ import annotations

import argparse
import collections
import json

from src.auditor.boundary.branch_mine import classify


def rows_of(segs):
    """严格前缀的决策行，特殊状态断开连边（与 g1_probe 同口径）。"""
    out, seen, prev, edges = [], [], None, collections.Counter()
    for s in segs:
        if s["seg_type"] != "interaction_node" or not s["event_id"]:
            prev = None                      # 不跨特殊状态连边
            continue
        n = s["event_id"]
        if seen:
            S = {t for (a, t) in edges if a == prev} if prev is not None else set()
            out.append({"candidates": list(seen), "prev_node": prev, "S": S,
                        "n_seen": len(seen), "gold_node": n,
                        "gold": n if n in seen else "NEW"})
        if prev is not None:
            edges[(prev, n)] += 1
        if n not in seen:
            seen.append(n)
        prev = n
    return out, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--windows", help="branch_windows.json，用来对照 proxy 的预测")
    a = ap.parse_args()

    doc = json.load(open(a.export, encoding="utf-8"))
    pred = {}
    if a.windows:
        pred = {r["rid"]: r for r in json.load(open(a.windows, encoding="utf-8"))}

    tot = collections.Counter()
    print("=== 逐条：图形状与产出 ===")
    print("%-20s %5s %5s %7s %-22s %7s %7s %7s"
          % ("video", "段", "节点", "自环", "出度分布", "simple", "branch", "novel"))
    per = []
    for v in doc["videos"]:
        segs = v["segments"]
        rows, edges = rows_of(segs)
        for r in rows:
            r["types"] = classify(r)
            for t in r["types"]:
                tot[t] += 1
            if not r["types"]:
                tot["（无）"] += 1
        od = collections.defaultdict(set)
        loops = 0
        for (x, y) in edges:
            if x == y:
                loops += edges[(x, y)]
            od[x].add(y)
        nodes = {s["event_id"] for s in segs
                 if s["seg_type"] == "interaction_node" and s["event_id"]}
        dist = collections.Counter(len(t) for t in od.values())
        c = collections.Counter()
        for r in rows:
            for t in r["types"]:
                c[t] += 1
        per.append({"vid": v["video_id"], "nodes": len(nodes), "dec": len(rows),
                    "branch": c["G-branch"], "novel": c["G-novel"],
                    "simple": c["G-simple"],
                    "maxout": max([len(t) for t in od.values()] or [0])})
        print("%-20s %5d %5d %7d %-22s %7d %7d %7d"
              % (v["video_id"], len(segs), len(nodes), loops,
                 " ".join("出度%d×%d" % (k, n) for k, n in sorted(dist.items())),
                 c["G-simple"], c["G-branch"], c["G-novel"]))

    print("\n=== 合计 ===")
    nd = sum(p["dec"] for p in per)
    print("  决策 %d 个 / %d 条片段" % (nd, len(per)))
    for k in ("G-simple", "G-branch", "G-novel", "（无）"):
        print("    %-10s %4d" % (k, tot[k]))
    print("  任何节点的最大出度：%d（出度>=2 的片段 %d 条）"
          % (max(p["maxout"] for p in per),
             sum(1 for p in per if p["maxout"] >= 2)))

    if pred:
        print("\n=== proxy 预测 vs 真实标注 ===")
        print("  proxy 用旧数据集的自由文本标签数「物件族切换」，这是它第一次被检验")
        print("%-20s %10s %10s %10s" % ("video", "proxy切换", "真实节点数", "真实切换"))
        import statistics
        xs, ys = [], []
        for v in doc["videos"]:
            segs = [s for s in v["segments"]
                    if s["seg_type"] == "interaction_node" and s["event_id"]]
            seq = [s["event_id"] for s in segs]
            sw = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
            p = pred.get(v["video_id"], {})
            xs.append(p.get("switches", 0))
            ys.append(sw)
            print("%-20s %10d %10d %10d"
                  % (v["video_id"], p.get("switches", -1),
                     len({s["event_id"] for s in segs}), sw))
        if len(xs) > 2 and statistics.pstdev(xs) and statistics.pstdev(ys):
            mx, my = statistics.mean(xs), statistics.mean(ys)
            r = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / (len(xs) * statistics.pstdev(xs) * statistics.pstdev(ys)))
            print("  相关系数 r = %+.2f" % r)

    print("\n=== 合同字段用了没有 ===")
    conf, sk, rep, st = (collections.Counter() for _ in range(4))
    late, tsn = 0, 0
    for v in doc["videos"]:
        first = {}
        for s in v["segments"]:
            st[s["seg_type"]] += 1
            conf[s.get("confidence", "")] += 1
            rep[bool(s.get("repeats_within_segment"))] += 1
            if s["seg_type"] == "special_state":
                sk[s.get("special_kind", "")] += 1
            if s.get("event_id"):
                first.setdefault(s["event_id"], float(s["start_s"]))
        for e in v["events"]:
            ts = e.get("evidence_timestamp")
            if ts is None:
                tsn += 1
                continue
            if e["event_id"] in first and float(ts) > first[e["event_id"]] + 1e-6:
                late += 1
                if not e.get("retrospective_confirmation"):
                    tot["late_not_flagged"] += 1
    print("  seg_type      %s" % dict(st))
    print("  confidence    %s" % dict(conf))
    print("  special_kind  %s" % dict(sk))
    print("  段内重复      %s" % dict(rep))
    ev = sum(len(v["events"]) for v in doc["videos"])
    print("  事件 %d 个；evidence_timestamp 晚于该节点首次出现的 %d 个，"
          "其中没有勾 retrospective 的 %d 个"
          % (ev, late, tot["late_not_flagged"]))
    print("  retrospective_confirmation 勾过的：%d"
          % sum(1 for v in doc["videos"] for e in v["events"]
                if e.get("retrospective_confirmation")))
    print("  inferred_goal 非空的：%d"
          % sum(1 for v in doc["videos"] for e in v["events"]
                if (e.get("inferred_goal") or "").strip()))
    print("  任务簇：每条片段的簇数 %s"
          % dict(collections.Counter(len(v["clusters"]) for v in doc["videos"])))


if __name__ == "__main__":
    main()
