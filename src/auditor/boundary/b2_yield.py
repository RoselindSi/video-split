"""Apply B2's frozen Go/No-Go to the coarse export, before any data exists.

WHY THIS IS WRITTEN FIRST. The thresholds in `docs/branching_pilot_b2_spec.md`
were frozen on 2026-09-28 specifically so they could not be adjusted after
seeing the yield, and a threshold that still needs code written around it is
not really frozen -- the code is where the quiet choices live. So this exists
before the recordings do, and the decision it prints is mechanical.

IT REUSES `branch_mine.classify` RATHER THAN RESTATING THE RULE. A G-branch is
not "out-degree >= 2": it is out-degree >= 2 **and** the reachable set smaller
than the candidates seen so far, under a strict prefix, with at least MIN_HIST
states behind it. The second clause is the one that matters and the one that
would be easy to drop here -- if `N+(A)` already contains everything seen, the
graph excludes nothing and the constraint is empty. One definition, imported.

THE SPEC'S THRESHOLDS ARE RANGES (15-20 decisions, 0.5-1 per recording) and
this does not quietly pick an end. It reports against both: GO only when the
strict end is met, NO-GO when the loose end is not, and BORDERLINE between,
which is a result to take to a person and not a number to round.

ONE RECORDING IS ONE SAMPLE. The spec requires the decisions to come from
several independent videos because a single unusually branchy recording is one
observation however many decisions it yields, so the concentration check is
part of the verdict rather than a diagnostic beside it.

CONDITION COVERAGE IS A GATE, NOT A STATISTIC. Criteria two and three exist
because branches invented to populate a graph would prove nothing about
sequential reasoning even if the graph then helped. A G-branch whose
`(from, to)` pair carries no recorded condition cannot be told apart from an
arbitrary route change, so those are counted separately and the verdict needs
some of them to be grounded.

Adjacent identical codes are merged exactly as the labelling page merges them.
The page shows the annotator a live G-branch count and claims to use the
analysis definition; that claim was checked against `branch_mine` and holds,
so a disagreement between the two is a bug in one of them, not a convention.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

# 来自 spec，冻结于 2026-09-28。宽端和严端都保留 —— 取中间值就等于事后改判据。
MIN_BRANCH_LOOSE, MIN_BRANCH_STRICT = 15, 20
PER_REC_LOOSE, PER_REC_STRICT = 0.5, 1.0
MAX_SHARE_FROM_ONE = 0.5          # 一条录像贡献超过一半，样本就只有它


def chain(video):
    """-> 相邻同码合并后的状态序列，和标注页同一个口径。"""
    out = []
    for seg in video.get("segments") or []:
        code = (seg.get("code") or "").strip()
        if not code or code == "?":
            continue
        if not out or out[-1] != code:
            out.append(code)
    return out


def rows_of(seq):
    """-> 严格前缀的决策行，字段和 branch_b1_yield.rows_of 对齐。

    在更新边和 seen **之前**判定，否则这一步的答案会进到自己的前缀里。
    """
    out, seen, edges = [], [], collections.Counter()
    for i, node in enumerate(seq):
        prev = seq[i - 1] if i else None
        if seen:
            S = {t for (a, t) in edges if a == prev} if prev is not None else set()
            out.append({"candidates": list(seen), "prev_node": prev, "S": S,
                        "n_seen": len(seen), "gold_node": node,
                        "gold": node if node in seen else "NEW"})
        if prev is not None:
            edges[(prev, node)] += 1
        if node not in seen:
            seen.append(node)
    return out


def conditions_of(video):
    return {(b.get("from"), b.get("to")): (b.get("condition") or "").strip()
            for b in video.get("branch_conditions") or []}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="append", required=True,
                    help="b2_coarse_export_v1 的 json，可给多个标注者")
    ap.add_argument("--min-hist", dest="min_hist", type=int, default=None,
                    help="默认用 branch_mine.MIN_HIST，别在这里改口径")
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.branch_mine import classify, MIN_HIST
    min_hist = MIN_HIST if a.min_hist is None else a.min_hist

    videos, annotators = {}, collections.Counter()
    for path in a.export:
        doc = json.load(open(path, encoding="utf-8"))
        if doc.get("schema_version") != "b2_coarse_export_v1":
            raise SystemExit("%s 不是 b2_coarse_export_v1：%s"
                             % (path, doc.get("schema_version")))
        who = doc.get("annotator") or path
        annotators[who] += 1
        for v in doc.get("videos") or []:
            if v.get("status") != "final":
                continue
            videos[(who, v["video_id"])] = v

    if not videos:
        raise SystemExit("没有 status=final 的录像；粗标还没交")

    # 2026-10-08 加的诊断，对应把问题收窄成 n_t = f(x_t, S(n_{t-1})) 之后
    # 真正要分的三个 regime。**冻结的 Go/No-Go 一个字没动**，下面这些只报告、
    # 不参与判决 —— 判据是 2026-09-28 冻的，加判据和改判据一样是事后改判。
    regime = collections.Counter()
    reduction = []
    per_rec, grounded, ungrounded, cand_sizes, novels = [], [], [], [], 0
    print("=== 逐条录像 ===")
    print("  %-26s %5s %7s %8s %8s" % ("录像", "状态", "决策", "G-branch", "G-novel"))
    for (who, vid), v in sorted(videos.items()):
        seq = chain(v)
        rows = rows_of(seq)
        cond = conditions_of(v)
        n_branch = n_novel = 0
        for r in rows:
            kinds = classify(r, min_hist=min_hist)
            # 三个 regime，按 |S| 和 gold 是否在 S 里分
            if r["gold_node"] not in r["S"] and r["S"]:
                regime["novel: gold 不在 S 里，hard constraint 会锁死"] += 1
            elif len(r["S"]) == 1:
                regime["cache 就够: |S|=1，方法无空间"] += 1
            elif len(r["S"]) >= 2:
                regime["cache 有歧义: |S|>=2，视觉必须发挥作用"] += 1
                reduction.append((len(r["S"]), len(r["candidates"])))
            else:
                regime["无前缀支持: S 为空"] += 1
            if "G-novel" in kinds:
                n_novel += 1
            if "G-branch" not in kinds:
                continue
            n_branch += 1
            cand_sizes.append(len(r["S"]))
            key = (r["prev_node"], r["gold_node"])
            (grounded if cond.get(key) else ungrounded).append((vid, key))
        novels += n_novel
        per_rec.append({"annotator": who, "video_id": vid,
                        "n_states": len(set(seq)), "n_decisions": len(rows),
                        "g_branch": n_branch, "g_novel": n_novel,
                        "max_outdegree": max(
                            [len({t for (p, t) in
                                  [(seq[i - 1], seq[i]) for i in range(1, len(seq))]
                                  if p == node}) for node in set(seq)] or [0])})
        print("  %-26s %5d %7d %8d %8d"
              % (vid[:26], len(set(seq)), len(rows), n_branch, n_novel))

    n_rec = len(per_rec)
    total = sum(r["g_branch"] for r in per_rec)
    with_branch = sum(1 for r in per_rec if r["max_outdegree"] >= 2)
    rate = total / n_rec
    by_rec = collections.Counter(r["video_id"] for r in per_rec
                                 for _ in range(r["g_branch"]))
    top_share = (max(by_rec.values()) / total) if total else 0.0
    n_contrib = sum(1 for r in per_rec if r["g_branch"] > 0)

    print("\n=== spec 要算的四个量 ===")
    print("  ① 出度 >= 2 的录像          %d / %d" % (with_branch, n_rec))
    print("  ② prefix-valid G-branch      %d（每条 %.2f）" % (total, rate))
    print("  ③ G-novel                    %d" % novels)
    print("  ④ 可达集大小 |N+|            %s"
          % ("中位 %.1f，分布 %s" % (statistics.median(cand_sizes),
                                 dict(collections.Counter(cand_sizes)))
             if cand_sizes else "无（没有 G-branch）"))

    print("\n=== 诊断：决策落在哪个 regime（不参与判决）===")
    n_dec = sum(regime.values())
    for name, n in regime.most_common():
        print("  %-38s %5d  %5.1f%%" % (name, n, 100 * n / max(1, n_dec)))
    if reduction:
        keep = statistics.median(s / c for s, c in reduction)
        print("  |S|>=2 那批里，prior 把候选收到 %.0f%%（中位）—— "
              "越小说明排除得越多，接近 100%% 则约束是空的"
              % (100 * keep))
    print("  >>> 论文主结果的那个 population 是「cache 有歧义」这一行。"
          "若它接近 0，successor support 在这批数据上无从检验。")

    print("\n=== Go / No-Go（判据冻结于 2026-09-28，此处不调整）===")
    checks = [
        ("决策数 >= %d（严）/ %d（宽）" % (MIN_BRANCH_STRICT, MIN_BRANCH_LOOSE),
         total >= MIN_BRANCH_STRICT, total >= MIN_BRANCH_LOOSE,
         "%d 个" % total),
        ("每条 >= %.1f（严）/ %.1f（宽）" % (PER_REC_STRICT, PER_REC_LOOSE),
         rate >= PER_REC_STRICT, rate >= PER_REC_LOOSE, "%.2f" % rate),
        ("来自多条独立录像（单条 <= %.0f%%）" % (100 * MAX_SHARE_FROM_ONE),
         total > 0 and top_share <= MAX_SHARE_FROM_ONE and n_contrib >= 2,
         total > 0 and n_contrib >= 2,
         "%d 条贡献，最大单条占 %.0f%%" % (n_contrib, 100 * top_share)),
        ("至少一部分由可观察条件驱动",
         bool(grounded) and len(grounded) >= max(1, total // 2),
         bool(grounded),
         "记了条件的 %d / 未记的 %d" % (len(grounded), len(ungrounded))),
    ]
    for name, strict, loose, got in checks:
        mark = "通过" if strict else ("宽端通过" if loose else "未过")
        print("  [%-4s] %-36s %s" % (mark, name, got))

    if all(c[1] for c in checks):
        verdict = "GO —— 扩到 50-60 条"
    elif not all(c[2] for c in checks):
        verdict = ("NO-GO —— 按 spec 最要紧的那一行：直接停掉 Graph 作为主方法"
                   "方向，不要变成「再拍 20 条看看」")
    else:
        verdict = ("BORDERLINE —— 宽端过严端不过。这是要带给人的结果，不是可以"
                   "四舍五入的数；spec 刻意留了两端就是为了不在这里私自取中值")
    print("\n  >>> %s" % verdict)

    if all(c[1] for c in checks):
        print("\n  过了 Go 也不要马上做完整 v2.1 标注。先在粗序列上确认可达集"
              "真的排除掉了东西：")
        empty = sum(1 for s in cand_sizes if s >= 2)
        print("    |N+| >= 2 且 < 候选总数的决策 %d 个（classify 已含后一半条件）"
              % empty)

    if a.out:
        json.dump({"min_hist": min_hist, "n_recordings": n_rec,
                   "g_branch_total": total, "per_recording_rate": rate,
                   "recordings_with_outdegree_2": with_branch,
                   "g_novel_total": novels,
                   "reachable_set_sizes": cand_sizes, "regime": dict(regime),
                   "grounded": len(grounded), "ungrounded": len(ungrounded),
                   "top_recording_share": top_share,
                   "verdict": verdict, "per_recording": per_rec},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
