"""Ask whether action-grain branching survives folding the transport states away.

WHY THIS IS THE DECIDING TOOL. Labelling the bike batch at action grain turned
0.00 G-branch per window at the dataset's own keystep grain into 0.50, which
looks like the instrument hypothesis confirmed. It is not, and the reason is
the failure that already disqualified EPIC-KITCHENS for this project: 60.2% of
EPIC's apparent branching sat on `pick-up` and `put-down`, because a node
called "pick up" is followed by whatever comes next and therefore has high
out-degree by construction rather than by decision. Here all five G-branch
decisions have the same predecessor, 拿工具, and every branch condition the
annotator wrote names a tool identity: "this time I picked up the air gun",
"this time the carving knife". Picking up the air gun before inflating is not
a decision point, it is the first moment of inflating.

A STATE IS WHAT IS ACTED ON AND HOW. A state that exists only to obtain the
means for the next state is not a state one can be in and then decide from, so
it folds into the state it serves. The test is not the leading verb: an earlier
version dropped every code beginning with 拿 and so deleted real work --
拿充气枪给轮胎充气 is inflating a tyre, 拿一个新的轮圈充气 is inflating a
replacement -- which made the resulting zero partly an artefact of the regex.
Acquiring a tool folds; acquiring a part does not, because choosing to replace
rather than patch is a decision. Tools are therefore named explicitly, the
whole classification is printed, and it can be disagreed with item by item.

THE REAL CONSTRAINT IS A VISIT COUNT, AND IT IS WORTH MORE THAN THE YIELD.
G-branch needs |S| >= 2, meaning the predecessor was departed from at least
twice before, so some state must be visited at least three times. Across ten
bike windows no real state is visited more than twice, while 拿工具 reaches
three or four. That is why folding sends the yield to zero, and it is a
statement about the task rather than about the labelling: a repair runs
forward, and one returns to the tool pile repeatedly but not to "inspect the
rim". Reporting the visit-count distribution says what a relabelling would
have to produce, which a yield number does not.

Finer grain cannot escape it. Splitting 拿工具 into reach / grip / lift raises
the visit count again and the yield with it, so the metric is one a coder can
inflate without bound by cutting transport motions thinner.

THE FROZEN PATH IS LEFT ALONE. `b2_yield.chain` skips `?` before de-duplicating
adjacent codes, so A ? A collapses to A and an unclear segment silently eats a
recurrence; `cmu_bike11_2` contains exactly that. The fix belongs in a
diagnostic, not in an evaluator whose thresholds were frozen on 2026-09-28, so
`--qmark-breaks` offers the other convention here and both are reported. On
this batch it changes nothing, which is the only reason the conclusion can
ignore it.
"""
from __future__ import annotations

import argparse
import collections
import json

from src.auditor.boundary.b2_yield import rows_of
from src.auditor.boundary.branch_mine import classify, MIN_HIST

# 纯取得类只认工具。取得零件（拿一个新轮圈）不折叠 —— 换胎而不是补胎是个决策。
TOOLS = ("工具", "充气枪", "刻刀", "螺丝刀", "钳子", "扳手", "起子")
ACQ_VERB = ("拿", "拿起", "取出", "放回", "整理")
# 出现这些标记说明名字里还有第二个作用于工件的动作，不是纯取得
WORK_MARK = ("给", "用", "往", "，", ",")


def is_tool_acq(name):
    """-> True iff the state is only getting or returning a tool."""
    n = (name or "").strip()
    if not any(n.startswith(v) for v in ACQ_VERB):
        return False
    if not any(t in n for t in TOOLS):
        return False
    return not any(m in n for m in WORK_MARK)


def seq_of(video, merge_dupe=False, qmark_breaks=False, drop_acq=False):
    """-> state sequence under the chosen conventions.

    `merge_dupe` treats identically named codes as one state, which matters
    because `cmu_bike20_3` gave 拆轮圈 two codes and so invented a transition
    between them. `qmark_breaks` stops an unclear segment from merging a
    revisit into its predecessor. `drop_acq` folds tool transport away.
    """
    name = {c["id"]: (c.get("name") or "").strip() for c in video.get("codes") or []}
    remap = {cid: cid for cid in name}
    if merge_dupe:
        first = {}
        for cid, nm in name.items():
            first.setdefault(nm, cid)
        remap = {cid: first[nm] for cid, nm in name.items()}

    out = []
    for seg in video.get("segments") or []:
        c = (seg.get("code") or "").strip()
        if not c:
            continue
        if c == "?":
            if qmark_breaks and out and out[-1] is not None:
                out.append(None)          # 占位，阻断跨越未知段的相邻合并
            continue
        c = remap.get(c, c)
        if drop_acq and is_tool_acq(name.get(c, "")):
            continue
        if not out or out[-1] != c:
            out.append(c)
    return [c for c in out if c is not None]


def measure(videos, label, **kw):
    seqs = [seq_of(v, **kw) for v in videos]
    n = ge2 = gbr = 0
    hubs = collections.Counter()
    for v, s in zip(videos, seqs):
        nm = {c["id"]: (c.get("name") or "").strip() for c in v.get("codes") or []}
        for r in rows_of(s):
            n += 1
            if len(r["S"]) >= 2:
                ge2 += 1
            if "G-branch" in classify(r, min_hist=MIN_HIST):
                gbr += 1
                hubs[nm.get(r["prev_node"], r["prev_node"])] += 1
    print("  %-40s 决策 %3d  |S|>=2 %2d  G-branch %2d（每窗口 %.2f）"
          % (label, n, ge2, gbr, gbr / max(1, len(seqs))))
    return {"label": label, "n_decisions": n, "n_S_ge_2": ge2, "g_branch": gbr,
            "per_window": gbr / max(1, len(seqs)), "hubs": dict(hubs)}


def visits(videos):
    """-> per recording, the most-visited real state and the most-visited tool state."""
    rows = []
    for v in videos:
        nm = {c["id"]: (c.get("name") or "").strip() for c in v.get("codes") or []}
        real = collections.Counter(seq_of(v, merge_dupe=True, qmark_breaks=True,
                                          drop_acq=True))
        allc = collections.Counter(seq_of(v, merge_dupe=True, qmark_breaks=True))
        acq = collections.Counter({c: k for c, k in allc.items()
                                   if is_tool_acq(nm.get(c, ""))})
        r_top = real.most_common(1)[0] if real else (None, 0)
        a_top = acq.most_common(1)[0] if acq else (None, 0)
        rows.append({"video_id": v["video_id"],
                     "real_state": nm.get(r_top[0], r_top[0]), "real_visits": r_top[1],
                     "acq_state": nm.get(a_top[0], a_top[0]), "acq_visits": a_top[1]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    doc = json.load(open(a.export, encoding="utf-8"))
    vids = [v for v in (doc.get("videos") or []) if v.get("status") == "final"]
    if not vids:
        raise SystemExit("没有 status=final 的录像")
    print("batch %s，status=final 的录像 %d 条"
          % (doc.get("batch_id", "?"), len(vids)))

    print("\n=== 代号分类（折叠 = 纯取得工具）===")
    seen, cls = set(), []
    for v in vids:
        for c in v.get("codes") or []:
            nm = (c.get("name") or "").strip()
            if nm in seen:
                continue
            seen.add(nm)
            cls.append({"name": nm, "fold": is_tool_acq(nm)})
            print("  %-5s %s" % ("折叠" if is_tool_acq(nm) else "保留", nm))

    print("\n=== 逐项修正的效果 ===")
    steps = [measure(vids, "① 按标注原样"),
             measure(vids, "② + 同名代号合并", merge_dupe=True),
             measure(vids, "③ + 「?」阻断相邻合并", merge_dupe=True,
                     qmark_breaks=True),
             measure(vids, "④ + 去掉纯取得工具状态", merge_dupe=True,
                     qmark_breaks=True, drop_acq=True)]
    print("\n  ①-③ 的分叉前驱：%s" % (steps[0]["hubs"] or "无"))
    print("  ④ 之后的分叉前驱：%s" % (steps[3]["hubs"] or "无"))

    print("\n=== G-branch 的真实前提：某个状态被访问 >=3 次 ===")
    print("  %-28s %-22s %5s   %-12s %s"
          % ("录像", "访问最多的真实状态", "次数", "取得类最多", "次数"))
    rows = visits(vids)
    for r in rows:
        print("  %-28s %-22s %5d   %-12s %d"
              % (r["video_id"][:28], (r["real_state"] or "—")[:22], r["real_visits"],
                 (r["acq_state"] or "无")[:12], r["acq_visits"]))
    dist = collections.Counter(r["real_visits"] for r in rows)
    print("\n  真实状态最大访问次数的分布：%s" % dict(sorted(dist.items())))
    print("  达到 >=3 次的录像：%d / %d —— 为 0 则 G-branch 不可能由真实状态产生"
          % (sum(n for k, n in dist.items() if k >= 3), len(rows)))
    adist = collections.Counter(r["acq_visits"] for r in rows)
    print("  取得类最大访问次数的分布：%s（>=3 次的录像 %d 条）"
          % (dict(sorted(adist.items())),
             sum(n for k, n in adist.items() if k >= 3)))

    if a.out:
        json.dump({"batch_id": doc.get("batch_id"), "n_final": len(vids),
                   "code_classification": cls, "steps": steps,
                   "visits": rows,
                   "real_visit_dist": dict(sorted(dist.items())),
                   "acq_visit_dist": dict(sorted(adist.items()))},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
