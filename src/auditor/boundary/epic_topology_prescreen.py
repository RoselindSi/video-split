"""Ask EPIC-KITCHENS' own annotations whether branching exists there.

WHY THIS DATASET AND NOT THE ONE ON DISK. The 699G of EgoDex sitting on the
server is unusable for B2 on two counts measured before writing this: episodes
run 20.6 seconds at the median, where B2 needs a decision state to recur
several times inside one unit, and the task list is `add_remove_lid`,
`charge_uncharge_airpods`, `stack_unstack_*` -- the do-it-then-undo-it shape
the spec names as producing zero. Its one sorting-shaped task, `declutter_desk`,
puts everything in a single box, so out-degree is one by construction.
EPIC-KITCHENS is the opposite: participants recorded whenever they entered
their own kitchen, nothing was scripted, and the videos are long.

THIS IS A PRE-SCREEN, NOT A VERDICT, and the distinction is load-bearing. B2's
Go/No-Go is defined on coarse state sequences a person labelled; what is
available here is verb/noun action segments, and turning those into state
nodes is a mapping *I* choose. This project has already paid for trusting such
a mapping once: the object-family proxy correlated with real node switching at
r = -0.46, predicting 16 switches for a recording that had 1. So nothing here
decides anything. It selects recordings worth labelling with
`b2_coarse_page.html`, after which `b2_yield.py` decides.

WHICH IS WHY IT REPORTS SEVERAL MAPPINGS AND THEIR SPREAD. Verb alone, noun
alone, and the verb-noun pair give different graphs from identical video, and
an earlier attempt on in-house labels saw a four-fold swing from a merge
choice. If the mappings agree, the signal is about the kitchen; if they
disagree, the number is about my mapping, and the honest output is to say so
rather than to quote the friendliest one.

THE UNIT IS A WINDOW, NOT A VIDEO. B2 labels units of at most three minutes,
so yield per window is the quantity that transfers; yield per whole video
flatters the result by letting a state recur across twenty minutes of
unrelated activity. Both are printed, and the window figure is the one to
read.

A G-branch is `branch_mine.classify`'s, imported rather than restated: out-
degree at least two **and** the reachable set strictly smaller than the
candidates seen so far, under a strict prefix. The second clause is what makes
the constraint non-empty, and it is the one that would be easy to lose here.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import statistics

WINDOW_S = 180.0          # B2 的标注单位上限
MIN_STATES_IN_WINDOW = 3  # 少于这个就谈不上前驱重现


def hms(t):
    """'00:01:02.34' -> 秒。"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read(paths):
    rows = []
    for path in paths:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows.append({
                "video_id": r["video_id"],
                "participant_id": r["participant_id"],
                "start_s": hms(r["start_timestamp"]),
                "verb": r["verb_class"], "noun": r["noun_class"],
                "narration": r["narration"],
            })
    rows.sort(key=lambda r: (r["video_id"], r["start_s"]))
    return rows


MAPPINGS = {
    "verb": lambda r: "v%s" % r["verb"],
    "noun": lambda r: "n%s" % r["noun"],
    "verb+noun": lambda r: "v%s_n%s" % (r["verb"], r["noun"]),
}


def merge_adjacent(seq):
    """和标注页同一个口径：相邻同码合并。"""
    out = []
    for code in seq:
        if not out or out[-1] != code:
            out.append(code)
    return out


def windows_of(rows, window_s):
    """-> [[row]]，按 <=window_s 切，不跨录像。"""
    out = []
    for _vid, group in itertools.groupby(rows, key=lambda r: r["video_id"]):
        group = list(group)
        cur, t0 = [], group[0]["start_s"]
        for r in group:
            if r["start_s"] - t0 > window_s and cur:
                out.append(cur)
                cur, t0 = [], r["start_s"]
            cur.append(r)
        if cur:
            out.append(cur)
    return out


def score(units, key, classify, min_hist):
    """-> (每单位 G-branch 列表, 可达集大小, G-novel 数)."""
    from src.auditor.boundary.b2_yield import rows_of
    per_unit, sizes, novel = [], [], 0
    for unit in units:
        seq = merge_adjacent([key(r) for r in unit])
        if len(set(seq)) < MIN_STATES_IN_WINDOW:
            per_unit.append(0)
            continue
        n = 0
        for row in rows_of(seq):
            kinds = classify(row, min_hist=min_hist)
            if "G-novel" in kinds:
                novel += 1
            if "G-branch" in kinds:
                n += 1
                sizes.append(len(row["S"]))
        per_unit.append(n)
    return per_unit, sizes, novel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--top", type=int, default=20, help="列出产出最高的几个窗口")
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.branch_mine import classify, MIN_HIST

    rows = read(a.csv)
    videos = collections.defaultdict(list)
    for r in rows:
        videos[r["video_id"]].append(r)
    wins = windows_of(rows, a.window)
    print("动作段 %d；录像 %d 条；<=%.0fs 窗口 %d 个"
          % (len(rows), len(videos), a.window, len(wins)))
    lens = [len(w) for w in wins]
    print("  每窗口动作段 中位 %d / 90 分位 %d"
          % (statistics.median(lens), sorted(lens)[int(len(lens) * 0.9)]))

    report = {}
    print("\n=== 三种映射，窗口口径（这一栏才是 B2 的单位）===")
    print("  %-10s %9s %10s %9s %9s" % ("映射", "G-branch", "每窗口", "有分叉窗口", "|N+|中位"))
    for name, key in MAPPINGS.items():
        per_unit, sizes, novel = score(wins, key, classify, MIN_HIST)
        total = sum(per_unit)
        hit = sum(1 for x in per_unit if x > 0)
        report[name] = {
            "window": {"g_branch": total, "per_window": total / len(wins),
                       "windows_with_branch": hit, "n_windows": len(wins),
                       "g_novel": novel,
                       "reachable_median": (statistics.median(sizes)
                                            if sizes else None)}}
        print("  %-10s %9d %10.2f %9d %9s"
              % (name, total, total / len(wins), hit,
                 "%.1f" % statistics.median(sizes) if sizes else "-"))

    print("\n=== 同样三种映射，整条录像口径（偏乐观，仅作对照）===")
    whole = [v for v in videos.values()]
    for name, key in MAPPINGS.items():
        per_unit, sizes, _novel = score(whole, key, classify, MIN_HIST)
        total = sum(per_unit)
        report[name]["video"] = {"g_branch": total,
                                 "per_video": total / len(whole)}
        print("  %-10s %9d 每条 %.2f" % (name, total, total / len(whole)))

    # 跨映射离散度 —— 结论是关于厨房的，还是关于我的映射选择的
    vals = [report[n]["window"]["per_window"] for n in MAPPINGS]
    lo, hi = min(vals), max(vals)
    print("\n=== 跨映射稳健性 ===")
    print("  每窗口 G-branch：%.2f – %.2f（%.1f 倍）" % (lo, hi, hi / max(lo, 1e-9)))
    if hi > 2 * max(lo, 1e-9):
        print("  >>> 换个映射就变两倍以上：这个数字是关于映射选择的，不是关于数据的。")
        print("      按本项目的先例（proxy r=-0.46），不能据此扩量。")
    else:
        print("  >>> 三种映射同量级，信号更可能来自数据本身。")

    print("\n=== 产出最高的窗口（候选，仍需人工粗标才算数）===")
    per_unit, _s, _n = score(wins, MAPPINGS["verb+noun"], classify, MIN_HIST)
    ranked = sorted(zip(per_unit, wins), key=lambda x: -x[0])[:a.top]
    print("  %-14s %6s %6s %8s  %s" % ("录像", "G-br", "段数", "起点s", "前几条叙述"))
    picks = []
    for n, w in ranked:
        if not n:
            continue
        says = " / ".join(r["narration"] for r in w[:3])
        picks.append({"video_id": w[0]["video_id"], "g_branch": n,
                      "start_s": round(w[0]["start_s"], 1),
                      "n_segments": len(w)})
        print("  %-14s %6d %6d %8.0f  %s"
              % (w[0]["video_id"], n, len(w), w[0]["start_s"], says[:60]))

    if a.out:
        json.dump({"window_s": a.window, "min_hist": MIN_HIST,
                   "n_segments": len(rows), "n_videos": len(videos),
                   "n_windows": len(wins), "mappings": report,
                   "spread_ratio": hi / max(lo, 1e-9), "top_windows": picks},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)
    print("\n注意：以上全部是选片用的预筛。B2 的判定只认 b2_coarse_page.html "
          "的粗标 + b2_yield.py，判据冻结于 2026-09-28。")


if __name__ == "__main__":
    main()
