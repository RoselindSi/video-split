"""Ask Ego-Exo4D's keysteps whether branching exists, at task level not primitive level.

WHY THIS DATASET AFTER EPIC-KITCHENS FAILED TO DECIDE. EPIC has the right
recording shape -- unscripted, 1998 windows of three minutes or less -- but it
annotates action primitives. 60.2% of its apparent branching sat on `pick-up`
and `put-down` alone and 70.8% on the top three nodes, because a node called
"pick up" is followed by everything and so has high out-degree by
construction, not by decision. Stripping those primitives dropped the yield
from 14.71 per window to 0.62-0.89, straddling B2's 0.5-1 threshold, and which
verbs count as primitive was my choice. A pre-screen that lands on the
threshold and moves with my own definitions cannot select anything.

Ego-Exo4D annotates keysteps: "actions that contribute towards the completion
of a procedural task". That is the level B2's state nodes are defined at, so
the primitive artifact does not arise the same way -- which is a reason to
look, not a guarantee, and the same concentration check runs here.

IT ALSO SOLVES THE HARDEST PART OF B2'S LABELLING CONTRACT FOR FREE.
`step_unique_id` is consistent across all scenario taxonomies, so "went back
to the thing I did before, use the same code" is given by the data rather than
maintained by an annotator. That clause is the one the coarse page was built
to practise because it is where codes drift, and drifted codes make every
out-degree 1 -- the exact failure that would look like a negative result.

PER-SCENARIO, BECAUSE THE SCENARIOS ARE NOT ONE POPULATION. Bike repair is
troubleshooting-shaped: inspect a component, and what follows depends on what
you found. Cooking recipes are closer to a fixed order. Pooling them would let
one hide inside the other, and B2 needs decisions from several independent
takes anyway, so the breakdown is the output and the total is the footnote.

STILL A PRE-SCREEN. B2's verdict comes only from coarse labels through
`b2_yield.py`, against thresholds frozen on 2026-09-28. The project has been
burned by a selection proxy that ran backwards (r = -0.46), so what this
produces is a shortlist and a reason, never a Go.

Written before the data exists on purpose: approval takes about 48 hours and
the AWS credentials then expire in 14 days, so the analysis has to be ready
before the clock starts. `--selftest` runs it against a synthetic file built
to the documented schema so the code is known to work beforehand.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import statistics

WINDOW_S = 180.0
MIN_STATES = 3

CODERS = {
    "step_unique_id": lambda s: str(s.get("step_unique_id")),
    "step_id": lambda s: str(s.get("step_id")),
    "step_name": lambda s: (s.get("step_name") or "").strip().lower(),
}


def load(path):
    """-> {take_uid: {"scenario":…, "take_name":…, "segments":[…]}}

    The file holds `annotations`, `taxonomy` and `vocabulary`; only the first
    is read. `annotations` may be a dict keyed by take_uid or a list of take
    records, so both are accepted rather than guessed at.
    """
    doc = json.load(open(path, encoding="utf-8"))
    ann = doc.get("annotations", doc)
    takes = {}
    items = ann.items() if isinstance(ann, dict) else (
        (t.get("take_uid"), t) for t in ann)
    for uid, rec in items:
        if not isinstance(rec, dict):
            continue
        segs = rec.get("segments") or rec.get("keysteps") or []
        if not segs:
            continue
        takes[uid or rec.get("take_uid")] = {
            "scenario": rec.get("scenario") or "?",
            "take_name": rec.get("take_name") or uid,
            "segments": sorted(segs, key=lambda s: float(s.get("start_time", 0))),
        }
    return takes


def merge_adjacent(seq):
    out = []
    for code in seq:
        if not out or out[-1] != code:
            out.append(code)
    return out


def windows(segments, window_s):
    out, cur = [], []
    t0 = float(segments[0].get("start_time", 0)) if segments else 0.0
    for s in segments:
        if float(s.get("start_time", 0)) - t0 > window_s and cur:
            out.append(cur)
            cur, t0 = [], float(s.get("start_time", 0))
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def yield_of(units, coder, classify, min_hist):
    from src.auditor.boundary.b2_yield import rows_of
    total = hit = novel = 0
    sizes, carriers = [], collections.Counter()
    for unit in units:
        seq = merge_adjacent([coder(s) for s in unit])
        if len(set(seq)) < MIN_STATES:
            continue
        n = 0
        for row in rows_of(seq):
            kinds = classify(row, min_hist=min_hist)
            novel += int("G-novel" in kinds)
            if "G-branch" in kinds:
                n += 1
                sizes.append(len(row["S"]))
                carriers[row["prev_node"]] += 1
        total += n
        hit += int(n > 0)
    return {"g_branch": total, "units": len(units), "units_with_branch": hit,
            "per_unit": total / max(1, len(units)), "g_novel": novel,
            "reachable_median": statistics.median(sizes) if sizes else None,
            "carriers": carriers}


def concentration(carriers):
    """前三个前驱占多少 —— EPIC 那次是 70.8%，全在搬运原语上。"""
    total = sum(carriers.values())
    if not total:
        return 0.0, []
    top = carriers.most_common(3)
    return sum(n for _c, n in top) / total, top


def selftest():
    """用文档给的字段名造一份合成标注，确认代码在真数据到之前就能跑。"""
    import os
    import tempfile

    def seg(uid, t, name):
        return {"start_time": t, "end_time": t + 5, "step_id": uid,
                "step_unique_id": uid, "step_name": name,
                "step_description": "", "is_essential": True}

    # 确定性环：出度恒为 1，应当给出 0
    det = [seg(c, i * 10, "s%s" % c) for i, c in enumerate("ABCDABCDABCD")]
    # 分叉：A 反复出现并进入不同后继
    br = []
    for i, nxt in enumerate("BCDBCDBC"):
        br.append(seg("A", i * 20, "inspect"))
        br.append(seg(nxt, i * 20 + 10, "route_%s" % nxt))
    doc = {"taxonomy": {}, "vocabulary": {}, "annotations": {
        "t_det": {"take_uid": "t_det", "take_name": "det",
                  "scenario": "cooking", "segments": det},
        "t_br": {"take_uid": "t_br", "take_name": "br",
                 "scenario": "bike repair", "segments": br}}}
    path = os.path.join(tempfile.mkdtemp(), "keystep.json")
    json.dump(doc, open(path, "w"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keystep", help="keystep 标注 json")
    ap.add_argument("--selftest", action="store_true",
                    help="跑合成数据，验证代码在拿到凭证之前就是对的")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.auditor.boundary.branch_mine import classify, MIN_HIST
    path = selftest() if a.selftest else a.keystep
    if not path:
        raise SystemExit("给 --keystep 或 --selftest")

    takes = load(path)
    if not takes:
        raise SystemExit("没读到 take；检查 annotations 的结构")
    by_scen = collections.defaultdict(list)
    for uid, t in takes.items():
        by_scen[t["scenario"]].append(t)
    all_wins = [w for t in takes.values()
                for w in windows(t["segments"], a.window)]
    print("take %d 条；scenario %d 个；<=%.0fs 窗口 %d 个"
          % (len(takes), len(by_scen), a.window, len(all_wins)))

    print("\n=== 三种节点编码，窗口口径 ===")
    print("  %-16s %9s %9s %12s %9s" % ("编码", "G-branch", "每窗口",
                                        "有分叉窗口", "|N+|中位"))
    spread, report = {}, {}
    for name, coder in CODERS.items():
        r = yield_of(all_wins, coder, classify, MIN_HIST)
        spread[name] = r["per_unit"]
        share, top = concentration(r["carriers"])
        report[name] = {k: v for k, v in r.items() if k != "carriers"}
        report[name]["top3_share"] = round(share, 3)
        report[name]["top_carriers"] = [[c, n] for c, n in top]
        print("  %-16s %9d %9.2f %12s %9s"
              % (name, r["g_branch"], r["per_unit"],
                 "%d/%d" % (r["units_with_branch"], r["units"]),
                 "%.1f" % r["reachable_median"] if r["reachable_median"] else "-"))

    best = "step_unique_id"
    # 用循环里算好的那个份额。拿只含三项的 top_carriers 重建 Counter 会让
    # 分母只剩那三项，份额恒为 100% —— 自测时就是这么露出来的。
    share = report[best]["top3_share"]
    top = report[best]["top_carriers"]
    print("\n=== 原语假象检查（EPIC 那次前三占 70.8%，全是搬运动作）===")
    print("  %s 下前三个前驱占 %.1f%%：%s"
          % (best, 100 * share, ", ".join("%s×%d" % (c, n) for c, n in top)))
    if share > 0.5:
        print("  >>> 集中度过半：先看这几个节点是不是 EPIC 里 pick-up 那种"
              "「后面能接一切」的通用步骤，是的话这个产出同样不可用。")

    print("\n=== 按 scenario（bike repair 是排障形态，cooking 偏固定顺序）===")
    print("  %-28s %6s %9s %9s" % ("scenario", "take", "G-branch", "每窗口"))
    per_scen = {}
    for scen, ts in sorted(by_scen.items(),
                           key=lambda kv: -len(kv[1]))[:a.top]:
        wins = [w for t in ts for w in windows(t["segments"], a.window)]
        r = yield_of(wins, CODERS[best], classify, MIN_HIST)
        per_scen[scen] = {"takes": len(ts), "g_branch": r["g_branch"],
                          "per_unit": r["per_unit"],
                          "units_with_branch": r["units_with_branch"]}
        print("  %-28s %6d %9d %9.2f"
              % (scen[:28], len(ts), r["g_branch"], r["per_unit"]))

    vals = list(spread.values())
    lo, hi = min(vals), max(vals)
    print("\n=== 跨编码稳健性 ===")
    print("  每窗口 %.2f – %.2f（%.1f 倍）" % (lo, hi, hi / max(lo, 1e-9)))
    if hi > 2 * max(lo, 1e-9):
        print("  >>> 换编码变两倍以上，数字是关于编码选择的。EPIC 就是在这里"
              "卡住的，不能据此扩量。")
    else:
        print("  >>> 三种编码同量级。")

    if a.selftest:
        det = yield_of([w for w in windows(takes["t_det"]["segments"], a.window)],
                       CODERS[best], classify, MIN_HIST)
        br = yield_of([w for w in windows(takes["t_br"]["segments"], a.window)],
                      CODERS[best], classify, MIN_HIST)
        ok = det["g_branch"] == 0 and br["g_branch"] > 0
        print("\n=== selftest ===")
        print("  确定性环 G-branch %d（应为 0）" % det["g_branch"])
        print("  分叉 take G-branch %d（应 > 0）" % br["g_branch"])
        print("  >>> %s" % ("通过" if ok else "失败：代码有问题，不要拿去跑真数据"))
        if not ok:
            raise SystemExit(1)

    if a.out:
        json.dump({"window_s": a.window, "min_hist": MIN_HIST,
                   "n_takes": len(takes), "n_windows": len(all_wins),
                   "codings": report, "per_scenario": per_scen,
                   "spread_ratio": hi / max(lo, 1e-9)},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)
    print("\n注意：预筛只产候选名单。B2 判定只认 b2_coarse_page.html 的粗标 + "
          "b2_yield.py，判据冻结于 2026-09-28。")


if __name__ == "__main__":
    main()
