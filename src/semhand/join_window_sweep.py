"""评测 track_join 的窗口参数 —— 对着手级 gold，同时数接对与接错。

为什么这个脚本只有在 R1/R2 之后才可能存在。以前所有 gold 都键在 `rec:tid`
上，所以「这两段本该是同一只手」这个问题提都提不出来，`track_join` 的效果只
能看碎片数降了多少 —— 一个把「并对了」和「把两只手并成一条」记成同样进步的
指标。手级 gold 指出每只手由哪些 tid 构成，于是接对与接错第一次能分开数。

该接：同一只手的 tid 按首帧排序，相邻两个构成一处「本该接上」。
接对：这两个 tid 在 join 之后落进同一组。
接错：两只不同的手的 tid 落进同一组 —— 这是比漏接更贵的错误，因为下游每个
      判定都是逐轨的，一条混了两只手的轨迹会把两只手的判定一起带偏。

注意这条曲线不单调，而且不单调是规则本身的性质不是噪声：`forced_pairs` 要求
「恰好一个候选」，窗口放大会让某些端点从唯一候选变成多候选，于是规则反而放
弃。所以**不要用这条曲线挑阈值** —— 它不是单峰的，取最大值等于对少数几个
事件过拟合。它的用途是回答一个二元问题：窗口是不是瓶颈。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.hand_level_metrics import assign_by_anchors, read_tracks  # noqa: E402
from rig.track_join import (DIST_FACTOR, centre, join_to_fixed_point,  # noqa: E402
                            width)


def frags_from_tracks(tracks):
    """track_join 的碎片结构，但框来自 detect2 而不是重读 csv。"""
    out = {}
    for tid, t in tracks.items():
        items = sorted((f, b) for f, b in t["boxes"].items() if b is not None)
        if not items:
            continue
        out[tid] = {"first": items[0][0], "last": items[-1][0],
                    "start_box": list(items[0][1]),
                    "end_box": list(items[-1][1]), "n": len(items)}
    return out


def should_join(assign, frag):
    """-> [(hand_key, a, b)]，同一只手里按首帧相邻的 tid 对。"""
    by_hand = collections.defaultdict(list)
    for tid, h in assign.items():
        if h is not None and tid in frag:
            by_hand[h].append(tid)
    pairs = []
    for h, tids in sorted(by_hand.items()):
        tids.sort(key=lambda t: frag[t]["first"])
        for a, b in zip(tids, tids[1:]):
            pairs.append((h, a, b))
    return pairs


def wrong_pairs(assign, frag, mapping):
    """-> [(group, [tid...])]，一组里混进了两只不同的手。"""
    groups = collections.defaultdict(list)
    for tid in frag:
        groups[mapping.get(tid, tid)].append(tid)
    bad = []
    for g, members in sorted(groups.items()):
        hands = {assign.get(t) for t in members} - {None}
        if len(hands) > 1:
            bad.append((g, sorted(members), sorted(hands)))
    return bad


def diagnose(frag, a, b, window, dist_factor=DIST_FACTOR):
    """这一对为什么没接上 —— 把窗口、距离、歧义三个原因分开。"""
    fa, fb = frag[a], frag[b]
    gap = fb["first"] - fa["last"]
    cx, cy = centre(fa["end_box"])
    bx, by = centre(fb["start_box"])
    dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
    budget = dist_factor * width(fa["end_box"])
    cands = []
    for c in frag:
        if c == a:
            continue
        fc = frag[c]
        g = fc["first"] - fa["last"]
        if g <= 0 or g > window:
            continue
        ux, uy = centre(fc["start_box"])
        if ((ux - cx) ** 2 + (uy - cy) ** 2) ** 0.5 <= budget:
            cands.append(c)
    if gap <= 0:
        why = "时间重叠（硬否）"
    elif gap > window:
        why = "间隔 %d > 窗口 %d，差 %d 帧" % (gap, window, gap - window)
    elif dist > budget:
        why = "中心距 %.0f > 预算 %.0f" % (dist, budget)
    elif len(cands) != 1:
        why = "多候选歧义（%d 个：%s），规则主动放弃" % (
            len(cands), ",".join(map(str, sorted(cands))))
    else:
        why = "应当接上"
    return {"gap": gap, "dist": round(dist, 1), "budget": round(budget, 1),
            "n_cands": len(cands), "why": why}


def run_one(gold_path, decisions_path, detect2_path, windows, min_iou=0.3):
    gold = json.load(open(gold_path, encoding="utf-8"))
    rec = gold["recording"]
    tracks = read_tracks(decisions_path, detect2_path, own_only=True)
    assign, diag = assign_by_anchors(gold["hands"], tracks, min_iou=min_iou)
    frag = frags_from_tracks(tracks)
    targets = should_join(assign, frag)

    rows = []
    for w in windows:
        mapping, rounds = join_to_fixed_point(frag, window=w,
                                              dist_factor=DIST_FACTOR)
        per_pair = []
        for h, a, b in targets:
            joined = mapping.get(a, a) == mapping.get(b, b)
            d = diagnose(frag, a, b, w)
            per_pair.append({"hand": h, "a": a, "b": b, "joined": joined,
                             **d})
        bad = wrong_pairs(assign, frag, mapping)
        rows.append({"window": w, "rounds": rounds,
                     "should": len(targets),
                     "ok": sum(1 for p in per_pair if p["joined"]),
                     "wrong": len(bad),
                     "wrong_detail": bad, "pairs": per_pair})
    return {"recording": rec, "n_tracks": len(frag),
            "assign_blind": diag["blind_no_anchor"],
            "assign_contaminating": diag["contaminating"],
            "targets": [{"hand": h, "a": a, "b": b} for h, a, b in targets],
            "sweep": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--detect2-dir", required=True)
    ap.add_argument("--windows", default="30,40,50,60,80,100,150")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out")
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",")]
    results = []
    for g in args.gold:
        rec = json.load(open(g, encoding="utf-8"))["recording"]
        dec = os.path.join(args.decisions_dir, "%s.decisions.csv" % rec)
        det = os.path.join(args.detect2_dir, "%s.detect2.csv" % rec)
        results.append(run_one(g, dec, det, windows, args.min_iou))

    print("逐录像逐对（为什么没接上）")
    for r in results:
        print("\n  %s  %d 条轨迹  该接 %d 处"
              % (r["recording"], r["n_tracks"], len(r["targets"])))
        for row in r["sweep"]:
            marks = []
            for p in row["pairs"]:
                marks.append("%s %s→%s %s" % (
                    p["hand"], p["a"], p["b"], "接上" if p["joined"] else p["why"]))
            print("    窗口 %3d | %s" % (row["window"], " | ".join(marks)))

    print("\n合计（%d 条录像）" % len(results))
    print("    窗口 |   该接   接对   接错")
    for i, w in enumerate(windows):
        should = sum(r["sweep"][i]["should"] for r in results)
        ok = sum(r["sweep"][i]["ok"] for r in results)
        wrong = sum(r["sweep"][i]["wrong"] for r in results)
        print("    %4d | %6d %6d %6d%s"
              % (w, should, ok, wrong, "  <- 现状" if w == 30 else ""))

    n_ev = sum(len(r["targets"]) for r in results)
    print("\n  事件总数 %d。少于 20 处时任何阈值结论都不成立 ——" % n_ev)
    print("  这条曲线不单调（「恰好一个候选」与窗口宽度交互），取最大值")
    print("  等于对这 %d 个事件过拟合。" % n_ev)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"windows": windows, "recordings": results},
                  open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\n  写出 %s" % args.out)


if __name__ == "__main__":
    main()
