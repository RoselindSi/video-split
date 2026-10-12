"""把手级 gold 上的每一个缺陷归类 —— 互斥、穷尽、数目对得上。

为什么要算而不是假设。「轨迹断了所以要 track merge」听上去是独立于窗口的一
类，其实不是：把两段接起来**正是** `track_join` 在做的事，窗口就是它的参数。
断裂里有多少是「单纯隔得远」（放宽窗口就能接）、有多少是「放宽之后候选不唯
一」（几何分不开两只手），只能数出来。十条录像上数出来的结果是后者占大头，
所以「断裂」不是一类，它要按**为什么没接上**拆开。

两个不同的计数单位，回答两个不同的问题，不能混：

  帧级   —— 每个「这只手可见的抽样帧」要么被正确持有，要么是一个缺陷。
            问的是「有多少帧丢了」，分母是 hand-frame。
  身份级 —— 每只手若由 n 条轨迹持有，就贡献 n-1 处「本该接上」。
            问的是「身份断了几次、为什么」，分母是 should-join 事件。

一只手可以帧级满分而身份级很差（覆盖 100% 但碎成 5 条），反过来也可以
（单一轨迹但漏了 6%）。十条里两种都出现了，所以两张表都要出。

归类用的框来自 **join 前的 detect2**（帧和框同源），因为「那一帧到底有没有
检测」是检测器的事实，与 join 无关；而「哪条轨迹持有这只手」要用
decisions + detect2_joined，那是部署状态。见 detect2-joined-is-the-right-file。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.hand_level_metrics import (assign_by_anchors, frames_of,  # noqa: E402
                                        iou, read_tracks)
from semhand.join_window_sweep import (diagnose, frags_from_tracks,  # noqa: E402
                                       read_pre_join, should_join)

# 帧级类别 —— 互斥且穷尽，覆盖每一个「这只手可见的抽样帧」
FRAME_KINDS = [
    ("held", "被唯一正确的轨迹持有"),
    ("held_duplicate", "被 >1 条属于这只手的轨迹同时持有（重复轨迹）"),
    ("miss_no_detection", "那一帧一个检测框都没有 —— 检测器漏了"),
    ("miss_box_unassigned", "有框且位置对，但那条轨迹没归到这只手 —— 关联丢了"),
    ("miss_box_elsewhere", "有框但位置都不对 —— 检测到了别的东西"),
]

# 身份级类别 —— 覆盖每一处「本该接上」
#
# `joined_now` 与 `would_join` 必须分开。前者是流水线**实际**并了的，后者是
# 「若窗口设成所问的那个值就会并」。混成一个 `joined` 会把这次分析唯一要回答
# 的问题抹掉：现状如何 vs 改了会怎样。
JOIN_KINDS = [
    ("joined_now", "流水线实际已经接上了"),
    ("would_join", "在所问的窗口下会接上（现状没接）"),
    ("blocked_window", "间隔超过所问的窗口"),
    ("blocked_ambiguity", "候选不唯一，规则弃权 —— 几何分不开"),
    ("blocked_overlap", "两段时间重叠 —— tracker 把一只手放成两条"),
    ("blocked_distance", "中心距超出预算"),
]


def frame_level(gold_hands, tracks, assign, pre, sampled, min_iou=0.3):
    """-> {hand_key: Counter}，每一个可见抽样帧归且仅归一类。"""
    samp = set(sampled)
    anchors = {h["hand_key"]: {int(a["frame"]): tuple(a["box"])
                               for a in h.get("anchors") or []}
               for h in gold_hands}
    out = {}
    for h in gold_hands:
        k = h["hand_key"]
        H = frames_of(h["intervals"]) & samp
        mine = [t for t, v in assign.items() if v == k]
        c = collections.Counter()
        detail = collections.defaultdict(list)
        for f in sorted(H):
            here = [t for t in mine if f in tracks[t]["boxes"]]
            if len(here) == 1:
                c["held"] += 1
                continue
            if len(here) > 1:
                c["held_duplicate"] += 1
                detail["held_duplicate"].append(f)
                continue
            # 没被持有 —— 分三种，看那一帧到底有没有框、框在哪
            boxes = [b for t in pre.values() for ff, b in t["boxes"].items()
                     if ff == f]
            if not boxes:
                c["miss_no_detection"] += 1
                detail["miss_no_detection"].append(f)
                continue
            gt = anchors[k].get(f)
            if gt is not None and any(iou(gt, b) >= min_iou for b in boxes):
                c["miss_box_unassigned"] += 1
                detail["miss_box_unassigned"].append(f)
            else:
                c["miss_box_elsewhere"] += 1
                detail["miss_box_elsewhere"].append(f)
        out[k] = {"n_visible": len(H), "counts": c, "detail": dict(detail)}
    return out


def identity_level(gold_hands, pre, frag, assign_pre, window, dist_factor=3.0,
                   joined_map=None):
    """-> [{hand, a, b, kind}]，每处「本该接上」归且仅归一类。

    用 join 前的碎片，因为问的就是 track_join 面对的那个选择。
    `joined_map` 给的是流水线实际的分组，用来判断这一处是不是已经接上了。
    """
    rows = []
    for k, a, b in should_join(assign_pre, frag):
        if joined_map is not None and joined_map.get(a) == joined_map.get(b):
            rows.append({"hand": k, "a": a, "b": b, "kind": "joined_now"})
            continue
        d = diagnose(frag, a, b, window, dist_factor)
        why = d["why"]
        if "时间重叠" in why:
            kind = "blocked_overlap"
        elif "窗口" in why:
            kind = "blocked_window"
        elif "多候选" in why:
            kind = "blocked_ambiguity"
        elif "中心距" in why:
            kind = "blocked_distance"
        else:
            kind = "would_join"
        rows.append({"hand": k, "a": a, "b": b, "kind": kind,
                     "gap": d["gap"], "n_cands": d["n_cands"]})
    return rows


def observed_map(joined_path, pre):
    key = {}
    for tid, fb in pre.items():
        for f, b in fb["boxes"].items():
            key[(f, round(b[0]), round(b[1]))] = tid
    out = {}
    for r in csv.DictReader(open(joined_path, encoding="utf-8-sig")):
        try:
            k = (int(r["frame"]), round(float(r["x0"])), round(float(r["y0"])))
        except (KeyError, ValueError):
            continue
        if k in key:
            out[key[k]] = str(r["tid"])
    return out


def run_one(gold_path, dec_dir, pre_dir, joined_dir, window, min_iou=0.3):
    gold = json.load(open(gold_path, encoding="utf-8"))
    rec = gold["recording"]
    dec = os.path.join(dec_dir, "%s.decisions.csv" % rec)
    pre_p = os.path.join(pre_dir, "%s.detect2.csv" % rec)
    jnd_p = os.path.join(joined_dir, "%s.detect2.csv" % rec)

    # 部署状态：谁持有这只手
    tracks = read_tracks(dec, jnd_p, own_only=True)
    assign, _ = assign_by_anchors(gold["hands"], tracks, min_iou=min_iou)
    # join 前：track_join 面对的选择
    pre = read_pre_join(pre_p)
    assign_pre, _ = assign_by_anchors(gold["hands"], pre, min_iou=min_iou)
    frag = frags_from_tracks(pre)

    fl = frame_level(gold["hands"], tracks, assign, pre,
                     gold["sampled_frames"], min_iou)
    il = identity_level(gold["hands"], pre, frag, assign_pre, window,
                        joined_map=observed_map(jnd_p, pre))
    return {"recording": rec, "frame": fl, "identity": il}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="glob")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--detect2-pre", required=True, help="join 前")
    ap.add_argument("--detect2-joined", required=True, help="join 后")
    ap.add_argument("--window", type=int, default=30,
                    help="用哪个窗口判「是不是被窗口卡住」；默认 30 = 旧值")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out")
    a = ap.parse_args()

    res = [run_one(g, a.decisions, a.detect2_pre, a.detect2_joined,
                   a.window, a.min_iou)
           for g in sorted(glob.glob(a.gold))]

    # ---- 帧级 ----
    tot = collections.Counter()
    for r in res:
        for k, v in r["frame"].items():
            tot.update(v["counts"])
    n = sum(tot.values())
    print("=== 帧级：%d 个「这只手可见的抽样帧」===" % n)
    print("  %-22s %6s %7s  %s" % ("类别", "帧数", "占比", "说明"))
    for key, desc in FRAME_KINDS:
        print("  %-22s %6d %6.1f%%  %s" % (key, tot[key], 100 * tot[key] / n,
                                           desc))
    miss = n - tot["held"] - tot["held_duplicate"]
    print("  %-22s %6d %6.1f%%" % ("—— 丢失合计", miss, 100 * miss / n))
    if miss:
        det = tot["miss_no_detection"]
        print("     其中检测器漏 %d（占丢失 %.0f%%），关联丢 %d（%.0f%%）"
              % (det, 100 * det / miss, tot["miss_box_unassigned"],
                 100 * tot["miss_box_unassigned"] / miss))

    # ---- 身份级 ----
    ident = collections.Counter()
    for r in res:
        for row in r["identity"]:
            ident[row["kind"]] += 1
    m = sum(ident.values())
    print("\n=== 身份级：%d 处「本该接上」（窗口按 %d 判）===" % (m, a.window))
    print("  %-22s %6s %7s  %s" % ("类别", "处数", "占比", "说明"))
    for key, desc in JOIN_KINDS:
        print("  %-22s %6d %6.1f%%  %s" % (key, ident[key],
                                           100 * ident[key] / m if m else 0,
                                           desc))
    block = m - ident["joined_now"] - ident["would_join"]
    if block:
        print("  %-22s %6d %6.1f%%" % ("—— 没接上合计", block, 100 * block / m))
        print("     窗口能解锁的 %d（占没接上 %.0f%%），"
              % (ident["blocked_window"],
                 100 * ident["blocked_window"] / block))
        print("     几何分不开的 %d（%.0f%%），重复轨迹 %d（%.0f%%）"
              % (ident["blocked_ambiguity"],
                 100 * ident["blocked_ambiguity"] / block,
                 ident["blocked_overlap"],
                 100 * ident["blocked_overlap"] / block))

    print("\n  两张表的分母不同，不能相加：帧级问「丢了多少帧」，"
          "身份级问「身份断了几次」。")
    print("  一只手可以帧级满分而身份级很差（覆盖 100% 碎成 5 条），反之亦然。")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"window": a.window,
                   "frame_total": dict(tot), "identity_total": dict(ident),
                   "recordings": [{"recording": r["recording"],
                                   "frame": {k: {"n_visible": v["n_visible"],
                                                 "counts": dict(v["counts"]),
                                                 "detail": v["detail"]}
                                             for k, v in r["frame"].items()},
                                   "identity": r["identity"]} for r in res]},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
