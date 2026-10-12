"""R2: per-hand tracking metrics, and the gold schema they require.

WHY THE UNIT HAD TO CHANGE. Every gold in this project is keyed on `rec:tid`,
that is on the tracker's own output, so a hand cut into five tracks is five
rows judged independently and "did we get one track per hand" cannot be asked
of it: the unit of the gold is the thing fragmentation damages. 96.6%
track-level ownership accuracy is consistent with a hand arriving in five
correctly-labelled pieces. These metrics take the hand as the unit instead.

THE WEARER HAS TWO HANDS, WHICH SIMPLIFIES THE GOLD. There is no need to
enumerate "hand instances": over a recording there are exactly two entities,
each visible in some intervals.

AND THEY ARE ANONYMOUS ON PURPOSE. The gold does not say which is the left
hand. `track_join` measured side's implied majority reliability at about 0.88
-- within one fragment 4.4% of frames disagree with that fragment's own
majority and 55.6% of fragments flip at least once -- and the human side of it
was left unresolved too. If the gold had to name left from right, the metric
would inherit that unreliability and could not separate "we lost the hand"
from "we called it the wrong hand". So the entities are A and B, distinct and
followed through the recording by the annotator, and side is a secondary check.

THE HARD PART IS ASSIGNMENT, AND INTERVALS ALONE CANNOT DO IT. Both hands are
often visible in the same frames, so "assign a track to the hand it shares the
most frames with" is undefined exactly when it matters. Per-frame gold boxes
would settle it and cost too much to annotate. So the gold carries sparse
ANCHORS -- a box at a few frames per interval -- and tracks match entities by
IoU there. Tracks that match nothing, or tie, stay unassigned and are counted:
an unassignable own-track is either contamination or a gap in the gold, and
assigning it to one hand to keep the books tidy is how a metric starts lying.

SELF-TEST BEFORE GOLD. The metric has to be known correct before annotation is
spent on it, so `--selftest` runs it against cases whose answers are obvious:
one perfect track, a hand split in two, a hand half covered, a hand with no
track at all.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics

GOLD_SCHEMA = """手级 gold 的 schema（R1 要产出的东西）

{
  "schema_version": "hand_gold_v1",
  "recording": "R26_0824_121025",
  "hands": [
    {
      "hand_key": "A",                   // 匿名但在本录像内恒定，不要求是左/右
      "side_note": "看着像右手",          // 可选，不参与计分
      "intervals": [[400, 612], [700, 755]],      // 可见帧区间，闭区间
      "anchors": [                        // 稀疏框，用来把轨迹分配到这只手
        {"frame": 400, "box": [908, 1078, 1122, 1290]},
        {"frame": 500, "box": [880, 1050, 1100, 1270]}
      ]
    },
    { "hand_key": "B", "intervals": [], "anchors": [] }
  ]
}

两条规则必须进标注合同，否则指标没有意义：
① hand_key 在整条录像内恒定 —— 手出画又回来仍是同一个 key。这一条正是在测
   跨遮挡的身份保持，所以它是标注者要做的判断，工具补不了。
② anchors 每个区间至少两个（两端），长区间按下面的密度加密。

anchor 密度不是风格问题，它决定指标能不能看见碎片：
**一条碎片若一个 anchor 帧都没盖住，它对指标不可见**，而漏记碎片会让系统
显得比实际好。所以间距必须不大于你在意的最短碎片。

R0 在 10 条录像上量到的 own 轨迹帧数分布：
    1-9 帧 10 条（占 own 帧 0.7%）/ 10-33 帧 17 条（6.0%）
    34-99 帧 20 条（24.5%）/ 100+ 帧 14 条（68.7%）

若接受放弃 <10 帧那一档（只占 0.7% 的帧），**间距 <=10 帧**即可覆盖其余全部。
录像约 400 帧 → 每只手约 40 个 anchor、每条录像约 80 个。
这就是测量所述目标的标注价格，工具会把实际盲区数报出来，不靠承诺。
"""


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def frames_of(intervals):
    out = set()
    for s, e in intervals:
        out.update(range(int(s), int(e) + 1))
    return out


def assign_by_anchors(gold_hands, tracks, min_iou=0.3):
    """-> ({tid: hand_key or None}, diagnostics)."""
    votes = collections.defaultdict(collections.Counter)
    for h in gold_hands:
        for a in h.get("anchors") or []:
            f, box = int(a["frame"]), tuple(a["box"])
            for tid, t in tracks.items():
                b = t["boxes"].get(f)
                if b is None:
                    continue
                if iou(box, b) >= min_iou:
                    votes[tid][h["hand_key"]] += 1
    anchor_frames = {int(a["frame"]) for h in gold_hands
                     for a in (h.get("anchors") or [])}
    out, ties, blind = {}, [], []
    for tid, t in tracks.items():
        c = votes.get(tid)
        if not c:
            out[tid] = None
            # 盲区与污染必须分开：一条轨迹若压根没盖住任何 anchor 帧，它不是
            # 「匹配不上」而是「没机会被匹配」。把两者并成 unassigned 会让
            # anchor 太稀造成的漏记看起来像系统的污染。
            if not (set(t["boxes"]) & anchor_frames):
                blind.append(tid)
            continue
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            out[tid] = None
            ties.append(tid)
            continue
        out[tid] = top[0][0]
    unassigned = [t for t, h in out.items() if h is None]
    return out, {"ties": ties, "unassigned": unassigned,
                 "blind_no_anchor": blind,
                 "contaminating": [t for t in unassigned if t not in blind]}


def assign_by_side(gold_hands, tracks):
    """Fallback before anchors exist. Inherits side's ~0.88 reliability."""
    keys = [h["hand_key"] for h in gold_hands]
    sides = sorted({(t["side"] or "?") for t in tracks.values()})
    m = {s: keys[i % len(keys)] for i, s in enumerate(sides)}
    return ({tid: m.get(t["side"] or "?") for tid, t in tracks.items()},
            {"ties": [], "unassigned": [],
             "warning": "side 分配继承 side 多数票的可靠性（约 0.88），"
                        "只能用来试跑，不能当结果"})


def per_hand(gold_hands, tracks, assign, sampled=None):
    """switches counts tid changes walking the hand's visible frames in order.

    A hole does not count as a switch by itself, but a different tid on the far
    side of a hole does -- that is precisely identity not being preserved.
    """
    rows = []
    for h in gold_hands:
        H = frames_of(h["intervals"])
        if sampled is not None:
            # 只在真的被标过的帧上算。区间是由抽样帧连成的，区间内的其它帧
            # 标注者没看过 —— 在那些帧上算覆盖率等于拿没核对过的东西当 gold。
            H &= set(sampled)
        mine = {tid: t for tid, t in tracks.items()
                if assign.get(tid) == h["hand_key"]}
        covered = set()
        for t in mine.values():
            covered |= (set(t["boxes"]) & H)
        seq = []
        for f in sorted(H):
            here = sorted(tid for tid, t in mine.items() if f in t["boxes"])
            seq.append("+".join(here) if here else None)
        sw, last = 0, None
        for v in seq:
            if v is None:
                continue
            if last is not None and v != last:
                sw += 1
            last = v
        rows.append({"hand_key": h["hand_key"], "visible_frames": len(H),
                     "n_tids": len(mine), "tids": sorted(mine),
                     "coverage": (len(covered) / len(H)) if H else float("nan"),
                     "switches": sw,
                     "multi_cover_frames": sum(1 for v in seq if v and "+" in v)})
    return rows


def per_track_purity(gold_hands, tracks, assign):
    vis = {h["hand_key"]: frames_of(h["intervals"]) for h in gold_hands}
    out = {}
    for tid, t in tracks.items():
        k = assign.get(tid)
        fs = set(t["boxes"])
        if not fs:
            continue
        out[tid] = (len(fs & vis[k]) / len(fs)) if k else 0.0
    return out


def read_tracks(decisions_path, detect2_path=None, own_only=True):
    """-> {tid: {"boxes": {frame: box or None}, "side": str}}

    Frames come from decisions, geometry from detect2 when given. Coverage and
    switches need only the frames; anchor assignment needs the boxes.
    """
    tracks = collections.defaultdict(lambda: {"boxes": {}, "side": ""})
    for r in csv.DictReader(open(decisions_path, encoding="utf-8-sig")):
        if own_only and (r.get("own") or "").strip() != "1":
            continue
        try:
            tid, f = str(r["tid"]), int(r["frame"])
        except (KeyError, ValueError):
            continue
        tracks[tid]["boxes"].setdefault(f, None)
        tracks[tid]["side"] = (r.get("side") or "").strip()
    if detect2_path:
        # 缺这个文件时不能静默继续：没有框就没有 IoU，于是每条轨迹都匹配不上，
        # 输出会是「全部污染」—— 一个看起来像系统失败的、自信的错误答案。
        # 第一次跑就栽在这里：文件叫 <rec>.detect2.csv 而代码找 <rec>.csv。
        if not os.path.exists(detect2_path):
            raise SystemExit("detect2 文件不存在：%s\n"
                             "  没有框就算不了 anchor 匹配，结果会全部变成「污染」，"
                             "所以这里直接停，不往下算。" % detect2_path)
        for r in csv.DictReader(open(detect2_path, encoding="utf-8-sig")):
            try:
                tid, f = str(r["tid"]), int(r["frame"])
                box = (float(r["x0"]), float(r["y0"]),
                       float(r["x1"]), float(r["y1"]))
            except (KeyError, ValueError):
                continue
            if tid in tracks and f in tracks[tid]["boxes"]:
                tracks[tid]["boxes"][f] = box
    return dict(tracks)


def headline(rows, thresholds=(0.5, 0.8, 0.9, 0.95)):
    """-> {threshold: fraction of hands held by ONE track at >= that coverage}

    A curve, not a point: this project has twice drawn a conclusion from one
    operating point that a curve then contradicted.

    从未出现过的手不进分母。153159 整条录像只有一只手露过面，另一只的
    visible_frames 是 0、coverage 是 nan，而 `nan >= x` 是 False —— 不排掉
    的话它会被记成一次跟踪失败，于是一条「唯一那只手跟得完美」的录像报成
    1/2。没出现的手不是跟丢的手，它在这条录像里根本不是一只手。
    """
    live = [r for r in rows if r["visible_frames"] > 0]
    out = {"n_hands": len(live), "n_never_visible": len(rows) - len(live)}
    for x in thresholds:
        ok = sum(1 for r in live if r["n_tids"] == 1 and r["coverage"] >= x)
        out[x] = (ok / len(live)) if live else float("nan")
    return out


def evaluate(gold, decisions_path, detect2_path, mode, min_iou):
    tracks = read_tracks(decisions_path, detect2_path)
    hands = gold["hands"]
    if mode == "side":
        assign, diag = assign_by_side(hands, tracks)
    else:
        assign, diag = assign_by_anchors(hands, tracks, min_iou)
    samp = gold.get("sampled_frames")
    rows = per_hand(hands, tracks, assign, samp)
    pur = per_track_purity(hands, tracks, assign)
    return {"recording": gold["recording"], "hands": rows,
            "at_sampled_only": samp is not None,
            "n_sampled": len(samp) if samp else None,
            "n_tracks": len(tracks),
            "n_unassigned": len(diag.get("unassigned", [])),
            "unassigned": diag.get("unassigned", []),
            "n_blind_no_anchor": len(diag.get("blind_no_anchor", [])),
            "blind_no_anchor": diag.get("blind_no_anchor", []),
            "n_contaminating": len(diag.get("contaminating", [])),
            "ties": diag.get("ties", []),
            "purity_median": (statistics.median(pur.values()) if pur
                              else float("nan")),
            "warning": diag.get("warning")}


def selftest():
    def T(spans, box=(0, 0, 10, 10), side="right"):
        return {"boxes": {f: box for s, e in spans for f in range(s, e + 1)},
                "side": side}

    # 按 schema：每个区间两端各一个 anchor。第一版自检只给了 frame 0 一个，
    # 于是帧 5-9 的碎片没机会被匹配 —— 自检因此失败，而失败是对的：
    # 它说明 anchor 太稀会让碎片对指标不可见。
    A = {"hand_key": "A", "intervals": [[0, 9]],
         "anchors": [{"frame": 0, "box": [0, 0, 10, 10]},
                     {"frame": 9, "box": [0, 0, 10, 10]}]}
    B = {"hand_key": "B", "intervals": [[0, 9]],
         "anchors": [{"frame": 0, "box": [100, 100, 110, 110]},
                     {"frame": 9, "box": [100, 100, 110, 110]}]}
    hands = [A, B]
    far = (100, 100, 110, 110)

    cases = [
        ("完美", {"t1": T([(0, 9)]), "t2": T([(0, 9)], far)},
         {"A_tids": 1, "A_cov": 1.0, "A_sw": 0}),
        ("A 切两段", {"t1": T([(0, 4)]), "t1b": T([(5, 9)]),
                   "t2": T([(0, 9)], far)},
         {"A_tids": 2, "A_cov": 1.0, "A_sw": 1}),
        ("A 覆盖一半", {"t1": T([(0, 4)]), "t2": T([(0, 9)], far)},
         {"A_tids": 1, "A_cov": 0.5, "A_sw": 0}),
        ("A 无轨迹", {"t2": T([(0, 9)], far)},
         {"A_tids": 0, "A_cov": 0.0, "A_sw": 0}),
    ]
    ok = True
    print("=== 自检（gold 还不存在时验证指标代码）===")
    for name, tr, want in cases:
        assign, _ = assign_by_anchors(hands, tr)
        rows = per_hand(hands, tr, assign)
        a = next(r for r in rows if r["hand_key"] == "A")
        got = {"A_tids": a["n_tids"], "A_cov": round(a["coverage"], 3),
               "A_sw": a["switches"]}
        good = all(abs(got[k] - want[k]) < 1e-6 for k in want)
        ok &= good
        print("  [%s] %-12s 期望 %s  得到 %s"
              % ("通过" if good else "失败", name, want, got))

    hd = headline([{"n_tids": 1, "coverage": 1.0, "visible_frames": 10},
                   {"n_tids": 2, "coverage": 1.0, "visible_frames": 10},
                   {"n_tids": 1, "coverage": 0.6, "visible_frames": 10}])
    want_hd = {0.5: 2 / 3, 0.8: 1 / 3, 0.9: 1 / 3, 0.95: 1 / 3}
    good = all(abs(hd[k] - want_hd[k]) < 1e-9 for k in want_hd)
    ok &= good
    print("  [%s] %-12s 期望 %s  得到 %s"
          % ("通过" if good else "失败", "主指标曲线",
             {k: round(v, 3) for k, v in want_hd.items()},
             {k: round(v, 3) for k, v in hd.items()}))

    # 从未出现的手不进分母 —— 153159 只有一只手露面，另一只 coverage 是 nan。
    # 不排掉的话 `nan >= x` 为假，会把「唯一那只手跟得完美」报成 1/2。
    hd2 = headline([{"n_tids": 1, "coverage": 1.0, "visible_frames": 24},
                    {"n_tids": 0, "coverage": float("nan"),
                     "visible_frames": 0}])
    good = (all(abs(hd2[k] - 1.0) < 1e-9 for k in (0.5, 0.8, 0.9, 0.95))
            and hd2["n_hands"] == 1 and hd2["n_never_visible"] == 1)
    ok &= good
    print("  [%s] %-12s 期望 1/1 且排掉 1 只  得到 %.3f@0.95 n=%d 排掉 %d"
          % ("通过" if good else "失败", "没露面的手",
             hd2[0.95], hd2["n_hands"], hd2["n_never_visible"]))

    # 污染：盖住了 anchor 帧但位置不对 -> unassigned 且算污染，不算盲区
    tr = {"t9": T([(0, 9)], (500, 500, 510, 510))}
    assign, diag = assign_by_anchors(hands, tr)
    good = (assign["t9"] is None and diag["contaminating"] == ["t9"]
            and diag["blind_no_anchor"] == [])
    ok &= good
    print("  [%s] %-12s 期望 污染，得到 污染=%s 盲区=%s"
          % ("通过" if good else "失败", "污染轨迹",
             diag["contaminating"], diag["blind_no_anchor"]))

    # 盲区：位置对但一个 anchor 帧都没盖住 -> 必须记成盲区，不算污染
    tr = {"t5": T([(4, 6)])}
    assign, diag = assign_by_anchors(hands, tr)
    good = (assign["t5"] is None and diag["blind_no_anchor"] == ["t5"]
            and diag["contaminating"] == [])
    ok &= good
    print("  [%s] %-12s 期望 盲区，得到 污染=%s 盲区=%s"
          % ("通过" if good else "失败", "anchor 盲区",
             diag["contaminating"], diag["blind_no_anchor"]))

    print("\n  %s" % ("全部通过" if ok else "** 有失败，指标代码不可用 **"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", help="hand_gold_v1 的 json，可 glob")
    ap.add_argument("--decisions", help="decisions csv 的目录")
    ap.add_argument("--detect2", help="detect2 csv 的目录（anchor 分配需要框）")
    ap.add_argument("--assign", choices=["anchors", "side"], default="anchors")
    ap.add_argument("--min-iou", dest="min_iou", type=float, default=0.3)
    ap.add_argument("--schema", action="store_true", help="打印 gold schema 后退出")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.schema:
        print(GOLD_SCHEMA)
        return 0
    if a.selftest:
        return selftest()
    if not (a.gold and a.decisions):
        raise SystemExit("要 --gold 与 --decisions；或者 --schema / --selftest")

    rows = []
    for gp in sorted(glob.glob(a.gold)):
        gold = json.load(open(gp, encoding="utf-8"))
        rec = gold["recording"]
        dp = os.path.join(a.decisions, "%s.decisions.csv" % rec)
        if not os.path.exists(dp):
            print("  跳过 %s：没有 %s" % (rec, dp))
            continue
        d2 = None
        if a.detect2:
            for nm in ("%s.csv" % rec, "%s.detect2.csv" % rec):
                cand = os.path.join(a.detect2, nm)
                if os.path.exists(cand):
                    d2 = cand
                    break
            d2 = d2 or os.path.join(a.detect2, "%s.csv" % rec)   # 让它去报错
        rows.append(evaluate(gold, dp, d2, a.assign, a.min_iou))
    if not rows:
        raise SystemExit("没有可评的录像")

    warn = next((r["warning"] for r in rows if r.get("warning")), None)
    if warn:
        print("！%s\n" % warn)

    print("=== 逐手 ===")
    print("  %-22s %4s %8s %7s %7s %7s"
          % ("录像", "手", "可见帧", "轨迹数", "覆盖", "切换"))
    flat = []
    for r in rows:
        for h in r["hands"]:
            flat.append(h)
            print("  %-22s %4s %8d %7d %6.1f%% %7d"
                  % (r["recording"][:22], h["hand_key"], h["visible_frames"],
                     h["n_tids"], 100 * h["coverage"], h["switches"]))

    hd = headline(flat)
    live = [h for h in flat if h["visible_frames"] > 0]
    print("\n=== 主指标：被单一轨迹覆盖了可见帧 >=X 的手，占露过面的手的比例 ===")
    for x in sorted(k for k in hd if isinstance(k, float)):
        print("  >=%3.0f%%   %5.1f%%   (%d / %d 只手)"
              % (100 * x, 100 * hd[x], round(hd[x] * hd["n_hands"]),
                 hd["n_hands"]))
    if hd["n_never_visible"]:
        print("  （另有 %d 只手整条录像没露过面，不进分母 —— 没出现的手不是跟丢的手）"
              % hd["n_never_visible"])
    print("\n  轨迹数中位 %.1f（理想 1）；切换中位 %.1f（理想 0）"
          % (statistics.median(h["n_tids"] for h in live),
             statistics.median(h["switches"] for h in live)))
    blind = sum(r["n_blind_no_anchor"] for r in rows)
    cont = sum(r["n_contaminating"] for r in rows)
    print("  分配不到任何一只手的 own 轨迹：%d = 污染 %d + anchor 盲区 %d"
          % (blind + cont, cont, blind))
    print("    盲区 = 这条轨迹一个 anchor 帧都没盖住，所以没机会被匹配。"
          "它不是系统的错，是 anchor 太稀 —— 盲区非零就说明主指标低估了碎片化。")

    if a.out:
        json.dump({"assign": a.assign, "min_iou": a.min_iou,
                   "headline": {str(k): v for k, v in hd.items()},
                   "recordings": rows},
                  open(a.out, "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
