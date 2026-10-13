"""Phase 0：把「候选不唯一」的决策点展开成带标签的 pair benchmark。

为什么单位是 (端点, 候选) 而不是「歧义事件」。taxonomy 数出的 24 处是
**本该接上却因歧义没接**，它们按定义全是正例 —— 拿它们当数据集，模型只会学到
「附近两个框就该并」。负例在候选集里：端点 a 面对 k 个候选，其中**至多一个**
是同一只手，其余 k−1 个是负例，而且是最难的那种负例（时间空间都说得通，
否则它们根本进不了候选集）。

负例还要分开数，因为它们的代价完全不同：
  另一只自己的手  —— 并错了，两只手被当成一只
  别人的手 / 非手 —— 并错了，**整条轨迹的 ownership 被污染**，这是最贵的

正例的定义严格跟着 gold：a 和 b 都被 anchor 匹配到**同一个 hand_key**。
assign 为 None 的轨迹不是负例也不是正例的对立面 —— 它可能是别人的手、
可能是非手、也可能只是没盖住任何 anchor 帧（盲区）。盲区必须单列，
把它当负例会把「我 anchor 太稀」记成「模型该说不」。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.hand_level_metrics import assign_by_anchors  # noqa: E402
from semhand.join_window_sweep import (frags_from_tracks,  # noqa: E402
                                       read_pre_join)
from rig.track_join import DIST_FACTOR, WINDOW, centre, width  # noqa: E402


def decision_points(frag, window, dist_factor):
    """-> [(a, [candidates])]，只留候选数 >= 2 的端点（唯一候选不是歧义）。"""
    order = sorted(frag, key=lambda t: frag[t]["first"])
    out = []
    for a in order:
        fa = frag[a]
        cx, cy = centre(fa["end_box"])
        w = width(fa["end_box"])
        cands = []
        for b in order:
            if b == a:
                continue
            fb = frag[b]
            gap = fb["first"] - fa["last"]
            if gap <= 0 or gap > window:
                continue
            bx, by = centre(fb["start_box"])
            if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 <= dist_factor * w:
                cands.append(b)
        if len(cands) >= 2:
            out.append((a, cands))
    return out


def motion_predict(fa, gap):
    """A 末尾速度外推 gap 帧后的中心。用最后两帧估速度，估不出就当静止。"""
    cx, cy = centre(fa["end_box"])
    v = fa.get("end_vel")
    if v is None:
        return cx, cy
    return cx + v[0] * gap, cy + v[1] * gap


def with_velocity(tracks, frag):
    """给每个碎片补上首尾速度 —— 外推要用，而 fragments() 不带。"""
    for tid, f in frag.items():
        pts = sorted(tracks[tid]["boxes"].items())
        if len(pts) >= 2:
            (f1, b1), (f0, b0) = pts[-1], pts[-2]
            dt = max(1, f1 - f0)
            c1, c0 = centre(b1), centre(b0)
            f["end_vel"] = ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)
        else:
            f["end_vel"] = None
    return frag


def build(gold_path, pre_path, window, dist_factor, min_iou=0.3):
    gold = json.load(open(gold_path, encoding="utf-8"))
    pre = read_pre_join(pre_path)
    assign, diag = assign_by_anchors(gold["hands"], pre, min_iou=min_iou)
    blind = set(diag["blind_no_anchor"])
    frag = with_velocity(pre, frags_from_tracks(pre))

    rows = []
    for a, cands in decision_points(frag, window, dist_factor):
        ha = assign.get(a)
        if ha is None:
            continue          # 起点身份未知，这个决策点无法判对错
        for b in cands:
            hb = assign.get(b)
            if hb == ha:
                label, kind = 1, "same_hand"
            elif hb is not None:
                label, kind = 0, "other_own_hand"
            elif b in blind:
                label, kind = None, "blind_no_anchor"
            else:
                label, kind = 0, "unassigned"
            fa, fb = frag[a], frag[b]
            gap = fb["first"] - fa["last"]
            px, py = motion_predict(fa, gap)
            bx, by = centre(fb["start_box"])
            cx, cy = centre(fa["end_box"])
            wa, wb = width(fa["end_box"]), width(fb["start_box"])
            rows.append({
                "recording": gold["recording"], "a": a, "b": b,
                "hand_a": ha, "label": label, "kind": kind,
                "gap": gap,
                "d_plain": round(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5, 1),
                "d_motion": round(((bx - px) ** 2 + (by - py) ** 2) ** 0.5, 1),
                # 归一化必须留在产物里：距离门是 d <= 3.0*width_a，是**相对**框宽
                # 的阈值，所以框大的录像允许更大的绝对距离。直接比绝对距离，
                # 「哪条录像正例多」这个构成差异会被当成区分力 —— pooled AUC
                # 0.879 高过决策点内 0.821 就是这么来的，是录像间捷径。
                "width_a": round(wa, 1),
                "s_plain": round(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / wa, 3),
                "s_motion": round(((bx - px) ** 2 + (by - py) ** 2) ** 0.5 / wa, 3),
                "log_scale": round(__import__("math").log(wb / wa), 3),
                "n_cands": len(cands),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="glob")
    ap.add_argument("--detect2-pre", required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--dist-factor", type=float, default=DIST_FACTOR)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for g in sorted(glob.glob(a.gold)):
        rec = json.load(open(g, encoding="utf-8"))["recording"]
        rows += build(g, os.path.join(a.detect2_pre, "%s.detect2.csv" % rec),
                      a.window, a.dist_factor)

    kinds = collections.Counter(r["kind"] for r in rows)
    pts = {(r["recording"], r["a"]) for r in rows}
    print("=== Phase 0：歧义决策点展开（窗口 %d）===" % a.window)
    print("  决策点 %d 个（候选 >= 2 且起点身份已知），展开成 %d 个 (端点, 候选) 对"
          % (len(pts), len(rows)))
    print()
    print("  %-18s %6s  %s" % ("候选是什么", "对数", "作为标签"))
    for k, desc, lab in [
            ("same_hand", "同一只手 —— 本该并", "正例"),
            ("other_own_hand", "自己的另一只手", "负例"),
            ("unassigned", "没匹配上任何一只手（别人的手 / 非手）", "负例"),
            ("blind_no_anchor", "一个 anchor 帧都没盖住 —— 盲区", "**不可判，排除**")]:
        print("  %-18s %6d  %-12s %s" % (k, kinds[k], lab, desc))
    usable = [r for r in rows if r["label"] is not None]
    pos = [r for r in usable if r["label"] == 1]
    print()
    print("  可用 %d 对：正 %d / 负 %d（正例率 %.1f%%）"
          % (len(usable), len(pos), len(usable) - len(pos),
             100 * len(pos) / len(usable) if usable else 0))
    if kinds["blind_no_anchor"]:
        print("  排除的盲区 %d 对 —— 把它们当负例等于把 anchor 太稀记成「模型该说不」"
              % kinds["blind_no_anchor"])

    # 每个决策点里有几个正例 —— 不该超过 1，超过就说明 gold 里一只手在同一端
    # 点有两个后继，那是 gold 或碎片结构的问题，要先看
    bypt = collections.defaultdict(int)
    for r in pos:
        bypt[(r["recording"], r["a"])] += 1
    multi = {k: v for k, v in bypt.items() if v > 1}
    nopos = len(pts) - len(bypt)
    print("  决策点里有正例的 %d 个，一个正例都没有的 %d 个"
          % (len(bypt), nopos))
    if multi:
        print("  ！%d 个决策点有多于一个正例：%s" % (len(multi), sorted(multi)[:5]))
    print("    「没有正例」= 这个端点的真后继根本不在候选集里（被窗口或距离挡掉），")
    print("    所以它不是歧义消解能救的 —— 消解器只能在候选集里挑。")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
