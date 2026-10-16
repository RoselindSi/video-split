"""主指标：track_join 实际把多少非目标框接进了目标手轨迹。

为什么要单独算这个。batch3 收口时只能停在「候选数」上：443 个别人的手，
按 gold 的手部 intervals 做时间重叠硬否，剩 19 个有资格进入距离判断。那是
**下界**——gold intervals 是最粗最长的 span，而 tracker 把佩戴者的手切成碎片，
span 一短，重叠硬否就拦得少。真值只能在 join 真正看到的输入上量：join 前的
detect2，按它自己的 tid 重建碎片，再原样跑 `join_to_fixed_point`。

单位跟 19 那次保持一致：**gold 里标出的非目标框**（`others`，抽样格上的框）。
每个框先找它在 detect2 里是哪条碎片（同帧 IoU >= 0.3），然后沿 `forced_pairs`
的四条判据逐级数：

  C1  与某条目标碎片时间不重叠（gap > 0，任一方向）   <- 19 那次只用了这一条
  C2  且 gap <= 150 帧
  C3  且距离：a 末框中心到 b 首框中心 <= 3.0 x width(a 末框)
  C4  且该端点恰好这一个候选（forced pair）
  并  迭代到不动点后，真的和目标碎片落在同一组

C1-C4 在**每一轮**的当前组上判、取并集，因为 join 是迭代的：碎片并成组之后，
组的跨度变长，下一轮的时间重叠硬否就按组来拦（`resolve` 里也有同样的组级硬否）。

**只有 track_join 造成的接入算进主指标。** 一个非目标框如果在 detect2 里本来就
和目标的手同一个 tid（tracker 自己混进来的），那不是 join 的错，单列为
「tracker 自身混入」——对训练数据一样有害，但修的地方不同。

目标轨迹 = 合并后含有至少一条目标碎片（assign 到 A 或 B）的组。受污染 = 组里还有
被判成 other_hand / not_hand / unsure 的碎片。组里若混着**判不了**的碎片
（盲区：没有一个框落在抽样格上；或落在格上却谁都没匹配上），单列成区间的上界，
不当成干净也不当成污染。A 与 B 被并进同一组单列为「自己两只手互换」，那不是
非目标框，但同样是身份错误。

**两臂都能量，输入决定是哪一臂。** 给 stage 1 的产物（`not_hand` 全 0）就是「未前置过滤」
那一臂。给 stage 3 的产物（先用 `--handness_boxes` 在 tracker 之前摘掉非手、再重跟踪），
同时用 `--scores` 给出那一遍用的逐框 hand-ness，就是部署顺序那一臂
（docs/track_join_gold_validation.md：在 fresh20 上，未过滤 15 组跨类别合并、过滤后 5 组）。
有了 `--scores`，在 detect2 里找不到碎片的 gold 框，会被分成「hand-ness 在 tracker 前摘掉」
和「本来就没检出」。佩戴者自己的手被摘掉多少，同样按 anchor 数报出来，那是过滤的代价。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.ambiguity_benchmark import lookup_kind, others_lookup  # noqa: E402
from semhand.hand_level_metrics import assign_by_anchors, iou  # noqa: E402
from semhand.join_window_sweep import frags_from_tracks, read_pre_join  # noqa: E402
from rig.track_join import (DIST_FACTOR, WINDOW, centre,  # noqa: E402
                            forced_pairs, join_to_fixed_point, resolve, width)

TARGET = ("A", "B")
NONTARGET = ("other_hand", "not_hand", "unsure")
UNKNOWN = ("blind", "unmatched", "tie")
MISSING = ("undetected", "handness_filtered", "detected_untracked", "absent_unscored")
STAGES = ("detected", "c1", "c2", "c3", "c4", "joined")


def label_fragments(gold, pre, min_iou):
    """-> {tid: 'A'|'B'|'other_hand'|'not_hand'|'unsure'|'blind'|'unmatched'|'tie'}。"""
    assign, diag = assign_by_anchors(gold["hands"], pre, min_iou=min_iou)
    oth = others_lookup(gold)
    complete = set(gold.get("others_complete_frames") or [])
    blind, ties = set(diag["blind_no_anchor"]), set(diag["ties"])
    sampled = set(gold.get("sampled_frames") or [])
    out = {}
    for tid, t in pre.items():
        if assign.get(tid) is not None:
            out[tid] = assign[tid]
        elif tid in ties:
            out[tid] = "tie"
        else:
            k = lookup_kind(t, oth, complete, min_iou)
            if k:
                out[tid] = k
            elif tid in blind and not (set(t["boxes"]) & sampled):
                # 一个框都没落在抽样格上：不是「匹配不上」，是「没机会被看见」
                out[tid] = "blind"
            else:
                out[tid] = "unmatched"
    return out


def candidate_relations(cur, window, dist_factor):
    """-> {(a, b): 最高通过到哪一级}，a 在前 b 在后，按 forced_pairs 的判据逐级记。

    与 `track_join.forced_pairs` 同一套判据，只是把被拦下的也记下来、并记到
    哪一级为止 —— forced_pairs 只返回最后幸存的那一对。
    """
    rel = {}
    cands = collections.defaultdict(list)
    order = sorted(cur, key=lambda t: cur[t]["first"])
    for a in order:
        fa = cur[a]
        cx, cy = centre(fa["end_box"])
        w = width(fa["end_box"])
        for b in order:
            if b == a:
                continue
            fb = cur[b]
            gap = fb["first"] - fa["last"]
            if gap <= 0:
                continue
            rel[(a, b)] = "c1"
            if gap > window:
                continue
            rel[(a, b)] = "c2"
            bx, by = centre(fb["start_box"])
            if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 <= dist_factor * w:
                rel[(a, b)] = "c3"
                cands[a].append(b)
    for a, bs in cands.items():
        if len(bs) == 1:
            rel[(a, bs[0])] = "c4"
    return rel


def rounds_trace(frag, window, dist_factor, max_rounds=20):
    """复刻 `join_to_fixed_point` 的循环，每一轮记下组间关系。

    -> (每轮 [(组成员表, 关系)], 最终映射)。最终映射另外直接调
    `join_to_fixed_point` 再算一遍并比对，保证这里量的就是部署的那个函数。
    """
    members = {t: {t} for t in frag}
    cur = {t: dict(f) for t, f in frag.items()}
    trace = []
    for _ in range(max_rounds):
        rel = candidate_relations(cur, window, dist_factor)
        trace.append(({g: set(m) for g, m in members.items()}, rel))
        # 合并用部署的那个函数给出的配对、按它的顺序：resolve 逐对处理且带
        # 组级重叠硬否，配对顺序不同，并出来的组就可能不同。
        pairs = forced_pairs(cur, window, dist_factor)
        if set(pairs) != {p for p, s in rel.items() if s == "c4"}:
            raise SystemExit("C4 的记录与 forced_pairs 不一致")
        mapping = resolve(cur, pairs)
        if all(mapping[t] == t for t in cur):
            break
        nxt, nmem = {}, collections.defaultdict(set)
        for t, f in cur.items():
            k = mapping[t]
            nmem[k] |= members[t]
            if k not in nxt:
                nxt[k] = dict(f)
            else:
                g = nxt[k]
                if f["first"] < g["first"]:
                    g["first"], g["start_box"] = f["first"], f["start_box"]
                if f["last"] > g["last"]:
                    g["last"], g["end_box"] = f["last"], f["end_box"]
                g["n"] += f["n"]
        cur, members = nxt, dict(nmem)
    final = {t: g for g, m in members.items() for t in m}
    ref, _ = join_to_fixed_point(frag, window=window, dist_factor=dist_factor)
    same = all((final[a] == final[b]) == (ref[a] == ref[b])
               for a in frag for b in frag)
    if not same:
        raise SystemExit("复刻的循环与 join_to_fixed_point 分组不一致，不能用")
    return trace, ref


def stage_of_fragment(tid, labels, trace):
    """-> 这条碎片和任一目标碎片之间曾经到达的最高判据级（跨轮取最高）。"""
    rank = {s: i for i, s in enumerate(("none",) + STAGES[1:5])}
    best = "none"
    for members, rel in trace:
        mine = next(g for g, m in members.items() if tid in m)
        if any(labels[t] in TARGET for t in members[mine]):
            continue        # 已经和目标同组，再往后是组内的事
        for (a, b), s in rel.items():
            other = b if a == mine else (a if b == mine else None)
            if other is None:
                continue
            if any(labels[t] in TARGET for t in members[other]):
                if rank[s] > rank[best]:
                    best = s
    return best


def match_box(pre, frame, box, min_iou):
    """-> 同帧 IoU 最高且 >= min_iou 的 detect2 碎片 tid，没有就 None。"""
    best, best_iou = None, min_iou
    for tid, t in pre.items():
        b = t["boxes"].get(frame)
        if b is None:
            continue
        v = iou(box, b)
        if v >= best_iou:
            best, best_iou = tid, v
    return best


def read_scores(path, thr):
    """-> {frame: [(box, p)]}，部署那一臂在 tracker 之前用的逐框 hand-ness；没给就是 None。

    这份打分是 detect1 那一遍对**全部原始检测框**打的（blur_check --handness_models），
    在 tracker 之前、在过滤之前，所以它同时也是「检测器到底出没出这个框」的记录。
    """
    if not path or not os.path.exists(path):
        return None
    out = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[int(r["frame"])].append((tuple(float(v) for v in r["box"].split(",")),
                                         float(r["p"])))
    return out


def why_missing(scores, frame, box, min_iou, thr):
    """一个 gold 框在 detect2 里找不到碎片，是在哪一层丢的。

      undetected          检测器根本没出这个框（原始检测里没有 IoU 够的）
      handness_filtered   出了框，但同位置的框 hand-ness 全部 < 门槛，在 tracker 前被摘掉
      detected_untracked  出了框、hand-ness 也过了门槛，却没进任何轨迹
                          —— 丢在 tracker：新建门槛、关联，或两遍检测之间的抖动
      absent_unscored     没有逐框打分，分不出是上面哪一种（stage 1 那一臂）

    第一版只分了前两种，把第三种也记成了「没检出」，于是 batch3 部署臂上
    「94 个漏掉的 anchor 里 89 个是检测器没检出」实际是检测器 58、tracker 33、过滤 3。
    「被摘掉」也从「同位置任一个框 < 门槛」改成了「全部 < 门槛」：只要有一个框过了门槛，
    过滤就会放行它，丢失就不是过滤造成的。
    """
    if scores is None:
        return "absent_unscored"
    ps = [p for b, p in scores.get(frame, []) if iou(box, b) >= min_iou]
    if not ps:
        return "undetected"
    if max(ps) < thr:
        return "handness_filtered"
    return "detected_untracked"


def run_one(gold_path, detect2_dir, window, dist_factor, min_iou,
            scores_dir=None, thr=0.10):
    gold = json.load(open(gold_path, encoding="utf-8"))
    rec = gold["recording"]
    pre = read_pre_join(os.path.join(detect2_dir, "%s.detect2.csv" % rec))
    scores = read_scores(scores_dir and os.path.join(scores_dir, "%s.jsonl" % rec), thr)
    labels = label_fragments(gold, pre, min_iou)
    frag = frags_from_tracks(pre)
    trace, mapping = rounds_trace(frag, window, dist_factor)
    groups = collections.defaultdict(list)
    for t in frag:
        groups[mapping[t]].append(t)
    gid = {t: mapping[t] for t in frag}
    target_groups = {g for g, ms in groups.items()
                     if any(labels[t] in TARGET for t in ms)}

    # ---- 单位一：gold 里的非目标框，逐级漏斗 ----
    boxes = []
    for o in gold.get("others") or []:
        f, box, kind = int(o["frame"]), tuple(o["box"]), o["kind"]
        tid = match_box(pre, f, box, min_iou)
        row = {"recording": rec, "frame": f, "kind": kind, "tid": tid,
               "frag_label": labels.get(tid), "reached": []}
        if tid is None:
            row["fate"] = why_missing(scores, f, box, min_iou, thr)
        elif labels[tid] in TARGET:
            row["fate"] = "tracker_mixed"     # tracker 自己就把它放进了目标碎片
            row["reached"] = ["detected"]
        else:
            s = stage_of_fragment(tid, labels, trace)
            row["reached"] = ["detected"] + list(STAGES[1:STAGES.index(s) + 1]
                                                 if s != "none" else [])
            if gid[tid] in target_groups:
                row["fate"] = "join_into_target"
                row["reached"].append("joined")
            else:
                row["fate"] = "clean"
        boxes.append(row)

    # ---- 单位二：目标轨迹（合并后的组） ----
    tgroups = []
    for g in sorted(target_groups, key=str):
        ms = groups[g]
        lab = collections.Counter(labels[t] for t in ms)
        rows = {t: frag[t]["n"] for t in ms}
        hands = sorted({labels[t] for t in ms if labels[t] in TARGET})
        tgroups.append({
            "recording": rec, "group": g, "hands": hands,
            "fragments": len(ms), "labels": dict(lab),
            "rows_total": sum(rows.values()),
            "rows_nontarget": sum(n for t, n in rows.items() if labels[t] in NONTARGET),
            "rows_unknown": sum(n for t, n in rows.items() if labels[t] in UNKNOWN),
            "nontarget_kinds": sorted({labels[t] for t in ms if labels[t] in NONTARGET}),
            "swap": len(hands) > 1,
        })
    # 合并前：同一批目标碎片各自独立时，有几条已经被 tracker 混进了非目标框
    pre_mixed = 0
    oth = others_lookup(gold)
    for t in frag:
        if labels[t] not in TARGET:
            continue
        for f, b in pre[t]["boxes"].items():
            if any(iou(b, ob) >= min_iou and k in NONTARGET
                   for ob, k in oth.get(f, [])):
                pre_mixed += 1
                break
    n_target_frags = sum(1 for t in frag if labels[t] in TARGET)
    # 过滤的代价：佩戴者自己的手在 detect2 里还在不在。按 anchor 数（抽样格上）。
    anchors = collections.Counter()
    for h in gold["hands"]:
        for an in h.get("anchors") or []:
            f, box = int(an["frame"]), tuple(an["box"])
            anchors["total"] += 1
            if match_box(pre, f, box, min_iou) is not None:
                anchors["kept"] += 1
            else:
                anchors[why_missing(scores, f, box, min_iou, thr)] += 1
    return {"recording": rec, "boxes": boxes, "target_groups": tgroups,
            "target_anchors": dict(anchors),
            "fragments": len(frag), "groups": len(groups),
            "target_fragments": n_target_frags,
            "target_fragments_tracker_mixed": pre_mixed,
            "fragment_labels": dict(collections.Counter(labels[t] for t in frag)),
            "rounds": len(trace)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="glob，v2 gold")
    ap.add_argument("--detect2", required=True, help="join 前的 detect2 目录，<rec>.detect2.csv")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--dist-factor", type=float, default=DIST_FACTOR)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--scores", help="部署那一臂：逐框 hand-ness 的目录，<rec>.jsonl。"
                                     "给了才能把「不在 detect2 里」分成被过滤掉的和本来没检出的")
    ap.add_argument("--handness-thr", type=float, default=0.10,
                    help="与 blur_check._load_handness_boxes 的默认门槛一致")
    ap.add_argument("--out")
    a = ap.parse_args()

    res = [run_one(g, a.detect2, a.window, a.dist_factor, a.min_iou,
                   a.scores, a.handness_thr)
           for g in sorted(glob.glob(a.gold))]
    boxes = [b for r in res for b in r["boxes"]]
    groups = [g for r in res for g in r["target_groups"]]

    print("=== 单位一：gold 里的非目标框（窗口 %d，距离 %.1f x 宽，IoU >= %.1f）==="
          % (a.window, a.dist_factor, a.min_iou))
    names = {"detected": "检测器检出（有碎片）", "c1": "C1 与目标碎片时间不重叠",
             "c2": "C2 且 gap <= %d" % a.window, "c3": "C3 且满足距离条件",
             "c4": "C4 且是唯一候选", "joined": "实际并进目标轨迹"}
    for kind in ("other_hand", "not_hand", "unsure"):
        ks = [b for b in boxes if b["kind"] == kind]
        if not ks:
            continue
        fate = collections.Counter(b["fate"] for b in ks)
        print("\n  %s：%d 个框（%d 条录像）" % (kind, len(ks), len({b["recording"] for b in ks})))
        print("    不在 detect2 里 %d（检测器没出框 %d / hand-ness 在 tracker 前摘掉 %d"
              " / 出了框没成轨迹 %d / 无打分分不出 %d）"
              % (sum(fate[k] for k in MISSING), fate["undetected"], fate["handness_filtered"],
                 fate["detected_untracked"], fate["absent_unscored"]))
        print("    tracker 自身混入目标碎片 %d / 在非目标碎片里 %d"
              % (fate["tracker_mixed"], fate["clean"] + fate["join_into_target"]))
        nt = [b for b in ks if b["fate"] in ("clean", "join_into_target")]
        for s in STAGES[1:]:
            hit = [b for b in nt if s in b["reached"]]
            recs = collections.Counter(b["recording"][-6:] for b in hit)
            print("    %-24s %4d  %s" % (names[s], len(hit),
                                         " ".join("%s:%d" % kv for kv in sorted(recs.items()))))

    print("\n=== 单位二：目标轨迹（合并后含 A 或 B 碎片的组）===")
    n = len(groups)
    cont = [g for g in groups if g["nontarget_kinds"]]
    unk = [g for g in groups if not g["nontarget_kinds"] and g["rows_unknown"]]
    swap = [g for g in groups if g["swap"]]
    print("  目标轨迹 %d 条（14 条录像）" % n)
    print("  被 join 接入非目标碎片的：%d（%.1f%%）  按来源 %s"
          % (len(cont), 100 * len(cont) / max(1, n),
             dict(collections.Counter(k for g in cont for k in g["nontarget_kinds"]))))
    print("  混有判不了的碎片（盲区/未匹配，上界用）：%d" % len(unk))
    print("  自己两只手被并进同一条：%d" % len(swap))
    tot = sum(g["rows_total"] for g in groups)
    bad = sum(g["rows_nontarget"] for g in groups)
    unr = sum(g["rows_unknown"] for g in groups)
    print("  目标轨迹的全部框 %d 行：来自非目标碎片 %d（%.2f%%），来自判不了的碎片 %d（%.2f%%）"
          % (tot, bad, 100 * bad / max(1, tot), unr, 100 * unr / max(1, tot)))
    mixed = sum(r["target_fragments_tracker_mixed"] for r in res)
    tf = sum(r["target_fragments"] for r in res)
    print("  对照（不是 join 的错）：%d / %d 条目标碎片在 join 之前就被 tracker 混进了非目标框"
          % (mixed, tf))
    an = collections.Counter()
    for r in res:
        an.update(r["target_anchors"])
    print("  佩戴者的手（gold anchor）：%d 个，仍在 detect2 里 %d；不在的 %d 个里"
          " 检测器没出框 %d / hand-ness 摘掉 %d / 出了框没成轨迹 %d / 无打分分不出 %d"
          % (an["total"], an["kept"], sum(an[k] for k in MISSING), an["undetected"],
             an["handness_filtered"], an["detected_untracked"], an["absent_unscored"]))

    print("\n  %-16s %5s %5s %5s %6s %6s %6s" % ("录像", "碎片", "合并后", "目标组", "受污染", "判不了", "互换"))
    for r in res:
        gs = r["target_groups"]
        print("  %-16s %5d %5d %5d %6d %6d %6d"
              % (r["recording"], r["fragments"], r["groups"], len(gs),
                 sum(1 for g in gs if g["nontarget_kinds"]),
                 sum(1 for g in gs if not g["nontarget_kinds"] and g["rows_unknown"]),
                 sum(1 for g in gs if g["swap"])))

    if a.out:
        json.dump({"window": a.window, "dist_factor": a.dist_factor, "min_iou": a.min_iou,
                   "recordings": res},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
