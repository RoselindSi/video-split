"""把 Phase 2 的合并决策应用下去，量它对「每只手由几条轨迹持有」的实际影响。

为什么必须单独量这个。pairwise 召回不是目标函数，因为**合并有传递性**：
手 A 若碎成 t1/t4/t6，从 t1 的末尾看过去 t4 和 t6 可能都是候选，从 t4 的末尾
看过去 t6 也是候选 —— 但并了 t1-t4 与 t4-t6 之后，t1-t6 自动成立。它仍然会被
记成一个「漏掉的正例」，而实际上没有任何东西漏掉。所以 57 个正例里有冗余，
54.4% 的 pairwise 召回既不等于「修好了 54.4% 的手」，剩下的 45.6% 也不等于
45.6% 的工作没做。

反过来也成立：一只手只要有**一处**该并没并，它就仍然是多条轨迹。所以 pairwise
召回高了未必够，低了也未必不够。唯一说得上话的是碎片数本身。

留一录像：给第 i 条录像打分的模型只在另外 9 条上训练，阈值也只在那 9 条上挑，
与 Phase 1/2 的规则一致。合并沿用 `track_join.resolve` 的时间重叠硬否
（一只手不可能同时在两处），所以这里只换「哪些对该并」，不换合并的语义。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.ambiguity_phase2 import (fit_logreg, pair_appearance,  # noqa: E402
                                      score_logreg)
from semhand.hand_level_metrics import assign_by_anchors, frames_of  # noqa: E402
from semhand.join_window_sweep import (frags_from_tracks,  # noqa: E402
                                       read_pre_join)
from rig.track_join import (DIST_FACTOR, WINDOW, forced_pairs,  # noqa: E402
                            join_to_fixed_point, resolve)

FEATS = ["s_plain", "cos_mean"]


def loro_merges(rows):
    """-> {(rec, a, b)}，每条录像用「只在另外 9 条上训练的模型 + 阈值」判。"""
    recs = sorted({r["recording"] for r in rows})
    keep = set()
    for held in recs:
        tr = [r for r in rows if r["recording"] != held]
        te = [r for r in rows if r["recording"] == held]
        model = fit_logreg([[r[f] for f in FEATS] for r in tr],
                           [r["label"] for r in tr])
        s_tr = score_logreg([[r[f] for f in FEATS] for r in tr], model)
        v = sorted(set(s_tr.tolist()))
        cand = ([max(v) + 0.01] + [(x + y) / 2 for x, y in zip(v, v[1:])]
                + [min(v) - 0.01])
        best, best_tp = None, -1
        for t in cand:
            tp = fu = 0
            for r, s in zip(tr, s_tr):
                if s >= t:
                    if r["label"] == 1:
                        tp += 1
                    elif r["kind"] != "other_own_hand":
                        fu += 1
            if fu == 0 and tp > best_tp:
                best, best_tp = t, tp
        if best is None:
            best = cand[0]
        s_te = score_logreg([[r[f] for f in FEATS] for r in te], model)
        for r, s in zip(te, s_te):
            if s >= best:
                keep.add((r["recording"], r["a"], r["b"]))
    return keep


def hand_tracks(gold, frag, assign, mapping, sampled):
    """-> [(hand_key, 可见帧数, 持有它的组数)]，组 = 合并之后的身份。"""
    out = []
    samp = set(sampled)
    for h in gold["hands"]:
        H = frames_of(h["intervals"]) & samp
        if not H:
            continue
        groups = set()
        for tid, k in assign.items():
            if k != h["hand_key"] or tid not in frag:
                continue
            if set(range(frag[tid]["first"], frag[tid]["last"] + 1)) & H:
                groups.add(mapping.get(tid, tid))
        out.append((h["hand_key"], len(H), len(groups)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--gold", required=True, help="glob")
    ap.add_argument("--detect2-pre", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    import glob
    emb = json.load(open(a.embeds, encoding="utf-8"))
    rows = []
    for r in json.load(open(a.pairs, encoding="utf-8")):
        if r["label"] is None:
            continue
        ap_ = pair_appearance(emb, r["recording"], r["a"], r["b"])
        if ap_ is None:
            continue
        r["cos_mean"], r["cos_junc"] = ap_
        rows.append(r)
    keep = loro_merges(rows)
    print("留一录像判出的合并：%d 对（其中真是同一只手的 %d 对）"
          % (len(keep), sum(1 for r in rows
                            if (r["recording"], r["a"], r["b"]) in keep
                            and r["label"] == 1)))

    print("\n  %-20s %3s %7s %9s %9s  %s"
          % ("录像", "手", "可见帧", "现状组数", "合并后组数", "变化"))
    before_one = after_one = total = 0
    rec_rows = []
    for gp in sorted(glob.glob(a.gold)):
        gold = json.load(open(gp, encoding="utf-8"))
        rec = gold["recording"]
        pre = read_pre_join(os.path.join(a.detect2_pre,
                                         "%s.detect2.csv" % rec))
        assign, _ = assign_by_anchors(gold["hands"], pre)
        frag = frags_from_tracks(pre)
        # 基准必须是**部署状态** —— 已经过 track_join（冻结窗口 150）。
        # 拿「完全不并」当基准会把 track_join 自己的功劳算到消解器头上。
        base, _ = join_to_fixed_point(frag, window=WINDOW,
                                      dist_factor=DIST_FACTOR)
        mine = [(x, y) for (r_, x, y) in keep if r_ == rec
                and x in frag and y in frag]
        # 必须**叠在 base 之上**，不能重新 resolve 一遍。base 是迭代到不动点的
        # 结果，而 `resolve(frag, forced_pairs(...))` 只有一轮，会比 base 并得少 ——
        # 两者不嵌套，于是「合并之后」能比「合并之前」组数还多，而合并在数学上
        # 不可能增加组数。第一版就是这么报出三只手「变差」的。
        group_frag, members = {}, collections.defaultdict(list)
        for t in frag:
            members[base[t]].append(t)
        for g, ms in members.items():
            fs = [frag[t] for t in ms]
            group_frag[g] = {"first": min(f["first"] for f in fs),
                             "last": max(f["last"] for f in fs),
                             "start_box": min(fs, key=lambda f: f["first"])["start_box"],
                             "end_box": max(fs, key=lambda f: f["last"])["end_box"],
                             "n": sum(f["n"] for f in fs)}
        extra = {(base[x], base[y]) for x, y in mine if base[x] != base[y]}
        gmap = resolve(group_frag, sorted(extra))
        merged = {t: gmap.get(base[t], base[t]) for t in frag}
        b = hand_tracks(gold, frag, assign, base, gold["sampled_frames"])
        m = hand_tracks(gold, frag, assign, merged, gold["sampled_frames"])
        for (k1, n1, g1), (k2, _, g2) in zip(b, m):
            total += 1
            before_one += g1 == 1
            after_one += g2 == 1
            mark = "" if g1 == g2 else ("  %d -> %d" % (g1, g2))
            print("  %-20s %3s %7d %9d %9d  %s"
                  % (rec[:20], k1, n1, g1, g2, mark))
            rec_rows.append({"recording": rec, "hand": k1, "visible": n1,
                             "groups_before": g1, "groups_after": g2})
    print("\n  单一轨迹持有的手：%d / %d -> %d / %d"
          % (before_one, total, after_one, total))
    print("  基准 = 部署状态（track_join 冻结窗口 %d）；之后 = 在它之上叠加消解器的合并。"
          % WINDOW)

    if a.out:
        json.dump({"n_merged": len(keep), "hands": rec_rows,
                   "single_before": before_one, "single_after": after_one,
                   "total": total},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
