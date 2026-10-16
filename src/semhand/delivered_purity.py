"""交付的 owner 手里，有多少不是佩戴者的手 —— 不管错框是从哪条路进来的。

为什么在 join 污染之后还要量这个。`join_contamination` 只回答了一条路：track_join 把
多少非目标框接进了目标轨迹（batch3 是 3 起、交付 55 帧）。但对机械臂训练数据要紧的是
**交付出去的 owner 框里有多少错框**，join 只是三条来路之一：

  join       非目标碎片被并进一条含目标碎片的组，随组一起判成 owner
  tracker    tracker 自己把非目标框放进了目标碎片（join 之前就混了）
  ownership  一条整条都是非目标的轨迹，被 post_pass 直接判成 owner，没经过任何合并

不知道三者的比例，就没法决定该修 join、tracker 还是归属判定。

**单位是 gold 抽样格上的交付框。** 抽样格每 10 帧一格，是系统抽样，所以格上的比例是
整段的无偏估计，不需要拿碎片多数票往没标的帧上外推。每个交付框（post_pass own=1）按同帧
IoU >= 阈值去对 gold：先对佩戴者的 A/B anchor，再对 `others`；都对不上的记为「没标到」，
单列，不当对也不当错。

**join 的净影响用同一份运行的两次 post_pass 量**：`decisions_before` 吃 join 之前的
detect2，`decisions` 吃 join 之后的。两者只差 join 这一步，所以差值就是 join 对交付的净
作用 —— 带进来的错框，和把碎片连起来以后多交付（或少交付）的真手，都在里面。

**召回**是另一面：gold 的 anchor（佩戴者的手）在同帧有没有被一个交付框盖住。没盖住的
先分成「有框但判成了 own=0」和「detect2 里就没有这个框」，后者再用 detect1 那一遍对全部原始
检测框打的分拆成三层（`join_contamination.why_missing`）：检测器没出框 / 出了框但 hand-ness
在 tracker 前摘掉 / 出了框也过了门槛却没成轨迹。最后一层丢在 tracker，不是检测器。

置信区间按录像聚类 bootstrap：同一条录像里的帧高度相关，按帧算会把区间算窄。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.hand_level_metrics import iou  # noqa: E402
from semhand.join_contamination import (MISSING, NONTARGET, TARGET,  # noqa: E402
                                        label_fragments, read_scores, why_missing)
from semhand.join_window_sweep import read_pre_join  # noqa: E402


def read_decisions(path):
    """-> {(frame, tid): own}"""
    return {(int(r["frame"]), r["tid"]): int(r["own"])
            for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def gold_evidence(gold):
    """-> {frame: [(box, label)]}，A/B anchor 和 others（只取判全了的帧）。"""
    ev = collections.defaultdict(list)
    for h in gold["hands"]:
        for an in h.get("anchors") or []:
            ev[int(an["frame"])].append((tuple(an["box"]), h["hand_key"]))
    complete = set(gold.get("others_complete_frames") or [])
    for o in gold.get("others") or []:
        if int(o["frame"]) in complete:
            ev[int(o["frame"])].append((tuple(o["box"]), o["kind"]))
    return ev


def box_label(ev, frame, box, min_iou):
    """-> 同帧 IoU 最高且 >= min_iou 的 gold 标签，没有就 'unlabelled'。"""
    best, lab = min_iou, "unlabelled"
    for b, k in ev.get(frame, []):
        v = iou(box, b)
        if v >= best:
            best, lab = v, k
    return lab


def run_one(gold_path, arm, min_iou, thr=0.10):
    gold = json.load(open(gold_path, encoding="utf-8"))
    rec = gold["recording"]
    scores = read_scores(os.path.join(arm, "scores", rec + ".jsonl"), thr)
    d2_path = os.path.join(arm, "detect2", rec + ".detect2.csv")
    rows = list(csv.DictReader(open(d2_path, encoding="utf-8-sig")))
    pre = read_pre_join(d2_path)
    frag_label = label_fragments(gold, pre, min_iou)
    mapping = json.load(open(os.path.join(arm, "joined_mapping", rec + ".mapping.json")))["mapping"]
    dec_after = read_decisions(os.path.join(arm, "decisions", rec + ".decisions.csv"))
    dec_before = read_decisions(os.path.join(arm, "decisions_before", rec + ".decisions.csv"))
    group_has_target = collections.defaultdict(bool)
    for t, g in mapping.items():
        if frag_label.get(t) in TARGET:
            group_has_target[g] = True

    ev = gold_evidence(gold)
    sampled = set(int(f) for f in gold.get("sampled_frames") or [])
    out = {"recording": rec, "before": collections.Counter(), "after": collections.Counter(),
           "wrong": []}
    delivered = {"before": collections.defaultdict(list), "after": collections.defaultdict(list)}
    present = collections.defaultdict(list)
    for r in rows:
        f = int(r["frame"])
        if f not in sampled:
            continue
        t = r["tid"]
        box = tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1"))
        present[f].append(box)
        lab = box_label(ev, f, box, min_iou)
        for arm_name, own in (("before", dec_before[(f, t)]), ("after", dec_after[(f, mapping[t])])):
            if not own:
                continue
            delivered[arm_name][f].append(box)
            c = out[arm_name]
            c["delivered"] += 1
            if lab in TARGET:
                c["correct"] += 1
            elif lab == "unlabelled":
                c["unlabelled"] += 1
            else:
                c["wrong"] += 1
                c["wrong_" + lab] += 1
                fl = frag_label.get(t)
                if fl in TARGET:
                    mech = "tracker"
                elif arm_name == "after" and group_has_target[mapping[t]]:
                    mech = "join"
                else:
                    mech = "ownership"
                c["mech_" + mech] += 1
                out["wrong"].append({"arm": arm_name, "frame": f, "tid": t,
                                     "joined_tid": mapping[t], "label": lab,
                                     "fragment_label": fl, "mechanism": mech,
                                     "box": [int(v) for v in box]})
    # 召回：佩戴者的手（anchor）在同帧有没有被交付框盖住
    for h in gold["hands"]:
        for an in h.get("anchors") or []:
            f, box = int(an["frame"]), tuple(an["box"])
            if f not in sampled:
                continue
            seen = any(iou(box, b) >= min_iou for b in present[f])
            for arm_name in ("before", "after"):
                c = out[arm_name]
                c["anchors"] += 1
                if any(iou(box, b) >= min_iou for b in delivered[arm_name][f]):
                    c["anchor_delivered"] += 1
                elif seen:
                    c["anchor_own0"] += 1
                else:
                    c["anchor_absent"] += 1
                    c["absent_" + why_missing(scores, f, box, min_iou, thr)] += 1
    return out


def ratio(rs, arm, num, den):
    n = sum(r[arm][num] for r in rs)
    d = sum(r[arm][den] if isinstance(den, str) else den(r[arm]) for r in rs)
    return n / d if d else float("nan")


def boot(res, fn, n=2000, seed=0):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [res[rnd.randrange(len(res))] for _ in res]
        v = fn(s)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="glob，v2 gold")
    ap.add_argument("--arm", required=True,
                    help="部署运行目录：detect2/ joined_mapping/ decisions/ decisions_before/")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out")
    a = ap.parse_args()

    res = [run_one(g, a.arm, a.min_iou) for g in sorted(glob.glob(a.gold))]
    judged = lambda c: c["delivered"] - c["unlabelled"]  # noqa: E731

    print("=== 交付的 owner 框（gold 抽样格，IoU >= %.1f，%d 条录像）===" % (a.min_iou, len(res)))
    for arm in ("before", "after"):
        T = collections.Counter()
        for r in res:
            T.update(r[arm])
        name = "join 之前（decisions_before）" if arm == "before" else "join 之后（decisions，部署）"
        w = T["wrong"] / max(1, judged(T))
        lo, hi = boot(res, lambda s, arm=arm: ratio(s, arm, "wrong", judged))
        print("\n  %s" % name)
        print("    交付框 %d：佩戴者的手 %d / 错框 %d / 没标到 %d"
              % (T["delivered"], T["correct"], T["wrong"], T["unlabelled"]))
        print("    错框率 %.2f%%  [%.2f%%, %.2f%%]（可判的里）；把没标到全算错的上界 %.2f%%"
              % (100 * w, 100 * lo, 100 * hi,
                 100 * (T["wrong"] + T["unlabelled"]) / max(1, T["delivered"])))
        print("    错框按类型：other_hand %d / not_hand %d / unsure %d"
              % (T["wrong_other_hand"], T["wrong_not_hand"], T["wrong_unsure"]))
        print("    错框按来路：join %d / tracker 混入 %d / 整条判错 %d"
              % (T["mech_join"], T["mech_tracker"], T["mech_ownership"]))
        rec_ = T["anchor_delivered"] / max(1, T["anchors"])
        lo, hi = boot(res, lambda s, arm=arm: ratio(s, arm, "anchor_delivered", "anchors"))
        print("    召回：anchor %d 个，被交付 %d（%.1f%% [%.1f%%, %.1f%%]），有框但判 own=0 %d，detect2 里没有 %d"
              % (T["anchors"], T["anchor_delivered"], 100 * rec_, 100 * lo, 100 * hi,
                 T["anchor_own0"], T["anchor_absent"]))
        print("      detect2 里没有的：检测器没出框 %d / hand-ness 摘掉 %d / 出了框没成轨迹 %d / 无打分分不出 %d"
              % tuple(T["absent_" + k] for k in MISSING))

    def delta(s, key, den):
        return ratio(s, "after", key, den) - ratio(s, "before", key, den)
    d_w = delta(res, "wrong", judged)
    lo, hi = boot(res, lambda s: delta(s, "wrong", judged))
    print("\n  join 的净作用（after − before）：错框率 %+.2f 点 [%+.2f, %+.2f]" % (100 * d_w, 100 * lo, 100 * hi))
    d_r = delta(res, "anchor_delivered", "anchors")
    lo, hi = boot(res, lambda s: delta(s, "anchor_delivered", "anchors"))
    print("                                  召回   %+.2f 点 [%+.2f, %+.2f]" % (100 * d_r, 100 * lo, 100 * hi))

    print("\n  %-16s %18s %18s" % ("录像", "错框 before/after", "召回 before/after"))
    for r in res:
        b, f = r["before"], r["after"]
        print("  %-16s %8d/%-4d %4d/%-4d %6d/%-4d %4d/%-4d"
              % (r["recording"], b["wrong"], judged(b), f["wrong"], judged(f),
                 b["anchor_delivered"], b["anchors"], f["anchor_delivered"], f["anchors"]))

    if a.out:
        json.dump({"min_iou": a.min_iou, "recordings": res}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
