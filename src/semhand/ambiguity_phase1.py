"""Phase 1：只用几何，看能并对多少 —— 留一录像交叉验证，两类负例分开报。

事前写死的判据（跑之前定的，不许事后改）：

    在「unassigned 负例一个都不并错」的约束下
      召回 >= 30%  -> 几何够用，Phase 2 的 appearance 要在这个基线上证明自己
      召回 <  30%  -> 几何到头，直接上 appearance

为什么召回门槛定得这么低。这不是 tracking benchmark，错误合并的代价不对称：
把自己的左右手并成一条，少一个身份，隐私上没有后果；把自己的手和**别人的
手**并成一条，整条轨迹被判成 owner，于是别人的手不打码。所以宁可少并。

为什么是打分加阈值，不是 argmax。35 个决策点里 20 个有**不止一个正例**
（一只手碎成三段，从第一段的末尾看过去，另外两段都是它自己）。argmax 每个
点最多并一个，按构造就漏掉一半。打分加阈值允许一个点并多个。

为什么按录像留一。同一条录像里的对共享场景、相机、同一双手，按对切会把
同一只手的信息同时放进训练和测试。10 条录像是唯一的独立单位。

为什么分数要除以 width_a。距离门是 `d <= 3.0 * width_a`，是**相对**框宽的
阈值，所以框大的录像允许更大的绝对距离。不归一化时 pooled AUC 0.879 高过
决策点内的 0.821，差值是录像间构成差异 —— 归一化后 pooled 0.822 与决策点内
0.821 重合，这就是「分数跨点可比」的证据。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics


def sweep(rows, feat, thresholds):
    """-> [(t, tp, fn, fp_own, fp_unassigned)]，分数 <= t 判为「并」。"""
    out = []
    for t in thresholds:
        tp = fn = fp_own = fp_un = 0
        for r in rows:
            merge = r[feat] <= t
            if r["label"] == 1:
                tp += merge
                fn += not merge
            elif r["kind"] == "other_own_hand":
                fp_own += merge
            else:
                fp_un += merge
        out.append((t, tp, fn, fp_own, fp_un))
    return out


def loro(rows, feat, thresholds):
    """留一录像：在 9 条上挑阈值，在第 10 条上用。

    挑阈值的规则也要事前定死，否则「挑」本身就是在测试集上调参：
    在训练折里取 **unassigned 误并为 0 的前提下召回最大** 的那个阈值。
    """
    recs = sorted({r["recording"] for r in rows})
    agg = {"tp": 0, "fn": 0, "fp_own": 0, "fp_un": 0}
    per = []
    for held in recs:
        tr = [r for r in rows if r["recording"] != held]
        te = [r for r in rows if r["recording"] == held]
        best, best_rec = None, -1
        for t, tp, fn, fo, fu in sweep(tr, feat, thresholds):
            if fu == 0 and tp > best_rec:
                best, best_rec = t, tp
        if best is None:
            best = min(thresholds)
        _, tp, fn, fo, fu = sweep(te, feat, [best])[0]
        agg["tp"] += tp
        agg["fn"] += fn
        agg["fp_own"] += fo
        agg["fp_un"] += fu
        per.append({"recording": held, "threshold": best, "tp": tp, "fn": fn,
                    "fp_own": fo, "fp_unassigned": fu, "n_pos": tp + fn})
    return agg, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--feat", default="s_plain")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.pairs, encoding="utf-8"))
            if r["label"] is not None]
    # 阈值取相邻取值的中点，不取 v-1e-9。取后者时打印出来的「0.363」其实是
    # 0.363-1e-9，恰好把分数等于 0.363 的那一对排除在外 —— 而这批数据里最危险
    # 的那个负例分数正好是 0.363，于是扫描表显示「0 个未归属误并」而实际有 1 个。
    # 我自己被这个显示误导过一次。中点落在样本之间，标签就不会骗人。
    vals = sorted({r[a.feat] for r in rows})
    thresholds = ([vals[0] - 0.01]
                  + [(x + y) / 2 for x, y in zip(vals, vals[1:])]
                  + [vals[-1] + 0.01])

    # 每条录像单独的 AUC —— 看 LORO 会不会被某一两条拖垮
    def auc(P, N):
        return (sum((p < n) + 0.5 * (p == n) for p in P for n in N)
                / (len(P) * len(N))) if P and N else None
    by = collections.defaultdict(lambda: ([], []))
    for r in rows:
        by[r["recording"]][0 if r["label"] == 1 else 1].append(r[a.feat])
    print("=== 逐录像 AUC（%s）===" % a.feat)
    aucs = []
    for rec in sorted(by):
        P, N = by[rec]
        v = auc(P, N)
        aucs.append(v) if v is not None else None
        print("  %-20s 正 %2d 负 %2d  AUC %s"
              % (rec, len(P), len(N), "%.3f" % v if v is not None else "不可算"))
    if aucs:
        print("  中位 %.3f，范围 %.3f-%.3f" % (statistics.median(aucs),
                                             min(aucs), max(aucs)))

    print("\n=== 全量阈值扫描（仅供看形状，不是结果）===")
    print("  %8s %5s %5s %8s %12s %8s" %
          ("阈值", "并对", "漏", "误并自手", "误并未归属", "召回"))
    shown = 0
    for t, tp, fn, fo, fu in sweep(rows, a.feat, thresholds):
        if shown % max(1, len(thresholds) // 12) == 0:
            print("  %8.3f %5d %5d %8d %12d %7.1f%%"
                  % (t, tp, fn, fo, fu, 100 * tp / (tp + fn)))
        shown += 1

    agg, per = loro(rows, a.feat, thresholds)
    n_pos = agg["tp"] + agg["fn"]
    recall = agg["tp"] / n_pos if n_pos else 0
    print("\n=== 留一录像（阈值在训练折里按「未归属误并=0 下召回最大」挑）===")
    print("  %-20s %8s %5s %5s %9s %12s"
          % ("留出的录像", "阈值", "并对", "漏", "误并自手", "误并未归属"))
    for p in per:
        print("  %-20s %8.3f %5d %5d %9d %12d"
              % (p["recording"], p["threshold"], p["tp"], p["fn"],
                 p["fp_own"], p["fp_unassigned"]))
    print("  %-20s %8s %5d %5d %9d %12d"
          % ("合计", "", agg["tp"], agg["fn"], agg["fp_own"],
             agg["fp_un"]))
    print("\n  召回 %d/%d = %.1f%%；误并自己的另一只手 %d；**误并未归属 %d**"
          % (agg["tp"], n_pos, 100 * recall, agg["fp_own"], agg["fp_un"]))

    ok = agg["fp_un"] == 0 and recall >= 0.30
    print("\n  事前判据：未归属误并 = 0 且 召回 >= 30%")
    print("  -> %s" % ("**通过** —— 几何够用，appearance 要在这个基线上证明自己"
                       if ok else
                       "**未通过** —— %s" % (
                           "未归属误并 %d > 0" % agg["fp_un"] if agg["fp_un"]
                           else "召回 %.1f%% < 30%%" % (100 * recall))))
    if not ok:
        print("     几何到头，下一步直接上 appearance embedding。")

    if a.out:
        json.dump({"feat": a.feat, "criterion_passed": ok,
                   "recall": recall, "aggregate": agg, "per_recording": per},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print("\n-> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
