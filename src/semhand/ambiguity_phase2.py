"""Phase 2：把 DINOv2 外观相似度加进来，判据与 Phase 1 完全相同。

事前判据（与 Phase 1 同一条，不许改）：
    unassigned 负例一个都不并错，且召回 >= 30%

两个外观特征，因为它们是两个不同的量，哪个有用要测：
    cos_mean  整轨均值的余弦 —— 稳，但跨越了整条轨迹的外观变化
    cos_junc  接头处的余弦（A 末尾 k 帧 vs B 开头 k 帧）—— 光照姿态最接近，
              但样本少

合并方式是逻辑回归，不是手调权重：111 对、2-3 个特征，逻辑回归是能用的最简
模型，而且系数可读。留一录像，阈值在训练折里按与 Phase 1 **完全相同**的规则
挑（未归属误并=0 下召回最大），否则两期不可比。

注意 `src/semhand/embed.py` 那套不能用在这里：它是 Qwen3-VL-Embedding，提示词
是「Represent what this hand is doing」，编码的是动作语义。两只手做同一件事会
被判成相似 —— 正好是这里最危险的负例。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics

import numpy as np

K_JUNC = 4


def pair_appearance(emb, rec, a, b):
    """-> (cos_mean, cos_junc) 或 None（任一条轨迹没有向量就不可打分）。"""
    T = emb["tracks"].get(rec, {})
    if a not in T or b not in T:
        return None
    va = np.asarray(T[a]["vecs"], dtype=np.float64)
    vb = np.asarray(T[b]["vecs"], dtype=np.float64)
    fa = np.asarray(T[a]["frames"])
    fb = np.asarray(T[b]["frames"])
    ma = va.mean(0)
    mb = vb.mean(0)
    ma /= np.linalg.norm(ma) or 1.0
    mb /= np.linalg.norm(mb) or 1.0
    cos_mean = float(ma @ mb)
    # 接头：A 最后 K 帧、B 最前 K 帧
    la = va[np.argsort(fa)][-K_JUNC:].mean(0)
    fb_ = vb[np.argsort(fb)][:K_JUNC].mean(0)
    la /= np.linalg.norm(la) or 1.0
    fb_ /= np.linalg.norm(fb_) or 1.0
    return cos_mean, float(la @ fb_)


def fit_logreg(X, y, iters=3000, lr=0.1, l2=1.0):
    """最小实现，省得为 3 个特征引入 sklearn。返回 (w, b)。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * g
        b -= lr * gb
    return w, b, mu, sd


def score_logreg(X, model):
    w, b, mu, sd = model
    Z = (np.asarray(X, dtype=np.float64) - mu) / sd
    return 1.0 / (1.0 + np.exp(-(Z @ w + b)))


def evaluate(rows, featnames, use_logreg, lower_is_merge):
    """留一录像。阈值规则与 Phase 1 相同：训练折里未归属误并=0 下召回最大。"""
    recs = sorted({r["recording"] for r in rows})
    agg = collections.Counter()
    per = []
    for held in recs:
        tr = [r for r in rows if r["recording"] != held]
        te = [r for r in rows if r["recording"] == held]
        if use_logreg:
            model = fit_logreg([[r[f] for f in featnames] for r in tr],
                               [r["label"] for r in tr])
            s_tr = score_logreg([[r[f] for f in featnames] for r in tr], model)
            s_te = score_logreg([[r[f] for f in featnames] for r in te], model)
            merge_if = lambda s, t: s >= t          # noqa: E731
            cand = sorted(set(s_tr.tolist()))
            cand = [(x + y) / 2 for x, y in zip(cand, cand[1:])] or [0.5]
            cand = [max(s_tr) + 0.01] + cand + [min(s_tr) - 0.01]
        else:
            f = featnames[0]
            s_tr = np.array([r[f] for r in tr])
            s_te = np.array([r[f] for r in te])
            merge_if = ((lambda s, t: s <= t) if lower_is_merge
                        else (lambda s, t: s >= t))
            v = sorted(set(s_tr.tolist()))
            cand = ([v[0] - 0.01] + [(x + y) / 2 for x, y in zip(v, v[1:])]
                    + [v[-1] + 0.01])
        best, best_tp = None, -1
        for t in cand:
            tp = fu = 0
            for r, s in zip(tr, s_tr):
                if merge_if(s, t):
                    if r["label"] == 1:
                        tp += 1
                    elif r["kind"] != "other_own_hand":
                        fu += 1
            if fu == 0 and tp > best_tp:
                best, best_tp = t, tp
        if best is None:
            best = cand[0]
        tp = fn = fo = fu = 0
        for r, s in zip(te, s_te):
            m = merge_if(s, best)
            if r["label"] == 1:
                tp += m
                fn += not m
            elif r["kind"] == "other_own_hand":
                fo += m
            else:
                fu += m
        tp, fn, fo, fu = int(tp), int(fn), int(fo), int(fu)
        agg.update({"tp": tp, "fn": fn, "fp_own": fo, "fp_un": fu})
        per.append({"recording": held, "tp": tp, "fn": fn, "fp_own": fo,
                    "fp_unassigned": fu})
    return agg, per


def report(name, agg, per):
    n_pos = agg["tp"] + agg["fn"]
    rec = agg["tp"] / n_pos if n_pos else 0.0
    ok = agg["fp_un"] == 0 and rec >= 0.30
    print("  %-30s 召回 %2d/%2d = %5.1f%%   误并自手 %2d   误并未归属 %2d   %s"
          % (name, agg["tp"], n_pos, 100 * rec, agg["fp_own"], agg["fp_un"],
             "**通过**" if ok else "未通过"))
    return {"name": name, "recall": rec, "fp_own": agg["fp_own"],
            "fp_unassigned": agg["fp_un"], "passed": ok,
            "per_recording": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    emb = json.load(open(a.embeds, encoding="utf-8"))
    rows = []
    skipped = 0
    for r in json.load(open(a.pairs, encoding="utf-8")):
        if r["label"] is None:
            continue
        ap_ = pair_appearance(emb, r["recording"], r["a"], r["b"])
        if ap_ is None:
            skipped += 1
            continue
        r["cos_mean"], r["cos_junc"] = ap_
        rows.append(r)
    print("可用 %d 对（没有向量而丢掉 %d 对）；正 %d / 负 %d"
          % (len(rows), skipped, sum(r["label"] == 1 for r in rows),
             sum(r["label"] == 0 for r in rows)))

    def auc_within_point(f, higher_is_same):
        bp = collections.defaultdict(lambda: ([], []))
        for r in rows:
            bp[(r["recording"], r["a"])][0 if r["label"] == 1 else 1].append(r[f])
        vals = []
        for P, N in bp.values():
            if not P or not N:
                continue
            w = sum((p > n) + 0.5 * (p == n) if higher_is_same
                    else (p < n) + 0.5 * (p == n) for p in P for n in N)
            vals.append(w / (len(P) * len(N)))
        return statistics.mean(vals), len(vals)

    print("\n=== 决策点内 AUC ===")
    for f, hi in [("s_plain", False), ("cos_mean", True), ("cos_junc", True)]:
        v, n = auc_within_point(f, hi)
        print("  %-10s %.3f   (%d 个决策点可算)" % (f, v, n))

    print("\n=== 留一录像（阈值规则与 Phase 1 完全相同）===")
    out = []
    out.append(report("几何 s_plain（Phase 1 基线）",
                      *evaluate(rows, ["s_plain"], False, True)))
    out.append(report("外观 cos_mean 单独",
                      *evaluate(rows, ["cos_mean"], False, False)))
    out.append(report("外观 cos_junc 单独",
                      *evaluate(rows, ["cos_junc"], False, False)))
    out.append(report("几何 + cos_mean（逻辑回归）",
                      *evaluate(rows, ["s_plain", "cos_mean"], True, False)))
    out.append(report("几何 + 两个外观（逻辑回归）",
                      *evaluate(rows, ["s_plain", "cos_mean", "cos_junc"],
                                True, False)))

    print("\n  事前判据：未归属误并 = 0 且 召回 >= 30%（与 Phase 1 同一条）")
    if a.out:
        json.dump({"k_junc": K_JUNC, "results": out},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
