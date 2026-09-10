"""Does the shape of a trajectory say anything a length and a score do not?

A RULE FAILING IS NOT A SIGNAL FAILING. The first probe applied one specific
rule -- MAD threshold, floor, round-trip -- and found it could not separate
non-hands inside the short low-confidence cell. That closes the rule, not the
question: a badly shaped threshold on a real signal fails exactly like a
threshold on noise. So the continuous quantities go in directly, and the test
is whether adding them to the two things already known to work buys anything.

B0 IS THE HONEST BASELINE AND IT IS ALREADY STRONG. Track length and median
detection confidence account for nearly all of the non-hand rate: 54.5% of
one-to-three-frame tracks are not hands, and nothing above 0.75 confidence is.
An incremental test against B0 is the only test that means anything, because
trajectory features correlate with length and would otherwise be rewarded for
rediscovering it.

FOLDS ARE RECORDINGS, NEVER TRACKS. A false detection is a property of a
scene -- a red cloth on a bench, a basket of greens -- so several tracks in
one recording are one observation of one mechanism. Splitting them across a
fold lets the model memorise the object and call it a trajectory feature.

AND THE SECOND TARGET IS THE ONE THAT COSTS ANYTHING. Of 61 non-hand tracks,
29 ever reach a blurred frame; the other 32 are classified `owner` and cost
nothing at all. A screen that cannot find non-hands in general might still
find the ones that produce false blur, and that is the version worth
deploying, so it gets its own column rather than being folded into the first.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import random
import statistics


def auroc(scores, labels):
    """Mann-Whitney, with ties at half credit. -> float"""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = r
        i = j + 1
    s = sum(rank[i] for i, y in enumerate(labels) if y)
    return (s - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def track_features(v):
    """One track's rows, sorted. -> (name -> value) or None if too short."""
    n = len(v)
    if n < 4:
        return None
    fr = [int(r["frame"]) for r in v]
    cen, area, conf = [], [], []
    for r in v:
        x0, y0 = float(r["x0"]), float(r["y0"])
        x1, y1 = float(r["x1"]), float(r["y1"])
        cen.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        area.append(max(1.0, (x1 - x0) * (y1 - y0)))
        conf.append(float(r["side_conf"]))
    size = statistics.median([math.sqrt(a) for a in area])
    gaps = [max(1, fr[i + 1] - fr[i]) for i in range(n - 1)]
    d = [math.hypot(cen[i + 1][0] - cen[i][0], cen[i + 1][1] - cen[i][1])
         / size / gaps[i] for i in range(n - 1)]
    ds = [abs(math.log(area[i + 1] / area[i])) / gaps[i] for i in range(n - 1)]
    acc = [abs(d[i + 1] - d[i]) for i in range(len(d) - 1)]
    jerk = [abs(acc[i + 1] - acc[i]) for i in range(len(acc) - 1)]

    def q(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    path = sum(d)
    net = math.hypot(cen[-1][0] - cen[0][0], cen[-1][1] - cen[0][1]) / size
    return {
        # B0 -- what is already known to work
        "len": float(n),
        "conf": statistics.median(conf),
        "size": size,
        # trajectory shape
        "d_max": max(d), "d_p90": q(d, 0.90), "d_med": statistics.median(d),
        "s_max": max(ds), "s_p90": q(ds, 0.90),
        "acc_max": max(acc) if acc else 0.0,
        "jerk_max": max(jerk) if jerk else 0.0,
        # A hand that wanders covers much more path than it displaces; a box
        # jittering on a static object does the same, so this is reported and
        # not assumed to point either way.
        "tortuosity": path / max(net, 1e-3),
        "path": path,
    }


B0 = ("len", "conf", "size")
TRAJ = ("d_max", "d_p90", "d_med", "s_max", "s_p90", "acc_max", "jerk_max",
        "tortuosity", "path")


def fit(X, y, iters=4000, lr=0.5, l2=1e-2):
    """Logistic regression, standardised, plain gradient descent.

    Written out rather than imported: this machine has no sklearn, and a model
    with twelve inputs and a few hundred rows does not need one. What it does
    need is standardisation inside the fold -- a mean taken over the held-out
    recording is a leak, small but exactly the kind that manufactures an
    increment where there is none."""
    import numpy as np
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(Z)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        e = p - y
        w -= lr * (Z.T @ e / n + l2 * w)
        b -= lr * e.mean()
    return (w, b, mu, sd)


def predict(model, row):
    import numpy as np
    w, b, mu, sd = model
    z = float(((np.asarray(row, float) - mu) / sd) @ w + b)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def grouped_cv(keys, feats, y, names):
    """Leave one recording out. -> out-of-fold probabilities."""
    recs = sorted({k[0] for k in keys})
    out = [0.0] * len(keys)
    for held in recs:
        tr = [i for i, k in enumerate(keys) if k[0] != held]
        te = [i for i, k in enumerate(keys) if k[0] == held]
        if not te or len(set(y[i] for i in tr)) < 2:
            continue
        X = [[feats[i][nm] for nm in names] for i in tr]
        model = fit(X, [y[i] for i in tr])
        for i in te:
            out[i] = predict(model, [feats[i][nm] for nm in names])
    return out


def logloss(p, y):
    return -statistics.mean(
        math.log(max(1e-9, pi if yi else 1 - pi)) for pi, yi in zip(p, y))


def recall_at_keep(p, y, keep=0.99):
    """How many positives are caught if 99% of the negatives are kept."""
    neg = sorted((pi for pi, yi in zip(p, y) if not yi), reverse=True)
    if not neg:
        return float("nan")
    thr = neg[min(len(neg) - 1, int((1 - keep) * len(neg)))]
    pos = [pi for pi, yi in zip(p, y) if yi]
    return sum(1 for pi in pos if pi > thr) / max(1, len(pos))


def matched(keys, feats, y, seed=0):
    """Each non-hand against the closest true hand, recording first.

    A CELL IS A COARSE MATCH AND THIS IS THE FINE ONE. `len 4-20 and conf<0.75`
    still lets a 4-frame 0.30 track sit beside a 20-frame 0.74 one. Pairing on
    the standardised distance in (length, confidence, size) removes what is
    left, and prefers a partner from the same recording so the scene is held
    fixed too."""
    pos = [i for i, yi in enumerate(y) if yi]
    neg = [i for i, yi in enumerate(y) if not yi]
    mu = {nm: statistics.mean(f[nm] for f in feats) for nm in B0}
    sd = {nm: statistics.pstdev([f[nm] for f in feats]) or 1.0 for nm in B0}

    def dist(i, j):
        d = sum(((feats[i][nm] - feats[j][nm]) / sd[nm]) ** 2 for nm in B0)
        return d + (0.0 if keys[i][0] == keys[j][0] else 4.0)

    # A CALIPER, BECAUSE NEAREST IS NOT NEAR. Without one the nearest true
    # hand to a 6-frame non-hand was an 18-frame hand -- the median lengths
    # came out 6 against 18 and the comparison was still measuring length.
    # Pairs that cannot be matched inside the caliper are dropped and counted,
    # since a match in name only is worse than a missing one.
    used, pairs, unmatched = set(), [], 0
    for i in pos:
        cand = [j for j in neg if j not in used
                and abs(feats[i]["len"] - feats[j]["len"]) <= 3
                and abs(feats[i]["conf"] - feats[j]["conf"]) <= 0.10]
        if not cand:
            unmatched += 1
            continue
        j = min(cand, key=lambda j: dist(i, j))
        used.add(j)
        pairs.append((i, j))
    return pairs, unmatched


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--precision", nargs="+", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    lab = {}
    for f in a.precision:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            lab[(r["rec"], r["tid"])] = r["verdict"]

    keys, feats, y_nh, y_blur = [], [], [], []
    for k, v in by.items():
        if k not in lab:
            continue
        f = track_features(v)
        if f is None:
            continue
        keys.append(k)
        feats.append(f)
        nh = lab[k] == "nohand"
        y_nh.append(int(nh))
        y_blur.append(int(nh and any(not int(r["final_owner_post_cap"])
                                     for r in v)))
    print(f"\n  {len(keys)} 条 n>=4 的轨迹（n<4 的判不了，我的特征对它们全零）")
    print(f"  不是手 {sum(y_nh)}   其中最终产生过糊帧的 {sum(y_blur)}")

    for target, y in (("不是手", y_nh), ("会造成误糊的非手", y_blur)):
        if sum(y) < 5:
            continue
        print(f"\n  === 目标：{target}（正例 {sum(y)}）===")
        print(f"  {'单特征 AUROC':<16}", end="")
        aus = [(nm, auroc([f[nm] for f in feats], y)) for nm in B0 + TRAJ]
        for nm, au in sorted(aus, key=lambda x: -abs(x[1] - 0.5)):
            print(f"{nm} {au:.3f}  ", end="")
        print()
        res = {}
        for name, cols in (("B0 长度+置信+框大小", B0),
                           ("B1 B0 + 轨迹特征", B0 + TRAJ),
                           ("仅轨迹特征", TRAJ)):
            p = grouped_cv(keys, feats, y, cols)
            res[name] = p
            print(f"    {name:<22} AUROC {auroc(p, y):.3f}   "
                  f"log-loss {logloss(p, y):.4f}   "
                  f"保住 99% 真手时抓到 {recall_at_keep(p, y):.1%}")
        d = auroc(res["B1 B0 + 轨迹特征"], y) - auroc(res["B0 长度+置信+框大小"], y)
        print(f"    B1 - B0 增量 AUROC {d:+.3f}"
              + ("   —— 没有增量" if abs(d) < 0.03 else ""))

    print(f"\n  === 匹配对照（每条非手配一条最接近的真手）===")
    pairs, unmatched = matched(keys, feats, y_nh)
    print(f"  卡钳：长度差 <=3 帧且置信差 <=0.10，同录像优先")
    print(f"  配成 {len(pairs)} 对，{unmatched} 条非手在卡钳内找不到对照")
    print(f"    {'':<12}{'非手中位':>10}{'真手中位':>10}{'非手更大的比例':>14}")
    for nm in TRAJ:
        a_ = [feats[i][nm] for i, _ in pairs]
        b_ = [feats[j][nm] for _, j in pairs]
        win = sum(1 for x, z in zip(a_, b_) if x > z) / max(1, len(pairs))
        print(f"    {nm:<12}{statistics.median(a_):>10.3f}"
              f"{statistics.median(b_):>10.3f}{win:>14.0%}")
    print("    「非手更大的比例」离 50% 越远越有判别力；贴着 50% = 分布重合。")
    for nm in B0:
        a_ = [feats[i][nm] for i, _ in pairs]
        b_ = [feats[j][nm] for _, j in pairs]
        print(f"    (匹配变量 {nm}: 非手 {statistics.median(a_):.2f} vs "
              f"真手 {statistics.median(b_):.2f})")


if __name__ == "__main__":
    main()
