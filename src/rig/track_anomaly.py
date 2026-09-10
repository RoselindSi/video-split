"""Does a wrong frame announce itself in the track it sits in?

DIAGNOSTIC ONLY. Nothing here changes a label. The question is whether the
errors that survived the frame-level stack -- 171 uncovered frames in fifteen
runs, and the four foreign tracks that a majority vote loses outright without
the cap -- carry a visible signature in the trajectory beforehand. If they do
not, no amount of trajectory filtering will find them and the idea is closed
cheaply. If they do, the signature is what an intervention gets defined on,
and the intervention is designed afterwards rather than fitted now.

THE MECHANISM COMES FROM WUJI'S `block_spike` AND IT IS NOT A HAMPEL FILTER.
A plain outlier test cannot tell a hand that moved from an estimate that
jumped, because both produce one large step. The round trip can: a run of
large steps is suspect only if the position AFTER it reconnects with the
position BEFORE it. 105 -> 390 -> 115 is an excursion; 105 -> 390 -> 395 is a
hand that moved.

TWO KINDS OF JUMP, AND THE SECOND IS NEW. Wuji runs this on metric 3D
translation, which this project does not have for most hands -- foreign hands
get usable stereo 45.7% of the time and the limit is the triangulation angle,
not software. But the panorama box is available every frame and so is the
classifier's probability, so the same test runs on a 2D trajectory and,
separately, on the ownership score. A hand whose box is perfectly continuous
while its ownership dips for three frames and comes back is the exact shape of
the residual failure the gold found.

THRESHOLDS ARE PER TRACK, WITH A FLOOR. `median + k * 1.4826 * MAD` over that
track's own steps, floored so a hand that barely moves cannot acquire a
hair-trigger. That shape is worth borrowing on its own: nearly every threshold
in this pipeline is a global constant, and the corpus has just been shown to
vary enormously between recordings.

MEASURED ON THE SAME 274 TRACKS THAT PRODUCED THE MAJORITY RULE. Anything
that looks good here is a hypothesis, not a gain: the gold has been seen. A
number from this file is only evidence about mechanism, and only a fresh
recording can turn it into evidence about performance.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics

# Wuji's block length: how far an excursion may run and still be treated as
# one. Kept at their value rather than tuned, since tuning it here would fit
# the corpus that is also being used to judge the result.
MAX_BLOCK = 15
# Floors, in units that survive a change of resolution: box widths for
# position, log-area for scale, probability for ownership.
FLOOR_CENTER = 0.60
FLOOR_SCALE = 0.35
FLOOR_OWNER = 0.30
# How close the far side of an excursion must come back to the near side.
RECOVERY = 0.60
K = 3.0

FIELDS = ("rec", "tid", "frame", "gold", "n_frames",
          "center_jump_score", "scale_jump_score", "owner_prob_jump",
          "round_trip_recovered", "is_temporal_outlier",
          "p_owner_raw", "p1_label", "p2_label", "leaked")


def mad_thresh(xs, k, floor):
    if len(xs) < 3:
        return floor
    med = statistics.median(xs)
    sigma = 1.4826 * statistics.median([abs(x - med) for x in xs])
    return max(floor, med + k * sigma)


def excursions(pos, size, big, max_block, recovery):
    """Wuji's block_spike, on whatever one-dimensional-in-time thing is given.

    `pos` is a list of scalars or 2-tuples, `big[i]` says step i->i+1 is
    large. Returns the set of indices inside a run that went out and came
    back."""
    n = len(pos)
    bad = set()

    def dist(a, b):
        if isinstance(a, tuple):
            return math.hypot(a[0] - b[0], a[1] - b[1])
        return abs(a - b)

    i = 1
    while i < n - 1:
        if not big[i - 1]:
            i += 1
            continue
        end = min(i + max_block, n - 1)
        j = i
        while j < end:
            if not big[j]:
                j += 1
                continue
            if dist(pos[j + 1], pos[i - 1]) < recovery * size:
                bad.update(range(i, j + 1))
                i = j + 1
            else:
                i += 1
            break
        else:
            i += 1
    return bad


def features(v):
    """One track's rows, sorted -> per-frame diagnostic dict."""
    n = len(v)
    fr = [int(r["frame"]) for r in v]
    cen, area = [], []
    for r in v:
        x0, y0 = float(r["x0"]), float(r["y0"])
        x1, y1 = float(r["x1"]), float(r["y1"])
        cen.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        area.append(max(1.0, (x1 - x0) * (y1 - y0)))
    p = [float(r["p_owner_raw"]) for r in v]
    size = statistics.median([math.sqrt(a) for a in area])
    out = [{"center_jump_score": 0.0, "scale_jump_score": 0.0,
            "owner_prob_jump": 0.0, "round_trip_recovered": 0,
            "is_temporal_outlier": 0} for _ in range(n)]
    if n < 4:
        return out
    # Steps are per FRAME, not per row: a track that was lost for five frames
    # is allowed to have moved five frames' worth without that counting as a
    # jump. This is the `/ gaps` in wuji's version and it matters here more,
    # because our tracker keeps identity across a gap.
    gaps = [max(1, fr[i + 1] - fr[i]) for i in range(n - 1)]
    dc = [math.hypot(cen[i + 1][0] - cen[i][0], cen[i + 1][1] - cen[i][1])
          / size / gaps[i] for i in range(n - 1)]
    ds = [abs(math.log(area[i + 1] / area[i])) / gaps[i] for i in range(n - 1)]
    do = [abs(p[i + 1] - p[i]) / gaps[i] for i in range(n - 1)]
    tc = mad_thresh(dc, K, FLOOR_CENTER)
    ts = mad_thresh(ds, K, FLOOR_SCALE)
    to = mad_thresh(do, K, FLOOR_OWNER)
    for i in range(n - 1):
        out[i + 1]["center_jump_score"] = round(dc[i] / tc, 3)
        out[i + 1]["scale_jump_score"] = round(ds[i] / ts, 3)
        out[i + 1]["owner_prob_jump"] = round(do[i] / to, 3)
    geo = excursions(cen, size, [x > tc for x in dc], MAX_BLOCK, RECOVERY)
    # The ownership score has no size to normalise by; its recovery bound is
    # in probability, and it is the same number as its floor for the same
    # reason -- coming back to within a third of a probability is coming back.
    own = excursions(p, 1.0, [x > to for x in do], MAX_BLOCK, FLOOR_OWNER)
    for i in geo | own:
        out[i]["round_trip_recovered"] = 1
        out[i]["is_temporal_outlier"] = 1
    for i in range(n):
        if (out[i]["center_jump_score"] > 1.0
                or out[i]["scale_jump_score"] > 1.0
                or out[i]["owner_prob_jump"] > 1.0):
            out[i]["is_temporal_outlier"] = 1
    return out


def screen(a):
    """Can the trajectory tell a hand from a thing the detector imagined?

    THE FIRST PROBE ASKED THE WRONG GOLD. It joined the excursion features to
    ownership misses and found nothing, which closed the question `does a jump
    predict a leak`. It did not touch the question a jump is actually shaped
    to answer: whether this observation is a hand at all. There are 335 tracks
    with a person's verdict on exactly that, and the features were never
    joined to them.

    THE ONLY TABLE THAT DECIDES ANYTHING IS THE HARD CELL. Non-hand tracks are
    short and low-confidence -- 54.5% of 1-3 frame tracks and 34.5% below 0.65
    confidence -- so an anomaly rate that is higher on non-hands proves
    nothing on its own: short tracks jitter more and short tracks are more
    often false. The trajectory earns its place only if it separates INSIDE
    the short, low-confidence cell, where length and confidence have already
    said everything they can.

    AND FAST IS NOT WRONG. The wearer's own hand swings, grabs and reaches;
    a rule that treats speed as error would delete the most useful hand in the
    corpus. What the round trip tests is different: a departure from the path
    that comes back, which is a detector or tracker mistake rather than a
    motion. 100 -> 150 -> 210 -> 270 is fast and continuous; 100 -> 110 ->
    420 -> 120 is not.

    DELETING A FOREIGN HAND IS THE EXPENSIVE MISTAKE, not deleting any true
    hand, so the true hands are split by ownership before anything is
    concluded about what a filter would cost."""
    by, _ = load(a, filter_hands=False)
    lab, conf = {}, {}
    for f in a.precision:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            lab[(r["rec"], r["tid"])] = r["verdict"]
            conf[(r["rec"], r["tid"])] = float(r["det_conf"])
    gold = {}
    for f in (a.gold or []):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    J = {k: v for k, v in by.items() if k in lab}
    feat = {k: features(v) for k, v in J.items()}

    def col(ks):
        n = len(ks)
        if not n:
            return None
        anom = sum(1 for k in ks if any(d["is_temporal_outlier"] for d in feat[k]))
        rt = sum(1 for k in ks if any(d["round_trip_recovered"] for d in feat[k]))
        frac = sorted(sum(d["is_temporal_outlier"] for d in feat[k]) / len(feat[k])
                      for k in ks)
        mc = sorted(max((d["center_jump_score"] for d in feat[k]), default=0)
                    for k in ks)
        ms = sorted(max((d["scale_jump_score"] for d in feat[k]), default=0)
                    for k in ks)
        return (n, anom, rt, frac[n // 2], mc[n // 2], ms[n // 2])

    def show(title, groups):
        print("\n  " + title)
        print(f"    {'':<22}" + "".join(f"{g:>12}" for g, _ in groups))
        rows = [col(ks) for _, ks in groups]
        names = ["轨迹数", "有异常帧", "有往返型", "异常帧占比中位",
                 "最大center跳中位", "最大scale跳中位"]
        for i, nm in enumerate(names):
            line = f"    {nm:<22}"
            for r in rows:
                if r is None:
                    line += f"{'—':>12}"
                elif i == 0:
                    line += f"{r[0]:>12}"
                elif i <= 2:
                    line += f"{r[i]}/{r[0]} = {r[i]/r[0]:.0%}".rjust(12)
                else:
                    line += f"{r[i]:>12.2f}"
            print(line)

    H = [k for k in J if lab[k] == "hand"]
    N = [k for k in J if lab[k] == "nohand"]
    show("全体", [("真手", H), ("不是手", N)])
    if gold:
        show("真手内部（误删别人的手才是贵的那个错）",
             [("自己的手", [k for k in H if gold.get(k) == "owner"]),
              ("别人的手", [k for k in H if gold.get(k) == "other"])])

    print("\n  === 困难格：len<=20 且 conf<0.75 ===")
    hard = [k for k in J if len(J[k]) <= 20 and conf.get(k, 1.0) < 0.75]
    hh = [k for k in hard if lab[k] == "hand"]
    hn = [k for k in hard if lab[k] == "nohand"]
    print(f"  这一格 {len(hard)} 条：真手 {len(hh)}，不是手 {len(hn)}"
          f"（基线非手率 {len(hn)/max(1,len(hard)):.1%}）")
    print(f"    {'':<14}{'轨迹干净':>10}{'轨迹异常':>10}{'异常率':>9}")
    for nm, ks in (("真手", hh), ("不是手", hn)):
        a_ = sum(1 for k in ks if any(d["is_temporal_outlier"] for d in feat[k]))
        print(f"    {nm:<14}{len(ks)-a_:>10}{a_:>10}"
              f"{a_/max(1,len(ks)):>9.1%}")
    ah = sum(1 for k in hh if any(d["is_temporal_outlier"] for d in feat[k]))
    an = sum(1 for k in hn if any(d["is_temporal_outlier"] for d in feat[k]))
    if ah + an:
        print(f"    「轨迹异常」里的非手比例 {an}/{an+ah} = {an/(an+ah):.1%}"
              f"   对比这一格基线 {len(hn)/max(1,len(hard)):.1%}")
        print("    两个数接近 = 轨迹没有提供长度和置信度之外的任何增量信息。")


def load(a, filter_hands=True):
    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    if a.hands and filter_hands:
        keep = set()
        for f in a.hands:
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                if r["verdict"] == "hand":
                    keep.add((r["rec"], r["tid"]))
        by = {k: v for k, v in by.items() if k in keep}
    gold = {}
    for f in (a.gold or []):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    return by, gold


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--hands", nargs="*")
    ap.add_argument("--gold", nargs="*")
    ap.add_argument("--precision", nargs="*",
                    help="hand_precision label csvs: run the screening probe")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.precision:
        screen(a)
        return
    by, gold = load(a)
    dump, feat = [], {}
    for k, v in by.items():
        f = features(v)
        feat[k] = f
        g = gold.get(k, "")
        for r, d in zip(v, f):
            row = {"rec": k[0], "tid": k[1], "frame": r["frame"], "gold": g,
                   "n_frames": len(v), "p_owner_raw": r["p_owner_raw"],
                   "p1_label": r["ownhold_pre_cap"],
                   "p2_label": r["final_owner_post_cap"],
                   "leaked": int(g == "other"
                                 and bool(int(r["final_owner_post_cap"])))}
            row.update(d)
            dump.append(row)
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(FIELDS))
            w.writeheader()
            w.writerows(dump)
        print(f"  {len(dump)} 帧-手 -> {a.out}")

    n = len(dump)
    flag = sum(r["is_temporal_outlier"] for r in dump)
    rt = sum(r["round_trip_recovered"] for r in dump)
    print(f"\n  === 基线 ===")
    print(f"  {len(by)} 条轨迹 / {n} 帧-手   标为时序异常 {flag} "
          f"({flag/n:.1%})   其中往返型 {rt} ({rt/n:.1%})")
    if not gold:
        return

    D = {k: v for k, v in by.items() if gold.get(k) in ("owner", "other")}
    print(f"\n  === 那 171 个漏帧，事先有没有征兆 ===")
    leak = [r for r in dump if r["leaked"]]
    ok = [r for r in dump if not r["leaked"] and r["gold"] == "other"]
    for name, s in (("漏掉的别人的手帧", leak), ("盖住的别人的手帧", ok)):
        if not s:
            continue
        fl = sum(r["is_temporal_outlier"] for r in s)
        rr = sum(r["round_trip_recovered"] for r in s)
        print(f"    {name:<16} n={len(s):<6} 异常 {fl:>4} = {fl/len(s):>6.1%}"
              f"    往返型 {rr:>4} = {rr/len(s):>6.1%}")
    print("    两行接近就说明这个信号对『这一帧会不会漏』没有判别力。")

    print(f"\n  === 多数票的安全边际 ===")
    def margin(v, key):
        o = sum(1 for r in v if int(r[key]))
        return abs(2 * o - len(v)) / len(v), (o * 2 > len(v))
    for key, arm in (("ownhold_pre_cap", "P1"),
                     ("final_owner_post_cap", "P2")):
        good, bad = [], []
        for k, v in D.items():
            if gold[k] != "other":
                continue
            m, maj_owner = margin(v, key)
            (bad if maj_owner else good).append((m, k, len(v)))
        gm = sorted(x[0] for x in good)
        print(f"    {arm}  判对的 {len(good)} 条  margin 中位 "
              f"{gm[len(gm)//2]:.2f}  p10 {gm[int(.1*len(gm))]:.2f}"
              f"   判错的 {len(bad)} 条")
        for m, k, ln in sorted(bad):
            fl = sum(d["is_temporal_outlier"] for d in feat[k])
            print(f"      × {k[0]}:{k[1]}  {ln} 帧  margin {m:.2f}  "
                  f"异常帧 {fl}/{ln} = {fl/ln:.0%}")
        if bad:
            lo = min(x[0] for x in bad)
            below = sum(1 for x in gm if x <= lo)
            print(f"      判错的最小 margin {lo:.2f}；"
                  f"判对的里有 {below}/{len(gm)} 条 margin 不高于它"
                  f" —— 光靠 margin 当闸门要误伤这么多。")

    print(f"\n  === 如果异常帧不参与投票（同一批 gold 上，只能当假设）===")
    for key, arm in (("ownhold_pre_cap", "P1"),
                     ("final_owner_post_cap", "P2")):
        chg = fixed = broke = 0
        for k, v in D.items():
            own = gold[k] == "owner"
            base = sum(1 for r in v if int(r[key])) * 2 > len(v)
            keep = [r for r, d in zip(v, feat[k])
                    if not d["is_temporal_outlier"]]
            if not keep:
                continue
            gated = sum(1 for r in keep if int(r[key])) * 2 > len(keep)
            if gated == base:
                continue
            chg += 1
            if gated == own:
                fixed += 1
            else:
                broke += 1
        print(f"    {arm}  改变了 {chg} 条整轨判定：修好 {fixed}，弄坏 {broke}")


if __name__ == "__main__":
    main()
