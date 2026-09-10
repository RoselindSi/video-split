"""Cover the frames where the detector blinked and the tracker did not.

WHAT IS LEFT AFTER CONSOLIDATION IS NOT A CLASSIFICATION ERROR. Broadcasting
the track's majority takes foreign-hand exposure from 32 events to 17, and
every one of the 17 survivors is a stretch where the detector produced no box
at all. The classifier has no opinion about those frames because they do not
exist; the tracker, meanwhile, is still holding the identity across them.

THE BOUND COMES FROM THE TRACKER, NOT FROM THE RESIDUAL. `MAX_LOST = 5` is
already this system's statement about how long a missing detection is still
the same hand -- past it the track is deleted. Using the same number here
means the interpolation never claims an identity the tracker would not claim,
and it is not a threshold read off the gaps that happen to be in this corpus.
That the observed gaps top out at exactly five is a fact about the corpus, and
it is why the improvement measured here cannot be the evidence that this
generalises.

INTERIOR ONLY. A gap with an observed detection at both ends is interpolation;
a gap running to the end of a track is extrapolation, and extrapolation is
already handled, causally and with a much shorter horizon, by `coasting`. The
distinction matters because the failure mode is different: interpolation can
be checked against a known destination and extrapolation cannot.

CENTRE AND LOG SIZE, NOT FOUR CORNERS. Interpolating corners lets a box change
aspect on the way across; interpolating cx, cy, log w and log h moves it the
way a hand approaching or receding actually scales.

THE RISK THIS CORPUS CANNOT SHOW. If the tracker bridged two different hands,
interpolation paints a box along the path between them and blurs everything on
it. `mixed_identity` came back 0 of 274 here, so that failure has no instance
in this data and its absence is not evidence -- it is the reason the endpoint
continuity of every filled gap is reported rather than assumed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics

# The tracker's own patience. Not chosen from the residual.
from src.rig.hand_track import MAX_LOST
INTERP_MAX_GAP = MAX_LOST
# What the deployed pipeline already covers forward from the last detection.
COAST = 2
FPS = 30.0


def load(a):
    rows = list(csv.DictReader(open(a.rows, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    gold = {}
    for f in a.gold:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    return {k: v for k, v in by.items()
            if gold.get(k) in ("owner", "other")}, gold


def box(r):
    x0, y0 = float(r["x0"]), float(r["y0"])
    x1, y1 = float(r["x1"]), float(r["y1"])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0,
            max(1.0, x1 - x0), max(1.0, y1 - y0))


def interpolate(b0, b1, alpha):
    """-> (cx, cy, w, h) a fraction alpha of the way from b0 to b1."""
    cx = (1 - alpha) * b0[0] + alpha * b1[0]
    cy = (1 - alpha) * b0[1] + alpha * b1[1]
    w = math.exp((1 - alpha) * math.log(b0[2]) + alpha * math.log(b1[2]))
    h = math.exp((1 - alpha) * math.log(b0[3]) + alpha * math.log(b1[3]))
    return (cx, cy, w, h)


def gaps_of(v):
    """-> [(i, j, gap_len)] interior gaps, both ends an observed detection."""
    out = []
    for i in range(len(v) - 1):
        g = int(v[i + 1]["frame"]) - int(v[i]["frame"]) - 1
        if 0 < g <= INTERP_MAX_GAP:
            out.append((i, i + 1, g))
    return out


def continuity(v, i, j):
    """How far the far endpoint sits from where the near side was heading.

    In box widths, so it means the same thing at any distance. A large value
    is what an identity switch across a gap would look like, and it is the one
    check that does not need a label."""
    b0, b1 = box(v[i]), box(v[j])
    f0, f1 = int(v[i]["frame"]), int(v[j]["frame"])
    size = math.sqrt(b0[2] * b0[3])
    if i == 0:
        vel = (0.0, 0.0)
    else:
        bp = box(v[i - 1])
        dt = max(1, f0 - int(v[i - 1]["frame"]))
        vel = ((b0[0] - bp[0]) / dt, (b0[1] - bp[1]) / dt)
    dt = f1 - f0
    pred = (b0[0] + vel[0] * dt, b0[1] + vel[1] * dt)
    return math.hypot(b1[0] - pred[0], b1[1] - pred[1]) / size


def coverage(v, label, fill):
    """Per frame over the track's span: is a foreign hand covered?"""
    seen = {int(r["frame"]): label(r) for r in v}
    filled = set()
    if fill:
        for i, j, g in gaps_of(v):
            for f in range(int(v[i]["frame"]) + 1, int(v[j]["frame"])):
                filled.add(f)
    lo, hi = int(v[0]["frame"]), int(v[-1]["frame"])
    out, last = [], None
    for f in range(lo, hi + 1):
        if f in seen:
            out.append(not seen[f])       # labelled other -> covered
            last = f
        elif f in filled:
            out.append(True)
        else:
            out.append(last is not None and f - last <= COAST)
    return out


def runs(flags):
    out, cur = [], 0
    for x in flags:
        if not x:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    D, gold = load(a)
    # TRUE MEANS THE TRACK'S MAJORITY SAYS `owner`. Named for what it holds:
    # reading it as `other` once made the fill run on exactly the wrong half.
    maj_owner = {k: sum(1 for r in v if int(r["final_owner_post_cap"])) * 2
                 > len(v) for k, v in D.items()}

    def lab(r):
        return maj_owner[(r["rec"], r["tid"])]

    print(f"\n  === D0 vs D1（{len(D)} 条：别人的手 "
          f"{sum(1 for k in D if gold[k] == 'other')}，自己的手 "
          f"{sum(1 for k in D if gold[k] == 'owner')}）===")
    print(f"  规则冻结：gap <= MAX_LOST = {INTERP_MAX_GAP}，只补 interior，"
          f"两端都必须是真实检测，\n  只在整轨判为 other 的轨迹上补，"
          f"cx/cy/log w/log h 线性插值")
    print(f"\n  {'':<34}{'暴露事件':>9}{'未覆盖帧':>9}{'秒':>7}{'最长s':>8}"
          f"{'录像':>6}")
    for name, fill in (("D0  P2 + 多数票 + coast2", False),
                       ("D1  D0 + 有界内插", True)):
        allr, recs = [], set()
        for k, v in D.items():
            if gold[k] != "other":
                continue
            rr = runs(coverage(v, lab, fill and not maj_owner[k]))
            allr += rr
            if rr:
                recs.add(k[0])
        allr.sort()
        print(f"  {name:<34}{len(allr):>9}{sum(allr):>9}{sum(allr)/FPS:>7.1f}"
              f"{(allr[-1]/FPS if allr else 0):>8.2f}{len(recs):>6}")

    n_gap = n_frame = 0
    bad_blur = bad_gap = 0
    cont = []
    for k, v in D.items():
        if maj_owner[k]:
            continue                       # only other-majority tracks fill
        for i, j, g in gaps_of(v):
            n_gap += 1
            n_frame += g
            cont.append(continuity(v, i, j))
            if gold[k] == "owner":
                bad_gap += 1
                bad_blur += g
    cont.sort()
    print(f"\n  补了 {n_gap} 个空洞 / {n_frame} 帧 ({n_frame/FPS:.1f}s)")
    print(f"  其中落在『整轨判 other 但人判是自己的手』的轨迹上：{bad_gap} 个 / "
          f"{bad_blur} 帧 = 新增误糊 {bad_blur/FPS:.2f}s")
    if cont:
        print(f"\n  端点连续性（用 gap 前速度外推的位置 vs 实际另一端，"
              f"框宽为单位）")
        print(f"    中位 {cont[len(cont)//2]:.2f}   p90 "
              f"{cont[int(.9*len(cont))]:.2f}   最大 {cont[-1]:.2f}   "
              f">2 个框宽的 {sum(1 for x in cont if x > 2)}")
        print("    这批里没有 mixed_identity，所以这个分布只说明规则在干净的"
              "\n    tracker 上不会乱接；它无法证明域漂移之后仍然安全。")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rec", "tid", "human", "track_majority", "frame",
                        "cx", "cy", "w", "h", "gap_len", "endpoint_continuity"])
            for k, v in sorted(D.items()):
                if maj_owner[k]:
                    continue
                for i, j, g in gaps_of(v):
                    b0, b1 = box(v[i]), box(v[j])
                    c = continuity(v, i, j)
                    f0 = int(v[i]["frame"])
                    for t in range(1, g + 1):
                        cx, cy, bw, bh = interpolate(b0, b1, t / (g + 1.0))
                        w.writerow([k[0], k[1], gold[k],
                                    "owner" if maj_owner[k] else "other",
                                    f0 + t,
                                    round(cx, 1), round(cy, 1), round(bw, 1),
                                    round(bh, 1), g, round(c, 3)])
        print(f"\n  插出来的框 -> {a.out}")


if __name__ == "__main__":
    main()
