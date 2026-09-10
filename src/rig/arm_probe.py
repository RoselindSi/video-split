"""Does where the forearm leaves the frame say anything the model has not used?

THE CUE IS NOT NEW; ONLY ITS UNTHRESHOLDED FORM IS. `forearm_exit` runs on
every detection and `rule_owner` is that ray reduced to `exit_y >= 0.55H`. So
the question is not whether arm attachment carries signal -- the rule reaches
85.4% at track level on its own -- but whether the direction, the edge and the
height carry more than the one bit that was kept, and specifically whether
they carry it where the deployed system is still wrong.

WHOLE-POPULATION SEPARATION WOULD PROVE NOTHING NOW. The deployed stack is at
96.6% on tracks and 96.2% on foreign-hand frames; a cue that separates the
easy 96% is a cue that agrees with a model that is already right. What matters
is the residual: nine tracks whose majority blurs the wearer's own hand, and
the seven foreign tracks that leak frames. Those get their own column.

AVAILABILITY IS A RESULT, NOT A PRELIMINARY. The ray needs twenty-one finite
keypoints, and a hand that is cut off at the frame edge or too small may not
have them. If the arm cue is missing exactly where the model is wrong, it
cannot be the repair however well it separates elsewhere.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics

H_PANO = 900.0
W_PANO = 1600.0


def load(a):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(a.rows, encoding="utf-8-sig")):
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    gold = {}
    for f in a.gold:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = (r.get("track_truth")
                                          or r.get("human_ownership"))
    return by, gold


def has_arm(r):
    return r["exit_y"] not in ("", None) and r["arm_angle"] not in ("", None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    a = ap.parse_args()

    by, gold = load(a)
    D = {k: v for k, v in by.items() if gold.get(k) in ("owner", "other")}
    OWN = [k for k in D if gold[k] == "owner"]
    OTH = [k for k in D if gold[k] == "other"]

    print(f"\n  === 手臂线索有没有 ===")
    for name, ks in (("自己的手", OWN), ("别人的手", OTH)):
        fr = [r for k in ks for r in D[k]]
        ok = sum(1 for r in fr if has_arm(r))
        tk = sum(1 for k in ks if any(has_arm(r) for r in D[k]))
        print(f"    {name}  帧 {ok}/{len(fr)} = {ok/len(fr):.1%}   "
              f"轨迹 至少一帧有 {tk}/{len(ks)}")

    print(f"\n  === 出口边（前臂射线离开画面的那条边）===")
    print(f"    {'':<10}{'bottom':>9}{'left':>8}{'right':>8}{'top':>7}{'无':>6}")
    for name, ks in (("自己的手", OWN), ("别人的手", OTH)):
        c = collections.Counter(r["exit_edge"] or "无"
                                for k in ks for r in D[k])
        n = sum(c.values())
        print(f"    {name:<10}" + "".join(
            f"{c[e]/n:>8.1%} " for e in ("bottom", "left", "right", "top", "无")))

    print(f"\n  === 连续量的分布（帧级中位 / p10 / p90）===")
    def col(ks, fn):
        v = sorted(fn(r) for k in ks for r in D[k] if has_arm(r))
        if not v:
            return "—"
        return (f"{v[len(v)//2]:>7.2f}{v[int(.1*len(v))]:>8.2f}"
                f"{v[int(.9*len(v))]:>8.2f}")
    for label, fn in (("exit_y / H", lambda r: float(r["exit_y"]) / H_PANO),
                      ("exit_x / W", lambda r: float(r["exit_x"]) / W_PANO),
                      ("arm_angle°", lambda r: float(r["arm_angle"]))):
        print(f"    {label:<12} 自己 {col(OWN, fn)}    别人 {col(OTH, fn)}")

    print(f"\n  === 轨迹级：出口高度的中位与它自己的稳定性 ===")
    def tmed(k):
        v = [float(r["exit_y"]) / H_PANO for r in D[k] if has_arm(r)]
        return statistics.median(v) if v else None

    def tspread(k):
        v = [float(r["exit_y"]) / H_PANO for r in D[k] if has_arm(r)]
        if len(v) < 3:
            return None
        v.sort()
        return v[int(.75 * len(v))] - v[int(.25 * len(v))]
    for name, ks in (("自己的手", OWN), ("别人的手", OTH)):
        m = sorted(x for x in (tmed(k) for k in ks) if x is not None)
        s = sorted(x for x in (tspread(k) for k in ks) if x is not None)
        print(f"    {name}  出口高度中位 {m[len(m)//2]:.2f} "
              f"[p10 {m[int(.1*len(m))]:.2f}, p90 {m[int(.9*len(m))]:.2f}]"
              f"   轨迹内四分位距中位 {s[len(s)//2]:.2f}")
    lo = [k for k in OTH if (tmed(k) or 0) >= 0.55]
    print(f"    别人的手里出口高度中位 >= 0.55（规则会判成自己的）"
          f" {len(lo)}/{len(OTH)} = {len(lo)/len(OTH):.1%}")

    print(f"\n  === 残余：模型还在错的地方，手臂线索在不在、指哪 ===")
    maj = {k: sum(1 for r in v if int(r["final_owner_post_cap"])) * 2 > len(v)
           for k, v in D.items()}
    groups = [
        ("误糊的自己的手（整轨判反）",
         [k for k in OWN if not maj[k]]),
        ("正常的自己的手", [k for k in OWN if maj[k]]),
        ("有漏帧的别人的手",
         [k for k in OTH if any(int(r["final_owner_post_cap"]) for r in D[k])]),
        ("无漏帧的别人的手",
         [k for k in OTH if not any(int(r["final_owner_post_cap"])
                                    for r in D[k])]),
    ]
    print(f"    {'':<22}{'条':>4}{'有臂线索帧':>10}{'出口高度中位':>12}"
          f"{'规则判成自己':>12}")
    for name, ks in groups:
        if not ks:
            continue
        fr = [r for k in ks for r in D[k]]
        ok = sum(1 for r in fr if has_arm(r)) / max(1, len(fr))
        m = [x for x in (tmed(k) for k in ks) if x is not None]
        rule = sum(1 for k in ks
                   if sum(1 for r in D[k] if int(r["reference_owner"])) * 2
                   >= len(D[k]))
        print(f"    {name:<22}{len(ks):>4}{ok:>10.1%}"
              f"{(statistics.median(m) if m else float('nan')):>12.2f}"
              f"{rule}/{len(ks):<11}")
    print("    最后一列是冻结规则在这一组上的判定；它和第三列一起说明"
          "\n    手臂线索在残余上是站在模型那边还是人那边。")


if __name__ == "__main__":
    main()
