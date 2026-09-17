"""The replacement decision for the GPT-6 second-teacher students, on batches 7-9.

THE RULE, as committed in `distil_train.py` before these arms trained:
at the post-processing the ablation chose for S_wide (geom_w 0.25, no cap),
an arm replaces S_wide only if, on batches 7+8+9 pooled,
    (a) its foreign-hand frames called self are FEWER than S_wide's, and
    (b) it passes the four ship criteria against V1 deployed there.
If both arms qualify, the one with fewer foreign frames called self wins.

AMENDED 2026-09-17, AFTER final_789 CHOSE ITS WINNER AND BEFORE ANY GPT-6
STUDENT HAS BEEN PREDICTED OR SCORED. `final_789` chose S_wide_h at geom_w 0,
no cap, and S_wide at geom_w 0.25 failed its criteria on 7-9 (204 foreign
frames against V1's 187). Beating a configuration that has just failed is no
bar, so the incumbent becomes the one that ships:
    INC     S_wide_h, geom_w 0, no cap
    arms    S_wide_g6 and S_wide_g6and read at that same post-processing
and (a) now reads "fewer foreign frames called self than S_wide_h". The rest
of the rule is unchanged. Known confound, stated now: S_wide_h carries V1's
1,013 human hands and the GPT-6 arms do not.

ADDED BEFORE TRAINING (2026-09-17), NOT A CHANGE TO THE RULE:
    gate    if Q1 agrees with GPT-6 on fewer than 80% of GPT-6's hands, the
            labels or the monocular view are broken and nothing is trained
            (checked in g6_run2.sh, printed by g6_report.agreement)
    report  the same arms beside the 7-9 winner of `final_789.py`, and on
            batches 4-9 pooled, descriptively
"""
from __future__ import annotations

import argparse
import json

from src.semhand.distil_ablate import load, name, show
from src.semhand.final_789 import BASE, check

INC = ("S_wide_h", (0.0, None))
ARMS = (("S_wide_g6", (0.0, None)), ("S_wide_g6and", (0.0, None)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", action="append", required=True)
    ap.add_argument("--old", action="append", default=[])
    ap.add_argument("--out", default="/workspace/distil/student/g6_final.json")
    a = ap.parse_args()
    students = ("S_wide", "S_wide_h", "S_wide_g6", "S_wide_g6and")
    tab, counts, recs = load(a.new, students)
    keep = {k: v for k, v in tab.items() if k == BASE or (k[0] in students and k[1] in ("raw", (0.25, None), (0.0, None)))}
    show(keep, "第七+八+九批")
    inc_m2 = tab[INC]["M2_other_frames_self"]
    res, winners = {}, []
    for k in ARMS:
        ok, c, d = check(tab, counts, recs, k)
        fewer = tab[k]["M2_other_frames_self"] < inc_m2
        res[name(k)] = {"passes_vs_v1": ok, "fewer_than_incumbent": fewer, "criteria": c, "diff_vs_v1": d}
        print(f"  {name(k):<26} 过四条 {ok}  别人帧 {tab[k]['M2_other_frames_self']} vs S_wide_h {inc_m2} → 更少 {fewer}"
              f"  ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  ΔM1 {d['M1'][0]:+.2f} ΔG {d['G'][0]:+.3f}")
        if ok and fewer:
            winners.append(k)
    ship = min(winners, key=lambda k: tab[k]["M2_other_frames_self"]) if winners else None
    print(f"\n=== 事先写定的规则：{'替换为 ' + name(ship) if ship else '不替换，保留 S_wide_h w0 无封顶'} ===")
    if a.old:
        t2, _, _ = load(a.new + a.old, students)
        show({k: v for k, v in t2.items() if k == BASE or (k[0] in students and k[1] in ((0.25, None), (0.0, None)))},
             "第四到九批合并（描述性）")
    json.dump({"results": res, "ship": name(ship) if ship else None}, open(a.out, "w"),
              indent=1, ensure_ascii=False, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
