"""The last decision: S_wide_g6 or S_wide_hg6and, on a batch nothing has touched.

WHY THIS BATCH EXISTS. On batches 7-9 the pre-registered rule chose
S_wide_hg6and (23 foreign frames against S_wide_g6's 26), but the two are
tied on that metric (difference +3, interval [-10,+19]) and S_wide_g6 keeps
0.7 points more of the wearer's own hands (interval excludes zero). Choosing
between them on 7-9 after seeing those numbers would be tuning on the test
set, so one more batch is drawn and labelled, and the rule below is written
before it exists.

THE BATCH. 30 recordings enriched for other people's hands by the shared
detector's count alone (a scanned frame with >= 3 hand detections), plus 12
drawn blind, on days and devices V1 never trained on and disjoint from every
earlier batch, the distillation pool, the GPT-6 label set and every human
gold set. Same zone tool, same 400-frame windows.

WRITTEN BEFORE THE BATCH IS LABELLED -- THE RULE. Both arms read at the frozen
post-processing (geom_w 0, no cap); V1 deployed is the reference.
  1. An arm may ship only if it passes the four criteria of `distil_train.py`
     against V1 deployed on this batch (M2 no worse, M1 no worse, G within
     GUARD, and at least one of M1/M2 better with a 95% interval below zero).
  2. Among the arms that pass: fewer foreign-hand frames called self wins, BUT
     only if the 95% recording-bootstrap interval of that difference excludes
     zero. Otherwise the two count as tied.
  3. Tied -> the arm with the higher own-hand recall G ships. Still tied ->
     S_wide_g6, the simpler recipe (teacher labels only, no human rows).
  4. If neither passes, nothing replaces the currently shipped V1 and the
     result is reported as a failure.
The enriched and blind halves are also printed apart, descriptively.
"""
from __future__ import annotations

import argparse
import json

from src.semhand.distil_ablate import load, name, show
from src.semhand.final_789 import BASE, check

ARMS = (("S_wide_g6", (0.0, None)), ("S_wide_hg6and", (0.0, None)))
STUDENTS = ("S_wide_g6", "S_wide_hg6and", "S_wide_h", "S_wide")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True)
    ap.add_argument("--out", default="/workspace/distil/student/final_b10.json")
    a = ap.parse_args()
    from src.semhand.evaluate import boot_diff
    tab, counts, recs = load(a.batch, STUDENTS)
    show({k: v for k, v in tab.items() if k == BASE or (k[0] in STUDENTS and k[1] in ("raw", (0.0, None)))},
         "新一批（富集 + 盲抽）")
    res, passed = {}, []
    for k in ARMS:
        ok, c, d = check(tab, counts, recs, k)
        res[name(k)] = {"passes": ok, "criteria": c, "diff_vs_v1": d}
        print(f"  {name(k):<26} 对 V1 {'通过' if ok else '未通过'}  ΔM2 {d['M2'][0]:+d} "
              f"[{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  ΔM1 {d['M1'][0]:+.2f}  ΔG {d['G'][0]:+.3f}  {c}")
        if ok:
            passed.append(k)
    ship, why = None, ""
    if not passed:
        why = "两个都没过判据，不替换"
    elif len(passed) == 1:
        ship, why = passed[0], "只有它过了判据"
    else:
        g6, hg = ARMS
        d = boot_diff(counts[g6], counts[hg], recs)
        lo, hi = d["M2"][1], d["M2"][2]
        if hi < 0:
            ship, why = g6, f"别人帧更少且区间不跨 0（Δ {d['M2'][0]:+d} [{lo:+.0f},{hi:+.0f}]）"
        elif lo > 0:
            ship, why = hg, f"别人帧更少且区间不跨 0（Δ {-d['M2'][0]:+d}）"
        else:
            gg, gh = tab[g6]["G_own_frames_self"], tab[hg]["G_own_frames_self"]
            ship = g6 if gg >= gh else hg
            why = (f"别人帧打平（Δ {d['M2'][0]:+d} [{lo:+.0f},{hi:+.0f}]），按自己的手召回选"
                   f"（{gg:.2%} vs {gh:.2%}）")
    print(f"\n=== 事先写定的规则：{name(ship) + ' 上线' if ship else '不替换'} —— {why} ===")
    json.dump({"results": res, "ship": name(ship) if ship else None, "reason": why},
              open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
