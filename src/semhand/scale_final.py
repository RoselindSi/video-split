"""More data for the shipped recipe: does S_wide_g6_v2 replace S_wide_g6?

WHAT CHANGES. Only the amount of data, not the recipe, the view or the
post-processing:
    teacher pool  22,622 hands (264 blind recordings)
                  + a second pool of ~17k hands from 201 recordings whose
                    windows the detector's own hand count says contain other
                    people (`distil_prep --mode pool_from_scans`), labelled by
                    the same Q1 teacher
    GPT-6         15,710 -> 57,612 hands (985 recordings, 24 images each),
                  its own labels, as in the shipped recipe
Everything else is the shipped student's: V1's architecture, whole frame with
the box, AdamW 3e-4, 12 epochs, three seeds averaged, geom_w 0, no cap.

WRITTEN BEFORE THE NEW DATA IS LABELLED OR TRAINED.
    test    zone batches 7+8+9+10+11 pooled (461 foreign-hand tracks)
    1. S_wide_g6_v2 must pass the four criteria of `distil_train.py` against
       V1 deployed there.
    2. It replaces S_wide_g6 only if it has FEWER foreign-hand frames called
       self AND the 95% recording-bootstrap interval of that difference
       excludes zero. A tie keeps S_wide_g6: it is the configuration already
       validated on a batch drawn for that purpose.
    3. Own-hand recall is reported and must not fall more than GUARD below
       S_wide_g6's; if it does, the arm is rejected whatever M2 says.

STATED PLAINLY: batches 7-11 have all been read before. This is a
same-recipe scale-up judged on used data, so it is evidence about the
direction, not a clean measurement. Nothing here re-tunes post-processing.
"""
from __future__ import annotations

import argparse
import json

from src.semhand import GUARD
from src.semhand.distil_ablate import load, name, show
from src.semhand.evaluate import boot_diff
from src.semhand.final_789 import BASE, check

CUR = ("S_wide_g6", (0.0, None))
NEW = ("S_wide_g6_v2", (0.0, None))
STUDENTS = ("S_wide_g6", "S_wide_g6_v2")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True)
    ap.add_argument("--out", default="/workspace/distil/student/scale_final.json")
    a = ap.parse_args()
    tab, counts, recs = load(a.batch, STUDENTS)
    show({k: v for k, v in tab.items() if k == BASE or (k[0] in STUDENTS and k[1] in ("raw", (0.0, None)))},
         "第七到十一批合并")
    ok, c, d_v1 = check(tab, counts, recs, NEW)
    print(f"  {name(NEW):<24} 对 V1 {'通过' if ok else '未通过'}  ΔM2 {d_v1['M2'][0]:+d} "
          f"[{d_v1['M2'][1]:+.0f},{d_v1['M2'][2]:+.0f}]  ΔM1 {d_v1['M1'][0]:+.2f}  ΔG {d_v1['G'][0]:+.3f}  {c}")
    d = boot_diff(counts[NEW], counts[CUR], recs)
    guard = tab[NEW]["G_own_frames_self"] >= tab[CUR]["G_own_frames_self"] - GUARD
    fewer = d["M2"][2] < 0
    ship = ok and fewer and guard
    print(f"  对现役 {name(CUR)}：ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  "
          f"ΔM1 {d['M1'][0]:+.2f}  ΔG {d['G'][0]:+.3f}  (显著更少 {fewer}，保护线 {guard})")
    print(f"\n=== 事先写定的规则：{'替换为 ' + name(NEW) if ship else '保留 ' + name(CUR)} ===")
    json.dump({"vs_v1": {"passes": ok, "criteria": c, "diff": d_v1}, "vs_current": d,
               "ship": name(NEW) if ship else name(CUR)}, open(a.out, "w"),
              indent=1, ensure_ascii=False, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
