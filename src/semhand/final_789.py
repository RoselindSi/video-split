"""The last choice between two student configurations, made on batches 7-9.

WHY THERE ARE TWO. The ablation's pre-registered rule, run on the dev sets
(e2e_main2 + fresh29), chose
    A  S_wide,   geom_w 0.25, no owner cap
and A passed the ship criteria on batches 4-6. On those same test batches
    B  S_wide_h, geom_w 0,    no owner cap
looked better on the privacy metric (18 foreign frames against 43), but the
dev sets could not tell them apart (3 frames each) and choosing B because of
the test batches would turn them into a tuning set. Batches 7-9 settle it.

WHAT 7-9 ARE. Detector-enriched: recordings with a scanned frame holding three
or more hand detections, chosen without any ownership model, so they carry the
foreign hands 4-6 lacked. They are labelled with the same zone tool.

WRITTEN BEFORE THE ANNOTATIONS EXIST -- THE RULE.
    1. On batches 7+8+9 pooled, each of A and B is checked against V1 deployed
       with the four ship criteria of `distil_train.py`.
    2. If both pass, the one with fewer foreign frames called self ships; an
       exact tie keeps A (the dev choice).
    3. If only one passes, it ships. If neither passes, the configuration
       already validated on 4-6 at V1's settings (S_wide, geom_w 0.5, cap 2)
       stays the candidate and nothing new ships.
Batches 4-9 pooled are printed afterwards, descriptively.
"""
from __future__ import annotations

import argparse
import json

from src.semhand import GUARD
from src.semhand.distil_ablate import load, name, show
from src.semhand.evaluate import boot_diff

A = ("S_wide", (0.25, None))
B = ("S_wide_h", (0.0, None))
BASE = ("V1 deployed", "shipped")


def check(tab, counts, recs, k):
    d = boot_diff(counts[k], counts[BASE], recs)
    s, b = tab[k], tab[BASE]
    c = {"M2": s["M2_other_frames_self"] <= b["M2_other_frames_self"],
         "M1": s["M1_own_flips_per100"] <= b["M1_own_flips_per100"],
         "G": s["G_own_frames_self"] >= b["G_own_frames_self"] - GUARD,
         "sig": [m for m in ("M1", "M2") if d[m][2] < 0]}
    return c["M2"] and c["M1"] and c["G"] and bool(c["sig"]), c, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", action="append", required=True, help="batches 7-9 specs")
    ap.add_argument("--old", action="append", default=[], help="batches 4-6 specs, descriptive")
    ap.add_argument("--out", default="/workspace/distil/student/final_789.json")
    a = ap.parse_args()
    tab, counts, recs = load(a.new)
    show({k: v for k, v in tab.items() if k in (A, B, BASE) or k[1] == "raw"}, "第七+八+九批（富集别人的手）")
    res = {}
    for k in (A, B):
        ok, c, d = check(tab, counts, recs, k)
        res[name(k)] = {"passes": ok, "criteria": c, "diff": d}
        print(f"  {name(k):<24} {'通过' if ok else '未通过'}  ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]"
              f"  ΔM1 {d['M1'][0]:+.2f} [{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  ΔG {d['G'][0]:+.3f}  {c}")
    pa, pb = res[name(A)]["passes"], res[name(B)]["passes"]
    if pa and pb:
        ship = B if tab[B]["M2_other_frames_self"] < tab[A]["M2_other_frames_self"] else A
    elif pa or pb:
        ship = A if pa else B
    else:
        ship = None
    print(f"\n=== 事先写定的规则选出：{name(ship) if ship else '都未通过，保持 S_wide w0.5 cap2 作为候选'} ===")
    if a.old:
        t2, _, _ = load(a.new + a.old)
        show({k: v for k, v in t2.items() if k in (A, B, BASE)}, "第四到九批合并（描述性）")
    json.dump({"results": res, "ship": name(ship) if ship else None}, open(a.out, "w"),
              indent=1, ensure_ascii=False, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
