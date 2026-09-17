"""Post-processing ablation for the distilled students: how much of V1's prior they need.

Runs in `/workspace/venv_rig`.

WHY. V1's geometric prior is blended at 0.5 because V1's classifier needed it:
on the held-out batches V1's raw classifier puts 393 foreign frames on "self"
and the blend brings that to 165. The students do not need rescuing, and the
same blend moved S_wide from 0 foreign frames to 17 on batch 5. So the weight
is re-chosen for the students, and the two-owner cap is checked alongside.

GRID. For each classifier (V1's raw P, S_v1, S_wide, S_wide_h):
    geom_w  in {0, 0.25, 0.5}
    cap     in {2, none}
and the raw form (no prior, no OwnHold, no cap) for reference.

WRITTEN BEFORE RUNNING -- CHOSEN ON DEV, READ ONCE ON TEST.
    dev   e2e_main2 + fresh29 pooled (human track gold; both already used)
    test  zone batches 4 + 5 + 6 pooled (held out; looked at once, for the
          pass/fail at geom_w 0.5 cap 2)
  1. For each student, pick its (geom_w, cap) on dev: fewest M2 frames, then
     lowest M1, subject to G >= V1 deployed's dev G - GUARD. Ties go to the
     simpler setting (lower geom_w, then cap 2 kept).
  2. Pick the student to ship on dev the same way, using each student's chosen
     setting.
  3. Report every setting on test for transparency, but the configuration that
     ships is the one step 2 names; its test numbers are checked against the
     four ship criteria in `distil_train.py`, and if it fails them there, the
     already-validated setting (geom_w 0.5, cap 2) stays.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

from src.semhand import GUARD
from src.semhand.distil_eval import one_batch
from src.semhand.evaluate import boot_diff, per_recording, summarise
from src.semhand.holdreplay import invert_prior, key, replay

WEIGHTS = (0.0, 0.25, 0.5)
CAPS = (2, None)
STUDENTS = ("S_v1", "S_wide", "S_wide_h")


def settings_for(rows, gold, preds, prior, by_key):
    """-> {(arm, setting): labels}; setting is 'raw' or (geom_w, cap)."""
    ident = lambda r: f"{r['rec']}|{r['frame']}|{r['tid']}"
    out = {("V1 deployed", "shipped"): {key(r): by_key[key(r)]["final_owner_post_cap"] == "1" for r in rows}}
    arms = {"V1 cls": {key(r): float(by_key[key(r)]["p_owner_raw"]) for r in rows}}
    for arm in STUDENTS:
        if arm in preds:
            arms[arm] = {key(r): preds[arm][ident(r)] for r in rows}
    for arm, p in arms.items():
        out[(arm, "raw")] = {k: v >= 0.5 for k, v in p.items()}
        for w, c in itertools.product(WEIGHTS, CAPS):
            out[(arm, (w, c))] = replay(rows, p, prior, geom_w=w, cap=c)
    return out


def load(specs):
    import csv
    rows_all, gold_all, labels_all = [], {}, {}
    for spec in specs:
        name, dump, labels_p, clips, pred, strata = spec.split(":")
        rows, gold, _, _ = one_batch(dump, labels_p, clips, pred, strata)
        dump_rows = list(csv.DictReader(open(dump, encoding="utf-8")))
        prior = invert_prior(dump_rows)
        by_key = {key(r): r for r in dump_rows}
        preds = {}
        for r in csv.DictReader(open(pred)):
            for arm, v in r.items():
                if arm != "id" and v != "":
                    preds.setdefault(arm, {})[r["id"]] = float(v)
        rows_all += rows
        gold_all.update(gold)
        for k, lab in settings_for(rows, gold, preds, prior, by_key).items():
            labels_all.setdefault(k, {}).update(lab)
    recs = sorted({r["rec"] for r in rows_all})
    counts = {k: per_recording(rows_all, lab, gold_all) for k, lab in labels_all.items()}
    return {k: summarise(c, recs) for k, c in counts.items()}, counts, recs


def name(k):
    arm, s = k
    if s in ("raw", "shipped"):
        return f"{arm} {s}"
    return f"{arm} w{s[0]:g} cap{s[1] or '-'}"


def pick(tab, arms, guard_g):
    def rank(k):
        m = tab[k]
        w, c = k[1]
        return (m["M2_other_frames_self"], m["M1_own_flips_per100"], w, c is None)
    cands = [k for k in tab if k[0] in arms and k[1] not in ("raw", "shipped")
             and tab[k]["G_own_frames_self"] >= guard_g]
    return min(cands, key=rank) if cands else None


def show(tab, title):
    print(f"\n=== {title} ===")
    print(f"  {'':<24}{'M1 翻转/100':>12}{'M2 别人帧判自己':>16}{'率':>8}{'别人轨迹有误':>12}{'G 自己帧召回':>14}")
    for k in sorted(tab, key=lambda k: (k[0], str(k[1]))):
        m = tab[k]
        o = m["other_tracks_any_self"]
        print(f"  {name(k):<24}{m['M1_own_flips_per100']:>12.2f}{m['M2_other_frames_self']:>16d}"
              f"{m['M2_other_frame_rate']:>8.2%}{f'{o[0]}/{o[1]}':>12}{m['G_own_frames_self']:>14.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="append", required=True, help="name:dump:labels:clips:pred:strata")
    ap.add_argument("--test", action="append", required=True)
    ap.add_argument("--out", default="/workspace/distil/student/ablation.json")
    a = ap.parse_args()
    dev, _, _ = load(a.dev)
    show(dev, "开发集 e2e_main2 + fresh29")
    guard = dev[("V1 deployed", "shipped")]["G_own_frames_self"] - GUARD
    chosen = {arm: pick(dev, [arm], guard) for arm in STUDENTS}
    print("\n=== 各学生在开发集上选出的设置（事先写定的规则）===")
    for arm, k in chosen.items():
        print(f"  {arm:<10} -> {name(k) if k else '无满足保护线的设置'}")
    ship = pick({k: dev[k] for k in chosen.values() if k}, STUDENTS, guard)
    if ship is None:
        raise SystemExit("没有学生设置满足开发集保护线，保留 w0.5 cap2")
    print(f"  上线候选 -> {name(ship)}")

    test, counts, recs = load(a.test)
    show(test, "测试集 第四+五+六批（只读一次）")
    base = ("V1 deployed", "shipped")
    d = boot_diff(counts[ship], counts[base], recs)
    s, b = test[ship], test[base]
    c1 = s["M2_other_frames_self"] <= b["M2_other_frames_self"]
    c2 = s["M1_own_flips_per100"] <= b["M1_own_flips_per100"]
    c3 = s["G_own_frames_self"] >= b["G_own_frames_self"] - GUARD
    c4 = [m for m in ("M1", "M2") if d[m][2] < 0]
    ok = c1 and c2 and c3 and bool(c4)
    print(f"\n=== 上线候选 {name(ship)} 对 V1 deployed（测试集）：{'通过' if ok else '未通过，保留 w0.5 cap2'} ===")
    print(f"  ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  ΔM1 {d['M1'][0]:+.2f} "
          f"[{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  ΔG {d['G'][0]:+.3f}  (1 {c1} 2 {c2} 3 {c3} 4 {c4})")
    json.dump({"dev": {name(k): v for k, v in dev.items()}, "test": {name(k): v for k, v in test.items()},
               "chosen": {arm: name(k) if k else None for arm, k in chosen.items()},
               "ship": name(ship), "ship_passes": ok, "diff": d},
              open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
