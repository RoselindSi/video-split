"""Why does per-hand Qwen beat V1? Swap their inputs and see which way the gap moves.

Runs in `/workspace/venv_rig`, per test root (e2e_main2 is the main one: V1's
raw classifier puts 336 foreign frames on "self" there and Q1 puts 5, so the
gap has room to move; fresh29 is reported beside it).

    Qwen, V1's view           Q0     Qwen's view, no description   (reference)
                              QV0    V1's view (hand 128 + context 128 +
                                     geometry as text), no description
                              Q1     Qwen's view + its own description (reference)
                              QV1    V1's view + a description written from V1's view
    V1's architecture         VV     V1's view, retrained on V1's 1,013 hands
                              VVng   V1's view without geometry
                              VQ     Qwen's view (whole frame + box, crop)
    references                V1 raw / V1 deployed from the dump; V1 (JPEG) is
                              the shipped V1 on the JPEG-cut crops, to show
                              what cutting from saved frames costs

WRITTEN BEFORE RUNNING -- HOW EACH SIDE IS READ (diagnostic, not a ship rule).
All on the raw form, e2e_main2, M2 = foreign frames called self, M1 = own
flips per 100 pairs.

  Qwen side. share = (QV1 - Q1) / (V1 raw - Q1), per metric; the same for
  QV0 against Q0.
      share >= 0.5 on M2   V1's view is what holds V1 back: most of the gap
                           comes back when Qwen is limited to it
      share <= 0.2 on M2   the view is not the reason; the model is
      in between           both
  V1 side. VQ - VV with a recording bootstrap interval, and VQ - VVng.
      VQ below VV on M2 with the interval under zero
                           Qwen's view helps even a small model trained on
                           1,013 hands -> distil into a wide-view student
      no difference        at this data size the architecture/data limit V1,
                           not what it sees
  Own-hand recognition (G) is printed with every row; a gain on M1 or M2
  that costs more than 2 points of G is reported as a trade, not a gain.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from src.semhand import GUARD
from src.semhand.evaluate import boot_diff, per_recording, summarise
from src.semhand.holdreplay import invert_prior, key, replay

QWEN = ("Q0", "QV0", "Q1", "QV1")
V1S = ("VV", "VVng", "VQ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from src.selfother.train import read_gold

    dump = list(csv.DictReader(open(a.dump, encoding="utf-8")))
    by_key = {key(r): r for r in dump}
    prior = invert_prior(dump)
    gold = read_gold(a.gold)
    probs = {}
    for arm in QWEN:
        for f in glob.glob(os.path.join(a.root, "qwen", f"{arm}_*.jsonl")):
            for line in open(f):
                if line.strip():
                    d = json.loads(line)
                    probs.setdefault(arm, {})[d["id"]] = d["p"]
    pc = os.path.join(a.root, "crossv1", "pred.csv")
    if os.path.exists(pc):
        for r in csv.DictReader(open(pc)):
            for arm in V1S:
                if arm in r:
                    probs.setdefault(arm, {})[r["id"]] = float(r[arm])
    cross = {}
    for p in glob.glob(os.path.join(a.root, "cross", "index_test_*.csv")):
        cross.update({r["id"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"})
    probs["V1 (JPEG)"] = {k: float(r["p_v1_jpeg"]) for k, r in cross.items()}

    fresh = []
    for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "index.csv"))):
        fresh += [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"]
    ident = lambda r: f"{r['rec']}|{r['frame']}|{r['tid']}"
    arms = [x for x in QWEN + V1S + ("V1 (JPEG)",) if x in probs]
    rows = [r for r in fresh if all(ident(r) in probs[x] for x in arms)]
    recs = sorted({r["rec"] for r in rows})
    tracks = {(r["rec"], str(r["tid"])) for r in rows} & set(gold)
    print(f"{a.root}: 共同集合 {len(rows)}/{len(fresh)} 只手，{len(recs)} 段录像，gold 轨迹 {len(tracks)}"
          f"（自己 {sum(gold[t] == 'owner' for t in tracks)} / 别人 {sum(gold[t] == 'other' for t in tracks)}）；"
          f"缺的臂 {[x for x in QWEN + V1S if x not in probs]}")

    labels = {}
    for arm in arms + ["V1 raw"]:
        p = ({key(r): float(by_key[key(r)]["p_owner_raw"]) for r in rows} if arm == "V1 raw"
             else {key(r): probs[arm][ident(r)] for r in rows})
        labels[(arm, "raw")] = {k: v >= 0.5 for k, v in p.items()}
        labels[(arm, "held")] = replay(rows, p, prior)
    labels[("V1 deployed", "held")] = {key(r): by_key[key(r)]["final_owner_post_cap"] == "1" for r in rows}
    counts = {k: per_recording(rows, lab, gold) for k, lab in labels.items()}
    report = {"root": a.root, "n": len(rows), "tables": {}, "reading": {}}
    for form in ("raw", "held"):
        tab = {arm: summarise(c, recs) for (arm, f), c in counts.items() if f == form}
        report["tables"][form] = tab
        print(f"\n  === {form} ===\n  {'':<12}{'M1 翻转/100':>12}{'M2 别人帧判自己':>16}{'率':>8}"
              f"{'别人轨迹有误':>12}{'G 自己帧召回':>14}")
        for arm, m in tab.items():
            o = m["other_tracks_any_self"]
            print(f"  {arm:<12}{m['M1_own_flips_per100']:>12.2f}{m['M2_other_frames_self']:>16d}"
                  f"{m['M2_other_frame_rate']:>8.2%}{f'{o[0]}/{o[1]}':>12}{m['G_own_frames_self']:>14.2%}")

    tab = report["tables"]["raw"]
    cr = {arm: c for (arm, f), c in counts.items() if f == "raw"}
    print("\n  === 读法（事先写定）===")
    for full, lim in (("Q1", "QV1"), ("Q0", "QV0")):
        if full in tab and lim in tab:
            sh = {}
            for m, name in (("M2", "M2_other_frames_self"), ("M1", "M1_own_flips_per100")):
                gap = tab["V1 raw"][name] - tab[full][name]
                sh[m] = (tab[lim][name] - tab[full][name]) / gap if gap else float("nan")
            d = boot_diff(cr[lim], cr[full], recs)
            verdict = ("V1 的输入是主因" if sh["M2"] >= 0.5 else
                       "输入不是主因，是模型" if sh["M2"] <= 0.2 else "两者都有")
            report["reading"][f"{lim}_vs_{full}"] = {"share": sh, "diff": d, "reading": verdict}
            print(f"  Qwen {lim} vs {full}: M2 收回差距 {sh['M2']:.0%}，M1 收回 {sh['M1']:.0%}  -> {verdict}   "
                  f"ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  "
                  f"ΔM1 {d['M1'][0]:+.2f} [{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  ΔG {d['G'][0]:+.3f}")
    for base in ("VV", "VVng"):
        if "VQ" in tab and base in tab:
            d = boot_diff(cr["VQ"], cr[base], recs)
            helps = d["M2"][2] < 0
            trade = d["G"][0] < -GUARD
            verdict = ("Qwen 的输入对小模型也有用" if helps else "看不出输入带来的差别") + ("（但以 G 为代价）" if trade else "")
            report["reading"][f"VQ_vs_{base}"] = {"diff": d, "reading": verdict}
            print(f"  V1  VQ vs {base}: ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  "
                  f"ΔM1 {d['M1'][0]:+.2f} [{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  ΔG {d['G'][0]:+.3f}  -> {verdict}")
    out = a.out or os.path.join(a.root, "crosscheck.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
