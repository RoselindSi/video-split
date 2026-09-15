"""Confirm per-hand Qwen Q1 on recordings it was not observed on.

WHY THIS SET. Q1 looked best on fresh29, but Q1 was not a registered verdict
arm there and fresh29 has been scored many times, so that is an observation.
e2e_main2 is outside fresh29 and testpkg_2, carries 274 human ownership
tracks (own_gold_p1-p3: 169 owner, 99 other, 6 ambiguous and dropped), and
Qwen has never seen any of it. It was V1's selection set, which can only flatter
V1 -- the comparison below is tilted against Qwen, not towards it.

WHAT IS SCORED. Every dumped detection on a gold track, every STRIDE-th frame
from the clip start, all cameras (no stereo filter). The same frames for every
row:

    Q1 raw        per-hand Qwen: one boxed hand + crop + its own description;
                  P(true) >= 0.5, no smoothing                     <- the claim
    Q1 held       the same P through V1's prior + OwnHold, replayed
    V1 deployed   the dump's own 30 fps verdict at these frames    <- the bar
    V1 raw        V1's classifier P >= 0.5

WRITTEN BEFORE RUNNING -- Q1 IS CONFIRMED iff, pooled over e2e_main2,

    (1) M2(Q1 raw) <= M2(V1 deployed)     foreign-hand frames labelled self
    (2) M1(Q1 raw) <= M1(V1 deployed)     own-hand flips per 100 pairs
    (3) G(Q1 raw)  >= G(V1 deployed) - 0.02   own-hand frames labelled self

on point estimates, with 95% recording-bootstrap intervals of each difference
printed beside them. Confirmed means Q1's labels are at least as good as V1's
shipped output without any smoothing, which is the bar for using them to
train V1. It does not by itself mean Q1 should replace V1.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

from src.semhand import GUARD
from src.semhand.evaluate import boot_diff, per_recording, summarise
from src.semhand.holdreplay import invert_prior, key, replay


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/semhand_e2e")
    ap.add_argument("--dump", default="/workspace/own_dump2.csv")
    ap.add_argument("--gold", action="append", default=[
        f"/workspace/selfother/gold/own_gold_p{i}.csv" for i in (1, 2, 3)])
    ap.add_argument("--arm", default="Q1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from src.selfother.train import read_gold

    dump = list(csv.DictReader(open(a.dump, encoding="utf-8")))
    by_key = {key(r): r for r in dump}
    prior = invert_prior(dump)
    gold = read_gold(a.gold)
    q = {}
    for f in glob.glob(os.path.join(a.root, "qwen", f"{a.arm}_*.jsonl")):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                q[d["id"]] = d["p"]
    rows = []
    n_index = 0
    for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "index.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n_index += 1
            if r["status"] == "ok" and f"{r['rec']}|{r['frame']}|{r['tid']}" in q:
                rows.append(r)
    recs = sorted({r["rec"] for r in rows})
    tracks = {(r["rec"], str(r["tid"])) for r in rows} & set(gold)
    print(f"{len(rows)}/{n_index} 只手有 {a.arm} 分数；{len(recs)} 段录像；gold 轨迹 {len(tracks)} "
          f"（自己 {sum(gold[t] == 'owner' for t in tracks)} / 别人 {sum(gold[t] == 'other' for t in tracks)}）")

    p_q = {key(r): q[f"{r['rec']}|{r['frame']}|{r['tid']}"] for r in rows}
    p_v1 = {key(r): float(by_key[key(r)]["p_owner_raw"]) for r in rows}
    labels = {
        f"{a.arm} raw": {k: v >= 0.5 for k, v in p_q.items()},
        f"{a.arm} held": replay(rows, p_q, prior),
        "V1 deployed": {key(r): by_key[key(r)]["final_owner_post_cap"] == "1" for r in rows},
        "V1 raw": {k: v >= 0.5 for k, v in p_v1.items()},
        "V1 held@stride": replay(rows, p_v1, prior),
    }
    counts = {name: per_recording(rows, lab, gold) for name, lab in labels.items()}
    tab = {name: summarise(c, recs) for name, c in counts.items()}
    print(f"\n  {'':<16}{'M1 翻转/100':>12}{'M2 别人帧判自己':>16}{'率':>8}{'别人轨迹有误':>12}"
          f"{'G 自己帧召回':>14}")
    for name, m in tab.items():
        o = m["other_tracks_any_self"]
        print(f"  {name:<16}{m['M1_own_flips_per100']:>12.2f}{m['M2_other_frames_self']:>16d}"
              f"{m['M2_other_frame_rate']:>8.2%}{f'{o[0]}/{o[1]}':>12}{m['G_own_frames_self']:>14.2%}")

    S, B = tab[f"{a.arm} raw"], tab["V1 deployed"]
    diff = boot_diff(counts[f"{a.arm} raw"], counts["V1 deployed"], recs)
    c1 = S["M2_other_frames_self"] <= B["M2_other_frames_self"]
    c2 = S["M1_own_flips_per100"] <= B["M1_own_flips_per100"]
    c3 = S["G_own_frames_self"] >= B["G_own_frames_self"] - GUARD
    ok = c1 and c2 and c3
    print(f"\n=== 判据（事先写定）：{a.arm} raw vs V1 deployed -> {'确认' if ok else '未确认'} ===")
    print(f"  (1) M2 {S['M2_other_frames_self']} vs {B['M2_other_frames_self']}  {c1}   "
          f"Δ {diff['M2'][0]:+d} [{diff['M2'][1]:+.0f}, {diff['M2'][2]:+.0f}]")
    print(f"  (2) M1 {S['M1_own_flips_per100']:.2f} vs {B['M1_own_flips_per100']:.2f}  {c2}   "
          f"Δ {diff['M1'][0]:+.2f} [{diff['M1'][1]:+.2f}, {diff['M1'][2]:+.2f}]")
    print(f"  (3) G {S['G_own_frames_self']:.2%} vs {B['G_own_frames_self']:.2%} (-{GUARD:.0%})  {c3}   "
          f"Δ {diff['G'][0]:+.3f} [{diff['G'][1]:+.3f}, {diff['G'][2]:+.3f}]")

    # Where the claim's errors are, track by track, so they can be looked at.
    errs = collections.defaultdict(list)
    for r in rows:
        t = (r["rec"], str(r["tid"]))
        s = labels[f"{a.arm} raw"][key(r)]
        if t in gold and (gold[t] == "other") == s:
            errs[(t, gold[t])].append(int(r["frame"]))
    print(f"\n{a.arm} raw 出错的轨迹（gold, 帧数, 帧范围）：")
    for (t, g), fr in sorted(errs.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t[0]} tid {t[1]:<4} {g:<6} {len(fr):>4} 帧  {min(fr)}-{max(fr)}")
    out = a.out or os.path.join(a.root, "confirm.json")
    json.dump({"table": tab, "diff_vs_v1_deployed": diff, "confirmed": ok,
               "criteria": {"M2": c1, "M1": c2, "G": c3},
               "errors": {f"{t[0]}|{t[1]}|{g}": fr for (t, g), fr in errs.items()}},
              open(out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
