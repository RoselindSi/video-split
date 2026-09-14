"""Score every arm on the same sampled fresh29 hands, and apply the verdicts.

Runs in `/workspace/venv_rig`. The conditions and the rule are in
`src/semhand/__init__.py` and were committed before any arm was scored.

THE COMMON SET. A hand is scored only if every arm listed has a P for it: the
V1 arms need recovered keypoints, the Qwen arms need an answer, and an arm
that has not finished yet is left out of the table rather than shrinking the
set for everyone -- `--require` names the arms that must be present.

TWO FORMS PER ARM. raw = P >= 0.5. held = V1's geometric prior (recovered from
the dump, see `holdreplay`) blended 0.5, then OwnHold, over the same sampled
frames. `V1 deployed` is the dump's own 30 fps verdict at those frames.

THE BOOTSTRAP resamples recordings, with replacement, and recomputes each
metric from summed counts, so a recording with many hands weighs what its
hands weigh.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

from src.semhand import BOOT, GUARD, QWEN_ARMS, STRIDE, V1_ARMS, VERDICTS
from src.semhand.holdreplay import invert_prior, key, replay

STRATA = "/workspace/fresh29_strata.txt"
COUNTS = ("own_n", "own_self", "own_pairs", "own_flips", "oth_n", "oth_self",
          "oth_tracks", "oth_tracks_any")


def per_recording(rows, lab, gold):
    tracks = collections.defaultdict(list)
    for r in rows:
        k = (r["rec"], str(r["tid"]))
        if k in gold:
            tracks[k].append((int(r["frame"]), lab[key(r)]))
    c = collections.defaultdict(collections.Counter)
    for k, seq in tracks.items():
        seq.sort()
        n = c[k[0]]
        if gold[k] == "owner":
            prev = None
            for f, s in seq:
                n["own_n"] += 1
                n["own_self"] += s
                if prev is not None and f == prev[0] + STRIDE:
                    n["own_pairs"] += 1
                    n["own_flips"] += s != prev[1]
                prev = (f, s)
        else:
            k_self = sum(s for _, s in seq)
            n["oth_n"] += len(seq)
            n["oth_self"] += k_self
            n["oth_tracks"] += 1
            n["oth_tracks_any"] += k_self > 0
    return c


def summarise(c, recs):
    t = collections.Counter()
    for r in recs:
        t.update(c.get(r, {}))
    return {"M1_own_flips_per100": 100.0 * t["own_flips"] / t["own_pairs"] if t["own_pairs"] else float("nan"),
            "M2_other_frames_self": t["oth_self"],
            "M2_other_frame_rate": t["oth_self"] / t["oth_n"] if t["oth_n"] else float("nan"),
            "other_tracks_any_self": [t["oth_tracks_any"], t["oth_tracks"]],
            "G_own_frames_self": t["own_self"] / t["own_n"] if t["own_n"] else float("nan"),
            "own_frames": t["own_n"], "own_pairs": t["own_pairs"], "other_frames": t["oth_n"]}


def boot_diff(ca, cb, recs, seed=0):
    """-> {metric: (point, lo, hi)} of a - b over recording resamples."""
    rng = np.random.default_rng(seed)
    recs = sorted(recs)
    pa, pb = summarise(ca, recs), summarise(cb, recs)
    d = {"M1": [], "M2": [], "G": []}
    for _ in range(BOOT):
        s = list(rng.choice(recs, len(recs), replace=True))
        a, b = summarise(ca, s), summarise(cb, s)
        d["M1"].append(a["M1_own_flips_per100"] - b["M1_own_flips_per100"])
        d["M2"].append(a["M2_other_frames_self"] - b["M2_other_frames_self"])
        d["G"].append(a["G_own_frames_self"] - b["G_own_frames_self"])
    point = {"M1": pa["M1_own_flips_per100"] - pb["M1_own_flips_per100"],
             "M2": pa["M2_other_frames_self"] - pb["M2_other_frames_self"],
             "G": pa["G_own_frames_self"] - pb["G_own_frames_self"]}
    return {m: (point[m], float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for m, v in d.items()}


def verdict(tab, counts, recs, S, base, controls):
    s, b = tab[S], tab[base]
    diff = boot_diff(counts[S], counts[base], recs)
    a_ok = (s["M1_own_flips_per100"] <= b["M1_own_flips_per100"]
            and s["M2_other_frames_self"] <= b["M2_other_frames_self"])
    names = {"M1": "M1_own_flips_per100", "M2": "M2_other_frames_self"}
    won = [m for m in ("M1", "M2") if diff[m][2] < 0]
    c_ok = [m for m in won if all(s[names[m]] < tab[c][names[m]] for c in controls if c in tab)]
    d_ok = s["G_own_frames_self"] >= b["G_own_frames_self"] - GUARD
    return {"helps": bool(a_ok and won and c_ok and d_ok), "a_no_worse": a_ok,
            "b_significant": won, "c_beats_controls": c_ok, "d_guard": d_ok,
            "diff_vs_base": diff, "controls_present": [c for c in controls if c in tab]}


def load_arm_probs(root):
    probs = {}
    p = os.path.join(root, "v1", "pred.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            for arm in V1_ARMS:
                if arm in r:
                    probs.setdefault(arm, {})[r["id"]] = float(r[arm])
    p = os.path.join(root, "refs", "pred.csv")
    if os.path.exists(p):                      # reference rows, never in a verdict
        for r in csv.DictReader(open(p)):
            for arm, v in r.items():
                if arm != "id":
                    probs.setdefault(arm, {})[r["id"]] = float(v)
    for arm in QWEN_ARMS:
        for f in glob.glob(os.path.join(root, "qwen", f"{arm}_*.jsonl")):
            for line in open(f):
                if line.strip():
                    d = json.loads(line)
                    probs.setdefault(arm, {})[d["id"]] = d["p"]
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/semhand")
    ap.add_argument("--dump", default="/workspace/own_dump_fresh.csv")
    ap.add_argument("--gold", action="append",
                    default=["/workspace/selfother/gold/fresh_gold_p1.csv",
                             "/workspace/selfother/gold/fresh_gold_p2.csv"])
    ap.add_argument("--require", default="Q0,Q2,Q2s,B0r,B2,B2s")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from src.selfother.evaluate import strata
    from src.selfother.train import read_gold

    dump = list(csv.DictReader(open(a.dump, encoding="utf-8")))
    prior = invert_prior(dump)
    by_key = {key(r): r for r in dump}
    fresh = []
    for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "index.csv"))):
        fresh += [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"]
    gold = read_gold(a.gold)
    probs = load_arm_probs(a.root)
    probs["V1"] = {f"{r['rec']}|{r['frame']}|{r['tid']}": float(r["p_dump"]) for r in fresh}
    need = [x for x in a.require.split(",") if x]
    missing = [x for x in need if x not in probs]
    if missing:
        raise SystemExit(f"缺少必需的臂：{missing}")
    ident = lambda r: f"{r['rec']}|{r['frame']}|{r['tid']}"
    rows = [r for r in fresh if all(ident(r) in probs[x] for x in need)]
    order = list(V1_ARMS + QWEN_ARMS) + sorted(set(probs) - set(V1_ARMS + QWEN_ARMS) - {"V1"})
    arms = ["V1"] + [x for x in order
                     if x in probs and all(ident(r) in probs[x] for r in rows)]
    dropped = sorted(set(probs) - set(arms))
    recs = sorted({r["rec"] for r in rows})
    n_gold = len({(r["rec"], str(r["tid"])) for r in rows} & set(gold))

    # keypoint recovery check: V1 recomputed from the re-render vs the dump
    dp = np.array([abs(float(r["p_v1"]) - float(r["p_dump"])) for r in rows])
    dprior = np.array([abs(float(r["prior"]) - prior[key(r)]) for r in rows if r["prior"] != ""])
    print(f"共同集合：{len(rows)} 只手 / {len(recs)} 段录像 / {n_gold} 条 gold 轨迹；"
          f"未完整进表的臂 {dropped}")
    print(f"核对：|V1 重算 P - dump P| 中位 {np.median(dp):.5f}  p99 {np.percentile(dp, 99):.4f}；"
          f"|先验 重算 - 反解| 中位 {np.median(dprior):.5f}  p99 {np.percentile(dprior, 99):.4f}")

    labels = {}
    for arm in arms:
        p = {key(r): probs[arm][ident(r)] for r in rows}
        labels[(arm, "raw")] = {k: v >= 0.5 for k, v in p.items()}
        labels[(arm, "held")] = replay(rows, p, prior)
    labels[("V1 deployed", "held")] = {key(r): by_key[key(r)]["final_owner_post_cap"] == "1"
                                       for r in rows}

    st = strata(STRATA)
    groups = {"all": recs}
    for s_ in sorted(set(st.values())):
        groups[s_] = [x for x in recs if st.get(x) == s_]
    report = {"n_hands": len(rows), "recordings": len(recs), "gold_tracks": n_gold,
              "arms": arms, "not_in_table": dropped, "tables": {}, "verdicts": {}}
    counts = {}
    for (arm, form), lab in labels.items():
        counts[(arm, form)] = per_recording(rows, lab, gold)
    for g, rs in groups.items():
        for form in ("held", "raw"):
            tab = {arm: summarise(counts[(arm, f)], rs)
                   for (arm, f) in counts if f == form}
            report["tables"].setdefault(g, {})[form] = tab
            print(f"\n=== {g}（{len(rs)} 段）  {form} ===")
            print(f"  {'arm':<12}{'M1 翻转/100':>12}{'M2 别人帧判自己':>16}{'率':>8}"
                  f"{'别人轨迹有误':>12}{'G 自己帧召回':>14}")
            for arm, m in tab.items():
                o = m["other_tracks_any_self"]
                print(f"  {arm:<12}{m['M1_own_flips_per100']:>12.2f}{m['M2_other_frames_self']:>16d}"
                      f"{m['M2_other_frame_rate']:>8.2%}{f'{o[0]}/{o[1]}':>12}"
                      f"{m['G_own_frames_self']:>14.2%}")

    print("\n=== 判据（事先写定，见 src/semhand/__init__.py）===")
    for form in ("held", "raw"):
        tab = report["tables"]["all"][form]
        cf = {arm: counts[(arm, f)] for (arm, f) in counts if f == form}
        for fam, (S, base, controls) in VERDICTS.items():
            if S not in tab or base not in tab:
                continue
            v = verdict(tab, cf, recs, S, base, controls)
            report["verdicts"][f"{fam}/{form}"] = v
            d = v["diff_vs_base"]
            print(f"  {fam:<5}{form:<5} {S} vs {base}: {'有帮助' if v['helps'] else '没有帮助'}  "
                  f"ΔM1 {d['M1'][0]:+.2f} [{d['M1'][1]:+.2f},{d['M1'][2]:+.2f}]  "
                  f"ΔM2 {d['M2'][0]:+d} [{d['M2'][1]:+.0f},{d['M2'][2]:+.0f}]  "
                  f"ΔG {d['G'][0]:+.3f}  (a {v['a_no_worse']} b {v['b_significant']} "
                  f"c {v['c_beats_controls']} d {v['d_guard']})")
    out = a.out or os.path.join(a.root, "report.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
