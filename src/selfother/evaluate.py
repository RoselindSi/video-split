"""Score every arm and V1 on fresh29, once, on exactly the same detections.

THE ONLY FILE HERE ALLOWED TO READ V1'S COLUMNS. They are the baseline rows,
never an input to anything that is trained.

SAME DETECTIONS FOR EVERY ROW. The set is the test crops that cut cleanly; a
detection with no crop is dropped from V1's rows too, so no row is scored on
hands another row never saw.

FOUR KINDS OF ROW.
  V1 deployed   per frame `final_owner_post_cap`; per track, its majority.
  V1 raw        V1's classifier probability before any post-processing,
                aggregated exactly as the arms are. The gap between this and
                `V1 deployed` is what V1's post-processing adds, so a gap
                between an arm and V1 can be attributed to the classifier or
                to the post-processing rather than to "V1".
  arm A/B x imagenet/v1   mean P(self) over seeds; per frame P >= 0.5, per
                track the mean >= 0.5. No smoothing: whatever an arm gets, it
                gets from the classifier.
  zone          the zone drawn on the test recording itself. A reference, not
                a competitor -- nothing deployable has that zone.

REPORTED BY STRATUM. fresh29 is fifteen recordings from days V1 never saw and
fourteen from days it did, and pooling them hides which kind of novelty an arm
fails on.

WHAT THE ARMS CANNOT SEE IS REPORTED BESIDE WHAT THEY CAN. A cam3|cam4 input
has no view of hands in the side modules' field, so some gold tracks have no
crop at all. They are scored for nobody in the main table -- and then counted
by class, with V1's result on exactly those tracks, so a pass on the covered
part cannot be read as a pass on the whole.

THE VERDICT IS THE ONE WRITTEN DOWN BEFORE RUNNING: an arm passes when it
misses at most one more foreign-hand track than V1 deployed AND recalls more
of the wearer's own tracks. The recording-bootstrap intervals are printed
beside it so a pass or fail on one or two tracks is read as what it is.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np

from src.selfother.crops import read_index
from src.selfother.model import load_trained
from src.selfother.train import predict, read_gold

V1_DEPLOYED = "final_owner_post_cap"
V1_RAW = "p_owner_raw"


def strata(path):
    """-> {recording tag: stratum} from a `databag start end stratum` file."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 4:
            out[p[0].replace("databag-26_", "R")] = p[3]
    return out


def arm_probs(ckpts, root, split, rows, device, batch, workers):
    """-> ({arm: mean P(self)}, {arm: [P(self) per seed]})"""
    per_seed = collections.defaultdict(list)
    for path in ckpts:
        net, meta = load_trained(path, device)
        arm = f"{meta['input']}/{meta['init']}"
        per_seed[arm].append(predict(net, root, split, rows, meta["input"],
                                     device, batch, workers))
        print(f"  {arm} seed {meta['seed']} ({meta['epochs']} epoch) 预测完",
              flush=True)
    return ({a: np.mean(np.stack(p), 0) for a, p in per_seed.items()},
            dict(per_seed))


def by_track(rows, values, agg):
    g = collections.defaultdict(list)
    for r, v in zip(rows, values):
        g[(r["rec"], r["tid"])].append(v)
    return {k: agg(v) for k, v in g.items()}


def metrics(rows, self_det, self_track, gold, recs, fps):
    """Track and frame metrics on the gold tracks inside `recs`."""
    tracks = sorted(k for k in self_track
                    if k in gold and k[0] in recs)
    own = [k for k in tracks if gold[k] == "owner"]
    oth = [k for k in tracks if gold[k] == "other"]
    frames = collections.defaultdict(list)
    for r, s in zip(rows, self_det):
        k = (r["rec"], r["tid"])
        if k in gold and r["rec"] in recs:
            frames[k].append((int(r["frame"]), bool(s)))
    longest = 0
    for k in oth:                     # a foreign hand shown as the wearer's
        run, prev = 0, None
        for f, s in sorted(frames[k]):
            run = (run + 1 if s and run and prev == f - 1 else 1) if s else 0
            prev = f
            longest = max(longest, run)
    return {
        "n": len(tracks),
        "other": [sum(1 for k in oth if not self_track[k]), len(oth)],
        "own": [sum(1 for k in own if self_track[k]), len(own)],
        "other_frames": [sum(1 for k in oth for _, s in frames[k] if not s),
                         sum(len(frames[k]) for k in oth)],
        "own_false_blur": [sum(1 for k in own for _, s in frames[k] if not s),
                           sum(len(frames[k]) for k in own)],
        "longest_exposure_s": longest / fps}


def boot(self_track_a, self_track_b, gold, recs, n, seed=0):
    """Recording bootstrap of (a - b) for foreign and own track recall."""
    rng = np.random.default_rng(seed)
    per = {}
    for rec in recs:
        ks = [k for k in gold if k[0] == rec and k in self_track_a
              and k in self_track_b]
        oth = [k for k in ks if gold[k] == "other"]
        own = [k for k in ks if gold[k] == "owner"]
        per[rec] = np.array([
            sum(1 for k in oth if not self_track_a[k]),
            sum(1 for k in oth if not self_track_b[k]), len(oth),
            sum(1 for k in own if self_track_a[k]),
            sum(1 for k in own if self_track_b[k]), len(own)], float)
    recs = sorted(per)
    d_oth, d_own = [], []
    for _ in range(n):
        s = sum(per[r] for r in rng.choice(recs, size=len(recs)))
        if s[2]:
            d_oth.append((s[0] - s[1]) / s[2])
        if s[5]:
            d_own.append((s[3] - s[4]) / s[5])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"other_recall_diff_ci": ci(d_oth), "own_recall_diff_ci": ci(d_own)}


def fmt(m):
    pct = lambda h, n: f"{h / n:6.1%} ({h}/{n})" if n else "   —"
    return (f"n={m['n']:<4} 别人的手 {pct(*m['other'])}  自己的手 {pct(*m['own'])}"
            f"  帧:别人的手糊到 {pct(*m['other_frames'])}"
            f"  自己的手误糊 {m['own_false_blur'][0]}"
            f"  最长暴露 {m['longest_exposure_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, help="test crops split")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--dump", required=True, help="the test recordings' dump")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--labels", required=True,
                    help="labels.py on the test recordings (zone row only)")
    ap.add_argument("--strata", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dump = {(r["rec"], r["frame"], r["tid"]): r
            for r in csv.DictReader(open(a.dump, encoding="utf-8"))}
    index = read_index(a.root, a.split)
    rows = [r for r in index
            if r["status"] == "ok" and (r["rec"], r["frame"], r["tid"]) in dump]
    gold = read_gold(a.gold)
    st = strata(a.strata)
    manifest = os.path.join(a.root, a.split, "manifest.json")
    if os.path.exists(manifest):
        print("切图设置:", json.load(open(manifest)))

    # -- coverage: gold tracks with no usable crop, and V1 on exactly those --
    in_recs = {r["rec"] for r in index}
    covered = {(r["rec"], r["tid"]) for r in rows}
    lost = {k: t for k, t in gold.items() if k[0] in in_recs and k not in covered}
    v1_by_track = collections.defaultdict(list)
    for (rec, _f, tid), d in dump.items():
        if (rec, tid) in lost:
            v1_by_track[(rec, tid)].append(d[V1_DEPLOYED] == "1")
    lost_oth = [k for k, t in lost.items() if t == "other"]
    lost_own = [k for k, t in lost.items() if t == "owner"]
    all_oth = sum(1 for k, t in gold.items() if k[0] in in_recs and t == "other")
    all_own = sum(1 for k, t in gold.items() if k[0] in in_recs and t == "owner")
    v1_oth = sum(1 for k in lost_oth if v1_by_track[k]
                 and np.mean(v1_by_track[k]) < 0.5)
    v1_own = sum(1 for k in lost_own if v1_by_track[k]
                 and np.mean(v1_by_track[k]) >= 0.5)
    print(f"覆盖: 别人的手 gold 轨迹 {all_oth - len(lost_oth)}/{all_oth} 有输入，"
          f"自己的手 {all_own - len(lost_own)}/{all_own}")
    print(f"  没有输入的轨迹上 V1 deployed: 别人的手 {v1_oth}/{len(lost_oth)} 判对，"
          f"自己的手 {v1_own}/{len(lost_own)} 判对（这部分新模型给不出判断）")
    coverage = {"other": [all_oth - len(lost_oth), all_oth],
                "own": [all_own - len(lost_own), all_own],
                "v1_on_lost_other": [v1_oth, len(lost_oth)],
                "v1_on_lost_own": [v1_own, len(lost_own)],
                "status": dict(collections.Counter(r["status"] for r in index))}
    print(f"{len(rows)} 个检测、{len({r['rec'] for r in rows})} 段录像；gold "
          f"覆盖 {len({(r['rec'], r['tid']) for r in rows} & set(gold))} 条轨迹")

    mean_p, seed_p = arm_probs(a.ckpt, a.root, a.split, rows, a.device,
                               a.batch, a.workers)

    row_defs = {}
    v1_det = [dump[(r["rec"], r["frame"], r["tid"])][V1_DEPLOYED] == "1"
              for r in rows]
    row_defs["V1 deployed"] = (
        v1_det, by_track(rows, v1_det, lambda v: np.mean(v) >= 0.5))
    v1_raw = [float(dump[(r["rec"], r["frame"], r["tid"])][V1_RAW])
              for r in rows]
    row_defs["V1 raw"] = ([p >= 0.5 for p in v1_raw],
                          by_track(rows, v1_raw, lambda v: np.mean(v) >= 0.5))
    for arm in sorted(mean_p):
        p = mean_p[arm]
        row_defs[arm] = ([x >= 0.5 for x in p],
                         by_track(rows, p, lambda v: np.mean(v) >= 0.5))
    seen = {(r["rec"], r["tid"]) for r in rows}
    zone = {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8")):
        if r["label"] in ("0", "1") and (r["rec"], r["tid"]) in seen:
            zone[(r["rec"], r["tid"])] = r["label"] == "1"
    zone_rows = [(r, zone[(r["rec"], r["tid"])]) for r in rows
                 if (r["rec"], r["tid"]) in zone]
    row_defs["zone (reference)"] = (
        [s for _, s in zone_rows], zone, [r for r, _ in zone_rows])

    all_recs = {r["rec"] for r in rows}
    groups = {"all": all_recs}
    for name in sorted(set(st.values())):
        groups[name] = {x for x in all_recs if st.get(x) == name}

    report = {"rows": {}, "decision": {}, "seed_spread": {},
              "coverage": coverage}
    for g, recs in groups.items():
        print(f"\n=== {g} ({len(recs)} 段) ===")
        for name, spec in row_defs.items():
            det, trk = spec[0], spec[1]
            rr = spec[2] if len(spec) == 3 else rows
            m = metrics(rr, det, trk, gold, recs, a.fps)
            report["rows"].setdefault(name, {})[g] = m
            print(f"  {name:<18} {fmt(m)}")

    v1 = report["rows"]["V1 deployed"]["all"]
    print("\n=== 判据（事先写定）: 别人的手最多比 V1 多漏 1 条，且自己的手召回更高 ===")
    for arm in sorted(mean_p):
        m = report["rows"][arm]["all"]
        miss_arm = m["other"][1] - m["other"][0]
        miss_v1 = v1["other"][1] - v1["other"][0]
        own_arm = m["own"][0] / m["own"][1] if m["own"][1] else float("nan")
        own_v1 = v1["own"][0] / v1["own"][1] if v1["own"][1] else float("nan")
        ok = miss_arm <= miss_v1 + 1 and own_arm > own_v1
        ci = boot(row_defs[arm][1], row_defs["V1 deployed"][1], gold,
                  all_recs, a.boot)
        spread = []
        for p in seed_p[arm]:
            t = by_track(rows, p, lambda v: np.mean(v) >= 0.5)
            s = metrics(rows, [x >= 0.5 for x in p], t, gold, all_recs, a.fps)
            spread.append([s["other"][0], s["own"][0]])
        report["decision"][arm] = {"pass": ok, "missed_other": miss_arm,
                                   "missed_other_v1": miss_v1,
                                   "own_recall": own_arm,
                                   "own_recall_v1": own_v1, **ci}
        report["seed_spread"][arm] = spread
        print(f"  {arm:<12} {'达标' if ok else '未达标'}  漏别人的手 {miss_arm} "
              f"(V1 {miss_v1})  自己的手召回 {own_arm:.1%} (V1 {own_v1:.1%})  "
              f"差值95%CI 别人的手 [{ci['other_recall_diff_ci'][0]:+.1%}, "
              f"{ci['other_recall_diff_ci'][1]:+.1%}] 自己的手 "
              f"[{ci['own_recall_diff_ci'][0]:+.1%}, "
              f"{ci['own_recall_diff_ci'][1]:+.1%}]  "
              f"各 seed (别人的手命中, 自己的手命中): {spread}")

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
