"""Draw the EPIC windows to coarse-label, without letting the pre-screen choose.

WHY NOT PICK THE HIGH-YIELD WINDOWS. The pre-screen ranks windows by predicted
G-branch, and on this project a selection proxy of exactly that kind ran
backwards: object-family switching correlated with real node switching at
r = -0.46, predicting sixteen switches for a recording that had one. The EPIC
pre-screen is worse behaved than that, not better -- 60.2% of its apparent
branching sits on `pick-up` and `put-down`, and the yield moves from 1.53 to
0.62 per window depending on which verbs I decide are primitives. Selecting on
it would raise the measured yield by construction and void the comparison
against a threshold that was frozen before any of this.

So the draw is uniform at random, and the only stratification is by
participant. That is about independence, not about yield: B2 requires the
decisions to come from several independent sources, and a sample concentrated
in one kitchen is one observation however many windows it holds. Nothing here
conditions on anything correlated with branching.

WHAT THIS SAMPLE CAN AND CANNOT CONCLUDE. B2's No-Go reads "having
deliberately chosen branching-native tasks, still only single digits", and
EPIC windows are a convenience sample of whatever people did in their
kitchens. A low yield here therefore does not trigger that clause -- it cannot
stop the Graph direction, because the premise it is conditioned on is not met.
A high yield, on the other hand, is informative: unscripted kitchen activity
would then be a usable branching regime, which is the question worth asking.

Windows that are too short to contain any decision are kept in the sample
rather than filtered out. Dropping them would inflate yield per window, and
yield per window is the quantity the threshold is stated in.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random

SEED = 20261008


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--window", type=float, default=180.0)
    ap.add_argument("--n", type=int, default=20, help="B2 的规模是 10-20 条")
    ap.add_argument("--max-per-participant", dest="cap", type=int, default=2,
                    help="独立性：一个厨房贡献再多也只是一个观察单位")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    from src.auditor.boundary.epic_topology_prescreen import read, windows_of

    rows = read(a.csv)
    wins = windows_of(rows, a.window)
    print("全部窗口 %d 个，来自 %d 条录像 / %d 位参与者"
          % (len(wins), len({w[0]["video_id"] for w in wins}),
             len({w[0]["participant_id"] for w in wins})))

    rng = random.Random(a.seed)
    order = list(range(len(wins)))
    rng.shuffle(order)                       # 均匀随机，不看任何预测量

    picked, per_p = [], collections.Counter()
    for i in order:
        w = wins[i]
        pid = w[0]["participant_id"]
        if per_p[pid] >= a.cap:
            continue
        per_p[pid] += 1
        start = w[0]["start_s"]
        picked.append({
            "window_id": "%s_w%06.0f" % (w[0]["video_id"], start),
            "video_id": w[0]["video_id"],
            "participant_id": pid,
            "start_s": round(start, 2),
            "end_s": round(w[-1]["start_s"], 2),
            "n_segments": len(w),
            "n_distinct_verbnoun": len({(r["verb"], r["noun"]) for r in w}),
            "first_narrations": [r["narration"] for r in w[:5]],
        })
        if len(picked) >= a.n:
            break

    picked.sort(key=lambda p: (p["participant_id"], p["video_id"], p["start_s"]))
    print("\n抽中 %d 个窗口，来自 %d 位参与者（单人上限 %d）"
          % (len(picked), len({p["participant_id"] for p in picked}), a.cap))
    print("  %-22s %8s %8s %6s  %s"
          % ("window", "start_s", "跨度s", "段数", "前两条叙述"))
    for p in picked:
        print("  %-22s %8.0f %8.0f %6d  %s"
              % (p["window_id"], p["start_s"], p["end_s"] - p["start_s"],
                 p["n_segments"], " / ".join(p["first_narrations"][:2])[:46]))

    segs = [p["n_segments"] for p in picked]
    print("\n  段数 中位 %d（全池中位 34）—— 偏离太多说明抽样有偏"
          % sorted(segs)[len(segs) // 2])
    json.dump({"seed": a.seed, "window_s": a.window,
               "cap_per_participant": a.cap,
               "n_all_windows": len(wins), "n_picked": len(picked),
               "selection": "uniform random, stratified only by participant; "
                            "no quantity correlated with branching was used",
               "cannot_trigger_b2_nogo": "B2 的 No-Go 条件是「刻意选了 "
                                         "branching-native 任务之后仍只有个位数」；"
                                         "本样本是便利样本，不满足该前提",
               "windows": picked},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
