"""Draw the Ego-Exo4D windows to coarse-label, by task family and not by score.

WHY A FAMILY AND NOT THE TOP SCENARIOS. The pre-screen puts Cooking Noodles at
1.05 G-branch per window and Remove a Wheel at 0.02, and it would be easy to
read that as "take the top two". That is selecting on the quantity being
measured. B2's spec does endorse choosing branching-native *task types* --
that is what "要拍什么" means -- so picking the cooking family is allowed;
ranking individual scenarios or takes by predicted yield is not, and the
difference is the whole reason this project writes selection rules down.

So the family is chosen on what the tasks are, and sampling inside it is
uniform. Cooking scenarios involve repeatedly checking a pan and deciding what
to do next; the bike tasks in this dataset turn out to be prescribed assembly
and disassembly rather than diagnose-then-decide, which is the opposite of
what I predicted and which the per-scenario numbers show plainly.

STRATIFIED BY TAKE, FOR INDEPENDENCE. B2 requires decisions from several
independent sources, and windows from one take share a kitchen, a person and a
camera. The cap is about that, not about yield.

WINDOWS TOO SHORT TO HOLD A DECISION ARE KEPT. Dropping them would inflate
yield per window, and yield per window is the unit the frozen threshold is
written in.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

SEED = 20261011

# 按任务形态选族，不按产出排序。关键词匹配 scenario 名，匹配到就算这一族。
COOKING = ("cook", "making", "salad", "omelet", "noodle", "egg", "tea",
           "coffee", "latte", "soup", "sandwich", "pasta", "rice")

# 第二次尝试（2026-10-11）。这一族是**按校准过的预筛排名**挑的，不是按任务形态 ——
# 形态依据在 docs/branching_pilot_b2_attempt2_3_preregistration.md §0 作废了，
# 它两次把选择引向全数据集最线性的一族。代价写在 §2：这一批的产出率不是任何
# 总体的无偏估计，只能回答「是否存在任何一族拓扑够用」这个存在性问题。
ANTIGEN = ("antigen", "covid",)

# 第三轮（2026-10-11）。厨房与抗原都被排除之后，带 keystep 标注的 17 个 scenario
# 里只剩修车。只取 Fix a Flat Tire 与 Clean and Lubricate the Chain —— Remove a
# Wheel（时长中位 34s）和 Install a Wheel（85s）按**时长可行性**排除，和当初排除
# CPR 用的是同一条规则，与预测产出无关。
BIKE = ("flat tire", "lubricate the chain")

FAMILIES = {"cooking": COOKING, "antigen": ANTIGEN, "bike": BIKE}


def family_of(scenario, keys=COOKING):
    s = (scenario or "").lower()
    return any(k in s for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keystep", required=True)
    ap.add_argument("--window", type=float, default=180.0)
    ap.add_argument("--n", type=int, default=20, help="B2 的规模是 10-20")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="cooking",
                    help="按形态/排名选定的任务族；族内一律均匀随机")
    ap.add_argument("--max-per-take", dest="cap", type=int, default=2)
    ap.add_argument("--skip-uid", action="append", default=[],
                    help="release 里没有视频的 take_uid。在遍历时跳过，"
                         "**不**从候选池里剔除 —— 池子大小一变 shuffle 的排列"
                         "就全变了，那等于重抽一遍。")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    from src.auditor.boundary.egoexo_topology_prescreen import load, windows

    takes = load(a.keystep)
    keys = FAMILIES[a.family]
    fam = {uid: t for uid, t in takes.items()
           if family_of(t["scenario"], keys)}
    print("take 共 %d；%s 族 %d（scenario %d 个）"
          % (len(takes), a.family, len(fam),
             len({t["scenario"] for t in fam.values()})))
    for s, n in collections.Counter(t["scenario"] for t in fam.values()).most_common():
        print("   %4d  %s" % (n, s))

    pool = []
    for uid, t in sorted(fam.items()):
        for i, w in enumerate(windows(t["segments"], a.window)):
            pool.append({"take_uid": uid, "take_name": t["take_name"],
                         "scenario": t["scenario"], "window_index": i,
                         "start_s": round(float(w[0].get("start_time", 0)), 2),
                         "end_s": round(float(w[-1].get("end_time", 0)), 2),
                         "n_keysteps": len(w),
                         "n_distinct": len({str(s.get("step_unique_id"))
                                            for s in w})})
    print("\n候选窗口 %d 个" % len(pool))

    rng = random.Random(a.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)                     # 均匀随机，不看任何预测量
    picked, per_take = [], collections.Counter()
    skip = set(a.skip_uid)
    for i in order:
        w = pool[i]
        if w["take_uid"] in skip:
            continue
        if per_take[w["take_uid"]] >= a.cap:
            continue
        per_take[w["take_uid"]] += 1
        picked.append(w)
        if len(picked) >= a.n:
            break

    picked.sort(key=lambda w: (w["scenario"], w["take_name"], w["start_s"]))
    print("\n抽中 %d 个窗口，来自 %d 个 take / %d 个 scenario（单 take 上限 %d）"
          % (len(picked), len({w["take_uid"] for w in picked}),
             len({w["scenario"] for w in picked}), a.cap))
    print("  %-34s %-26s %7s %6s" % ("scenario", "take", "起点s", "keystep"))
    for w in picked:
        print("  %-34s %-26s %7.0f %6d"
              % (w["scenario"][:34], w["take_name"][:26], w["start_s"],
                 w["n_keysteps"]))

    ks = [w["n_keysteps"] for w in picked]
    allks = [w["n_keysteps"] for w in pool]
    print("\n  每窗口 keystep 数：抽样中位 %d，全池中位 %d —— 偏离太多说明有偏"
          % (sorted(ks)[len(ks) // 2], sorted(allks)[len(allks) // 2]))
    json.dump({"seed": a.seed, "window_s": a.window, "cap_per_take": a.cap,
               "skipped_uids_no_video": sorted(skip),
               "family": a.family, "family_keys": list(keys),
               "n_pool_windows": len(pool), "n_picked": len(picked),
               "selection": "task family chosen by what the tasks are; "
                            "uniform random within it, capped per take. "
                            "No per-scenario or per-take ranking by predicted "
                            "branching was used.",
               "windows": picked},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
