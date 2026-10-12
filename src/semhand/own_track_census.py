"""R0: how many tracks does one pair of hands arrive as, from the deliverable alone.

WHY THIS NEEDS NO ANNOTATION. The goal is stable tracking of the wearer's own
hands, so the deliverable is one `own=1` track per hand, continuous. A wearer
has two hands. If a recording's output carries nine `own=1` tids, the
fragmentation is visible without anyone labelling anything, and if two of them
occupy the same frame then at least one is wrong or duplicated -- a person has
two hands and they cannot be in three places.

WHY THE EXISTING GOLD CANNOT DO THIS. Every gold in the project is keyed on
`rec:tid`, that is, on the tracker's own output. A hand split into five tids
appears as five rows judged independently, so "did we get one track per hand"
is unmeasurable from it by construction: the unit of the gold is the thing
fragmentation damages. The 96.6% track-level ownership accuracy is consistent
with a hand being cut into five correctly-labelled pieces.

WHAT IT CANNOT SEE, SO THAT NOBODY READS MORE INTO IT. Two pieces of one hand
that do not overlap in time look exactly like one hand that left the frame and
came back. This census counts tracks and finds simultaneity; it does not know
which pieces belong to the same hand. It is necessary and not sufficient, and
hand-level gold is still required -- this is the free half.

CONCURRENCY IS THE PART THAT NEEDS NO INTERPRETATION. Frames carrying three or
more `own=1` tids are wrong on their face. `cap-coupling-was-off` measured
16.5% of frames judging more than two of the wearer's own hands when the cap
was accidentally disabled; under the old blurring goal that was a symptom of
that bug, and under this goal it is a direct component of the headline.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics


def read_decisions(path):
    """-> {tid: {"frames": sorted list, "own": int, "side": str, "is_hand": int}}

    Values are whole-track constant by contract (post_pass makes one verdict
    per track and the per-frame file is an expansion for the renderer), so a
    tid whose rows disagree is a defect and is reported rather than averaged.
    """
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        try:
            rows[str(r["tid"])].append((int(r["frame"]),
                                        (r.get("own") or "").strip(),
                                        (r.get("side") or "").strip(),
                                        (r.get("is_hand") or "").strip()))
        except (KeyError, ValueError):
            continue
    out, inconsistent = {}, []
    for tid, rs in rows.items():
        rs.sort()
        owns = {x[1] for x in rs}
        sides = {x[2] for x in rs}
        hands = {x[3] for x in rs}
        if len(owns) > 1 or len(hands) > 1:
            inconsistent.append(tid)
        out[tid] = {"frames": [x[0] for x in rs],
                    "own": sorted(owns)[-1], "side": sorted(sides)[-1],
                    "is_hand": sorted(hands)[-1],
                    "n_frames": len(rs),
                    "span": (rs[0][0], rs[-1][0]),
                    "holes": (rs[-1][0] - rs[0][0] + 1) - len(rs)}
    return out, inconsistent


def census(path):
    tracks, inconsistent = read_decisions(path)
    own = {t: v for t, v in tracks.items() if v["own"] == "1"}
    per_frame = collections.Counter()
    for v in own.values():
        for f in v["frames"]:
            per_frame[f] += 1
    frames_any = len(per_frame)
    over2 = sum(1 for n in per_frame.values() if n > 2)
    over1 = sum(1 for n in per_frame.values() if n > 1)
    spans = [v["span"] for v in own.values()]
    first = min((s[0] for s in spans), default=0)
    last = max((s[1] for s in spans), default=0)
    sides = collections.Counter(v["side"] for v in own.values())
    return {"recording": os.path.basename(path).split(".")[0],
            "n_tracks": len(tracks), "n_own": len(own),
            "n_not_hand": sum(1 for v in tracks.values() if v["is_hand"] == "0"),
            "own_frames": frames_any,
            "span_frames": (last - first + 1) if own else 0,
            "frames_gt2": over2, "frames_gt1": over1,
            "frac_gt2": (over2 / frames_any) if frames_any else 0.0,
            "frac_gt1": (over1 / frames_any) if frames_any else 0.0,
            "own_track_frames": sorted(v["n_frames"] for v in own.values()),
            "own_holes": sum(v["holes"] for v in own.values()),
            "sides": dict(sides),
            "inconsistent_tids": inconsistent}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True,
                    help="decisions csv 的目录或 glob")
    ap.add_argument("--hands", type=int, default=2, help="佩戴者的手数")
    ap.add_argument("--out")
    a = ap.parse_args()

    pat = a.decisions
    if os.path.isdir(pat):
        pat = os.path.join(pat, "*.decisions.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit("没找到 decisions csv：%s" % pat)

    rows = [census(p) for p in paths]
    print("=== 逐录像：两只手到达时是几条 own=1 轨迹 ===")
    print("  %-22s %6s %6s %7s %9s %9s %s"
          % ("录像", "总轨迹", "own=1", "是手帧", ">2只手", ">1只手", "own 轨迹帧数"))
    for r in rows:
        print("  %-22s %6d %6d %7d %8.1f%% %8.1f%%  %s"
              % (r["recording"][:22], r["n_tracks"], r["n_own"], r["own_frames"],
                 100 * r["frac_gt2"], 100 * r["frac_gt1"],
                 ",".join(str(x) for x in r["own_track_frames"][:8])
                 + ("…" if len(r["own_track_frames"]) > 8 else "")))

    n_own = [r["n_own"] for r in rows]
    print("\n=== 汇总（%d 条录像，手数按 %d 计）===" % (len(rows), a.hands))
    print("  own=1 轨迹数：中位 %.1f  均值 %.1f  范围 %d-%d"
          % (statistics.median(n_own), sum(n_own) / len(n_own),
             min(n_own), max(n_own)))
    print("  碎片化倍数（own 轨迹数 / 手数）：中位 %.1f×"
          % (statistics.median(n_own) / a.hands))
    exact = sum(1 for n in n_own if n == a.hands)
    print("  恰好 %d 条的录像：%d / %d" % (a.hands, exact, len(rows)))
    g2 = [r["frac_gt2"] for r in rows]
    g1 = [r["frac_gt1"] for r in rows]
    print("  同时 >2 只自己的手的帧：中位 %.1f%%，最大 %.1f%%"
          % (100 * statistics.median(g2), 100 * max(g2)))
    print("    （一个人只有两只手，所以这一列里任何非零都是错的）")
    print("  同时 >1 只的帧：中位 %.1f%% —— 两只手同时在画面里是正常的，这列只是参照"
          % (100 * statistics.median(g1)))
    holes = [r["own_holes"] for r in rows]
    print("  own 轨迹内部的空洞帧合计：中位 %d" % statistics.median(holes))
    bad = [r for r in rows if r["inconsistent_tids"]]
    print("  整轨值不一致的 tid（契约说应当恒定）：%d 条录像有，共 %d 个"
          % (len(bad), sum(len(r["inconsistent_tids"]) for r in bad)))
    print("\n  这个普查看不到的：两段不重叠的碎片，和「一只手离开又回来」长得一样。"
          "\n  所以它是必要不充分，手级 gold 仍然要做。")

    if a.out:
        json.dump({"hands": a.hands, "n_recordings": len(rows),
                   "median_own_tracks": statistics.median(n_own),
                   "recordings": rows},
                  open(a.out, "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
