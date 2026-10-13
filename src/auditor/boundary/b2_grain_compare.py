"""Run one classifier over the same windows at two labelling grains.

WHY THIS EXISTS. Batch 1's coarse labels produced 0 G-branch, and a zero is
ambiguous: it can mean the domain has no branching, or it can mean the
labelling unit was too coarse to hold any. Those have opposite consequences,
so the measurement is worthless until they are separated. Ego-Exo4D ships its
own keystep annotation of the identical windows at a much finer grain, which
makes the separation a substitution rather than a new annotation round: same
`classify`, same windows, two grains.

WHAT IT FOUND, AND WHY THE `|S|` COLUMN IS THE POINT. The answer was 0 at
coarse grain and 9 at keystep grain, so the grain is the binding constraint --
but reporting only those two totals invites the wrong fix. The `|S|`
distribution says where the constraint binds: G-branch requires `|S| >= 2`, the
predecessor having been observed going to two DIFFERENT successors, and at
coarse grain `|S|` never exceeded 1 in 54 decisions. The failure is a clause,
not a count, and no amount of extra coarse labelling would change it.

THE `gold in S` RATE IS THE ONE TO CARRY FORWARD. It answers a different
question from yield: when the prefix graph has an opinion at all, how often
does its successor set contain the truth? Across two unrelated task families
it came back at 11% and 10%, with G-novel outnumbering G-branch about 3.7 : 1
in both. That is the mechanism behind the online failure measured separately
in `g1_online` -- the successor set usually does not contain the answer even
given a perfect previous state -- so it belongs next to the yield rather than
in a footnote.

Finer grain does not escape this. It raises G-branch and G-novel together,
because more states means more states never seen before.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

from src.auditor.boundary.b2_yield import chain, rows_of
from src.auditor.boundary.branch_mine import classify, MIN_HIST


def tally(seqs, label, min_hist=MIN_HIST):
    """-> dict. Pure counting over already-built state sequences."""
    S_dist, n_dec = collections.Counter(), 0
    gbr = gsim = gnov = in_S = out_S = 0
    for seq in seqs:
        for r in rows_of(seq):
            n_dec += 1
            S_dist[len(r["S"])] += 1
            k = classify(r, min_hist=min_hist)
            gbr += "G-branch" in k
            gsim += "G-simple" in k
            gnov += "G-novel" in k
            if r["S"]:
                if r["gold_node"] in r["S"]:
                    in_S += 1
                else:
                    out_S += 1
    n_w = max(1, len(seqs))
    out = {"label": label, "n_windows": len(seqs), "n_decisions": n_dec,
           "states_median": statistics.median([len(set(s)) for s in seqs]) if seqs else 0,
           "S_dist": dict(sorted(S_dist.items())),
           "n_S_ge_2": sum(v for k, v in S_dist.items() if k >= 2),
           "g_branch": gbr, "g_simple": gsim, "g_novel": gnov,
           "per_window": gbr / n_w,
           "gold_in_S": in_S, "gold_out_S": out_S,
           "hit_rate_when_opinionated": (in_S / (in_S + out_S)
                                         if in_S + out_S else None)}
    return out


def keystep_seqs(windows, ann):
    """-> state sequences at the dataset's own grain, one per window.

    Windows are matched by time against the take's segments, same as the
    pre-screen does, so a window holds exactly the keysteps it contains.
    """
    seqs = []
    for w in windows:
        segs = sorted((s for s in (ann.get(w["take_uid"], {}).get("segments") or [])
                       if float(s.get("start_time", 0)) >= w["start_s"] - 1e-6
                       and float(s.get("end_time", 0)) <= w["end_s"] + 1e-6),
                      key=lambda s: float(s.get("start_time", 0)))
        seq = []
        for s in segs:
            c = str(s.get("step_unique_id"))
            if not seq or seq[-1] != c:
                seq.append(c)
        seqs.append(seq)
    return seqs


def show(t):
    print("\n=== %s ===" % t["label"])
    print("  窗口 %d，决策 %d，每窗口状态数中位 %.0f"
          % (t["n_windows"], t["n_decisions"], t["states_median"]))
    print("  |S| 分布        %s" % t["S_dist"])
    print("  |S|>=2 的决策   %d   <- G-branch 的必要条件，为 0 则与数量无关"
          % t["n_S_ge_2"])
    print("  G-branch %d（每窗口 %.2f） / G-simple %d / G-novel %d"
          % (t["g_branch"], t["per_window"], t["g_simple"], t["g_novel"]))
    if t["hit_rate_when_opinionated"] is not None:
        print("  S 非空时 gold 在 S 里 %d / 不在 %d  ->  图开口时命中 %.0f%%"
              % (t["gold_in_S"], t["gold_out_S"],
                 100 * t["hit_rate_when_opinionated"]))
        if t["g_branch"]:
            print("  G-novel : G-branch = %.1f : 1"
                  % (t["g_novel"] / t["g_branch"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True,
                    help="b2_coarse_export_v1 的人工粗标 json")
    ap.add_argument("--sample", help="egoexo_sample_windows 的窗口清单")
    ap.add_argument("--keystep", help="Ego-Exo4D keystep 标注 json")
    ap.add_argument("--min-hist", dest="min_hist", type=int, default=MIN_HIST)
    ap.add_argument("--out")
    a = ap.parse_args()

    doc = json.load(open(a.export, encoding="utf-8"))
    vids = [v for v in (doc.get("videos") or []) if v.get("status") == "final"]
    if not vids:
        raise SystemExit("没有 status=final 的录像")
    rows = [tally([chain(v) for v in vids], "人工粗标粒度", a.min_hist)]
    show(rows[0])

    if a.sample and a.keystep:
        wins = json.load(open(a.sample, encoding="utf-8"))["windows"]
        ann = json.load(open(a.keystep, encoding="utf-8")).get("annotations", {})
        # 只比人工真的标过的那些窗口，否则两边的总体不是一个
        done = {v["video_id"] for v in vids}
        wins = [w for w in wins
                if "%s_w%06.0f" % (w["take_name"], w["start_s"]) in done]
        if len(wins) != len(vids):
            print("\n  注意：清单与导出只对上 %d / %d 个窗口，"
                  "下面的对照只用对上的那些" % (len(wins), len(vids)))
        rows.append(tally(keystep_seqs(wins, ann),
                          "数据集自带 keystep 粒度（同一批窗口）", a.min_hist))
        show(rows[1])
        print("\n  >>> 粗标为 0 而 keystep 不为 0 ⇒ 瓶颈是粒度；"
              "两个粒度都是个位数 ⇒ 结论不因粒度翻转。")

    if a.out:
        json.dump({"min_hist": a.min_hist, "grains": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
