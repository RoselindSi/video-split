"""Rank recordings by how many chances they give the semantic layer.

THE PROBLEM THIS SOLVES. On the 20 frozen recordings the semantic gate fixed
3 tracks out of 3 that only it could fix, and that is the whole evidence base:
117 of 137 OWNER tracks were already right, 17 more were fixed for free by a
whole-track majority vote, and 3 were left. A paired bootstrap over 3 chances
has a confidence interval that touches zero no matter how the 3 come out, so
the next batch has to be chosen for chances, not at random.

WHAT COUNTS AS A CHANCE, and why it is observable without labels. The layer
defends against one failure: base calls a track the wearer's for part of its
life and somebody else's for the rest, and the wrong side wins the vote. The
part that needs a label is "which side was right". The part that does not is
"did base disagree with itself, and did the declining side win" -- that is
just base's own per-frame column, read per track. So:

    A   base never flips within the track          majority vote unnecessary
    B   base flips, the owner side wins the vote   majority vote fixes it free
    C   base flips, the declining side wins        only the semantic layer

C is the opportunity set. It is a candidate, not a verdict: a track where base
declines overall and truly is somebody else's hand is base being right, and
only labelling separates the two. That is the point -- the batch needs both,
because the guardrail number comes from the second kind.

SELECTION USES BASE'S OWN DECISIONS AND THE THREE MECHANISMS, NOTHING ELSE.
No ownership model probability enters the ranking. Scoring candidates with the
model whose benefit is being measured is how a selection proxy ends up
ordering recordings backwards -- it happened on this project when hand
visibility was ranked by raw detection rate and the top pick had 30px boxes.
The mechanisms named in the frozen spec are occlusion, a single hand entering
frame, and track re-acquisition; each is counted here from geometry and frame
continuity.

ONE RECORDING IS ONE SAMPLE. Tracks inside a recording share a scene, a
person and a lighting condition, so a batch that gets its C tracks from two
recordings has two samples and not forty. The quota below therefore caps how
much any single recording may contribute.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

MAX_PER_REC = 8          # 一条录像最多贡献这么多 C 候选，其余不计入排名


def tracks_of(path):
    """-> {tid: [(frame, own, n_det, covered, box)]}，按帧排好。"""
    by_tid = collections.defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8")):
        tid = row.get("tid")
        if tid in (None, ""):
            continue
        try:
            frame = int(row["frame"])
            own = int(float(row["own"]))
            box = [int(float(row[c])) for c in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            continue
        try:
            n_det = int(float(row.get("n_det") or 0))
        except (TypeError, ValueError):
            n_det = 0
        covered = str(row.get("covered", "")) in ("1", "True", "true")
        by_tid[tid].append((frame, own, n_det, covered, box))
    for tid in by_tid:
        by_tid[tid].sort()
    return by_tid


def classify(rows):
    """-> ("A"|"B"|"C", 机制字典). 只用 base 自己的逐帧判定。"""
    owns = [r[1] for r in rows]
    n_own = sum(owns)
    flips = sum(1 for a, b in zip(owns, owns[1:]) if a != b)
    frames = [r[0] for r in rows]
    gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
    solo = sum(1 for r in rows if r[2] == 1)
    covered = sum(1 for r in rows if r[3])
    marks = {"n_frames": len(rows), "flips": flips, "gaps": gaps,
             "solo_frames": solo, "covered_frames": covered,
             "own_fraction": round(n_own / len(rows), 3)}
    if flips == 0:
        return "A", marks
    return ("B" if n_own * 2 > len(rows) else "C"), marks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="jobs 文件；这些录像不参选（已用过的）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", dest="min_frames", type=int, default=10,
                    help="短于此的轨迹不计入 C —— 碎片本身是另一个问题，"
                         "<10 帧的轨迹有 8% 判错率，混进来测的是碎片不是语义")
    ap.add_argument("--pick", type=int, default=25)
    a = ap.parse_args()

    skip = set()
    for path in a.exclude:
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("|")
            if parts:
                skip.add(parts[0])

    rows = []
    for path in sorted(glob.glob(os.path.join(a.arm, "*.csv"))):
        rec = os.path.basename(path)[:-4]
        if rec in skip:
            continue
        by_tid = tracks_of(path)
        census = collections.Counter()
        cands = []
        for tid, items in by_tid.items():
            kind, marks = classify(items)
            census[kind] += 1
            if kind == "C" and marks["n_frames"] >= a.min_frames:
                census["C_long"] += 1
                cands.append({"tid": tid, **marks})
        cands.sort(key=lambda c: -c["n_frames"])
        rows.append({
            "rec": rec, "n_tracks": sum(census[k] for k in "ABC"),
            "A": census["A"], "B": census["B"], "C": census["C"],
            "C_long": census["C_long"],
            "scored": min(census["C_long"], MAX_PER_REC),
            "gaps": sum(c["gaps"] for c in cands),
            "solo": sum(c["solo_frames"] for c in cands),
            "covered": sum(c["covered_frames"] for c in cands),
            "cands": cands[:MAX_PER_REC],
        })
        print("  %-22s 轨 %3d  A %3d B %2d C %2d (>=%d帧 %2d)  空洞 %3d 单手 %4d 遮挡 %3d"
              % (rec, rows[-1]["n_tracks"], census["A"], census["B"],
                 census["C"], a.min_frames, census["C_long"],
                 rows[-1]["gaps"], rows[-1]["solo"], rows[-1]["covered"]),
              flush=True)

    rows.sort(key=lambda r: (-r["scored"], -r["gaps"]))
    pick = rows[:a.pick]
    total_c = sum(r["scored"] for r in pick)
    print("\n=== 候选 %d 条录像，按可计入的 C 候选排序 ===" % len(rows))
    print("全池 C(>=%d帧) 合计 %d 条；取前 %d 条录像得 %d 条 C 候选"
          % (a.min_frames, sum(r["C_long"] for r in rows), len(pick), total_c))
    print("  （fresh20 给出的 C 类是 3 条，这是要超过的那个数）")
    print("\n  %-22s C候选 计入 空洞 单手 遮挡  轨数" % "录像")
    for r in pick:
        print("  %-22s %4d %4d %5d %4d %5d %5d"
              % (r["rec"], r["C_long"], r["scored"], r["gaps"], r["solo"],
                 r["covered"], r["n_tracks"]))
    json.dump({"min_frames": a.min_frames, "max_per_rec": MAX_PER_REC,
               "picked": pick, "all": rows},
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
