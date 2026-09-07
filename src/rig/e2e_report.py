"""The end-to-end table: what the finished video gets wrong, per hour.

RATES PER FRAME ARE THE WRONG UNIT AND THIS PROJECT HAS ALREADY BEEN CAUGHT BY
IT. A mosaic that sits on a bench for half a second is one mistake; sampled
every five frames of a 30 fps source it produces three flagged rows, and nine
of the fourteen face false positives in the candidate audit came from a single
continuous event. So consecutive flagged samples of the same kind, in the same
recording, are joined into one EVENT, and events are what the headline reports.
Frames are still reported alongside, because a leak that lasts four seconds is
worse than one that lasts a tenth of a second and the event count cannot say
which happened.

THE MAIN SET AND THE STRESS SET ARE NEVER POOLED. Every clip rendered before
this audit was chosen -- for having hands in it, for reproducing a complaint.
A rate estimated over clips selected for containing the phenomenon is not a
rate. Named stress cases are reported one by one and excluded from every
aggregate.

WHAT THIS CAN SEE THAT NOTHING ELSE COULD. The auditor was shown the source
frame and the delivered frame and no model output at all, so a face the
detector never proposed is visible here and is in no denominator any
candidate-level tool can build. That is the entire reason for the exercise;
the candidate-level numbers elsewhere in this project are all conditioned on a
proposal existing.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os

FLAGS = ("face_miss", "other_miss", "owner_blur", "junk_blur")
NAMES = {"face_miss": "漏了脸", "other_miss": "漏了别人的手",
         "owner_blur": "误糊自己的手", "junk_blur": "误糊非脸非手"}
# Privacy-critical: something identifiable was delivered uncovered.
CRITICAL = ("face_miss", "other_miss")
SRC_FPS = 30.0


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load(paths):
    """-> {recording: [row]} sorted by frame, one row per audited sample."""
    by = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            rec = r["recording"]
            by.setdefault(rec, []).append(
                {"frame": int(r["frame"]), "stride": int(r["stride"]),
                 **{k: int(r[k]) for k in FLAGS}})
    for v in by.values():
        v.sort(key=lambda x: x["frame"])
    return by


def events(rows, flag):
    """Consecutive flagged samples joined. -> [(first frame, last, n samples)]

    Consecutive means the next AUDITED sample, not the next frame: the gap
    between samples is the stride, and a run broken by an unflagged sample is
    two events."""
    out, cur = [], None
    for i, r in enumerate(rows):
        if r[flag]:
            if cur is None:
                cur = [r["frame"], r["frame"], 1]
            else:
                cur[1] = r["frame"]
                cur[2] += 1
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def summarise(by, fps=SRC_FPS):
    """-> (per-recording rows, totals)"""
    per, tot = [], {"samples": 0, "src_frames": 0.0}
    for f in FLAGS:
        tot[f + "_ev"] = 0
        tot[f + "_fr"] = 0
    for rec, rows in sorted(by.items()):
        stride = rows[0]["stride"]
        # Each audited sample stands for `stride` source frames.
        src = len(rows) * stride
        d = {"rec": rec, "samples": len(rows), "src_frames": src,
             "minutes": src / fps / 60.0}
        for f in FLAGS:
            ev = events(rows, f)
            d[f + "_ev"] = len(ev)
            d[f + "_fr"] = sum(e[2] for e in ev) * stride
            tot[f + "_ev"] += len(ev)
            tot[f + "_fr"] += d[f + "_fr"]
        d["critical_ev"] = sum(d[f + "_ev"] for f in CRITICAL)
        per.append(d)
        tot["samples"] += len(rows)
        tot["src_frames"] += src
    tot["minutes"] = tot["src_frames"] / fps / 60.0
    return per, tot


def show(per, tot, title, fps=SRC_FPS):
    print(f"\n=== {title} ===")
    print(f"  {len(per)} 段录像   {tot['samples']} 个采样对   "
          f"{tot['src_frames']:.0f} 源帧 = {tot['minutes']:.1f} 分钟")
    print(f"\n  {'录像':<16} {'对':>4} {'漏脸':>10} {'漏他手':>10} "
          f"{'误糊自手':>10} {'误糊杂物':>10}")
    for d in per:
        cells = "".join(
            f"{d[f + '_ev']:>4}ev/{d[f + '_fr']:<5}" for f in FLAGS)
        print(f"  {d['rec']:<16} {d['samples']:>4} {cells}")
    print(f"\n  {'':<16} {'事件':>6} {'帧':>8} {'每小时事件':>12}")
    hours = tot["minutes"] / 60.0
    for f in FLAGS:
        rate = tot[f + "_ev"] / hours if hours else float("nan")
        print(f"  {NAMES[f]:<16} {tot[f + '_ev']:>6} {tot[f + '_fr']:>8} "
              f"{rate:>12.1f}")
    crit = sum(tot[f + "_ev"] for f in CRITICAL)
    bad = sum(1 for d in per if d["critical_ev"] > 0)
    lo, hi = wilson(bad, len(per))
    print(f"\n  隐私关键事件（漏脸+漏他手） {crit} 次 = "
          f"{crit / hours if hours else float('nan'):.1f}/小时")
    print(f"  至少一次隐私关键漏检的录像 {bad}/{len(per)} = "
          f"{bad / len(per) if per else float('nan'):.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"\n  每小时的数按 {fps:g} fps 源、采样步长换算；"
          f"一个采样对代表 stride 个源帧。\n  事件 = 连续被标记的采样合并；"
          f"帧 = 那些事件覆盖的源帧数。")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True,
                    help="audited csv files, or globs")
    ap.add_argument("--stress", action="append", default=[],
                    help="recording tags to hold out of the aggregate and "
                         "report individually")
    ap.add_argument("--fps", type=float, default=SRC_FPS)
    a = ap.parse_args()

    paths = []
    for c in a.csv:
        paths += sorted(glob.glob(c)) or [c]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no csv found")
    by = load(paths)
    stress = set(a.stress)
    main_by = {k: v for k, v in by.items() if k not in stress}
    stress_by = {k: v for k, v in by.items() if k in stress}

    if main_by:
        per, tot = summarise(main_by, a.fps)
        show(per, tot, "MAIN 随机录像", a.fps)
    if stress_by:
        per, tot = summarise(stress_by, a.fps)
        show(per, tot, "STRESS 已知难例（不进合并）", a.fps)
        print("  压力集是挑出来的，它的率不是率；逐段读通过与否。")
    if not stress_by and stress:
        print(f"\n  （指定的压力段 {sorted(stress)} 没有对应的 csv）")


if __name__ == "__main__":
    main()
