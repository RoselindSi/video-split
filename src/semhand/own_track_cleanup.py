"""R3: drop the own-tracks that are short AND redundant, and name what causes >2 hands.

TWO THINGS, AND THE SECOND NEEDS NO GOLD. A wearer has two hands, so a frame
carrying three or more `own=1` tracks is wrong on its face; R0 measured a
median of 1.3% of own frames and up to 8.0%, nonzero in six of ten recordings.
This finds which tracks are responsible, ranked by how many over-count frames
each one accounts for.

WHY "SHORT" IS NOT ENOUGH TO DROP ON. Tracks under ten frames are 16.4% of own
tracks and 0.7% of own frames, so discarding them looks free. It is not free if
one of them is the only coverage a hand has in that stretch -- then dropping it
loses the hand rather than tidying the count. So a track is droppable only when
it is short AND redundant: its frames are already covered by another own track.
A short track standing alone is kept, however inconvenient for the count.

WHY THIS HELPS THE PER-HAND METRIC AND NOT JUST THE COSMETICS. Under the
per-hand-identity reading the denominator is tracks per hand, so a redundant
fragment of a hand already covered elsewhere is a direct failure: that hand now
has two tracks. Removing it moves the metric for the right reason. Removing a
sole-coverage fragment would move the same metric for the wrong one, which is
why the redundancy test is not optional.

IT IS A PROPOSAL, NOT A PIPELINE CHANGE. This reports before and after on the
deliverable's own output. Nothing is rewritten in place -- the pipeline change
only earns its way in if the per-hand metric (R2) exists to judge it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics

MIN_FRAMES = 10
REDUNDANT_FRAC = 0.80


def read_own(path):
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if (r.get("own") or "").strip() != "1":
            continue
        try:
            rows[str(r["tid"])].append(int(r["frame"]))
        except (KeyError, ValueError):
            continue
    return {t: sorted(set(f)) for t, f in rows.items()}


def over_count(own, keep=None):
    """-> (frames with any own, frames with >2, frames with >1, per-frame counts)."""
    cnt = collections.Counter()
    for t, fs in own.items():
        if keep is not None and t not in keep:
            continue
        for f in fs:
            cnt[f] += 1
    return (len(cnt), sum(1 for n in cnt.values() if n > 2),
            sum(1 for n in cnt.values() if n > 1), cnt)


def blame(own):
    """-> [(tid, frames it spends in an over-counted frame)], worst first."""
    _, _, _, cnt = over_count(own)
    hot = {f for f, n in cnt.items() if n > 2}
    sc = [(t, sum(1 for f in fs if f in hot)) for t, fs in own.items()]
    return sorted((x for x in sc if x[1]), key=lambda x: -x[1])


def droppable(own, min_frames=MIN_FRAMES, frac=REDUNDANT_FRAC):
    """-> (droppable tids, kept-short tids). Short and covered elsewhere."""
    drop, kept = [], []
    for t, fs in own.items():
        if len(fs) >= min_frames:
            continue
        others = set()
        for o, ofs in own.items():
            if o != t:
                others.update(ofs)
        cov = sum(1 for f in fs if f in others) / len(fs)
        (drop if cov >= frac else kept).append((t, len(fs), cov))
    return drop, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--min-frames", dest="minf", type=int, default=MIN_FRAMES)
    ap.add_argument("--redundant-frac", dest="frac", type=float,
                    default=REDUNDANT_FRAC)
    ap.add_argument("--out")
    a = ap.parse_args()

    pat = a.decisions
    if os.path.isdir(pat):
        pat = os.path.join(pat, "*.decisions.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit("没找到 decisions csv")

    print("=== ② 谁造成了「同时 >2 只自己的手」===")
    print("  %-22s %6s %7s  %s" % ("录像", ">2帧", "占own帧", "按责任帧数排的 tid"))
    rows = []
    for p in paths:
        own = read_own(p)
        n_any, n2, n1, _ = over_count(own)
        b = blame(own)
        rec = os.path.basename(p).split(".")[0]
        if n2:
            print("  %-22s %6d %6.1f%%  %s"
                  % (rec[:22], n2, 100 * n2 / max(1, n_any),
                     "，".join("%s(%d帧,长%d)" % (t, k, len(own[t]))
                               for t, k in b[:4])))
        drop, kept = droppable(own, a.minf, a.frac)
        rows.append({"recording": rec, "own_tracks": len(own),
                     "own_frames": n_any, "frames_gt2": n2, "frames_gt1": n1,
                     "blame": b[:6],
                     "droppable": drop, "kept_short": kept})

    print("\n=== ① 短且冗余的 own 轨迹（<%d 帧，且 >=%.0f%% 的帧已被别的 own 轨迹覆盖）==="
          % (a.minf, 100 * a.frac))
    print("  %-22s %6s %8s %9s %8s" % ("录像", "own轨", "可丢", "短但唯一", "丢后轨数"))
    tot_drop = tot_kept = 0
    for p, r in zip(paths, rows):
        own = read_own(p)
        keep = {t for t in own if t not in {d[0] for d in r["droppable"]}}
        n_any2, n2_2, n1_2, _ = over_count(own, keep)
        r["after"] = {"own_tracks": len(keep), "own_frames": n_any2,
                      "frames_gt2": n2_2, "frames_gt1": n1_2}
        tot_drop += len(r["droppable"])
        tot_kept += len(r["kept_short"])
        print("  %-22s %6d %8d %9d %8d"
              % (r["recording"][:22], r["own_tracks"], len(r["droppable"]),
                 len(r["kept_short"]), len(keep)))

    before_t = [r["own_tracks"] for r in rows]
    after_t = [r["after"]["own_tracks"] for r in rows]
    before_f = sum(r["own_frames"] for r in rows)
    after_f = sum(r["after"]["own_frames"] for r in rows)
    before_2 = sum(r["frames_gt2"] for r in rows)
    after_2 = sum(r["after"]["frames_gt2"] for r in rows)

    print("\n=== 汇总（%d 条录像）===" % len(rows))
    print("  own 轨迹数 中位      %.1f  ->  %.1f"
          % (statistics.median(before_t), statistics.median(after_t)))
    print("  own 轨迹数 合计      %d  ->  %d（丢 %d 条，另有 %d 条短但唯一、保留）"
          % (sum(before_t), sum(after_t), tot_drop, tot_kept))
    print("  own 覆盖帧 合计      %d  ->  %d（损失 %.2f%%）"
          % (before_f, after_f, 100 * (before_f - after_f) / max(1, before_f)))
    print("  >2 只手的帧 合计     %d  ->  %d（降 %.0f%%）"
          % (before_2, after_2,
             100 * (before_2 - after_2) / max(1, before_2)))
    print("  恰好 2 条的录像      %d  ->  %d / %d"
          % (sum(1 for n in before_t if n == 2),
             sum(1 for n in after_t if n == 2), len(rows)))
    if tot_kept:
        print("\n  短但唯一、因此不丢的那 %d 条是这条规则的要点：" % tot_kept)
        for r in rows:
            for t, n, cov in r["kept_short"]:
                print("    %-22s tid %-4s %d 帧，仅 %.0f%% 被别的 own 轨迹覆盖"
                      % (r["recording"][:22], t, n, 100 * cov))

    if a.out:
        json.dump({"min_frames": a.minf, "redundant_frac": a.frac,
                   "recordings": rows}, open(a.out, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
