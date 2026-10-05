"""Measure what tracker fragmentation is made of, before proposing to fix it.

WHY THIS IS THE TARGET. The semantic gate defends 2% of OWNER tracks and its
evidence base is 3 chances. Fragmentation touches 50.7%: two hands arrive as a
median of 5 tracks, 42% of tracks have holes, and the errors are concentrated
in the short pieces -- below 10 frames 8% are misclassified and 7.6% of frames
are wrongly blurred, at 34 frames and above those are 0% and 0.66%. Joining
five pieces back into two acts directly on the bigger number.

WHAT THIS DOES NOT DO. It does not interpolate inside a track's hole. That was
measured and rejected: across real holes of 5 frames, 37.9% of the endpoint
pairs have IoU below 0.5, against 3.3% for synthetic holes, with a 5.0% id
switch rate. Joining two separate tracks across a gap is a different operation
-- it decides identity rather than inventing boxes -- and it is measured here
on its own terms, not inherited from that result.

THE QUESTION IS AMBIGUITY, NOT DISTANCE. A join is only free when the piece
that ends has exactly one plausible continuation. Where several pieces could
follow, picking the nearest is a guess dressed as geometry, and this project
has already seen what that costs: in the interaction graph 68% of decisions
had a single candidate and collapsed into "wherever it went last time", which
made a prior look like a model. So the headline number here is the share of
track ends with exactly one candidate inside the window, and the features are
reported as distributions rather than folded into a score.

NO LABELS ARE USED AND NONE ARE NEEDED YET. Whether two pieces are the same
hand needs a person. What does not is how often the choice is forced. If most
ends are forced, the join is cheap to build and cheap to audit; if most are
contested, this needs annotation before any code gets written -- and that
answer is worth having before spending the annotation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics

GAP_WINDOW = 30          # 帧。6fps 下 5 秒，比「手离开画面再回来」的尺度宽
DIST_FACTOR = 3.0        # 候选中心距 <= 这个倍数的框宽，才算说得通


def hand_tracks(path):
    """-> {tid: [(frame, box, own, side)]}，只取 base 认作手的框。"""
    by_tid = collections.defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8")):
        tid = row.get("tid")
        if tid in (None, ""):
            continue
        if str(row.get("not_hand")) in ("1", "True", "true"):
            continue
        try:
            frame = int(row["frame"])
            box = [float(row[c]) for c in ("x0", "y0", "x1", "y1")]
            own = int(float(row["own"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_tid[tid].append((frame, box, own, row.get("side", "")))
    for tid in by_tid:
        by_tid[tid].sort()
    return by_tid


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def width(box):
    return max(1.0, box[2] - box[0])


def summarise(items):
    frames = [i[0] for i in items]
    owns = [i[2] for i in items]
    holes = [b - a - 1 for a, b in zip(frames, frames[1:]) if b - a > 1]
    return {
        "first": frames[0], "last": frames[-1], "n": len(items),
        "span": frames[-1] - frames[0] + 1,
        "holes": len(holes), "hole_frames": sum(holes),
        "own_fraction": sum(owns) / len(owns),
        "end_box": items[-1][1], "start_box": items[0][1],
        "side": collections.Counter(i[3] for i in items).most_common(1)[0][0],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=GAP_WINDOW)
    ap.add_argument("--owner-only", dest="owner_only", action="store_true",
                    help="只看 base 多数判 owner 的碎片 —— 目标是 OWNER 轨迹")
    a = ap.parse_args()

    per_rec, ends = [], []
    for path in sorted(glob.glob(os.path.join(a.arm, "*.csv"))):
        rec = os.path.basename(path)[:-4]
        by_tid = hand_tracks(path)
        if not by_tid:
            continue
        info = {tid: summarise(items) for tid, items in by_tid.items()}
        if a.owner_only:
            info = {t: s for t, s in info.items() if s["own_fraction"] > 0.5}
        if not info:
            continue
        order = sorted(info, key=lambda t: info[t]["first"])
        # 每条碎片结束后，窗口内有几条碎片可以接上去
        for tid in order:
            s = info[tid]
            cx, cy = centre(s["end_box"])
            w = width(s["end_box"])
            cands = []
            for other in order:
                if other == tid:
                    continue
                o = info[other]
                gap = o["first"] - s["last"]
                if not (0 < gap <= a.window):
                    continue
                ox, oy = centre(o["start_box"])
                dist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
                if dist <= DIST_FACTOR * w:
                    cands.append({"tid": other, "gap": gap,
                                  "dist_px": round(dist, 1),
                                  "dist_rel": round(dist / w, 2),
                                  "size_ratio": round(width(o["start_box"]) / w, 2),
                                  "same_side": o["side"] == s["side"]})
            cands.sort(key=lambda c: (c["gap"], c["dist_px"]))
            ends.append({"rec": rec, "tid": tid, "n": s["n"],
                         "n_cands": len(cands), "cands": cands[:4]})
        own_tracks = len(info)
        per_rec.append({
            "rec": rec, "n_tracks": own_tracks,
            "short": sum(1 for s in info.values() if s["n"] < 10),
            "with_holes": sum(1 for s in info.values() if s["holes"] > 0),
            "median_n": statistics.median(s["n"] for s in info.values()),
        })
        print("  %-22s 碎片 %3d  <10帧 %3d  有空洞 %3d  中位长 %5.0f 帧"
              % (rec, own_tracks, per_rec[-1]["short"],
                 per_rec[-1]["with_holes"], per_rec[-1]["median_n"]),
              flush=True)

    counts = collections.Counter(e["n_cands"] for e in ends)
    total = len(ends)
    print("\n=== %d 条录像，%d 个碎片结束点 ===" % (len(per_rec), total))
    print("窗口 %d 帧、中心距 <= %.1f 倍框宽" % (a.window, DIST_FACTOR))
    print("\n窗口内可接续的碎片数：")
    for k in sorted(counts):
        label = {0: "（无接续 —— 手真的离开了，或后面没碎片）",
                 1: "← 强制选择，接起来是免费的"}.get(k, "← 有歧义，需要判定")
        print("  %d 个候选  %4d  %5.1f%%  %s"
              % (k, counts[k], 100 * counts[k] / max(1, total), label))
    forced = counts[1]
    contested = sum(v for k, v in counts.items() if k >= 2)
    print("\n  强制 %d (%.1f%%) / 有歧义 %d (%.1f%%) / 无接续 %d (%.1f%%)"
          % (forced, 100 * forced / max(1, total),
             contested, 100 * contested / max(1, total),
             counts[0], 100 * counts[0] / max(1, total)))
    singles = [e["cands"][0] for e in ends if e["n_cands"] == 1]
    if singles:
        print("\n  强制那批的间隔/距离分布（中位 / 90 分位）：")
        for key in ("gap", "dist_rel", "size_ratio"):
            vals = sorted(c[key] for c in singles)
            print("    %-11s %6.2f / %6.2f"
                  % (key, statistics.median(vals),
                     vals[int(len(vals) * 0.9)] if len(vals) > 1 else vals[0]))
        same = sum(1 for c in singles if c["same_side"])
        print("    左右手一致     %d/%d (%.1f%%)"
              % (same, len(singles), 100 * same / len(singles)))
    tracks = sum(r["n_tracks"] for r in per_rec)
    print("\n  碎片合计 %d；若把所有强制接续都接上，减少 %d 条 (%.1f%%)"
          % (tracks, forced, 100 * forced / max(1, tracks)))
    json.dump({"window": a.window, "dist_factor": DIST_FACTOR,
               "per_rec": per_rec, "cand_counts": dict(counts),
               "ends": ends}, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
