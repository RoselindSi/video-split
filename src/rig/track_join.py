"""Join the fragments whose continuation is not in doubt, before verdicts are made.

WHY HERE AND NOT INSIDE THE TRACKER. The tracker has already decided; this
reads its output and merges pieces it left separate. That is a worse place to
fix association in principle and a better place in practice -- it is reversible,
it is measurable against the tracker's own output, and it cannot destabilise the
association logic that the rest of the pipeline was tuned against.

WHY BEFORE `post_pass` AND NOT AFTER. Every verdict downstream is per track:
hand-ness, ownership, handedness. Joining changes what a track *is*, so doing
it afterwards would mean reconciling verdicts that were formed on pieces, which
is the ambiguity this is meant to remove. Measured on 52 recordings, two hands
arrive as a median of five tracks and 42% of tracks have holes, and the errors
concentrate in the short pieces: below 10 frames 8% are misclassified against
0% at 34 frames and above.

WHAT COUNTS AS NOT IN DOUBT. A fragment end is joined only when exactly one
other fragment starts inside the window and near enough. Where two could
follow, picking the nearest is a guess dressed as geometry; on 52 recordings
57.9% of ends had no candidate at all, 31.8% had exactly one, and 10.3% had
several, so this takes the 31.8% and leaves the rest alone. Joining all the
forced ones took fragments from 321 to 219.

LEFT/RIGHT IS NOT A GATE, AND THAT IS A MEASUREMENT NOT A PREFERENCE. 21.6% of
these joins have disagreeing side majorities, which first looked like a fifth
of them being wrong. But inside a single fragment, where identity is not in
doubt, 4.4% of frames already disagree with that fragment's own majority and
55.6% of fragments flip at least once; the implied majority-level reliability
is about 0.88, which accounts for the 21.6% on its own. Gating on a label that
is 12% unreliable would throw away good joins to avoid a problem it cannot
see, so the join ignores `side` and recomputes it afterwards from the merged
track -- strictly more evidence than either piece had.

TIME OVERLAP IS A HARD VETO. One hand cannot be in two places, so fragments
whose spans overlap are never merged however close they are. The audit only
ever looked at positive gaps, so this never fired there, but a rule that is
only correct because of how its input was filtered is not a rule.

THE OPERATION IS MERGE-ONLY AND IS ITERATED TO A FIXED POINT, which is how it
earns the right to be idempotent rather than by assertion. A single pass is
not idempotent, and the property test is what showed it: given three pieces at
10-frame spacing the first sees two candidates and is correctly left alone,
but once the second and third merge it sees one, so a second run merges more
than the first. A transform that gives a different answer the second time is a
hazard in front of the decisions, so the merging loops until no new forced
pair appears and one call is the fixed point.

That loop does change the semantics, and the change is deliberate: a piece
whose two candidates turn out to be one hand no longer faces a choice, so
"what follows A" has a single answer. Ambiguity that dissolves because the
alternatives were the same track was never ambiguity about identity.

Track count can only go down, and `--dry-run` prints what would change without
writing. Those are what make it safe to put in front of the decisions rather
than behind them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

WINDOW = 30            # 帧。6fps 下 5 秒，比「手离开画面再回来」的尺度宽
DIST_FACTOR = 3.0      # 候选中心距 <= 这个倍数的框宽才算说得通


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def width(box):
    return max(1.0, box[2] - box[0])


def fragments(rows):
    """-> {tid: {first, last, start_box, end_box, n}}，只看 base 认作手的框。"""
    by_tid = collections.defaultdict(list)
    for row in rows:
        tid = row.get("tid")
        if tid in (None, ""):
            continue
        if str(row.get("not_hand")) in ("1", "True", "true"):
            continue
        try:
            frame = int(row["frame"])
            box = [float(row[c]) for c in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            continue
        by_tid[tid].append((frame, box))
    out = {}
    for tid, items in by_tid.items():
        items.sort()
        out[tid] = {"first": items[0][0], "last": items[-1][0],
                    "start_box": items[0][1], "end_box": items[-1][1],
                    "n": len(items)}
    return out


def forced_pairs(frag, window=WINDOW, dist_factor=DIST_FACTOR):
    """-> [(a, b)]，a 之后窗口内只有 b 一个说得通的接续。"""
    order = sorted(frag, key=lambda t: frag[t]["first"])
    pairs = []
    for a in order:
        fa = frag[a]
        cx, cy = centre(fa["end_box"])
        w = width(fa["end_box"])
        cands = []
        for b in order:
            if b == a:
                continue
            fb = frag[b]
            gap = fb["first"] - fa["last"]
            if gap <= 0:
                continue          # 时间重叠：一只手不可能同时在两处，硬否
            if gap > window:
                continue
            bx, by = centre(fb["start_box"])
            if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 <= dist_factor * w:
                cands.append(b)
        if len(cands) == 1:
            pairs.append((a, cands[0]))
    return pairs


def resolve(frag, pairs):
    """-> {tid: 合并后的 tid}。传递闭包，但不允许产生时间重叠的并。"""
    parent = {t: t for t in frag}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    # 每个组记已占用的时间区间，避免 A→B、C→B 之类并出重叠
    span = {t: (frag[t]["first"], frag[t]["last"]) for t in frag}
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        (a0, a1), (b0, b1) = span[ra], span[rb]
        if not (a1 < b0 or b1 < a0):
            continue              # 两组时间重叠，不并
        parent[rb] = ra
        span[ra] = (min(a0, b0), max(a1, b1))
    # 组代表取组内最早出现的那个 tid，输出才稳定
    groups = collections.defaultdict(list)
    for t in frag:
        groups[find(t)].append(t)
    mapping = {}
    for members in groups.values():
        keep = min(members, key=lambda t: (frag[t]["first"], str(t)))
        for t in members:
            mapping[t] = keep
    return mapping


def apply_to(rows, mapping):
    """-> (新 rows, 改了多少行)。只改 tid 和 side，别的列原样。"""
    out = [dict(r) for r in rows]
    changed = 0
    for r in out:
        tid = r.get("tid")
        if tid in mapping and mapping[tid] != tid:
            r["tid"] = mapping[tid]
            changed += 1
    # side 用合并后的整轨多数票重算 —— 接续带来的证据比任一碎片都多
    votes = collections.defaultdict(collections.Counter)
    for r in out:
        if r.get("tid") and r.get("side"):
            votes[r["tid"]][r["side"]] += 1
    for r in out:
        v = votes.get(r.get("tid"))
        if v:
            r["side"] = v.most_common(1)[0][0]
    return out, changed


def join_to_fixed_point(frag, window=WINDOW, dist_factor=DIST_FACTOR,
                        max_rounds=20):
    """-> ({tid: 合并后 tid}, 轮数)。循环到没有新的强制接续为止。

    一遍不是幂等的：两个候选并成一条之后，原本有歧义的那个端点就只剩一个
    候选了。所以在这里迭代，让「一次调用」就是不动点。
    """
    total = {t: t for t in frag}
    cur = dict(frag)
    for rounds in range(1, max_rounds + 1):
        mapping = resolve(cur, forced_pairs(cur, window, dist_factor))
        if all(mapping[t] == t for t in cur):
            return total, rounds
        # 把这一轮的合并折进总映射，并按合并后的组重建碎片
        for t in total:
            total[t] = mapping.get(total[t], total[t])
        nxt = {}
        for t, f in cur.items():
            k = mapping[t]
            if k not in nxt:
                nxt[k] = dict(f)
            else:
                g = nxt[k]
                if f["first"] < g["first"]:
                    g["first"], g["start_box"] = f["first"], f["start_box"]
                if f["last"] > g["last"]:
                    g["last"], g["end_box"] = f["last"], f["end_box"]
                g["n"] += f["n"]
        cur = nxt
    return total, max_rounds


def join_csv(path, window=WINDOW, dist_factor=DIST_FACTOR):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    frag = fragments(rows)
    pairs = forced_pairs(frag, window, dist_factor)
    mapping, rounds = join_to_fixed_point(frag, window, dist_factor)
    merged = sum(1 for t, k in mapping.items() if k != t)
    new_rows, changed = apply_to(rows, mapping)
    before = len(frag)
    after = len({mapping[t] for t in frag})
    return new_rows, {"fragments_before": before, "fragments_after": after,
                      "forced_pairs": len(pairs), "merged_tracks": merged,
                      "rows_retagged": changed, "rounds": rounds}, mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="stage 3 的输出目录")
    ap.add_argument("--out", help="写到这里；不给就只报不写")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--dist_factor", type=float, default=DIST_FACTOR)
    ap.add_argument("--dry-run", dest="dry", action="store_true")
    a = ap.parse_args()

    if a.out and not a.dry:
        os.makedirs(a.out, exist_ok=True)
    total = collections.Counter()
    tid_maps = {}
    print("  %-26s %7s %7s %8s" % ("录像", "碎片前", "碎片后", "接了"))
    for path in sorted(glob.glob(os.path.join(a.arm, "*.csv"))):
        rec = os.path.basename(path)
        new_rows, st, mapping = join_csv(path, a.window, a.dist_factor)
        tid_maps[rec[:-4]] = mapping
        for k, v in st.items():
            total[k] += v
        print("  %-26s %7d %7d %8d"
              % (rec[:26], st["fragments_before"], st["fragments_after"],
                 st["merged_tracks"]))
        if a.out and not a.dry and new_rows:
            with open(os.path.join(a.out, rec), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(new_rows[0]))
                w.writeheader()
                w.writerows(new_rows)
    print("\n碎片 %d -> %d（降 %.1f%%），强制接续 %d 对，重打 tid 的行 %d"
          % (total["fragments_before"], total["fragments_after"],
             100 * (total["fragments_before"] - total["fragments_after"])
             / max(1, total["fragments_before"]),
             total["forced_pairs"], total["rows_retagged"]))
    if a.out and not a.dry:
        json.dump(dict(total), open(os.path.join(a.out, "_join_report.json"),
                                    "w"), indent=2)
        # 渲染重放的是接续**之前**的 track cache，里面是原始 tid，而
        # post_pass 的判定会按合并后的 tid 写出。没有这张映射，渲染器会
        # （正确地）报 "track ids are not in the decisions"。
        json.dump(tid_maps, open(os.path.join(a.out, "_tid_map.json"), "w"),
                  indent=2)
        print("-> %s（含 _tid_map.json，渲染前必须用它把判定展开回原始 tid）"
              % a.out)
    elif a.dry:
        print("（dry-run，没有写出）")


if __name__ == "__main__":
    main()
