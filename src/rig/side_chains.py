"""Own-hand chains, with the wearer's two hands told apart.

WHY THE CHAINS SO FAR OVERSTATE CONTINUITY. They are built by overlap between
consecutive frames, and the wearer has two hands that repeatedly come close
and cross. When they do, a chain can carry on down the other arm and still
count as one continuous observation of one hand -- which is the failure the
whole identity layer exists to prevent, invisible to the measurement meant to
show it. The detector has been emitting `left` and `right` all along and
nothing recorded it; on 134 gold boxes those classes are 91.8% right.

A SUSTAINED FLIP, NOT ANY FLIP. Per-frame labels are 8.2% wrong, so two
consecutive frames disagree about 15% of the time from noise alone and
counting single flips would measure the label error. A run of `min_run`
frames carrying the other label is what independent noise almost never
produces, and what an arm swap produces by construction.

THREE COUNTS, AND THE MIDDLE ONE IS NOT THE GOAL. Chaining on overlap alone
is today's number; chaining on overlap AND the same label is stricter than
the truth, because an 8.2% label error cuts real chains; the flips inside the
overlap chains are the estimate that does not inherit either bias.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="a run directory whose CSVs carry a `side` column")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--min_run", type=int, default=3)
    a = ap.parse_args()

    gone = set()
    for r in csv.DictReader(open("/workspace/visiblepkg/hands.csv")):
        if r["label"] == "other":
            rec, rest = r["stem"].rsplit("_f", 1)
            gone.add((rec, int(rest.split("_h")[0])))
    N = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            N[f[0]] = (int(f[2]), int(f[3]))

    print("  %-12s %9s %8s %7s %11s %9s %8s %12s" % (
        "臂", "重叠链", "中位", "p90", "重叠+同侧链", "中位", "p90", "持续翻边的链"))
    for d in a.arm:
        plain, strict = [], []
        flipped = flips = chains_with_side = 0
        for rec, (s, n) in sorted(N.items()):
            p = os.path.join(d, rec + ".csv")
            if not os.path.exists(p):
                continue
            own = collections.defaultdict(list)
            has_side = False
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if r["own"] != "1":
                    continue
                sd = (r.get("side") or "")
                has_side = has_side or sd in ("left", "right")
                own[int(r["frame"])].append(
                    ([float(r[c]) for c in ("x0", "y0", "x1", "y1")], sd))
            if not has_side:
                print("  !! %s/%s 没有 side 列" % (d, rec))
                continue
            for mode in ("plain", "strict"):
                live = []          # [(box, side, length, [labels])]
                out = []
                for f in range(s, s + n):
                    if (rec, f) in gone:
                        continue
                    cur = own.get(f, [])
                    used, nxt = set(), []
                    for box, sd, L, seq in live:
                        best, bi = 0.0, None
                        for i, (c, cs) in enumerate(cur):
                            if i in used:
                                continue
                            if mode == "strict" and cs and sd and cs != sd:
                                continue
                            v = iou(box, c)
                            if v > best:
                                best, bi = v, i
                        if bi is not None and best >= 0.3:
                            used.add(bi)
                            nxt.append((cur[bi][0], cur[bi][1] or sd, L + 1,
                                        seq + [cur[bi][1]]))
                        else:
                            out.append((L, seq))
                    for i, (c, cs) in enumerate(cur):
                        if i not in used:
                            nxt.append((c, cs, 1, [cs]))
                    live = nxt
                out += [(L, seq) for _, _, L, seq in live]
                if mode == "plain":
                    plain += [L for L, _ in out]
                    for L, seq in out:
                        lab = [x for x in seq if x in ("left", "right")]
                        if len(lab) < 2 * a.min_run:
                            continue
                        chains_with_side += 1
                        # a sustained run of the other label inside the chain
                        runs, cur_lab, cur_len, hit = [], None, 0, 0
                        for x in lab:
                            if x == cur_lab:
                                cur_len += 1
                            else:
                                runs.append((cur_lab, cur_len))
                                cur_lab, cur_len = x, 1
                        runs.append((cur_lab, cur_len))
                        runs = [r for r in runs if r[0]]
                        long = [r for r in runs if r[1] >= a.min_run]
                        switches = sum(1 for i in range(1, len(long))
                                       if long[i][0] != long[i - 1][0])
                        if switches:
                            flipped += 1
                            flips += switches
                else:
                    strict += [L for L, _ in out]
        pl, st = sorted(plain), sorted(strict)
        print("  %-12s %9d %8d %7d %11d %9d %8d %6d/%-5d" % (
            os.path.basename(d), len(pl), pl[len(pl) // 2], pl[int(len(pl) * .9)],
            len(st), st[len(st) // 2], st[int(len(st) * .9)],
            flipped, chains_with_side))
        print("      持续翻边共 %d 次（min_run=%d 帧）；能判的链 %d 条"
              % (flips, a.min_run, chains_with_side))


if __name__ == "__main__":
    main()
