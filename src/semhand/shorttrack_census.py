"""A track that lives one or two frames: mostly a split hand, or mostly a colleague?

WHY ASK. The detector sometimes puts a second box on part of a hand it has
already found -- the fingers of a hand whose palm is another box. The tracker
cannot give that box the hand's id, because the hand already took it, so the
extra box starts a track of its own with no history; `OwnHold` then has only
that one frame's probability to go on, and on the clip that prompted this it
read 0.32 and covered the wearer's own fingers. Dropping tracks that never
live longer than a frame or two would fix that -- and would also stop covering
a colleague's hand that the detector only saw once.

WHICH OF THE TWO IS MORE COMMON IS AN EMPIRICAL QUESTION, and this answers it
on every recording with a human or zone label, at full frame rate, using the
deployed verdict already in the dump:

    gold = owner, covered   the wearer's pixels this would save
    gold = other, covered   a colleague's frames this would expose

Caveat carried into the reading: the labels are per track, and a one-frame
track is exactly where a labeller has least to go on. One of the two frames
this was written for is a colleague's hand that the gold calls the wearer's.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from src.semhand.distil_eval import read_zone_gold


def touching(a, b):
    """-> (IoU, gap in pixels). A hand split into two boxes leaves them
    touching or overlapping; a colleague's hand elsewhere in the frame does
    not. `gap` is the distance between the boxes, 0 when they touch."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    dx, dy = x0 - x1, y0 - y1                     # negative where they overlap
    i = max(0.0, -dx) * max(0.0, -dy)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return (i / u if u > 0 else 0.0), max(0.0, max(dx, dy))


def adjacency(dump_path, labels_path, lifetime=2, near=0.25, causal=False):
    """For every SHORT track's frames: was it next to a hand already called the
    wearer's, in the same frame? `near` is the gap allowed, as a fraction of
    the short box's longer side.

    `causal` asks the question a renderer can actually ask: not "did this track
    turn out to be short" -- which needs the future -- but "is it this young
    right now" (`track_age`), which is on the row."""
    gold = read_zone_gold(labels_path)
    rows = list(csv.DictReader(open(dump_path, encoding="utf-8")))
    seen = collections.defaultdict(list)
    by_frame = collections.defaultdict(list)
    for r in rows:
        seen[(r["rec"], str(r["tid"]))].append(r)
        by_frame[(r["rec"], int(r["frame"]))].append(r)
    c = collections.Counter()
    for t, v in seen.items():
        if not causal and len(v) > lifetime:
            continue
        g = gold.get(t, "unlabelled")
        for r in v:
            if causal and int(r["track_age"]) > lifetime:
                continue
            if r["final_owner_post_cap"] == "1":
                continue                            # not covered: nothing to save
            box = [float(r[x]) for x in ("x0", "y0", "x1", "y1")]
            side = max(box[2] - box[0], box[3] - box[1])
            adj = False
            for q in by_frame[(r["rec"], int(r["frame"]))]:
                if q is r or q["final_owner_post_cap"] != "1":
                    continue
                io, gap = touching(box, [float(q[x]) for x in ("x0", "y0", "x1", "y1")])
                if io > 0 or gap <= near * side:
                    adj = True
                    break
            c[f"{g}_{'adjacent' if adj else 'apart'}"] += 1
    return dict(c)


def census(dump_path, labels_path, lifetimes=(1, 2, 3)):
    gold = read_zone_gold(labels_path)
    rows = list(csv.DictReader(open(dump_path, encoding="utf-8")))
    seen = collections.defaultdict(list)
    for r in rows:
        seen[(r["rec"], str(r["tid"]))].append(r)
    out = {}
    for n in lifetimes:
        short = {t: v for t, v in seen.items() if len(v) <= n}
        c = collections.Counter()
        for t, v in short.items():
            g = gold.get(t, "unlabelled")
            c[f"{g}_tracks"] += 1
            for r in v:
                c[f"{g}_frames"] += 1
                if r["final_owner_post_cap"] != "1":
                    c[f"{g}_covered"] += 1
        out[n] = dict(c)
    return {"tracks": len(seen), "hand_frames": len(rows),
            "labelled_tracks": sum(1 for t in seen if t in gold), "by_lifetime": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True, help="name:dump:labels")
    ap.add_argument("--lifetime", default="1,2,3")
    ap.add_argument("--adjacent_lifetime", type=int, default=2,
                    help="also split the short tracks by whether they touch a hand "
                         "already called the wearer's")
    ap.add_argument("--out", default="/workspace/distil/student/shorttrack.json")
    a = ap.parse_args()
    lts = [int(x) for x in a.lifetime.split(",")]
    report, tot = {}, {n: collections.Counter() for n in lts}
    print(f"  {'':<10}{'轨迹':>7}{'手帧':>8}" +
          "".join(f"{'活<=' + str(n) + '帧: 救回自己/漏掉别人/无标注':>34}" for n in lts))
    for spec in a.batch:
        name, dump, labels = spec.split(":")[:3]
        r = census(dump, labels, lts)
        r["adjacency"] = adjacency(dump, labels, a.adjacent_lifetime)
        r["adjacency_causal"] = adjacency(dump, labels, a.adjacent_lifetime, causal=True)
        report[name] = r
        cells = ""
        for n in lts:
            c = r["by_lifetime"][n]
            tot[n].update(c)
            cells += (f"{c.get('owner_covered', 0):>10}"
                      f"{c.get('other_covered', 0):>11}"
                      f"{c.get('unlabelled_covered', 0):>13}")
        print(f"  {name:<10}{r['tracks']:>7}{r['hand_frames']:>8}{cells}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"batches": report, "totals": {n: dict(c) for n, c in tot.items()}},
              open(a.out, "w"), indent=1)
    print()
    for n in lts:
        c = tot[n]
        print(f"活 <= {n} 帧的轨迹：{c.get('owner_tracks', 0) + c.get('other_tracks', 0) + c.get('unlabelled_tracks', 0)} 条"
              f"（自己 {c.get('owner_tracks', 0)} / 别人 {c.get('other_tracks', 0)} / 无标注 "
              f"{c.get('unlabelled_tracks', 0)}）；若不参与打码："
              f"救回自己 {c.get('owner_covered', 0)} 帧，漏掉别人 {c.get('other_covered', 0)} 帧，"
              f"无标注的少糊 {c.get('unlabelled_covered', 0)} 帧")
    for k, title in (("adjacency", f"活 <= {a.adjacent_lifetime} 帧（事后才知道）"),
                     ("adjacency_causal", f"track_age <= {a.adjacent_lifetime}（渲染时就知道）")):
        adj = collections.Counter()
        for r in report.values():
            adj.update(r[k])
        print(f"\n{title}、且当前被糊掉的手帧，按「是否紧贴一只已判为自己的手」分："
              f"\n  {'':<12}{'贴着':>8}{'不贴':>8}")
        for g in ("owner", "other", "unlabelled"):
            print(f"  {g:<12}{adj.get(g + '_adjacent', 0):>8}{adj.get(g + '_apart', 0):>8}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
