"""Which of the six E0 structures each development video actually contains.

WHY THIS RUNS BEFORE THE SCHEMA IS WRITTEN. E0 asks for 10-15 videos that
between them exhibit sustained single-state repeat, single-state recurrence,
a linear multi-step procedure, a multi-step cycle, a cross-task transition,
and a task left and returned to. A sample cannot be chosen for coverage
before anyone knows where the coverage is, and two of the six are rare enough
in the v1 release that they may decide the sample by themselves: the release
reports 9 cross-group transitions across 4 videos, and 2 videos that leave a
procedure group and come back. If that holds, those videos are not选-able --
they are compulsory, and the remaining slots are the only free choice.

IT ALSO ANSWERS A PRIOR QUESTION: whether any of the six is ABSENT. E0's job
is to show the schema can express all six; a structure that occurs nowhere in
the 60 cannot be audited from them at all and has to be collected, which is a
different task with a different cost.

THE COUNTS HERE ARE NOT THE RELEASE'S COUNTS AND MUST NOT BE MIXED WITH THEM.
The report is explicit that v1's 161 exact state repeats and 171 intra-group
transitions belong to the v1 contract, and that changing the self-loop and
Task Cluster semantics requires a fresh conversion and audit rather than
carrying the old numbers over. What this file produces is a per-video census
of STRUCTURAL PRESENCE under stated rules, for choosing a sample. It is not a
v2 measurement and it is not a re-statement of v1's.

THE RULES, WRITTEN DOWN BECAUSE THE SAMPLE'S RATIONALE HAS TO BE KEPT:

  special states are not endpoints. `未开始`, `idle` and occlusion are timed
    states in the annotation, and §8.3④ forbids synthesising a direct edge
    across their interval. So an adjacent pair is a transition only when
    NEITHER end is a special node, and a recurrence that spans a special gap
    is recorded separately rather than silently counted or silently dropped.

  A -> A is recurrence, not a reset. §8.3① is explicit that a self edge means
    the state came back, not that a new episode started. The field is named
    for what it observes. Whether an occurrence is a new instance needs
    completion or reset evidence that the v1 annotation does not carry, and
    supplying it is E0's job, not this file's.

  a linear procedure needs three distinct nodes inside ONE group, and a
    multi-step cycle needs a node to come back after at least one different
    node of that same group. One node returning to itself is check 1 or 2,
    never check 4.

  task return is at GROUP level and is not implied by check 4. T1 -> T2 -> T1
    is a task revisited; A -> B -> A inside T1 is a cycle within one task.
    §8.3③ says a cycle in the topology is not evidence of task continuity, so
    the two are counted apart and never summed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

RELEASE = "results/auditor/interaction_graph_release_v1"

CHECKS = [
    ("single_state_repeat", "持续重复（段内往返）"),
    ("single_state_recurrence", "同节点再次出现（A→A）"),
    ("linear_procedure", "线性多步骤（同簇≥3节点）"),
    ("multi_state_recurrence", "多步骤循环（A→B→…→A）"),
    ("cross_task_transition", "跨任务切换"),
    ("task_return", "离开后返回原任务"),
]


def load_groups(release):
    """-> {(batch, video, event_id): procedure_group_id}"""
    out = {}
    p = os.path.join(release, "local_nodes.csv")
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[(r["batch_id"], r["video_id"], r["event_id"])] = \
                r["procedure_group_id"] or ""
    return out


def load_special(release, reclass=None, include_undecided=False):
    """-> {(batch, video, event_id)} for states that are not interactions.

    `reclass` is the v1->v2 node map. Nodes it moves to `special_state` join
    v1's own special nodes, which is the whole point of running the census
    twice: the rarest structure in the release is two videos long and one of
    the two passes through a waiting state, so where that line is drawn moves
    the evidence base rather than merely renaming it."""
    d = json.load(open(os.path.join(release, "special_nodes.json"),
                     encoding="utf-8"))
    out = {(x["batch_id"], x["video_id"], x["event_id"]) for x in d}
    if reclass:
        m = json.load(open(reclass, encoding="utf-8"))
        rows = list(m.get("decided", []))
        if include_undecided:
            rows += m.get("undecided", [])
        for x in rows:
            b, v, e = x["local_node_id"].split("/")
            out.add((b, v, e))
    return out


def census_one(doc, batch, groups, special):
    """-> {check: evidence} for one frozen annotation."""
    vid = doc["video_id"]
    segs = doc["segments"]
    key = lambda e: (batch, vid, e)

    found = {k: [] for k, _ in CHECKS}

    # 1. A span the annotator marked as containing its own back-and-forth.
    for i, s in enumerate(segs):
        if s.get("repeats_within_segment") and key(s["event_id"]) not in special:
            found["single_state_repeat"].append(
                {"at_s": s["start_s"], "node": s["event_id"]})

    # THE TIMELINE IS KEPT WHOLE. Special states stay in the sequence with
    # their own spans; nothing is deleted and nothing is spliced. An earlier
    # version built the occurrence list by dropping special segments and
    # taking adjacent pairs of what was left, which gates transitions
    # correctly only because a flag was carried alongside -- the
    # representation still said two states were neighbours when a waiting
    # interval sat between them. Two different facts live here and both have
    # to survive: a direct edge needs an observed transition with no special
    # gap, while a task revisit is defined over time-ordered task occurrences
    # and does not care that one intervened.
    timeline = [{"node": s["event_id"], "start_s": s["start_s"],
                 "end_s": s["end_s"], "group": groups.get(key(s["event_id"]), ""),
                 "special": key(s["event_id"]) in special} for s in segs]
    occ = []
    for i, t in enumerate(timeline):
        if t["special"]:
            continue
        prev = next((j for j in range(i - 1, -1, -1)
                     if not timeline[j]["special"]), None)
        occ.append({**t, "ti": i,
                    "after_gap": prev is not None and prev != i - 1})

    for i in range(1, len(occ)):
        a, b = occ[i - 1], occ[i]
        # 2. The same state again. Adjacent only: a later return with other
        # states in between is check 4, and conflating them would let one
        # video satisfy two checks on one piece of evidence.
        if a["node"] == b["node"]:
            found["single_state_recurrence"].append(
                {"at_s": b["start_s"], "node": b["node"],
                 "across_special_gap": b["after_gap"]})
        # 5. A transition whose two ends belong to different groups. Skipped
        # when either group is unassigned, because "no group" is not "a
        # different task".
        if (a["group"] and b["group"] and a["group"] != b["group"]
                and not b["after_gap"]):
            found["cross_task_transition"].append(
                {"at_s": b["start_s"], "from": a["group"], "to": b["group"]})

    # 3. Three distinct nodes of one group seen in this video.
    per_group = collections.defaultdict(list)
    for o in occ:
        if o["group"]:
            per_group[o["group"]].append(o["node"])
    for g, seq in per_group.items():
        if len(set(seq)) >= 3:
            found["linear_procedure"].append(
                {"group": g, "nodes": sorted(set(seq))})

    # 4. Inside one group, a node returns after at least one DIFFERENT node
    # of that group -- adjacent repeats do not qualify.
    for g, seq in per_group.items():
        seen = {}
        for i, n in enumerate(seq):
            if n in seen and any(m != n for m in seq[seen[n] + 1:i]):
                found["multi_state_recurrence"].append(
                    {"group": g, "node": n, "first_i": seen[n], "again_i": i})
                break
            seen.setdefault(n, i)

    # 6. Group level: a group is left and later comes back. Collapse runs
    # first, so staying inside one group across several nodes is not a return.
    gseq = [o["group"] for o in occ if o["group"]]
    runs = [g for i, g in enumerate(gseq) if i == 0 or g != gseq[i - 1]]
    seen = set()
    for i, g in enumerate(runs):
        if g in seen:
            found["task_return"].append({"group": g, "run_index": i})
            break
        seen.add(g)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--out", help="write the per-video census as CSV")
    ap.add_argument("--reclass", help="v1->v2 node map; its `decided` rows "
                                      "become special states")
    ap.add_argument("--with_undecided", action="store_true",
                    help="also apply the map's `undecided` rows, to price "
                         "the sub-rule that has not been written yet")
    a = ap.parse_args()

    groups = load_groups(a.release)
    special = load_special(a.release, a.reclass, a.with_undecided)
    rows = []
    for batch_dir in sorted(glob.glob(os.path.join(
            a.release, "frozen_annotations", "*"))):
        batch = os.path.basename(batch_dir)
        for p in sorted(glob.glob(os.path.join(batch_dir, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            f = census_one(doc, batch, groups, special)
            rows.append({"batch": batch, "video": doc["video_id"],
                         "n_segments": len(doc["segments"]),
                         "n_nodes": len(doc["events"]),
                         **{k: len(f[k]) for k, _ in CHECKS},
                         "_evidence": f})

    print("录像 %d 条\n" % len(rows))
    print("%-26s %6s %8s" % ("结构", "有的录像", "出现次数"))
    for k, label in CHECKS:
        vids = [r for r in rows if r[k]]
        print("%-26s %6d %8d" % (label, len(vids), sum(r[k] for r in rows)))

    print()
    print("按录像数从少到多，稀有的结构决定样本：")
    for k, label in sorted(CHECKS, key=lambda c: len([r for r in rows if r[c[0]]])):
        vids = [r["video"] for r in rows if r[k]]
        if len(vids) <= 8:
            print("  %-26s %d 条：%s" % (label, len(vids), " ".join(vids)))

    # A greedy cover, reported as a starting point and not as the sample: the
    # plan requires the selection rationale to be kept, and "greedy picked it"
    # is a rationale only if what it optimised is stated. This maximises newly
    # covered checks, breaking ties toward more segments.
    need = {k for k, _ in CHECKS}
    picked = []
    pool = list(rows)
    while need and pool:
        best = max(pool, key=lambda r: (len({k for k in need if r[k]}),
                                        r["n_segments"]))
        got = {k for k in need if best[k]}
        if not got:
            break
        picked.append((best, got))
        need -= got
        pool.remove(best)
    print()
    print("贪心最小覆盖（%d 条即可覆盖 %d/%d 个结构）："
          % (len(picked), len(CHECKS) - len(need), len(CHECKS)))
    for r, got in picked:
        print("  %-22s %-18s 段 %2d 节点 %2d  首次覆盖：%s"
              % (r["video"], r["batch"], r["n_segments"], r["n_nodes"],
                 "、".join(dict(CHECKS)[k] for k in got)))
    if need:
        print("  仍未覆盖：%s" % "、".join(dict(CHECKS)[k] for k in need))

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            cols = ["batch", "video", "n_segments", "n_nodes"] + [k for k, _ in CHECKS]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        js = a.out.rsplit(".", 1)[0] + "_evidence.json"
        json.dump({r["video"]: r["_evidence"] for r in rows},
                  open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n-> %s\n-> %s" % (a.out, js))


if __name__ == "__main__":
    main()
