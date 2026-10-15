"""E1a's population and its audit sample, frozen before any model output exists.

WHY THIS RUNS FIRST AND ON ITS OWN. The sample has to be chosen without
reference to what Qwen said, or the audit measures how interesting the errors
looked rather than how often they happen. So this script writes the ids and
the seed, and the inference script is a separate file that cannot reach back
and change them.

THE POPULATION IS EVERY INTERACTION SEGMENT, not the 307 boundary candidates.
E1a asks whether Qwen can label local visual semantics at all; its universe is
the segments themselves. The 307 candidates are E1b's universe, and mixing the
two would quietly restrict E1a to the 49 videos that happen to carry a usable
boundary.

TWO SAMPLES, REPORTED SEPARATELY AND NEVER POOLED:

  80 representative   stratified by batch x duration tertile, drawn at random
                      under a fixed seed. At N≈378 and n=80 the worst-case
                      95% margin is about ±10 points, which is what this
                      stage needs.
  20 challenge        the visual conditions E0 exposed -- transport and
                      locomotion, a result that never becomes visible inside
                      the segment, long repetition, and objects the v1
                      annotator could not pin down. Chosen from annotation
                      structure, never from model output.

Pooling them would report a number for a population that does not exist.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re

from src.auditor.boundary.interaction_graph_structure_census import (
    load_groups, load_special)

RELEASE = "results/auditor/interaction_graph_release_v1"
NODEMAP = "results/auditor/interaction_graph_schema_v2/v1_to_v2_node_map.json"
SEED = 20260926
N_REPR = 80
N_EACH_CHALLENGE = 5
# Movement with an object in hand: the family E0 found the contract could not
# separate. Matched on the v1 description, which is annotation structure and
# not model output.
TRANSPORT = re.compile(r"拿着|携带|搬|送|端|走到|走去|前往|取.*回|放到|运")


def population(release, nodemap):
    groups = load_groups(release)
    special = load_special(release, nodemap)
    rows = []
    for bd in sorted(glob.glob(os.path.join(release, "frozen_annotations", "*"))):
        batch = os.path.basename(bd)
        for p in sorted(glob.glob(os.path.join(bd, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            vid = doc["video_id"]
            desc = {e["event_id"]: e["description"] for e in doc["events"]}
            n = len(doc["segments"])
            for i, s in enumerate(doc["segments"]):
                if (batch, vid, s["event_id"]) in special:
                    continue
                rows.append({
                    "segment_id": "%s#%02d" % (vid, i + 1),
                    "batch": batch, "video_id": vid, "seg_index": i + 1,
                    "start_s": s["start_s"], "end_s": s["end_s"],
                    "duration_s": round(s["end_s"] - s["start_s"], 2),
                    "event_id": s["event_id"],
                    "repeats": bool(s.get("repeats_within_segment")),
                    "is_last": i == n - 1,
                    "v1_description": desc.get(s["event_id"], ""),
                    "v1_group": groups.get((batch, vid, s["event_id"]), ""),
                })
    return rows


def tertiles(values):
    v = sorted(values)
    return v[len(v) // 3], v[2 * len(v) // 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = population(a.release, a.nodemap)
    by_id = {r["segment_id"]: r for r in rows}
    lo, hi = tertiles([r["duration_s"] for r in rows])
    for r in rows:
        r["dur_tertile"] = ("short" if r["duration_s"] <= lo else
                            "long" if r["duration_s"] > hi else "mid")
    print("总体：%d 个交互段 / %d 条视频" % (rows and len(rows), len({r["video_id"] for r in rows})))
    print("  时长三分位切点 %.1fs / %.1fs" % (lo, hi))
    cells = collections.Counter((r["batch"], r["dur_tertile"]) for r in rows)
    for k in sorted(cells):
        print("    %-20s %-6s %3d" % (k[0], k[1], cells[k]))

    # ---- 80 代表性：按 batch x 时长三分位 比例分配，固定种子 -----------------
    rng = random.Random(SEED)
    repr_ids, alloc = [], {}
    total = len(rows)
    for k in sorted(cells):
        alloc[k] = max(1, round(N_REPR * cells[k] / total))
    # 四舍五入后凑齐 80：从最大的格子里增减，保持比例最接近
    while sum(alloc.values()) != N_REPR:
        d = 1 if sum(alloc.values()) < N_REPR else -1
        k = max(alloc, key=lambda x: cells[x] / max(1, alloc[x]) * d)
        alloc[k] += d
    for k in sorted(cells):
        pool = sorted(r["segment_id"] for r in rows
                      if (r["batch"], r["dur_tertile"]) == k)
        repr_ids += rng.sample(pool, min(alloc[k], len(pool)))
    repr_ids = sorted(set(repr_ids))

    # ---- 20 困难：四类各 5，全部来自标注结构，不来自模型 ---------------------
    taken = set(repr_ids)
    chal = []

    def take(name, pool, n=N_EACH_CHALLENGE):
        got = [x for x in pool if x not in taken][:n]
        for x in got:
            taken.add(x)
            chal.append({"segment_id": x, "challenge": name})
        print("  困难-%-22s 取 %d（候选 %d）" % (name, len(got), len(pool)))

    print("\n困难样本：")
    take("transport_locomotion",
         [r["segment_id"] for r in sorted(rows, key=lambda r: -r["duration_s"])
          if TRANSPORT.search(r["v1_description"])])
    take("result_truncated_at_end",
         [r["segment_id"] for r in sorted(rows, key=lambda r: r["duration_s"])
          if r["is_last"] and r["duration_s"] < 12])
    take("long_repetition",
         [r["segment_id"] for r in sorted(rows, key=lambda r: -r["duration_s"])
          if r["repeats"]])
    mq = set()
    with open(os.path.join(a.release, "review_queue.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            mq.add((r["video_id"], r["event_id"]))
    take("ambiguous_object_v1_medium",
         [r["segment_id"] for r in sorted(rows, key=lambda r: -r["duration_s"])
          if (r["video_id"], r["event_id"]) in mq])

    out = {
        "schema_version": "e1a_sample_v1",
        "frozen": "2026-09-26",
        "seed": SEED,
        "population_n": len(rows),
        "population_videos": len({r["video_id"] for r in rows}),
        "duration_tertiles": [lo, hi],
        "note": "在任何模型输出存在之前冻结。代表性与困难样本分开报告，不合并计算总体正确率。",
        "representative": repr_ids,
        "challenge": chal,
        "population": rows,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out)
    print("   代表性 %d 个，困难 %d 个，互不重叠，合计 %d"
          % (len(repr_ids), len(chal), len(set(repr_ids) | {c["segment_id"] for c in chal})))


if __name__ == "__main__":
    main()
