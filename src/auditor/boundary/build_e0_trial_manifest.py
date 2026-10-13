"""The eleven videos of the E0 trial, as one manifest drawn from the two batch manifests.

EIGHT COVER THE STRUCTURES, THREE COVER THE AMBIGUITY. `SELECTION.md` picks
eight for structural coverage -- every cross-task and task-return video that
exists, plus three instances each of the four common structures. Three more
are added here because the transport/locomotion cases the trial is meant to
probe do not live in those eight: only `mosaic_graph_0002` does, and the other
three ambiguity nodes sit in 0022, 0026 and 0010. Eleven is inside the plan's
10-15.

THE ORDER IS SHUFFLED UNDER A FIXED SEED, and the ambiguity videos are not
marked. An annotator told which four cases are the interesting ones is being
asked to agree rather than to judge, and the whole point is whether two people
reach the same call on them independently.

NOTHING FROM v1 TRAVELS WITH IT. No node descriptions, no cluster names, no
boundaries, no model output -- the packet builder copies only the video and
its duration. That is what makes the trial a test of the manual.
"""
from __future__ import annotations

import argparse
import json
import os
import random

SAMPLE = [
    # 结构覆盖（SELECTION.md）
    "mosaic_graph_0002", "mosaic_graph_0006", "mosaic_graph_0020",
    "mosaic_graph_0030", "graph_review_0023", "graph_review_0002",
    "mosaic_graph_0008", "graph_review_0016",
    # 运输 / 位移歧义（v1_to_v2_node_map.json 的 ambiguity_cases_for_e0）
    "mosaic_graph_0022", "mosaic_graph_0026", "mosaic_graph_0010",
]
SOURCES = [
    "results/auditor/interaction_graph_mosaic_complement_v1/manifest.json",
    "results/auditor/interaction_graph_timeline_demo/manifest.json",
]
SEED = 20260924


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = {}
    for p in a.sources:
        for v in json.load(open(p, encoding="utf-8"))["videos"]:
            rows[v["video_id"]] = v
    missing = [v for v in SAMPLE if v not in rows]
    if missing:
        raise SystemExit("清单里找不到：%s" % " ".join(missing))

    picked = [rows[v] for v in SAMPLE]
    for v in picked:
        if not os.path.isfile(v["video"]):
            raise SystemExit("视频文件不存在：%s" % v["video"])
    random.Random(SEED).shuffle(picked)

    out = {"schema_version": "interaction_graph_timeline_manifest_v1",
           "n_videos": len(picked),
           "source_pool": "e0_trial_v2",
           "contract_version": "graph_schema_v2_draft",
           "selection_basis":
               "results/auditor/interaction_graph_schema_v2/SELECTION.md",
           "shuffle_seed": SEED,
           "videos": picked}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("-> %s（%d 条）" % (a.out, len(picked)))
    for i, v in enumerate(picked, 1):
        print("   %2d  %-20s %6.1fs" % (i, v["video_id"], v["duration_s"]))


if __name__ == "__main__":
    main()
