"""E1b's 307 candidate boundaries: gold relations, and a window that does not leak.

THE WINDOW IS PER-CANDIDATE, NOT FIXED. The shorter of the two adjacent
segments has a median of 5.6 s and a minimum of 0.3 s, so a fixed +-8 s window
would stay inside both segments for only 37% of candidates and would quietly
show a third state for the rest. Here

    W = clamp(min(prev_duration, next_duration), 3.0, 8.0)

and the clip is one continuous stretch [t-W, t+W] with the boundary at its
exact midpoint. Continuous rather than two concatenated windows: joining two
clips manufactures a visual cut, and a model asked "did the state change"
should not be handed a splice as evidence.

WHERE THE FLOOR BITES, IT IS RECORDED RATHER THAN HIDDEN. Below 3 s a window
would hold too few frames at 1 fps to judge anything, so the floor wins and
the clip reaches into the segment beyond. `window_clean` marks those, and
results are reported split by it. 74% of candidates are clean at this floor.

FRAME COUNT IS RECORDED TOO. At 1 fps a 6 s clip is six frames, and whether
accuracy tracks frame count is a question the run can answer for itself --
which is the evidence that would justify the 1/2 fps sensitivity check the
plan allows, rather than changing the rate on a hunch.

GOLD COMES FROM THE FROZEN v1 RELEASE WITH THE v2.1 NODE MAP APPLIED. Pairs
where either end became a special state are not candidates at all: v2.1 §4.1
forbids synthesising an edge across that interval.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess

from src.auditor.boundary.interaction_graph_structure_census import (
    load_groups, load_special)

RELEASE = "results/auditor/interaction_graph_release_v1"
NODEMAP = "results/auditor/interaction_graph_schema_v2/v1_to_v2_node_map.json"
W_MIN, W_MAX = 3.0, 8.0


def candidates(release, nodemap):
    groups = load_groups(release)
    special = load_special(release, nodemap)
    out = []
    for bd in sorted(glob.glob(os.path.join(release, "frozen_annotations", "*"))):
        batch = os.path.basename(bd)
        for p in sorted(glob.glob(os.path.join(bd, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            vid = doc["video_id"]
            desc = {e["event_id"]: e["description"] for e in doc["events"]}
            segs = doc["segments"]
            seen_groups = set()
            for i in range(len(segs)):
                a = segs[i - 1] if i else None
                c = segs[i]
                ka = (batch, vid, a["event_id"]) if a else None
                kc = (batch, vid, c["event_id"])
                if a is not None and ka not in special and kc not in special:
                    ga, gc = groups.get(ka, ""), groups.get(kc, "")
                    if not ga or not gc:
                        task = "NO_CLUSTER"
                    elif ga == gc:
                        task = "SAME_TASK"
                    elif gc in seen_groups:
                        task = "RETURN_TASK"
                    else:
                        task = "DIFFERENT_TASK"
                    da = a["end_s"] - a["start_s"]
                    dc = c["end_s"] - c["start_s"]
                    w = min(max(min(da, dc), W_MIN), W_MAX)
                    t = c["start_s"]
                    out.append({
                        "cand_id": "%s@%06.1f" % (vid, t),
                        "batch": batch, "video_id": vid, "boundary_s": round(t, 2),
                        "prev_event": a["event_id"], "next_event": c["event_id"],
                        "prev_duration_s": round(da, 2), "next_duration_s": round(dc, 2),
                        "window_s": round(w, 2),
                        "clip_start_s": round(max(0.0, t - w), 2),
                        "clip_end_s": round(t + w, 2),
                        "window_clean": min(da, dc) >= W_MIN,
                        "n_frames_1fps": int(2 * w),
                        "gold_state": ("SAME_STATE" if a["event_id"] == c["event_id"]
                                       else "DIFFERENT_STATE"),
                        "gold_task": task,
                        "prev_group": ga, "next_group": gc,
                        # 仅用于 selftext 条件的 join，不进任何提示词
                        "prev_seg_index": i, "next_seg_index": i + 1,
                        "v1_prev_desc": desc.get(a["event_id"], ""),
                        "v1_next_desc": desc.get(c["event_id"], ""),
                    })
                if kc not in special and groups.get(kc):
                    seen_groups.add(groups[kc])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", help="给了就切窗口片段")
    ap.add_argument("--videos", default="/workspace/ig_videos")
    a = ap.parse_args()

    rows = candidates(a.release, a.nodemap)
    st = collections.Counter(r["gold_state"] for r in rows)
    tk = collections.Counter(r["gold_task"] for r in rows)
    clean = sum(1 for r in rows if r["window_clean"])
    fr = sorted(r["n_frames_1fps"] for r in rows)
    print("候选 %d 个 / %d 条视频" % (len(rows), len({r["video_id"] for r in rows})))
    print("  gold_state %s   多数类 %.1f%%"
          % (dict(st), 100 * max(st.values()) / len(rows)))
    print("  gold_task  %s   多数类 %.1f%%"
          % (dict(tk), 100 * max(tk.values()) / len(rows)))
    print("  窗口干净 %d/%d (%.0f%%)；1fps 帧数 中位 %d，最少 %d，最多 %d"
          % (clean, len(rows), 100 * clean / len(rows),
             fr[len(fr) // 2], fr[0], fr[-1]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"schema_version": "e1b_candidates_v1",
               "frozen": "2026-09-27",
               "window_rule": "W = clamp(min(prev,next), %.1f, %.1f)" % (W_MIN, W_MAX),
               "n": len(rows), "candidates": rows},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> %s" % a.out)

    if a.clips:
        os.makedirs(a.clips, exist_ok=True)
        made = skipped = 0
        for r in rows:
            p = os.path.join(a.clips, r["cand_id"].replace("@", "_t") + ".mp4")
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                skipped += 1
                continue
            src = os.path.join(a.videos, r["video_id"] + ".mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", "%.2f" % r["clip_start_s"],
                 "-i", src, "-t", "%.2f" % (r["clip_end_s"] - r["clip_start_s"]),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-movflags", "+faststart", "-an", p, "-y"], check=True)
            made += 1
        print("切片 %d（已存在 %d）-> %s" % (made, skipped, a.clips))


if __name__ == "__main__":
    main()
