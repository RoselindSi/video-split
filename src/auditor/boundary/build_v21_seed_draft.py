"""Last round's answers, mapped into the v2.1 schema, to seed the re-label as a draft.

WHY SEED AT ALL. The same annotator is re-labelling the same eleven videos
under a revised contract. They are no longer blind to this material -- they
made these calls themselves -- so retyping eighty-five minutes of
segmentation is cost without information.

WHAT CARRIES AND WHAT DOES NOT. Everything structural carries: cut points,
segment types, special kinds, event identity, task membership and its
evidence. Four fields deliberately arrive empty:

  observed_result           the split between what was seen and what was
                            inferred IS the revision. Last round's
                            `visible_result` is where the defect lives -- one
                            event recorded "selected a suitable motorcycle"
                            as an observation -- and a pre-filled wrong answer
                            is one almost nobody edits.
  inferred_goal             pre-filling it would presume which of last
                            round's results were inferences, which is the
                            judgement being asked for.
  evidence_timestamp        no v2 equivalent exists.
  retrospective_confirmation  likewise.

`interaction` DOES carry into `observed_action`, because "手取出" and "插入并
测试" are descriptions of visible movement; the field was never the one that
attracted inference. If that turns out to be wrong the re-label will show it,
since the annotator is asked to re-read every segment anyway.

THE SEED IS NOT A FLOOR. The repetition rule changes when a cycle becomes a
new occurrence, so some of the seeded cut points are exactly what the
revision is meant to move. The page says so at the top rather than pretending
the draft is a starting point that only needs additions.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="上一轮的导出 JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    exp = json.load(open(a.export, encoding="utf-8"))
    out, kept, blanked = {}, 0, 0
    for v in exp["videos"]:
        events = []
        for e in v.get("events", []):
            events.append({
                "event_id": e["event_id"],
                "description": e.get("description", ""),
                "object": e.get("object", ""),
                "observed_action": e.get("interaction", ""),
                "observed_result": "",          # 见 docstring
                "inferred_goal": "",
                "evidence_timestamp": None,
                "retrospective_confirmation": False,
                "cluster_id": e.get("cluster_id", ""),
            })
            kept += 1
            blanked += 1
        out[v["video_id"]] = {
            "schema_version": "interaction_graph_timeline_v2_1",
            "video_id": v["video_id"],
            "events": events,
            "clusters": [{"cluster_id": c["cluster_id"],
                          "goal_evidence": c.get("goal_evidence", "")}
                         for c in v.get("clusters", [])],
            "segments": [{"start_s": s["start_s"], "end_s": s["end_s"],
                          "seg_type": s.get("seg_type", ""),
                          "event_id": s.get("event_id", ""),
                          "special_kind": s.get("special_kind", ""),
                          "repeats_within_segment":
                              bool(s.get("repeats_within_segment")),
                          "confidence": s.get("confidence", "high"),
                          "note": s.get("note", "")}
                         for s in v["segments"]],
            # NOT carried over: the round is not finished until it is
            # re-confirmed under the new rules.
            "confirmed_full_video": False,
            "status": "draft",
            "revision": 0,
        }
    json.dump(out, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_seg = sum(len(x["segments"]) for x in out.values())
    print("-> %s" % a.out)
    print("   %d 条录像 / %d 段 / %d 个事件带过来" % (len(out), n_seg, kept))
    print("   每个事件留空 4 个字段（看到的结果、推测的目的、证据第几秒、回顾性证据）")
    print("   confirmed_full_video 一律复位为 false")


if __name__ == "__main__":
    main()
