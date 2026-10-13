"""The E0 sample's eight videos written out as v1-against-v2 case sheets.

WHAT A CASE HAS TO CARRY, per §9.1's deliverable list: the visual evidence, the
v1 and v2 representation side by side, the reason for the call, and the
ambiguity that remains. A case that shows only the v2 answer teaches the
answer rather than the rule, and the point of E0 is that a second annotator
can reach the same answer from the manual alone.

THE TWO REPRESENTATIONS ARE PRINTED FROM THE SAME TIMELINE. Nothing is
retyped: v1 is the frozen annotation's own sequence, v2 is that sequence with
the node map applied and the special-gap rule enforced. Where they differ, the
difference is a consequence of the contract and can be traced to one node.

SPECIAL STATES ARE PRINTED IN PLACE, not removed, because the whole
distinction the schema turns on -- a direct edge needs no intervening special
state, a task revisit tolerates one -- is invisible in a sequence that has
already dropped them.
"""
from __future__ import annotations

import argparse
import json
import os

from src.auditor.boundary.interaction_graph_structure_census import (
    CHECKS, census_one, load_groups, load_special)

RELEASE = "results/auditor/interaction_graph_release_v1"
SAMPLE = [
    ("mosaic_complement", "mosaic_graph_0002"),
    ("mosaic_complement", "mosaic_graph_0006"),
    ("mosaic_complement", "mosaic_graph_0020"),
    ("mosaic_complement", "mosaic_graph_0030"),
    ("repeated_actions", "graph_review_0023"),
    ("repeated_actions", "graph_review_0002"),
    ("mosaic_complement", "mosaic_graph_0008"),
    ("repeated_actions", "graph_review_0016"),
]


def timeline(doc, batch, groups, special):
    out = []
    for s in doc["segments"]:
        k = (batch, doc["video_id"], s["event_id"])
        out.append({"node": s["event_id"], "t0": s["start_s"], "t1": s["end_s"],
                    "group": groups.get(k, ""), "special": k in special,
                    "repeat": bool(s.get("repeats_within_segment"))})
    return out


def render(tl, short=True):
    """The sequence as a reader sees it, special states in place."""
    out = []
    for t in tl:
        name = t["node"]
        if t["special"]:
            out.append("[%s]" % name)
        elif t["repeat"]:
            out.append("%s*" % name)
        else:
            out.append(name)
    return " → ".join(out)


def edges(tl):
    """Direct edges only: adjacent, neither end special."""
    out = []
    for i in range(1, len(tl)):
        a, b = tl[i - 1], tl[i]
        if a["special"] or b["special"]:
            continue
        kind = ("self_transition" if a["node"] == b["node"] else
                "cross_cluster" if a["group"] and b["group"]
                and a["group"] != b["group"] else "intra_cluster")
        out.append((b["t0"], a["node"], b["node"], kind))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--reclass",
                    default="results/auditor/interaction_graph_schema_v2/"
                            "v1_to_v2_node_map.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups = load_groups(a.release)
    sp1 = load_special(a.release)
    sp2 = load_special(a.release, a.reclass)
    m = json.load(open(a.reclass, encoding="utf-8"))
    notes = {}
    for bucket in ("decided", "kept", "undecided"):
        for x in m.get(bucket, []):
            notes[x["local_node_id"]] = (bucket, x)

    lines = ["# E0 八条样本：v1 / v2 对照案例", "",
             "由 `src/auditor/boundary/interaction_graph_case_sheet.py` 生成，"
             "2026-09-24。", "",
             "读法：`A` 交互节点，`A*` 该段内有往返重复，`[B]` 特殊状态。",
             "直接边要求相邻且两端都非特殊状态；任务回访定义在时间序的任务出现上，"
             "中间隔一个特殊状态不打断它。", ""]

    for batch, vid in SAMPLE:
        p = os.path.join(a.release, "frozen_annotations", batch, vid + ".json")
        doc = json.load(open(p, encoding="utf-8"))
        desc = {e["event_id"]: e["description"] for e in doc["events"]}
        t1, t2 = (timeline(doc, batch, groups, s) for s in (sp1, sp2))
        f1 = census_one(doc, batch, groups, sp1)
        f2 = census_one(doc, batch, groups, sp2)

        lines += ["---", "", "## %s（%s）" % (vid, batch), "",
                  "| 节点 | 描述 | v1 cluster | v2 |", "|---|---|---|---|"]
        for e in sorted(desc):
            k = "%s/%s/%s" % (batch, vid, e)
            g = groups.get((batch, vid, e), "") or "—"
            isp = any(t["special"] for t in t2 if t["node"] == e)
            tag = "**special**" if isp else "interaction node"
            if k in notes:
                tag += "（%s）" % notes[k][0]
            lines.append("| %s | %s | %s | %s |" % (e, desc[e], g, tag))

        lines += ["", "**v1 时间轴**", "", "```", render(t1), "```", ""]
        if render(t1) != render(t2):
            lines += ["**v2 时间轴**", "", "```", render(t2), "```", ""]
        else:
            lines += ["v2 时间轴与 v1 相同。", ""]

        e1, e2 = edges(t1), edges(t2)
        lines += ["**直接边** v1 %d 条 → v2 %d 条" % (len(e1), len(e2)), ""]
        lost = [x for x in e1 if x not in e2]
        if lost:
            lines.append("被特殊状态间隔切断的：")
            for t, s, d, k in lost:
                lines.append("- `%.1fs  %s → %s`（%s）" % (t, s, d, k))
            lines.append("")
        cross = [x for x in e2 if x[3] == "cross_cluster"]
        if cross:
            lines.append("v2 保留的跨任务边：")
            for t, s, d, k in cross:
                lines.append("- `%.1fs  %s → %s`" % (t, s, d))
            lines.append("")

        names = dict(CHECKS)
        g1 = [names[k] for k, _ in CHECKS if f1[k]]
        g2 = [names[k] for k, _ in CHECKS if f2[k]]
        lines += ["**结构** v1：%s" % ("、".join(g1) or "无"), "",
                  "**结构** v2：%s" % ("、".join(g2) or "无"), ""]
        if f2["task_return"]:
            lines += ["任务回访成立于 `%s`。"
                      % f2["task_return"][0]["group"], ""]
        for e in sorted(desc):
            k = "%s/%s/%s" % (batch, vid, e)
            if k in notes:
                b, x = notes[k]
                lines += ["> **%s 的判断依据（%s）**：%s"
                          % (e, b, x.get("evidence", x.get("why_uncertain", ""))), ""]
                if x.get("granularity_note"):
                    lines += ["> 遗留歧义：%s" % x["granularity_note"], ""]
                if x.get("overrides"):
                    lines += ["> 推翻记录：%s" % x["overrides"], ""]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("-> %s（%d 条样本）" % (a.out, len(SAMPLE)))


if __name__ == "__main__":
    main()
