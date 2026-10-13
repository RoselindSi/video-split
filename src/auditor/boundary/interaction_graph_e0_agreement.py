"""What the E0 trial says about the v2 contract, not about the annotator.

WHAT THIS IS NOT. It is not inter-annotator agreement. The v1 release was made
by reviewer_1 under the v1 contract, whose self-loop and Task Cluster
semantics have since changed; scoring one against the other as two arms would
report contract drift as human disagreement. Real inter-annotator agreement
needs reviewer_1 to re-label under v2, which is P1.

WHAT IT IS. Three things the trial can answer on its own:

  uptake       did the contract's new values get used at all -- special_state,
               its kinds, uncertain, within-node repeat, task clusters. A value
               nobody ever selects is either unnecessary or unreachable from
               the manual, and those need opposite fixes.

  the four     the transport/locomotion nodes were designated ambiguity cases
  cases        before the packet was built. The comparison here is against the
               calls written in `v1_to_v2_node_map.json` WITH their video
               evidence -- so it measures whether the manual carried the rule,
               not whether the annotator agreed with a person.

  drift        where the v2 segmentation departs from v1's on the same video.
               Extra boundaries are EXPECTED wherever v2 splits a special state
               out of a v1 node, so a raw boundary F1 would punish the contract
               for working. Boundaries are therefore reported split by whether
               a special state sits at them.

TOLERANCE IS 1.0 s. Not 0.5 -- that is a standing rule in this project and the
published f1@0.5 numbers belong to a different question.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

TOL = 1.0
RELEASE = "results/auditor/interaction_graph_release_v1"
NODEMAP = "results/auditor/interaction_graph_schema_v2/v1_to_v2_node_map.json"


def v1_docs(release):
    out = {}
    for batch in os.listdir(os.path.join(release, "frozen_annotations")):
        d = os.path.join(release, "frozen_annotations", batch)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".json"):
                doc = json.load(open(os.path.join(d, name), encoding="utf-8"))
                out[doc["video_id"]] = (batch, doc)
    return out


def cuts(segments):
    """Interior boundaries only: 0 and the end are not decisions."""
    return [s["start_s"] for s in segments[1:]]


def match(a, b, tol=TOL):
    """Greedy nearest matching within tol. -> (matched pairs, a only, b only)"""
    used, pairs = set(), []
    for x in a:
        best, bi = tol + 1e-9, None
        for j, y in enumerate(b):
            if j in used:
                continue
            if abs(x - y) < best:
                best, bi = abs(x - y), j
        if bi is not None:
            used.add(bi)
            pairs.append((x, b[bi]))
    return (pairs,
            [x for x in a if all(abs(x - p[0]) > 1e-9 for p in pairs)],
            [y for j, y in enumerate(b) if j not in used])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--out")
    a = ap.parse_args()

    exp = json.load(open(a.export, encoding="utf-8"))
    v1 = v1_docs(a.release)
    got = {v["video_id"]: v for v in exp["videos"]}

    print("标注者 %s，%d 条，全部 %s"
          % (exp["annotator"], len(got),
             "final" if all(v["status"] == "final" for v in got.values())
             else "有未提交"))

    # ---- 1. 契约取值的使用率 -------------------------------------------------
    types = collections.Counter()
    kinds = collections.Counter()
    n_rep = n_seg = 0
    for v in got.values():
        for s in v["segments"]:
            types[s.get("seg_type") or "(空)"] += 1
            n_seg += 1
            if s.get("seg_type") == "special_state":
                kinds[s.get("special_kind") or "(空)"] += 1
            if s.get("repeats_within_segment"):
                n_rep += 1
    print("\n== 1. 新取值用了多少 ==")
    print("段 %d 个" % n_seg)
    for k in ("interaction_node", "special_state", "uncertain", "(空)"):
        if types.get(k):
            print("   %-18s %4d (%.0f%%)" % (k, types[k], 100 * types[k] / n_seg))
    for k in ("interaction_node", "special_state", "uncertain"):
        if not types.get(k):
            print("   %-18s %4d   ← 一次都没用" % (k, 0))
    if kinds:
        print("   special_kind：" + "、".join("%s×%d" % kv for kv in kinds.most_common()))
    print("   段内反复 %d 段" % n_rep)

    # 无主的事件与簇：validate 只检查被段引用到的，孤儿会漏过去
    orphan_ev = orphan_cl = 0
    for vid, v in got.items():
        used_ev = {s.get("event_id") for s in v["segments"] if s.get("event_id")}
        for e in v.get("events", []):
            if e["event_id"] not in used_ev:
                orphan_ev += 1
                print("   孤儿事件  %-20s %s%s" % (vid, e["event_id"],
                      "（全空）" if not any(e.get(k) for k in
                      ("description", "object", "interaction", "visible_result")) else ""))
        used_cl = {e.get("cluster_id") for e in v.get("events", [])
                   if e["event_id"] in used_ev and e.get("cluster_id")}
        for c in v.get("clusters", []):
            if c["cluster_id"] not in used_cl:
                orphan_cl += 1
                print("   孤儿任务  %-20s %s%s" % (vid, c["cluster_id"],
                      "（无依据）" if not c.get("goal_evidence") else ""))
    if orphan_ev or orphan_cl:
        print("   → 提交校验只看被段引用到的事件和任务，孤儿能通过。这是工具缺口。")

    # ---- 2. 四个歧义案例 -----------------------------------------------------
    nm = json.load(open(a.nodemap, encoding="utf-8"))
    calls = {}
    for bucket in ("decided", "kept", "undecided"):
        for x in nm.get(bucket, []):
            calls[x["local_node_id"]] = x.get("v2", bucket)
    print("\n== 2. 四个事先指定的歧义节点 ==")
    print("%-24s %-9s %-18s %s" % ("节点", "合同判定", "标注者判定", "落在哪几段"))
    for nid in nm["ambiguity_cases_for_e0"]:
        batch, vid, ev = nid.split("/")
        _, doc = v1[vid]
        spans = [(s["start_s"], s["end_s"]) for s in doc["segments"]
                 if s["event_id"] == ev]
        v = got.get(vid)
        overl = []
        for s in v["segments"]:
            for t0, t1 in spans:
                if min(s["end_s"], t1) - max(s["start_s"], t0) > 0.5:
                    tag = s.get("seg_type")
                    if tag == "special_state":
                        tag += "/" + (s.get("special_kind") or "?")
                    overl.append("%.1f-%.1f %s" % (s["start_s"], s["end_s"], tag))
        theirs = sorted({x.split(" ", 1)[1] for x in overl})
        print("%-24s %-9s %-18s %s"
              % (vid + "/" + ev, calls.get(nid, "?"), "、".join(theirs) or "—",
                 " | ".join(overl)))

    # ---- 3. 与 v1 的分段差异 -------------------------------------------------
    print("\n== 3. 与 v1 分段的差异（容差 %.1fs）==" % TOL)
    print("%-20s %6s %6s %7s %7s %7s  %s"
          % ("录像", "v1段", "v2段", "共有界", "仅v1", "仅v2", "仅v2的界旁有特殊状态"))
    tot = collections.Counter()
    for vid in sorted(got):
        _, doc = v1[vid]
        av, bv = cuts(doc["segments"]), cuts(got[vid]["segments"])
        pairs, only_a, only_b = match(av, bv)
        near_special = 0
        for y in only_b:
            for s in got[vid]["segments"]:
                if s.get("seg_type") == "special_state" and (
                        abs(s["start_s"] - y) < 1e-6 or abs(s["end_s"] - y) < 1e-6):
                    near_special += 1
                    break
        print("%-20s %6d %6d %7d %7d %7d  %d"
              % (vid, len(doc["segments"]), len(got[vid]["segments"]),
                 len(pairs), len(only_a), len(only_b), near_special))
        tot["m"] += len(pairs); tot["a"] += len(only_a)
        tot["b"] += len(only_b); tot["sp"] += near_special
    # ---- 3b. 丢失的界里有多少落在「段内反复」的跨度内 ----------------------
    #
    # ONE MATCHING IMPLEMENTATION, USED TWICE. The first version of this count
    # lived in a throwaway script with `abs(x - y) > tol` instead of the
    # epsilon here, and two boundaries exactly 1.0 s apart came out at
    # 1.0000000000000004 -- so it reported 46 unmatched where the tool
    # reported 44. Same tolerance, two spellings, two answers.
    #
    # AND THE COUNT IS A CANDIDATE, NOT AN EXPLANATION. A v1 boundary inside a
    # span the annotator marked as repeating is consistent with "v1 cut each
    # cycle, v2 marked the span" -- but it is equally consistent with a real
    # restart the annotator missed. Telling those apart needs the video.
    print("\n== 3b. 丢失的 v1 界有多少落在 v2 的「段内反复」跨度里 ==")
    print("%-20s %6s %10s %8s" % ("录像", "丢失", "在反复段内", "占比"))
    rep_tot = collections.Counter()
    for vid in sorted(got):
        _, doc = v1[vid]
        av, bv = cuts(doc["segments"]), cuts(got[vid]["segments"])
        _, only_a, _ = match(av, bv)
        reps = [(s["start_s"], s["end_s"]) for s in got[vid]["segments"]
                if s.get("repeats_within_segment")]
        inside = [x for x in only_a if any(t0 < x < t1 for t0, t1 in reps)]
        if only_a:
            print("%-20s %6d %10d %7.0f%%"
                  % (vid, len(only_a), len(inside),
                     100 * len(inside) / len(only_a)))
        rep_tot["lost"] += len(only_a)
        rep_tot["in"] += len(inside)
    print("%-20s %6d %10d %7.0f%%"
          % ("合计", rep_tot["lost"], rep_tot["in"],
             100 * rep_tot["in"] / max(1, rep_tot["lost"])))
    print("   这是候选归因，不是解释：落在反复段内也可能是标注者漏掉了一次真实的重新开始。"
          "\n   要分开必须回看视频。")

    p = tot["m"] / max(1, tot["m"] + tot["b"])
    r = tot["m"] / max(1, tot["m"] + tot["a"])
    print("%-20s %6s %6s %7d %7d %7d  %d"
          % ("合计", "", "", tot["m"], tot["a"], tot["b"], tot["sp"]))
    print("   对 v1 的 P %.3f  R %.3f  F1 %.3f"
          % (p, r, 2 * p * r / max(1e-9, p + r)))
    print("   其中 %d/%d 个「仅 v2」的边界紧挨着一个特殊状态 —— v2 把它从 v1 的节点里切出来，"
          "这是合同在起作用，不是分歧。" % (tot["sp"], tot["b"]))

    if a.out:
        rows = []
        for vid, v in got.items():
            for i, s in enumerate(v["segments"], 1):
                ev = next((e for e in v.get("events", [])
                           if e["event_id"] == s.get("event_id")), {})
                rows.append({"video": vid, "i": i, "start_s": s["start_s"],
                             "end_s": s["end_s"], "seg_type": s.get("seg_type"),
                             "special_kind": s.get("special_kind"),
                             "event_id": s.get("event_id"),
                             "description": ev.get("description", ""),
                             "object": ev.get("object", ""),
                             "visible_result": ev.get("visible_result", ""),
                             "cluster_id": ev.get("cluster_id", ""),
                             "repeat": s.get("repeats_within_segment"),
                             "note": s.get("note", "")})
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
