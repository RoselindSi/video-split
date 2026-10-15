"""E2's history prefixes: the same 307 boundaries, five ways of remembering what came before.

THE LADDER, AND WHY IT IS A 2x2 AND NOT A LINE.

                       no explicit ID      explicit ID
    raw wording        M1raw               M2raw
    canonical wording  M1canon             M2canon
                                           M3 = M2canon + edges

The earlier plan went flat text -> node memory -> graph in one step each,
which changes two things at once: an occurrence written twice in different
words becomes one reused string AND gains an identifier. E1b showed that
matters -- given independently written text the model swung hard toward
DIFFERENT_STATE, because 71% of same-state pairs were worded differently. So
`M2canon - M1canon` alone cannot tell an identity effect from a paraphrase
effect, and the four planned contrasts below separate them:

    M1canon - M1raw     what normalising the wording is worth on its own
    M2raw   - M1raw     whether an explicit id survives noisy language
    M2canon - M1canon   what the id adds once wording is controlled
    M3      - M2canon   what the edges add once identity is stable

M0 IS ALREADY RUN. E1b-video is this same prompt with no history block at
all, on the same clips and the same 307 candidates, so it is the zero rung
rather than a separate experiment. The prompt here is that prompt with a
history section prepended and nothing else changed.

CANONICALISATION USES ORACLE IDENTITY, AND THAT IS NOT A DEPLOYABLE BASELINE.
Rewriting "拿出下水道塞子" and "拔出塞子" as one string requires already
knowing they are the same node. M1canon, M2*, M3 are oracle-controlled
representation ablations for measuring what structure contributes; none of
them is a claim that canonicalisation is cheap. E4 is where the state
tracking becomes predicted.

PREFIX-ONLY. Every history stops strictly before the boundary being judged.
Special states stay in the timeline as an unlabelled marker rather than being
deleted and the neighbours spliced (v2.1 §4.1), identically in all five
conditions, so they cannot favour one.
"""
from __future__ import annotations

import argparse
import collections
import glob
import random
import json
import os

from src.auditor.boundary.interaction_graph_structure_census import (
    load_groups, load_special)

RELEASE = "results/auditor/interaction_graph_release_v1"
NODEMAP = "results/auditor/interaction_graph_schema_v2/v1_to_v2_node_map.json"
SPECIAL_MARK = "（特殊状态：等待/遮挡/走动一类，无交互）"


def occurrence_text(p):
    """E1a's per-occurrence description, as one short line."""
    if not p:
        return "（无描述）"
    o = str(p.get("object") or "").strip()
    act = str(p.get("observed_action") or "").strip()
    return ("%s%s" % (act, o and "（%s）" % o)) or "（无描述）"


def build(release, nodemap, e1a_path):
    groups = load_groups(release)
    special = load_special(release, nodemap)
    seg_pred = {}
    for line in open(e1a_path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("parsed"):
            seg_pred[d["segment_id"]] = d["parsed"]

    out = []
    for bd in sorted(glob.glob(os.path.join(release, "frozen_annotations", "*"))):
        batch = os.path.basename(bd)
        for p in sorted(glob.glob(os.path.join(bd, "*.json"))):
            doc = json.load(open(p, encoding="utf-8"))
            vid = doc["video_id"]
            canon = {e["event_id"]: e["description"] for e in doc["events"]}
            segs = doc["segments"]
            for i in range(1, len(segs)):
                a, c = segs[i - 1], segs[i]
                if ((batch, vid, a["event_id"]) in special
                        or (batch, vid, c["event_id"]) in special):
                    continue
                # 前缀：这个边界之前的全部 occurrence（不含 a 之后的任何东西）
                hist = []
                for j in range(i):  # segs[0..i-1]，a 是 segs[i-1]
                    s = segs[j]
                    sp = (batch, vid, s["event_id"]) in special
                    hist.append({
                        "k": j + 1, "event": s["event_id"], "special": sp,
                        "raw": occurrence_text(seg_pred.get("%s#%02d" % (vid, j + 1))),
                        "canon": canon.get(s["event_id"], ""),
                    })
                # 节点字典与观测到的转移，只用前缀里出现过的
                seen = []
                for h in hist:
                    if not h["special"] and h["event"] not in seen:
                        seen.append(h["event"])
                edges = collections.Counter()
                prev = None
                for h in hist:
                    if h["special"]:
                        prev = None          # 不跨特殊状态合成边
                        continue
                    if prev is not None:
                        edges[(prev, h["event"])] += 1
                    prev = h["event"]
                out.append({
                    "cand_id": "%s@%06.1f" % (vid, c["start_s"]),
                    "video_id": vid, "boundary_s": round(c["start_s"], 2),
                    "n_history": len(hist),
                    "n_nodes_seen": len(seen),
                    "history": hist,
                    "nodes": [{"id": e, "canon": canon.get(e, "")} for e in seen],
                    "edges": [{"from": k[0], "to": k[1], "count": v}
                              for k, v in sorted(edges.items())],
                    "current_prev_event": a["event_id"],
                })
    return out


def add_length_controls(rows, seed=20260928):
    """Two histories the same shape as M3raw but carrying no usable structure.

    EVERY NEGATIVE RESULT IN E2 IS OTHERWISE UNINTERPRETABLE. Prompt length
    and structure are perfectly collinear across M0..M3raw -- M0 sends zero
    characters and scores best, M3raw sends ~850 and scores worst -- so a drop
    could be the transition list distracting from the clip, or could be
    context length alone. These two hold length and format fixed and destroy
    only the content:

      shuffled  this video's own occurrences, with the id labels permuted so
                the dictionary and the edges describe a structure that never
                happened. Internally consistent, factually wrong.
      other     another video's history entirely, picked from the donor whose
                history length is closest, so the character count matches as
                nearly as the corpus allows.

    M3raw - shuffled is then the structure effect with length controlled, and
    shuffled - M0 is the price of the context itself.
    """
    rng = random.Random(seed)
    by_len = sorted(rows, key=lambda r: r["n_history"])
    for r in rows:
        live = [x for x in r["history"] if not x["special"]]
        labels = [x["event"] for x in live]
        perm = labels[:]
        for _ in range(8):                       # 尽量打乱，短序列可能打不动
            rng.shuffle(perm)
            if perm != labels or len(set(labels)) == 1:
                break
        sh, k = [], 0
        for x in r["history"]:
            if x["special"]:
                sh.append(dict(x))
            else:
                y = dict(x); y["event"] = perm[k]; k += 1
                sh.append(y)
        seen = []
        for x in sh:
            if not x["special"] and x["event"] not in seen:
                seen.append(x["event"])
        ed = collections.Counter(); prev = None
        for x in sh:
            if x["special"]:
                prev = None; continue
            if prev is not None:
                ed[(prev, x["event"])] += 1
            prev = x["event"]
        r["shuffled"] = {
            "history": sh,
            "nodes": [{"id": e, "canon": ""} for e in seen],
            "edges": [{"from": a, "to": b, "count": c} for (a, b), c in sorted(ed.items())],
            "current_prev_event": sh[-1]["event"] if sh and not sh[-1]["special"] else (seen[-1] if seen else "A"),
        }
        # 捐赠者：别的视频里历史长度最接近的一条
        i = by_len.index(r)
        donor = None
        for d in range(1, len(by_len)):
            for j in (i - d, i + d):
                if 0 <= j < len(by_len) and by_len[j]["video_id"] != r["video_id"]:
                    donor = by_len[j]; break
            if donor:
                break
        r["donor_cand_id"] = donor["cand_id"] if donor else None
        r["donor"] = ({"history": donor["history"], "nodes": donor["nodes"],
                       "edges": donor["edges"],
                       "current_prev_event": donor["current_prev_event"]}
                      if donor else None)
    return rows


def render(row, cond):
    """-> the history block for one condition, or '' for M0."""
    h = row["history"]
    if cond == "M0":
        return ""
    if cond in ("M1raw", "M1canon"):
        key = "raw" if cond == "M1raw" else "canon"
        lines = [("%d. %s" % (x["k"], SPECIAL_MARK if x["special"] else x[key]))
                 for x in h]
        return "在这一刻之前，这条视频里依次发生过：\n" + "\n".join(lines)
    if cond in ("M2raw", "M2canon"):
        key = "raw" if cond == "M2raw" else "canon"
        lines = [("%d. %s" % (x["k"], SPECIAL_MARK) if x["special"]
                  else "%d. [%s] %s" % (x["k"], x["event"], x[key])) for x in h]
        dic = "已知的状态：\n" + "\n".join(
            "  %s = %s" % (n["id"], n["canon"]) for n in row["nodes"])
        return (dic + "\n\n在这一刻之前，这条视频里依次发生过：\n" + "\n".join(lines)
                + "\n\n方括号里是状态编号。同一个编号就是同一个状态，"
                  "**即使两次的文字写法不同**。")
    # RAW FAMILY: the dictionary lists the node's own occurrence wordings, not
    # a canonical name. The first M2raw shipped a canonical name in its
    # dictionary, so "M2raw - M1raw" mixed explicit identity with an oracle
    # label and could not be read as an identity effect. These three keep the
    # raw text everywhere, so the only thing that changes across them is
    # identity, then edges, then whether a trivial self-loop is shown.
    # M2raw IS THE ONLY CONDITION THAT BEAT "no history at all", and it differs
    # from M2rawd in two ways at once: its dictionary gives ONE line per node
    # instead of every wording that node ever had, and that line is an oracle
    # canonical name. Three ablations separate those:
    #
    #   perm    the canonical names are real but attached to the wrong nodes.
    #           Same length, same shape, the label's meaning broken.
    #   opaque  one line per node carrying no content at all.
    #   first   one line per node, but the node's own FIRST raw wording --
    #           short and stable like M2raw, with no oracle vocabulary.
    #
    # If `first` matches M2raw the gain is "one short anchor per node"; if it
    # matches M2rawd the gain is the canonical wording itself.
    if cond in ("M2lab_perm", "M2lab_opaque", "M2lab_first"):
        lines = [("%d. %s" % (x["k"], SPECIAL_MARK) if x["special"]
                  else "%d. [%s] %s" % (x["k"], x["event"], x["raw"])) for x in h]
        ids = [n["id"] for n in row["nodes"]]
        if cond == "M2lab_perm":
            names = [row["nodes"][(i + 1) % len(ids)]["canon"] for i in range(len(ids))]
        elif cond == "M2lab_opaque":
            names = ["状态%s" % n for n in ids]
        else:
            firsts = {}
            for x in h:
                if not x["special"]:
                    firsts.setdefault(x["event"], x["raw"])
            names = [firsts.get(i, "") for i in ids]
        dic = "已知的状态：\n" + "\n".join(
            "  %s = %s" % (i, nm) for i, nm in zip(ids, names))
        return (dic + "\n\n在这一刻之前，这条视频里依次发生过：\n" + "\n".join(lines)
                + "\n\n方括号里是状态编号。同一个编号就是同一个状态，"
                  "**即使两次的文字写法不同**。")
    if cond in ("M3raw_shuffled", "M3raw_other"):
        sub = row["shuffled"] if cond == "M3raw_shuffled" else row.get("donor")
        if not sub:
            return ""
        return render({**row, **sub}, "M3raw")
    if cond in ("M2rawd", "M3raw", "M3raw_noloop"):
        lines = [("%d. %s" % (x["k"], SPECIAL_MARK) if x["special"]
                  else "%d. [%s] %s" % (x["k"], x["event"], x["raw"])) for x in h]
        by = collections.OrderedDict()
        for x in h:
            if not x["special"]:
                by.setdefault(x["event"], [])
                if x["raw"] not in by[x["event"]]:
                    by[x["event"]].append(x["raw"])
        dic = "已知的状态（每个状态下面是它出现过的几次写法）：\n" + "\n".join(
            "  %s：\n%s" % (k, "\n".join("    - %s" % t for t in v))
            for k, v in by.items())
        body = (dic + "\n\n在这一刻之前，这条视频里依次发生过：\n" + "\n".join(lines)
                + "\n\n方括号里是状态编号。同一个编号就是同一个状态，"
                  "**即使几次的文字写法不同**。")
        if cond == "M2rawd":
            return body
        edges = row["edges"]
        if cond == "M3raw_noloop":
            edges = [e for e in edges if e["from"] != e["to"]]
        ed = ("已观测到的转移：\n" + "\n".join(
            "  %s → %s   ×%d" % (e["from"], e["to"], e["count"]) for e in edges)
              ) if edges else "已观测到的转移：（还没有）"
        return body + "\n\n" + ed + "\n\n当前所处的状态：%s" % row["current_prev_event"]
    if cond == "M3":
        dic = "已知的状态：\n" + "\n".join(
            "  %s = %s" % (n["id"], n["canon"]) for n in row["nodes"])
        seq = " → ".join(SPECIAL_MARK if x["special"] else x["event"] for x in h)
        ed = ("已观测到的转移：\n" + "\n".join(
            "  %s → %s   ×%d" % (e["from"], e["to"], e["count"])
            for e in row["edges"])) if row["edges"] else "已观测到的转移：（还没有）"
        return (dic + "\n\n实际经过的顺序：\n  " + seq + "\n\n" + ed
                + "\n\n当前所处的状态：%s" % row["current_prev_event"])
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--nodemap", default=NODEMAP)
    ap.add_argument("--e1a", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", help="打印某个 cand_id 的五种渲染，人工检查")
    a = ap.parse_args()

    rows = add_length_controls(build(a.release, a.nodemap, a.e1a))
    print("前缀 %d 个 / %d 条视频" % (len(rows), len({r["video_id"] for r in rows})))
    hs = sorted(r["n_history"] for r in rows)
    ns = sorted(r["n_nodes_seen"] for r in rows)
    print("  历史长度  中位 %d，最短 %d，最长 %d" % (hs[len(hs) // 2], hs[0], hs[-1]))
    print("  已见节点数 中位 %d，最少 %d，最多 %d" % (ns[len(ns) // 2], ns[0], ns[-1]))
    # 措辞混淆的规模：同一节点的两次出现，raw 写法相同的比例
    same_node_pairs = diff_word = 0
    for r in rows:
        by = collections.defaultdict(list)
        for x in r["history"]:
            if not x["special"]:
                by[x["event"]].append(x["raw"])
        for k, v in by.items():
            for i in range(1, len(v)):
                same_node_pairs += 1
                diff_word += v[i] != v[i - 1]
    if same_node_pairs:
        print("  历史里同一节点的相邻两次出现：%d 对，其中**措辞不同** %d (%.0f%%)"
              % (same_node_pairs, diff_word, 100 * diff_word / same_node_pairs))
        print("    → 这就是 M1raw 与 M1canon 之间被控制掉的那个变量的规模")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"schema_version": "e2_prefix_v1", "frozen": "2026-09-28",
               "conditions": ["M0", "M1raw", "M1canon", "M2raw", "M2canon", "M3",
                              "M2rawd", "M3raw", "M3raw_noloop",
                              "M3raw_shuffled", "M3raw_other",
                              "M2lab_perm", "M2lab_opaque", "M2lab_first"],
               "n": len(rows), "rows": rows},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> %s" % a.out)

    if a.show:
        r = next((x for x in rows if x["cand_id"] == a.show), None) or rows[len(rows) // 2]
        print("\n" + "=" * 70)
        print("样例 %s（历史 %d 条，%d 个节点）" % (r["cand_id"], r["n_history"], r["n_nodes_seen"]))
        for cond in ("M1raw", "M1canon", "M2raw", "M2canon", "M3", "M2rawd", "M3raw"):
            print("\n----- %s -----" % cond)
            print(render(r, cond))


if __name__ == "__main__":
    main()
