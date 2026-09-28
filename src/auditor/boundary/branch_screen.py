"""Which unannotated recordings would actually yield branching decisions.

THE SELECTION MUST NOT USE THE MODEL. Picking recordings by what the visual
scorer or the transition prior says about them would build the regime it claims
to find, and the same rule already applies to picking decisions. So the screen
runs entirely on `human_ego_recording_segments_v1`, a human segmentation with
its own label per segment, produced under a different contract and long before
any of this. Nothing here reads an embedding or a model score.

THE FIRST VERSION OF THIS SCREEN SELECTED FOR INCONSISTENT ANNOTATION, which
is worse than selecting for nothing. Node identity was taken from the label
text, and in `recording_000157` two states are written four ways -- `Remove
smartphone protective case`, `Remove phone case`, `attach phone case`,
`reattach phone case`. Counted literally that recording has seventeen nodes
and fifty-five branching decisions; it actually oscillates between install and
remove and has none. The branches were wording drift, so ranking on them would
have picked the recordings whose annotator was least consistent.

TWO GATES, BOTH AIMED AT THAT FAILURE:

    reuse      how often labels repeat within the recording, as
               `unique / segments`. Across the corpus this runs from 0.16 at
               the tenth percentile to 1.00 at the ninetieth -- at the top end
               every segment is worded differently and node identity simply
               cannot be read off the text. Only recordings whose annotator
               demonstrably reused labels are admitted.
    collapse   within a recording, labels whose content words overlap enough
               are merged, so `remove phone case` and `remove smartphone
               protective case` stop being two nodes. Branch counts are
               reported before and after, and selection uses the count that
               survives, because the collapse can only remove fake branches
               and never invents one.

`main_objects` would have been a better identity than free text and is empty
in all 25904 segments, so it is not used.

IT IS A PROXY, AND ONLY FOR SELECTION. These labels are not schema v2.1 node
identities -- the granularity is someone else's -- so whatever this picks
still gets annotated from scratch under v2.1. The proxy only has to order
recordings correctly, not label them.

WHAT IS COUNTED, under the same strict prefix rule the analysis uses:

    opportunity   a node left two or more times, so it *could* branch
    realisation   it went somewhere different at least once
    G-branch      |N+(A)| >= 2 before the decision, support still excludes a
                  candidate, >=3 nodes known
    G-novel       the answer is not in N+(A), which a hard mask would forbid
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

from src.auditor.boundary.branch_mine import classify

# 序号标记：英文 set/round/... + 数字，中文第N组/次/遍，括号数字，结尾裸数字
OCC = [
    re.compile(r"\b(set|round|part|trial|rep|repetition|time|attempt|batch|cycle|pass|stage|step)\s*[#no.]*\s*\d+\b", re.I),
    re.compile(r"第\s*[0-9一二三四五六七八九十]+\s*[组次遍轮个段]"),
    re.compile(r"[（(]\s*\d+\s*[)）]"),
    re.compile(r"\s+\d+\s*$"),
]
STOP = {"the", "a", "an", "of", "to", "into", "from", "with", "on", "in", "at",
        "and", "or", "for", "by", "out", "up", "down", "again", "then", "it",
        "its", "his", "her", "their", "some", "one", "another", "back"}


def norm(lab):
    s = (lab or "").strip().lower()
    for _ in range(3):
        for rx in OCC:
            s = rx.sub(" ", s)
    s = re.sub(r"[\s\-_,.]+", " ", s).strip(" -_,.")
    return s or (lab or "").strip().lower()


def content(lab):
    return frozenset(t for t in re.findall(r"[a-z]+", norm(lab)) if t not in STOP)


def collapse(labels, thr):
    """把措辞变体并成一个节点。-> {label: 代表label}

    并的判据是内容词 Jaccard >= thr。合并只会去掉假分叉，不会凭空造出一个，
    所以选片用合并后的数字是保守方向。"""
    uniq = sorted({norm(x) for x in labels if norm(x)})
    parent = {u: u for u in uniq}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cw = {u: content(u) for u in uniq}
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            ca, cb = cw[a], cw[b]
            if not ca or not cb:
                continue
            j = len(ca & cb) / len(ca | cb)
            if j >= thr:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    return {u: find(u) for u in uniq}


def rows_of(nodes):
    """节点序列 -> 严格前缀的决策行，字段与 branch_mine.classify 一致。"""
    out, seen, prev, edges = [], [], None, collections.Counter()
    for n in nodes:
        if seen:
            S = {t for (s, t) in edges if s == prev} if prev is not None else set()
            out.append({"candidates": list(seen), "prev_node": prev, "S": S,
                        "n_seen": len(seen), "gold_node": n,
                        "gold": n if n in seen else "NEW"})
        if prev is not None:
            edges[(prev, n)] += 1
        if n not in seen:
            seen.append(n)
        prev = n
    return out


def topology(nodes):
    coll = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
    rows = rows_of(coll)
    for r in rows:
        r["types"] = classify(r)
    succ = collections.defaultdict(list)
    for a, b in zip(coll, coll[1:]):
        succ[a].append(b)
    return {"n_seg": len(coll), "n_node": len(set(coll)), "dec": len(rows),
            "opp": sum(1 for t in succ.values() if len(t) >= 2),
            "real": sum(1 for t in succ.values() if len(set(t)) >= 2),
            "branch": sum(1 for r in rows if "G-branch" in r["types"]),
            "novel": sum(1 for r in rows if "G-novel" in r["types"]),
            "simple": sum(1 for r in rows if "G-simple" in r["types"])}


def scan(path, thr):
    d = json.load(open(path, encoding="utf-8"))
    segs = d.get("segments") or []
    raw = [s.get("label_en") or s.get("label_zh") or "" for s in segs]
    nn = [norm(x) for x in raw]
    m = collapse(raw, thr)
    a = topology(nn)
    b = topology([m.get(x, x) for x in nn])
    dur = sum(float(s.get("end_s", 0)) - float(s.get("start_s", 0)) for s in segs)
    return {"rid": d.get("recording_id") or os.path.basename(os.path.dirname(path)),
            "path": path, "n_seg_raw": len(segs), "dur_s": round(dur, 1),
            "reuse": round(len({x for x in nn if x}) / max(1, len(segs)), 3),
            "n_node_raw": a["n_node"], "n_node": b["n_node"],
            "branch_raw": a["branch"], "branch": b["branch"],
            "novel": b["novel"], "simple": b["simple"], "dec": b["dec"],
            "opp": b["opp"], "real": b["real"], "n_seg": b["n_seg"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--max_reuse", type=float, default=0.5,
                    help="unique/段数 的上限；越低表示标注者越一致")
    ap.add_argument("--min_node", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()

    seen, paths = set(), []
    for r in a.roots:
        for p in sorted(glob.glob(os.path.join(r, "**", "segments.json"),
                                  recursive=True)):
            rid = os.path.basename(os.path.dirname(p))
            if rid in seen:                 # 父目录与 part_01/02 是同一批
                continue
            seen.add(rid)
            paths.append(p)
    recs = []
    for p in paths:
        try:
            recs.append(scan(p, a.jaccard))
        except Exception:
            pass
    print("去重后 %d 条录像（Jaccard 合并阈值 %.2f）\n" % (len(recs), a.jaccard))

    t = lambda k, g=None: sum(r[k] for r in (g if g is not None else recs))
    print("=== 合并措辞变体前后 ===")
    print("  节点数中位：合并前 %d → 合并后 %d"
          % (sorted(r["n_node_raw"] for r in recs)[len(recs) // 2],
             sorted(r["n_node"] for r in recs)[len(recs) // 2]))
    print("  G-branch 总数：合并前 %d → 合并后 %d（差额 %d 是措辞漂移造的假分叉）"
          % (t("branch_raw"), t("branch"), t("branch_raw") - t("branch")))

    print("\n=== 标签复用率分箱（unique/段数，越低标注越一致）===")
    print("%-12s %6s %9s %11s %11s" % ("reuse", "录像", "节点中位", "branch/录像", "假分叉占比"))
    for lo, hi, lab in ((0, .2, "<=0.2"), (.2, .35, "0.2-0.35"),
                        (.35, .5, "0.35-0.5"), (.5, .75, "0.5-0.75"),
                        (.75, 1.01, ">0.75")):
        g = [r for r in recs if lo <= r["reuse"] < hi]
        if not g:
            continue
        br, brr = t("branch", g), t("branch_raw", g)
        print("%-12s %6d %9d %11.2f %10s"
              % (lab, len(g), sorted(x["n_node"] for x in g)[len(g) // 2],
                 br / len(g), ("%.0f%%" % (100 * (brr - br) / brr)) if brr else "-"))

    ok = [r for r in recs
          if r["reuse"] <= a.max_reuse and r["n_node"] >= a.min_node
          and r["branch"] >= 1]
    ok.sort(key=lambda r: (-r["branch"], -r["novel"], r["reuse"]))
    print("\n=== 入选（reuse<=%.2f 且 节点>=%d 且 合并后 branch>=1）：%d 条 ==="
          % (a.max_reuse, a.min_node, len(ok)))
    print("%-20s %6s %6s %6s %7s %8s %7s %7s"
          % ("recording", "reuse", "段", "节点", "时长s", "G-branch", "novel", "simple"))
    for r in ok[:a.top]:
        print("%-20s %6.2f %6d %6d %7.0f %8d %7d %7d"
              % (r["rid"], r["reuse"], r["n_seg"], r["n_node"], r["dur_s"],
                 r["branch"], r["novel"], r["simple"]))
    if ok:
        o, rl = t("opp", ok), t("real", ok)
        print("\n  入选集合：G-branch %d，G-novel %d，兑现率 %.0f%%（现有语料 16.4%%）"
              % (t("branch", ok), t("novel", ok), 100 * rl / max(1, o)))
        for n in (50, 60):
            c = k = 0
            for r in ok:
                if c >= n:
                    break
                c += r["branch"]
                k += 1
            print("  凑满 %d 个 G-branch 需要前 %d 条（共 %.0f 分钟素材）"
                  % (n, k, sum(x["dur_s"] for x in ok[:k]) / 60))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(sorted(recs, key=lambda r: -r["branch"]))
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
