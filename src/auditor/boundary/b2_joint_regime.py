"""Put every dataset on one definition and measure the two axes separately.

WHY THIS FILE EXISTS. The project had been quoting two different quantities as
"branching". `g1_ladder`'s branching regime is `|S| >= 2 and gold in S`, which
on the internal release is 11 of 322 decisions. `branch_mine`'s G-branch adds
the two clauses that decide whether the structure is usable at all -- the
successor set must not already contain every candidate seen, and there must be
at least MIN_HIST states of history -- and that is 2 of 322. Comparing 11
against Ego-Exo4D's 9 would have put two definitions on one axis. Under one
definition the internal release is 0.62% and Ego-Exo4D 3.75%, so the ordering
survives, but it had to be checked rather than assumed.

THE NINE VACUOUS ONES ARE A RESULT, NOT BOOKKEEPING. The gap between 11 and 2
is almost entirely the `|S| < |candidates|` clause: in nine of the eleven
decisions where a predecessor had two or more observed successors, the
successor set already contained everything seen so far, so conditioning on it
excludes nothing. A graph that permits every candidate is not a constraint,
however high its out-degree.

TWO AXES, NOT ONE SCORE. A transition prior needs state recurrence AND
branching at once, and the two datasets fail on different axes, so collapsing
them into one number hides the finding. Recurrence is measured as how often a
predecessor has been departed before and, given that, how often its successor
set holds the truth. Branching is measured as how often the set holds two
different successors, and how often that set also narrows the candidates.

Sequences arrive from different places -- the internal release through
`task_rows`, Ego-Exo4D through keystep ids, human labels through the coarse
export -- so each loader produces state sequences and everything downstream is
shared. That is the only way the columns mean the same thing.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

from src.auditor.boundary.b2_yield import chain, rows_of
from src.auditor.boundary.branch_mine import classify, MIN_HIST


def measure(rows, label, n_units, min_hist=MIN_HIST):
    """-> dict. `rows` already carry S / candidates / n_seen / gold_node."""
    S_dist = collections.Counter()
    gbr = gnov = gsim = in_S = out_S = ge2 = vacuous = 0
    for r in rows:
        S_dist[len(r["S"])] += 1
        k = classify(r, min_hist=min_hist)
        gbr += "G-branch" in k
        gnov += "G-novel" in k
        gsim += "G-simple" in k
        if len(r["S"]) >= 2:
            ge2 += 1
            if len(r["S"]) >= len(r["candidates"]):
                vacuous += 1
        if r["S"]:
            if r["gold_node"] in r["S"]:
                in_S += 1
            else:
                out_S += 1
    n = max(1, len(rows))
    non_empty = in_S + out_S
    return {
        "label": label, "n_units": n_units, "n_decisions": len(rows),
        # --- 轴一：复现 ---
        "p_support": non_empty / n,
        "hit_given_support": (in_S / non_empty) if non_empty else None,
        # --- 轴二：分叉 ---
        "p_S_ge_2": ge2 / n,
        "n_S_ge_2": ge2,
        "n_vacuous": vacuous,
        "p_g_branch": gbr / n,
        "g_branch": gbr, "g_simple": gsim, "g_novel": gnov,
        "novel_per_branch": (gnov / gbr) if gbr else None,
        "S_dist": dict(sorted(S_dist.items())),
    }


def from_sequences(seqs, label, min_hist=MIN_HIST):
    rows = [r for s in seqs for r in rows_of(s)]
    return measure(rows, label, len(seqs), min_hist)


def internal_rows(release, nodemap):
    """-> the internal release's decisions, built exactly as branch_mine does."""
    from src.auditor.boundary.g1_probe import task_rows
    from src.auditor.boundary.g1b_support import edge_list, support_of
    rows = task_rows(release, nodemap)
    for r in rows:
        r["S"] = (support_of(edge_list(r["edges"]), r["prev_node"])
                  if r["prev_node"] is not None else set())
    return rows


def keystep_seqs(windows, ann):
    seqs = []
    for w in windows:
        segs = sorted((s for s in (ann.get(w["take_uid"], {}).get("segments") or [])
                       if float(s.get("start_time", 0)) >= w["start_s"] - 1e-6
                       and float(s.get("end_time", 0)) <= w["end_s"] + 1e-6),
                      key=lambda s: float(s.get("start_time", 0)))
        seq = []
        for s in segs:
            c = str(s.get("step_unique_id"))
            if not seq or seq[-1] != c:
                seq.append(c)
        seqs.append(seq)
    return seqs


def table(rs):
    print("\n=== 同一个 classify、同一套严格前缀，两条轴分开报 ===")
    print("  %-30s %6s %7s | %8s %8s | %8s %9s %8s"
          % ("数据", "单位", "决策", "S非空", "命中|S非空",
             "|S|>=2", "其中空约束", "G-branch"))
    for r in rs:
        print("  %-30s %6d %7d | %7.1f%% %9s | %7.2f%% %9s %7.2f%%"
              % (r["label"][:30], r["n_units"], r["n_decisions"],
                 100 * r["p_support"],
                 "%.1f%%" % (100 * r["hit_given_support"])
                 if r["hit_given_support"] is not None else "—",
                 100 * r["p_S_ge_2"],
                 "%d/%d" % (r["n_vacuous"], r["n_S_ge_2"]) if r["n_S_ge_2"] else "—",
                 100 * r["p_g_branch"]))
    print("\n  复现 = S 非空的比例，以及 S 非空时 gold 真在里面的比例")
    print("  分叉 = 前驱被观察到走向两个不同后继的比例；"
          "「空约束」是其中可达集已含全部候选、条件化排除不掉任何东西的")
    print("  G-branch = |S|>=2 且 |S|<候选数 且 n_seen>=%d —— 真正可用的那一类"
          % MIN_HIST)
    for r in rs:
        if r["novel_per_branch"]:
            print("  %-30s G-novel : G-branch = %.1f : 1"
                  % (r["label"][:30], r["novel_per_branch"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", help="内部 release 目录（给了就算这一行）")
    ap.add_argument("--nodemap")
    ap.add_argument("--export", action="append", default=[],
                    help="b2_coarse_export_v1 人工粗标，可多个")
    ap.add_argument("--sample", action="append", default=[],
                    help="与 --keystep 配对的窗口清单，可多个")
    ap.add_argument("--keystep", help="Ego-Exo4D keystep 标注")
    ap.add_argument("--min-hist", dest="min_hist", type=int, default=MIN_HIST)
    ap.add_argument("--out")
    a = ap.parse_args()

    rs = []
    if a.release:
        from src.auditor.boundary.g1_probe import RELEASE, NODEMAP
        rows = internal_rows(a.release or RELEASE, a.nodemap or NODEMAP)
        rs.append(measure(rows, "内部 release（人工合同、强制 id 复用）",
                          len({r["video_id"] for r in rows}), a.min_hist))
    for p in a.export:
        doc = json.load(open(p, encoding="utf-8"))
        vids = [v for v in (doc.get("videos") or []) if v.get("status") == "final"]
        rs.append(from_sequences([chain(v) for v in vids],
                                 "人工粗标 %s" % doc.get("batch_id", p),
                                 a.min_hist))
    if a.keystep:
        ann = json.load(open(a.keystep, encoding="utf-8")).get("annotations", {})
        for p in a.sample:
            doc = json.load(open(p, encoding="utf-8"))
            rs.append(from_sequences(keystep_seqs(doc["windows"], ann),
                                     "Ego-Exo4D keystep %s"
                                     % doc.get("family", p), a.min_hist))
    if not rs:
        raise SystemExit("至少给一个数据源")
    table(rs)
    if a.out:
        json.dump({"min_hist": a.min_hist, "rows": rs},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
