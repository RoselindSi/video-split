"""Per-scenario coordinates for the paper's regime figure.

WHY PER SCENARIO AND NOT PER DATASET. The earlier figure put Ego-Exo4D at one
point, and two uniform draws from its cooking family alone returned a 10.5%
and a 33.8% hit rate (n=38 against n=77, intervals not overlapping). The
per-scenario yields span 0.02 to 1.05, so a dataset mean hides the only thing
the figure is about: whether the two properties a transition prior needs ever
co-occur. The spread is the result, so every scenario gets its own point.

THE TWO AXES, AND WHAT THEY ARE NOT. Recurrence is measured as how often the
predecessor has been departed before (`P(S non-empty)`) and, given that, how
often its successor set contains the truth (`hit | S non-empty`). It is NOT
`P(n_t = n_{t-1})`, which is a self-loop rate -- staying put rather than
returning. Branching is `P(|S| >= 2)`, the predecessor observed going to two
different successors, reported next to the share of those whose successor set
already admits every candidate seen, because a graph that excludes nothing is
not a constraint however high its out-degree.

GRAIN IS CARRIED, NOT AVERAGED OVER. Ego-Exo4D scenarios are measured at the
dataset's own keystep grain, the internal release under its annotation
contract, and the human batches at whatever grain they were labelled. Those
are different instruments and the figure has to say so per point rather than
mixing them into one cloud; `grain` is emitted for exactly that.
"""
from __future__ import annotations

import argparse
import collections
import json

from src.auditor.boundary.b2_yield import chain, rows_of
from src.auditor.boundary.branch_mine import classify, MIN_HIST


def axes(seqs, min_hist=MIN_HIST):
    """-> the two axes plus the counts needed to judge them."""
    n = ge2 = gbr = in_S = out_S = vac = 0
    for s in seqs:
        for r in rows_of(s):
            n += 1
            if len(r["S"]) >= 2:
                ge2 += 1
                if len(r["S"]) >= len(r["candidates"]):
                    vac += 1
            if "G-branch" in classify(r, min_hist=min_hist):
                gbr += 1
            if r["S"]:
                if r["gold_node"] in r["S"]:
                    in_S += 1
                else:
                    out_S += 1
    ne = in_S + out_S
    return {"n_windows": len(seqs), "n_decisions": n,
            "p_support": (ne / n) if n else 0.0,
            "hit": (in_S / ne) if ne else None, "n_support": ne,
            "p_S_ge_2": (ge2 / n) if n else 0.0, "n_S_ge_2": ge2,
            "n_vacuous": vac,
            "p_g_branch": (gbr / n) if n else 0.0, "g_branch": gbr,
            "per_window": gbr / max(1, len(seqs))}


def keystep_scenarios(keystep_path, window_s, min_windows, coder="step_unique_id"):
    """`coder` picks what counts as the same state. The default indexes which
    step, so returning to the same physical state draws a fresh id and
    recurrence is suppressed by construction; `step_name` merges by wording and
    so over-counts it. Running both brackets the bias rather than asserting it.
    """
    from src.auditor.boundary.egoexo_topology_prescreen import load, windows, CODERS
    key = CODERS[coder]
    takes = load(keystep_path)
    by = collections.defaultdict(list)
    for uid, t in takes.items():
        by[t["scenario"]].append(t)
    out = []
    for scen, ts in sorted(by.items()):
        seqs = []
        for t in ts:
            for w in windows(t["segments"], window_s):
                seq = []
                for s in w:
                    c = key(s)
                    if not seq or seq[-1] != c:
                        seq.append(c)
                seqs.append(seq)
        if len(seqs) < min_windows:
            continue
        r = axes(seqs)
        r.update({"label": scen, "source": "Ego-Exo4D",
                  "grain": "keystep", "coder": coder, "window_s": window_s})
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keystep")
    ap.add_argument("--window", type=float, default=180.0)
    ap.add_argument("--coder", default="step_unique_id",
                    choices=["step_unique_id", "step_id", "step_name"],
                    help="状态身份用哪个字段；见 keystep_scenarios 的说明")
    ap.add_argument("--min-windows", dest="minw", type=int, default=10,
                    help="窗口太少的 scenario 两条轴都没有精度，不画")
    ap.add_argument("--release", help="内部 release 目录")
    ap.add_argument("--nodemap")
    ap.add_argument("--export", action="append", default=[],
                    help="人工粗标 export，可多个；格式 path[:标签]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    if a.keystep:
        rows += keystep_scenarios(a.keystep, a.window, a.minw, a.coder)

    if a.release:
        from src.auditor.boundary.b2_joint_regime import internal_rows, measure
        from src.auditor.boundary.g1_probe import NODEMAP
        ir = internal_rows(a.release, a.nodemap or NODEMAP)
        m = measure(ir, "内部 release", len({r["video_id"] for r in ir}))
        rows.append({"label": "内部 release（人工合同）", "source": "internal",
                     "grain": "contract", "n_windows": m["n_units"],
                     "n_decisions": m["n_decisions"],
                     "p_support": m["p_support"], "hit": m["hit_given_support"],
                     "n_support": round(m["p_support"] * m["n_decisions"]),
                     "p_S_ge_2": m["p_S_ge_2"], "n_S_ge_2": m["n_S_ge_2"],
                     "n_vacuous": m["n_vacuous"],
                     "p_g_branch": m["p_g_branch"], "g_branch": m["g_branch"],
                     "per_window": m["g_branch"] / max(1, m["n_units"])})

    for spec in a.export:
        path, _, lab = spec.partition(":")
        doc = json.load(open(path, encoding="utf-8"))
        vids = [v for v in (doc.get("videos") or []) if v.get("status") == "final"]
        r = axes([chain(v) for v in vids])
        r.update({"label": lab or doc.get("batch_id", path), "source": "human",
                  "grain": "human"})
        rows.append(r)

    print("  %-40s %-10s %4s %5s | %7s %7s | %7s %8s"
          % ("点", "粒度", "窗口", "决策", "S非空", "命中", "|S|>=2", "G-branch"))
    for r in sorted(rows, key=lambda x: -x["p_g_branch"]):
        print("  %-40s %-10s %4d %5d | %6.1f%% %7s | %6.2f%% %7.2f%%"
              % (r["label"][:40], r["grain"], r["n_windows"], r["n_decisions"],
                 100 * r["p_support"],
                 "%.1f%%" % (100 * r["hit"]) if r["hit"] is not None else "—",
                 100 * r["p_S_ge_2"], 100 * r["p_g_branch"]))
    hi = [r for r in rows if r["hit"] is not None and r["hit"] >= 0.5
          and r["p_g_branch"] >= 0.03]
    print("\n  同时满足 命中>=50%% 且 G-branch>=3%% 的点：%d / %d  <- 期望区间"
          % (len(hi), len(rows)))
    json.dump({"window_s": a.window, "min_windows": a.minw, "points": rows},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("  -> %s" % a.out)


if __name__ == "__main__":
    main()
