"""The guardrail-versus-utility curve the gate has only ever been reported as two points.

THIS LIVES IN THE RESEARCH TREE, NOT THE DELIVERABLE. It was briefly committed
to the ownergate package in the delivery repository, which was wrong: that
repository holds what ships, and a sweep whose conclusion is "the rule is
already on its own frontier" is an experiment. Only results that change the
pipeline belong there.

WHY A CURVE AND NOT A NUMBER. The privacy requirement is asymmetric: letting
someone else's hand through is a privacy failure, blurring the wearer's own
hand is a utility loss. So the operating point should be chosen by fixing the
guardrail and reading off what utility survives, which needs a curve. What has
been reported instead is two points -- with description 49.3%/24.4%, without
37.3%/15.0% -- and two points cannot say whether 0% leakage costs everything or
almost nothing.

WHAT THE SWEEP REPLACES. 45% of the wearer's own tracks are blurred not because
the gate judged them someone else's but because it abstained, and 128 of 129
abstentions say `too_few_context_views`: the view count `n` has a median of 2
against a setting of 5. Thresholding a score instead forces a decision from
whatever views exist, so the curve answers a specific question -- if the gate
stopped abstaining and committed on the evidence it has, what would the
trade-off be -- rather than being a generic ROC.

THE DENOMINATOR INCLUDES TRACKS THE RUN NEVER SAW. A gold track inside a
recording the arm covered, but missing from the arm's output, is a coverage
failure: it gets blurred in deployment and no threshold can rescue it. Dropping
it would flatter every operating point on the curve. `against_gold` keys on
gold ∩ runs, which excludes them, so both denominators are printed and the gap
between them is the coverage gap.

RECORDINGS ARE THE SAMPLE UNIT. Ten recordings carry 202 tracks, so a
track-level interval would be far too narrow. The bootstrap resamples
recordings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random

SCORES = ("p_median", "p_max", "p_min", "positive_frac",
          "base_p_median", "base_strong_frac")


def read_gold(pattern):
    gold = {}
    for path in glob.glob(pattern):
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            truth = (row.get("track_truth") or row.get("human_ownership")
                     or "").strip()
            if truth in ("owner", "other"):
                gold[(row["rec"], str(row["tid"]))] = truth
    return gold


def read_run(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[(r["rec"], str(r["tid"]))] = r
    return out


def rates(gold, run, keys, col, thr):
    """-> (owner pass rate, other pass rate). Absent or unparsable never passes."""
    own = [k for k in keys if gold[k] == "owner"]
    oth = [k for k in keys if gold[k] == "other"]

    def passes(k):
        r = run.get(k)
        if r is None:
            return False                 # 没跑到 —— 部署里会被打码
        v = (r.get(col) or "").strip()
        try:
            return float(v) >= thr
        except ValueError:
            return False
        return False

    return ((sum(1 for k in own if passes(k)) / len(own)) if own else float("nan"),
            (sum(1 for k in oth if passes(k)) / len(oth)) if oth else float("nan"))


def curve(gold, run, keys, col):
    vals = set()
    for k in keys:
        r = run.get(k)
        if not r:
            continue
        try:
            vals.add(float((r.get(col) or "").strip()))
        except ValueError:
            pass
    pts = []
    for t in sorted(vals | {0.0, 1.0000001}):
        u, g = rates(gold, run, keys, col, t)
        pts.append({"thr": t, "owner_pass": u, "other_pass": g})
    return pts


def best_at(pts, bound):
    """-> the point with the highest owner_pass whose leak is within bound."""
    ok = [p for p in pts if p["other_pass"] <= bound + 1e-12]
    return max(ok, key=lambda p: p["owner_pass"]) if ok else None


def boot_owner_at(gold, run, keys, col, thr, bound, n_boot, seed):
    """Recording-clustered CI for owner_pass, and how often the leak bound holds."""
    units = collections.defaultdict(list)
    for k in keys:
        units[k[0]].append(k)
    recs = sorted(units)
    rng = random.Random(seed)
    us, held = [], 0
    for _ in range(n_boot):
        ks = [k for r in (rng.choice(recs) for _ in recs) for k in units[r]]
        u, g = rates(gold, run, ks, col, thr)
        if u == u:
            us.append(u)
        if g == g and g <= bound + 1e-12:
            held += 1
    us.sort()
    if not us:
        return None
    lo = us[int(0.025 * len(us))]
    hi = us[int(0.975 * len(us))]
    return {"ci": [lo, hi], "bound_held_frac": held / max(1, n_boot)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--run", action="append", required=True, help="名字=路径")
    ap.add_argument("--bounds", default="0,0.05,0.10,0.15,0.20",
                    help="护栏上界（别人的手被放行率）")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    a = ap.parse_args()

    gold = read_gold(a.gold)
    runs = {}
    for spec in a.run:
        name, _, path = spec.partition("=")
        runs[name] = read_run(path)
    bounds = [float(x) for x in a.bounds.split(",")]

    report = {"bounds": bounds, "arms": {}}
    for name, run in runs.items():
        covered = {k[0] for k in run}
        keys_rec = [k for k in gold if k[0] in covered]          # 含没跑到
        keys_int = [k for k in gold if k in run]                 # against_gold 的口径
        n_own = sum(1 for k in keys_rec if gold[k] == "owner")
        print("=== %s ===" % name)
        print("  覆盖 %d 条录像。分母两种：录像内全部 gold %d 条（OWNER %d）"
              % (len(covered), len(keys_rec), n_own))
        print("  　　　　　　　　　 与 run 的交集 %d 条 —— 差 %d 条就是覆盖缺口"
              % (len(keys_int), len(keys_rec) - len(keys_int)))
        rule_own = sum(1 for k in keys_rec
                       if gold[k] == "owner"
                       and (run.get(k, {}).get("verdict") or "") == "owner")
        rule_oth = sum(1 for k in keys_rec
                       if gold[k] == "other"
                       and (run.get(k, {}).get("verdict") or "") == "owner")
        n_oth = len(keys_rec) - n_own
        print("  现在的规则操作点：OWNER 放行 %.1f%%   别人的手被放行 %.1f%%"
              % (100 * rule_own / max(1, n_own), 100 * rule_oth / max(1, n_oth)))
        arm = {"n_recordings": len(covered), "n_keys_rec": len(keys_rec),
               "n_keys_intersect": len(keys_int), "n_owner": n_own,
               "n_other": n_oth,
               "rule": {"owner_pass": rule_own / max(1, n_own),
                        "other_pass": rule_oth / max(1, n_oth)},
               "scores": {}}
        print("\n  %-18s %s" % ("分数列", "  ".join("护栏<=%.0f%%" % (100 * b)
                                                 for b in bounds)))
        for col in SCORES:
            pts = curve(gold, run, keys_rec, col)
            if not pts:
                continue
            cells, detail = [], {}
            for b in bounds:
                p = best_at(pts, b)
                if p is None:
                    cells.append("   —   ")
                    continue
                cells.append("  %5.1f%%" % (100 * p["owner_pass"]))
                detail["%.2f" % b] = {"thr": p["thr"],
                                      "owner_pass": p["owner_pass"],
                                      "other_pass": p["other_pass"]}
            print("  %-18s %s" % (col, "".join(cells)))
            arm["scores"][col] = {"points": pts, "at_bound": detail}
        report["arms"][name] = arm

        # 最好的那一列，在最严护栏上给区间
        tight = bounds[0]
        cand = [(c, v["at_bound"].get("%.2f" % tight))
                for c, v in arm["scores"].items()]
        cand = [(c, d) for c, d in cand if d]
        if cand:
            col, d = max(cand, key=lambda x: x[1]["owner_pass"])
            bs = boot_owner_at(gold, run, keys_rec, col, d["thr"], tight,
                               a.boot, a.seed)
            print("\n  护栏 <=%.0f%% 时最好的是 %s（阈值 %.4f）：OWNER 放行 %.1f%%"
                  % (100 * tight, col, d["thr"], 100 * d["owner_pass"]))
            if bs:
                print("    按录像聚类 bootstrap（%d 次）CI [%.1f%%, %.1f%%]；"
                      "重抽样里护栏仍然守住的比例 %.0f%%"
                      % (a.boot, 100 * bs["ci"][0], 100 * bs["ci"][1],
                         100 * bs["bound_held_frac"]))
                print("    （护栏守住的比例低，说明这个操作点是在这 10 条录像上"
                      "调出来的，换一批会漏）")
                arm["tight_best"] = {"score": col, **d, **bs}
        print()

    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), indent=1)
        print("-> %s" % a.out)


if __name__ == "__main__":
    main()
