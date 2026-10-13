"""E1a's seven numbers, defined before any answer exists.

WRITTEN AHEAD OF THE AUDIT ON PURPOSE. Every denominator here was fixed while
the sheet was still blank, so none of them can be chosen later to suit the
result. In particular `correct + partial = acceptable` and `correct only =
strict` are both reported every time, and neither is allowed to become "the"
accuracy depending on which reads better.

THE TWO ARMS ARE NEVER POOLED. 80 representative segments estimate a rate for
the 378; 20 challenge segments are enriched by construction and estimate
nothing. Summing them would report a number for a population that does not
exist, so this file refuses to compute a combined figure at all.

RESULT PRECISION EXCLUDES `not_observable` FROM ITS DENOMINATOR, because
answering null when nothing concludes is a right answer, not an abstention.
Including it would let a model buy precision by saying null more often; the
`missing` count is what pushes back on that, and it is reported beside it.

CONFIDENCE INTERVALS CLUSTER ON VIDEO. Segments from one recording share a
scene, an operator and a camera; treating them as independent would shrink
every interval. The bootstrap resamples videos, not segments.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

BOOT = 2000
SEED = 20260926


def cluster_ci(units, stat, n_boot=BOOT, seed=SEED):
    """95% CI by resampling whole videos. -> (lo, hi) or None when degenerate."""
    keys = sorted(units)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        pick = [units[rng.choice(keys)] for _ in keys]
        rows = [r for grp in pick for r in grp]
        v = stat(rows)
        if v is not None:
            out.append(v)
    if not out:
        return None
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def rate(rows, num, den):
    d = [r for r in rows if den(r)]
    if not d:
        return None
    return sum(1 for r in d if num(r)) / len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--items", help="核验包里的 items（可选，用来取模型的 uncertain）")
    a = ap.parse_args()

    key = {x["id"]: x for x in json.load(open(a.key, encoding="utf-8"))["items"]}
    ans = json.load(open(a.answers, encoding="utf-8"))["answers"]
    for r in ans:
        r.update({k: v for k, v in key.get(r["id"], {}).items() if k not in r})

    ACC = {"correct", "partial"}
    METRICS = [
        ("物体 acceptable", lambda r: r.get("object_status") in ACC,
         lambda r: r.get("object_status")),
        ("物体 strict", lambda r: r.get("object_status") == "correct",
         lambda r: r.get("object_status")),
        ("动作 acceptable", lambda r: r.get("action_status") in ACC,
         lambda r: r.get("action_status")),
        ("动作 strict", lambda r: r.get("action_status") == "correct",
         lambda r: r.get("action_status")),
        ("物体+动作 joint usable",
         lambda r: r.get("object_status") in ACC and r.get("action_status") in ACC,
         lambda r: r.get("object_status") and r.get("action_status")),
        # 分母只取「模型声称有结果」的：not_observable 是判对，不进分母
        ("结果 precision", lambda r: r.get("result_status") == "correct",
         lambda r: r.get("result_status") in ("correct", "unsupported")),
        ("臆测率 unsupported_inference", lambda r: r.get("unsupported_inference") == "yes",
         lambda r: r.get("unsupported_inference")),
        ("整体可用 overall_usable", lambda r: r.get("overall_usable") == "yes",
         lambda r: r.get("overall_usable")),
    ]

    for arm in ("representative", "challenge"):
        rows = [r for r in ans if r.get("arm") == arm]
        if not rows:
            continue
        units = collections.defaultdict(list)
        for r in rows:
            units[r.get("video_id", r["id"])].append(r)
        print("\n=== %s（%d 段 / %d 条视频）===" % (arm, len(rows), len(units)))
        if arm == "challenge":
            print("    按构造富集，不估计总体；下列数字只描述这 20 段本身。")
        print("%-28s %7s %7s  %s" % ("指标", "值", "分母", "95% CI（按视频聚类）"))
        for name, num, den in METRICS:
            v = rate(rows, num, den)
            n = sum(1 for r in rows if den(r))
            if v is None:
                print("%-28s %7s %7d" % (name, "—", n))
                continue
            ci = (cluster_ci(units, lambda rr: rate(rr, num, den))
                  if arm == "representative" else None)
            print("%-28s %6.1f%% %7d  %s"
                  % (name, 100 * v, n,
                     "[%.1f%%, %.1f%%]" % (100 * ci[0], 100 * ci[1]) if ci else ""))

        print("  分布：")
        for field in ("object_status", "action_status", "result_status",
                      "uncertainty_status"):
            c = collections.Counter(r.get(field) for r in rows if r.get(field))
            print("    %-20s %s" % (field, dict(c)))

        if arm == "challenge":
            print("  按困难类型：")
            for ch in sorted({r.get("challenge") for r in rows if r.get("challenge")}):
                g = [r for r in rows if r.get("challenge") == ch]
                bad = sum(1 for r in g if r.get("overall_usable") == "no")
                uns = sum(1 for r in g if r.get("result_status") == "unsupported")
                print("    %-28s n=%d  不够用 %d  没有依据 %d" % (ch, len(g), bad, uns))

    # 模型自己的弃权率来自全量 378，不是这 100 个
    print("\n注：Abstention rate 与 Accuracy-when-non-uncertain 需要模型的 uncertain 字段；")
    print("    E1a 全量 378 段里 uncertain=true 为 0，因此这两项在本轮无法计算，")
    print("    并且这个 0 本身就是结果——不是缺测。")


if __name__ == "__main__":
    main()
