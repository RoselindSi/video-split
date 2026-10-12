"""Train the owner head to answer "should I overrule the detector", not "whose hand is this".

WHY THE TARGET CHANGES. Asked to predict ownership directly, the student
reaches 69.9% against a detector prior that is free and reaches 88.5%. Feeding
it that prior as an extra feature helped only where the teacher disagrees --
26.9% to 34.6% on those tracks -- while costing 51 of the 200 tracks the
detector already had right, because the student treated the prior as one more
thing to blend rather than as the answer to start from.

MAKING IT A RESIDUAL REMOVES THAT FAILURE BY CONSTRUCTION. The target becomes
`the teacher disagrees with the detector`, and the ownership that gets used is
the detector's, flipped when and only when the head fires. A head that never
fires scores exactly what the detector scores; it cannot do worse, so every
point it earns is a point added rather than traded.

THE PRIOR STAYS AN INPUT even though it is no longer the output, because
overruling `owner` and overruling `other` are different visual questions --
one asks whether this hand belongs to someone else in the scene, the other
whether it is the wearer's after all -- and a head that cannot tell which
question it is being asked has to answer both at once.

ONLY TRACKS WITH A DETECTOR VERDICT CAN BE USED. Without a prior there is
nothing to overrule and the target is undefined, so those rows are dropped
rather than given a default; a default would teach the head that "no prior"
means "do not fire", which is a fact about the manifest and not about hands.

THE POSITIVE CLASS IS 13.5% and the metric that matters is not accuracy.
A head that never fires is 86.5% accurate and worth nothing, so what gets
reported is recall on the tracks the teacher flipped and the count of
non-flipped tracks it breaks.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="带 det_own 的 owner manifest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    fields = list(rows[0].keys())

    # 轨迹级裁决：教师取中位数，和分析端口径一致
    per = collections.defaultdict(list)
    for r in rows:
        per[(r["rec"], r["canonical_tid"])].append(r)
    keep, dropped = [], 0
    stat = collections.Counter()
    for key, group in per.items():
        det = (group[0].get("det_own") or "").strip()
        if det not in ("owner", "other"):
            dropped += len(group)
            continue
        med = statistics.median(float(r["teacher_p"]) for r in group)
        teacher = "owner" if med >= 0.5 else "other"
        override = 1.0 if teacher != det else 0.0
        stat[(det, teacher)] += 1
        for r in group:
            r = dict(r)
            # 目标改成「要不要推翻」；归属本身不再是输出
            r["teacher_p"] = "%.1f" % override
            r["human_label"] = ""
            keep.append(r)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)

    print("保留 %d 行 / %d 条轨迹；丢掉 %d 行（没有检测器先验，推翻无从谈起）"
          % (len(keep), sum(stat.values()), dropped))
    print("\n检测器 × 教师（轨迹级）")
    print("%-10s %10s %10s" % ("det \\ 教师", "owner", "other"))
    for d in ("owner", "other"):
        print("%-10s %10d %10d" % (d, stat[(d, "owner")], stat[(d, "other")]))
    pos = stat[("owner", "other")] + stat[("other", "owner")]
    print("正类（要推翻）%d / %d = %.1f%%" % (pos, sum(stat.values()),
                                          100 * pos / max(1, sum(stat.values()))))
    print("\n按 split 的正类比例")
    for s in ("train", "val", "test"):
        g = [r for r in keep if r["split"] == s]
        if not g:
            continue
        t = {(r["rec"], r["canonical_tid"]) for r in g}
        p = len({(r["rec"], r["canonical_tid"]) for r in g
                 if float(r["teacher_p"]) >= 0.5})
        print("  %-6s %5d 行 / %4d 轨迹，要推翻的 %3d (%.1f%%)"
              % (s, len(g), len(t), p, 100 * p / max(1, len(t))))
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
