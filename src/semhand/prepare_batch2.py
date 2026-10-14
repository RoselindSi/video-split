"""把第二批的检测 csv 导成与第一批同构的三件产物，并统一 join 的窗口。

为什么不能直接用两批现成的东西。第一批的 `detect2_joined` 是 e2e_v4 用**旧窗口
30** 跑出来的（那次运行早于窗口冻结），第二批若现在跑 track_join 就是 150 ——
两批的「部署状态」会是不同参数下的产物，合并统计就是在比两个不同的系统。

所以两批都由这里按**同一个窗口**算 join。这么做的前提是模拟能复现流水线，而
那一条核对过：窗口 30 下模拟分组与 e2e_v4 实际分组在 7/7 条录像上一致
（见 join_window_sweep 的 check30）。

第二批的检测 csv 已经带 own / side / tid / not_hand，所以 decisions 可以直接导出，
不必再跑一遍渲染阶段。decisions 的 tid 必须是 **join 之后**的，和第一批一致。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semhand.join_window_sweep import frags_from_tracks, read_pre_join  # noqa: E402
from rig.track_join import DIST_FACTOR, join_to_fixed_point  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-dir", required=True, help="join 前的检测 csv 目录")
    ap.add_argument("--out-joined", required=True)
    ap.add_argument("--out-decisions", required=True)
    ap.add_argument("--window", type=int, required=True,
                    help="两批必须用同一个值，否则不可比")
    a = ap.parse_args()
    os.makedirs(a.out_joined, exist_ok=True)
    os.makedirs(a.out_decisions, exist_ok=True)

    print("窗口 %d（两批统一）" % a.window)
    print("  %-20s %7s %7s %9s" % ("录像", "join前", "join后", "decisions 行"))
    for p in sorted(glob.glob(os.path.join(a.pre_dir, "*.detect2.csv"))):
        rec = os.path.basename(p)[:-len(".detect2.csv")]
        pre = read_pre_join(p)
        frag = frags_from_tracks(pre)
        mapping, _ = join_to_fixed_point(frag, window=a.window,
                                         dist_factor=DIST_FACTOR)
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        for r in rows:
            r["tid"] = mapping.get(str(r["tid"]), str(r["tid"]))
        with open(os.path.join(a.out_joined, "%s.detect2.csv" % rec), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        # decisions：只留 is_hand 的行，与第一批一致（那边 is_hand 全是 1，
        # 判非手的轨迹整条不进文件，不是标成 0）
        seen = set()
        out = []
        for r in rows:
            if str(r.get("not_hand")) in ("1", "True", "true"):
                continue
            k = (r["frame"], r["tid"])
            if k in seen:
                continue
            seen.add(k)
            out.append({"frame": r["frame"], "tid": r["tid"],
                        "own": r.get("own", ""), "side": r.get("side", ""),
                        "is_hand": "1"})
        dp = os.path.join(a.out_decisions, "%s.decisions.csv" % rec)
        with open(dp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "tid", "own", "side",
                                               "is_hand"])
            w.writeheader()
            w.writerows(out)
        print("  %-20s %7d %7d %9d"
              % (rec, len(frag), len({mapping[t] for t in frag}), len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
