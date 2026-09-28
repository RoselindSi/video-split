"""Turn the owner teacher's per-view scores into a manifest a student can train on.

WHAT THE TEACHER PRODUCES is one `P(wearer)` per view, from a four-panel
temporal context with the tracked hand boxed plus a crop of the current frame.
The student is given the same two pictures, so the context image arrives as
`full_path`; it already carries the green box, which is why this head needs no
separate box channel the way the handness head did.

SPLIT BY RECORDING, NEVER BY VIEW. Five views of one track differ by a second
or two of the same hand, and a track shares a scene, a person and a lighting
with every other track in its recording. Splitting anywhere below the
recording would let the student memorise the room and read back a number that
says nothing about a new one.

THE DETECTOR'S VERDICT RIDES ALONG AND IS NOT THE LABEL. `det_own` is what the
pipeline already believed, and the teacher exists to correct it; keeping both
is what makes it possible to ask later whether the student learned the teacher
or merely reproduced the detector. It is also the only way to notice the
failure that matters here -- a student that agrees with the detector
everywhere has learned nothing, however good its accuracy looks, because the
detector's own recall on other people's hands is 70.5%.

`human_label` IS LEFT EMPTY ON PURPOSE. `target_of` falls through to
`teacher_p` when it is, which is what distillation wants; the column stays in
the schema so human verdicts can be merged in later without changing anything.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True,
                    help="owner_gate 打分输出的 jsonl（可给多个分片）")
    ap.add_argument("--arm", default="/workspace/crowd10_base",
                    help="检测器逐帧 csv 的目录，用来取 det_own 先验")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    records = {}
    for pattern in a.scores:
        for path in glob.glob(pattern):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    records[r["id"]] = r
    print("打分视图 %d 个" % len(records))
    if not records:
        raise SystemExit("没有读到任何打分记录")

    # 检测器先验：轨迹级多数票，和送进教师时用的是同一个口径
    det = {}
    for path in glob.glob(os.path.join(a.arm, "*.csv")):
        base = os.path.basename(path)
        if base.endswith(".faces.csv") or base.endswith(".decisions.csv"):
            continue
        rec = base[:-4]
        per = collections.defaultdict(collections.Counter)
        for row in csv.DictReader(open(path, encoding="utf-8")):
            try:
                tid = str(int(float(row["tid"])))
            except (KeyError, ValueError, TypeError):
                continue
            if str(row.get("not_hand", "0")) == "1":
                continue
            per[tid][str(row.get("own"))] += 1
        for tid, c in per.items():
            det[(rec, tid)] = ("owner" if c.get("1", 0) * 2 >= sum(c.values())
                               else "other")

    tracks = sorted({(r["rec"], str(r["tid"])) for r in records.values()})
    recs = sorted({rec for rec, _ in tracks})
    import random
    rng = random.Random(a.seed)
    rng.shuffle(recs)
    n_val = max(1, round(a.val_frac * len(recs)))
    n_test = max(1, round(a.test_frac * len(recs)))
    split_of = {}
    for i, rec in enumerate(recs):
        split_of[rec] = ("val" if i < n_val
                         else "test" if i < n_val + n_test else "train")

    fields = ["item_id", "rec", "canonical_tid", "frame", "crop_path",
              "full_path", "context_path", "teacher_p", "human_label",
              "split", "det_own", "base_p", "description_ok"]
    out_rows = []
    for r in records.values():
        rec, tid = r["rec"], str(r["tid"])
        out_rows.append({
            "item_id": r["id"], "rec": rec, "canonical_tid": tid,
            "frame": r.get("frame", ""),
            "crop_path": r["crop"],
            # 学生看到的第二张图就是老师看到的那张四格上下文，框已经画在里面
            "full_path": r.get("context") or r.get("full"),
            "context_path": r.get("context", ""),
            "teacher_p": "%.6f" % float(r["p"]),
            "human_label": "",
            "split": split_of[rec],
            "det_own": det.get((rec, tid), ""),
            "base_p": r.get("base_p", ""),
            "description_ok": r.get("description_ok", ""),
        })

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print("\n%-8s %7s %7s %10s %10s" % ("split", "录像", "轨迹", "视图", "det_other"))
    for s in ("train", "val", "test"):
        g = [r for r in out_rows if r["split"] == s]
        t = {(r["rec"], r["canonical_tid"]) for r in g}
        print("%-8s %7d %7d %10d %10d"
              % (s, len({r["rec"] for r in g}), len(t), len(g),
                 len({k for k in t if det.get(k) == "other"})))

    ps = sorted(float(r["teacher_p"]) for r in out_rows)
    print("\n教师 P(wearer) 分布：中位 %.3f，<0.4 的 %d，>0.6 的 %d，中间 %d"
          % (ps[len(ps) // 2], sum(1 for p in ps if p < 0.4),
             sum(1 for p in ps if p > 0.6),
             sum(1 for p in ps if 0.4 <= p <= 0.6)))

    # 教师与检测器先验的关系：全一致说明教师什么也没纠正
    per_track = collections.defaultdict(list)
    for r in out_rows:
        per_track[(r["rec"], r["canonical_tid"])].append(float(r["teacher_p"]))
    tab = collections.Counter()
    for k, vs in per_track.items():
        vs.sort()
        med = vs[len(vs) // 2]
        tab[(det.get(k, "?"), "owner" if med >= 0.5 else "other")] += 1
    print("\n检测器先验 × 教师裁决（轨迹级，教师取中位数）")
    print("%-10s %10s %10s" % ("det \\ 教师", "owner", "other"))
    for d in ("owner", "other"):
        print("%-10s %10d %10d" % (d, tab[(d, "owner")], tab[(d, "other")]))
    flips = tab[("owner", "other")] + tab[("other", "owner")]
    print("教师改判了 %d / %d 条轨迹 —— 这个数为 0 就说明教师没带来新信息"
          % (flips, sum(tab.values())))
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
