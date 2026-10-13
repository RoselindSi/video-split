"""Paired comparison of both arms against the string baseline, on the same 307.

WHY THIS FILE EXISTS. The first reading of E1b said neither arm "clearly beat"
the 65.8% string rule, and reached that by looking at a macro-F1 confidence
interval while the baseline was quoted as an accuracy. Those are different
quantities and one interval says nothing about the other. Three methods score
the same 307 candidates, so the comparison is paired and has to be run as
one: resample recordings, recompute every method on the resampled set, and
take the difference inside each replicate.

THE BASELINE IS A RULE, NOT A MODEL. "Predict SAME_STATE when the two sides'
(object, action) strings are byte-identical." It reads E1a's per-occurrence
text, which is the same text the selftext arm reads, so selftext-minus-
baseline isolates what the model adds over string equality on identical
input.

Both deltas are reported for both metrics. Reporting only the one that looks
better is how the first version of this went wrong.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

BOOT, SEED = 2000, 20260928
STATES = ["SAME_STATE", "DIFFERENT_STATE", "UNCERTAIN"]


def macro_f1(rows, pred):
    present = [l for l in STATES if any(r["gold_state"] == l for r in rows)]
    fs = []
    for l in present:
        tp = sum(1 for r in rows if r["gold_state"] == l and pred(r) == l)
        fp = sum(1 for r in rows if r["gold_state"] != l and pred(r) == l)
        fn = sum(1 for r in rows if r["gold_state"] == l and pred(r) != l)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(fs) / len(fs) if fs else 0.0


def acc(rows, pred):
    return sum(1 for r in rows if pred(r) == r["gold_state"]) / len(rows) if rows else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--selftext", required=True)
    ap.add_argument("--e1a", required=True)
    a = ap.parse_args()

    cands = {c["cand_id"]: c for c in
             json.load(open(a.candidates, encoding="utf-8"))["candidates"]}
    seg = {}
    for line in open(a.e1a, encoding="utf-8"):
        d = json.loads(line)
        if d.get("parsed"):
            seg[d["segment_id"]] = d["parsed"]

    def load(p, tag):
        out = {}
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            out[d["cand_id"]] = (d.get("parsed") or {}).get("state_relation")
        return out

    pv, pt = load(a.video, "video"), load(a.selftext, "selftext")

    rows = []
    for cid, c in cands.items():
        ka = "%s#%02d" % (c["video_id"], c["prev_seg_index"])
        kb = "%s#%02d" % (c["video_id"], c["next_seg_index"])
        pa, pb = seg.get(ka), seg.get(kb)
        if not pa or not pb or cid not in pv or cid not in pt:
            continue
        ident = ((str(pa.get("object")).strip(), str(pa.get("observed_action")).strip())
                 == (str(pb.get("object")).strip(), str(pb.get("observed_action")).strip()))
        rows.append({"cand_id": cid, "video_id": c["video_id"],
                     "gold_state": c["gold_state"],
                     "p_string": "SAME_STATE" if ident else "DIFFERENT_STATE",
                     "p_video": pv[cid], "p_text": pt[cid]})
    print("配对样本 %d 个 / %d 条视频\n" % (rows and len(rows), len({r["video_id"] for r in rows})))

    METH = [("字符串基线", lambda r: r["p_string"]),
            ("video", lambda r: r["p_video"]),
            ("selftext", lambda r: r["p_text"])]
    print("%-12s %8s %10s" % ("方法", "准确率", "宏平均F1"))
    for name, f in METH:
        print("%-12s %7.1f%% %10.3f" % (name, 100 * acc(rows, f), macro_f1(rows, f)))

    units = collections.defaultdict(list)
    for r in rows:
        units[r["video_id"]].append(r)
    keys = sorted(units)
    rng = random.Random(SEED)
    reps = []
    for _ in range(BOOT):
        pick = [units[rng.choice(keys)] for _ in keys]
        rr = [r for g in pick for r in g]
        reps.append({name: (acc(rr, f), macro_f1(rr, f)) for name, f in METH})

    def ci(name, base, idx):
        d = sorted(r[name][idx] - r[base][idx] for r in reps)
        lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
        point = (dict(METH) and 0)
        return lo, hi

    print("\n按录像聚类的配对 bootstrap（%d 次重采样，49 条录像）" % BOOT)
    print("%-26s %10s %22s" % ("对比", "Δ点估计", "95% CI"))
    for name in ("video", "selftext"):
        for idx, label in ((0, "Δ准确率"), (1, "Δ宏F1")):
            obs = ([acc(rows, dict(METH)[name]), macro_f1(rows, dict(METH)[name])][idx]
                   - [acc(rows, dict(METH)["字符串基线"]),
                      macro_f1(rows, dict(METH)["字符串基线"])][idx])
            lo, hi = ci(name, "字符串基线", idx)
            sig = "" if lo < 0 < hi else "   ← CI 不跨 0"
            fmt = "%+6.1f%%" if idx == 0 else "%+6.3f"
            cifmt = "[%+.1f%%, %+.1f%%]" if idx == 0 else "[%+.3f, %+.3f]"
            s = (100, 100) if idx == 0 else (1, 1)
            print("%-26s %10s %22s%s"
                  % ("%s − 基线  %s" % (name, label),
                     fmt % (obs * s[0]), cifmt % (lo * s[0], hi * s[1]), sig))

    # video 对 selftext 也配对一次：两条臂用的是同一批候选
    print()
    for idx, label in ((0, "Δ准确率"), (1, "Δ宏F1")):
        d = sorted(r["video"][idx] - r["selftext"][idx] for r in reps)
        lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
        obs = ([acc(rows, dict(METH)["video"]), macro_f1(rows, dict(METH)["video"])][idx]
               - [acc(rows, dict(METH)["selftext"]),
                  macro_f1(rows, dict(METH)["selftext"])][idx])
        sig = "" if lo < 0 < hi else "   ← CI 不跨 0"
        s = 100 if idx == 0 else 1
        f = "%+6.1f%%" if idx == 0 else "%+6.3f"
        c = "[%+.1f%%, %+.1f%%]" if idx == 0 else "[%+.3f, %+.3f]"
        print("%-26s %10s %22s%s" % ("video − selftext  %s" % label,
                                     f % (obs * s), c % (lo * s, hi * s), sig))


if __name__ == "__main__":
    main()
