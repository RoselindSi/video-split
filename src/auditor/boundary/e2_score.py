"""E2's four planned contrasts, paired on candidate and clustered on recording.

THE FOUR CONTRASTS WERE CHOSEN BEFORE THE RUN, and they are the report:

    M1canon - M1raw     normalising the wording, on its own
    M2raw   - M1raw     does an explicit id survive noisy language
    M2canon - M1canon   what the id adds once wording is controlled
    M3      - M2canon   what the edges add once identity is stable

Reporting "which condition scored highest" instead would answer a question
nobody asked, and would let a difference in prompt length pass for a
difference in structure.

STRATIFIED BY HOW MANY NODES THE HISTORY HAS ALREADY SHOWN. The median
prefix has seen exactly one node, and where only one node exists the node
dictionary is one line and the transition list is a self-loop: M2 and M3
carry almost no information M1 does not. Pooling would let that half of the
candidates flatten any real structural effect, so every contrast is reported
for `1 node` and `>=2 nodes` separately as well as overall.

PAIRED, BECAUSE ALL CONDITIONS SCORE THE SAME 307. The bootstrap resamples
recordings once per replicate and recomputes every condition on that same
resample, so the difference is taken within the replicate.

PROMPT LENGTH IS REPORTED BESIDE EVERY CONDITION. A condition that helps only
by being longer is a finding about length.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

BOOT, SEED = 2000, 20260928
STATES = ["SAME_STATE", "DIFFERENT_STATE", "UNCERTAIN"]
# `M2raw` shipped a canonical node name in its dictionary, so `M2raw - M1raw`
# mixed explicit identity with an oracle label. `M2rawd` carries the node's own
# occurrence wordings instead, and is the clean identity contrast. The original
# pair stays in the table so the correction is visible rather than quietly
# swapped in.
CONTRASTS = [("M1canon", "M1raw", "措辞规范化的作用"),
             ("M2raw", "M1raw", "旧口径：身份 + 规范名（混淆，见下）"),
             ("M2rawd", "M1raw", "干净的身份效应（字典也是原始措辞）"),
             ("M2canon", "M1canon", "控制措辞后，身份还值多少"),
             ("M3", "M2canon", "旧口径：canonical 之上的边"),
             ("M3raw", "M2rawd", "最强表示之上，边的净贡献"),
             ("M3raw_noloop", "M3raw", "去掉平凡自环"),
             # 长度与结构在 M0..M3raw 上完全共线，所以这两条是解释前面一切的前提
             ("M3raw", "M3raw_shuffled", "结构效应（长度/格式已对齐）"),
             ("M3raw_shuffled", "M0", "上下文长度本身的代价"),
             ("M3raw_other", "M3raw_shuffled", "无关历史 vs 打乱历史"),
             # E2 里唯一的正面结果挂在 M2raw 的字典那一行上，先查实它
             ("M2raw", "M2lab_perm", "规范名的语义（字数逐字相同）"),
             ("M2raw", "M2lab_opaque", "语义 vs 只是多一行"),
             ("M2lab_first", "M2rawd", "一行短锚点 vs 多行原始措辞"),
             ("M2raw", "M2lab_first", "oracle 词汇 vs 自己的第一次措辞")]


def macro_f1(rows, cond):
    present = [l for l in STATES if any(r["gold_state"] == l for r in rows)]
    fs = []
    for l in present:
        tp = sum(1 for r in rows if r["gold_state"] == l and r["pred"][cond] == l)
        fp = sum(1 for r in rows if r["gold_state"] != l and r["pred"][cond] == l)
        fn = sum(1 for r in rows if r["gold_state"] == l and r["pred"][cond] != l)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(fs) / len(fs) if fs else 0.0


def acc(rows, cond):
    return (sum(1 for r in rows if r["pred"][cond] == r["gold_state"]) / len(rows)
            if rows else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/workspace")
    ap.add_argument("--m0", help="E1b-video，作为 M0")
    a = ap.parse_args()

    conds = ["M1raw", "M1canon", "M2raw", "M2canon", "M3",
             "M2rawd", "M3raw", "M3raw_noloop",
             "M3raw_shuffled", "M3raw_other",
             "M2lab_perm", "M2lab_opaque", "M2lab_first"]
    data = collections.defaultdict(dict)
    meta = {}
    plen = collections.defaultdict(list)
    for c in conds:
        p = os.path.join(a.dir, "e2_%s.jsonl" % c)
        if not os.path.exists(p):
            print("缺 %s" % p)
            continue
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            data[d["cand_id"]][c] = (d.get("parsed") or {}).get("state_relation")
            plen[c].append(d.get("prompt_chars", 0))
            meta[d["cand_id"]] = d
    if a.m0 and os.path.exists(a.m0):
        conds = ["M0"] + conds
        for line in open(a.m0, encoding="utf-8"):
            d = json.loads(line)
            data[d["cand_id"]]["M0"] = (d.get("parsed") or {}).get("state_relation")
            plen["M0"].append(0)
            meta.setdefault(d["cand_id"], d)

    rows = [{"cand_id": k, "pred": v, **{f: meta[k][f] for f in
             ("video_id", "gold_state", "n_nodes_seen", "n_history")
             if f in meta[k]}}
            for k, v in data.items() if all(c in v for c in conds)]
    print("配对样本 %d 个 / %d 条视频，条件 %s\n"
          % (len(rows), len({r["video_id"] for r in rows}), conds))

    STRATA = [("全部", lambda r: True),
              ("历史只见过 1 个节点", lambda r: r.get("n_nodes_seen", 1) == 1),
              ("历史见过 >=2 个节点", lambda r: r.get("n_nodes_seen", 1) >= 2)]

    for label, sel in STRATA:
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        print("=== %s（n=%d / %d 条视频）===" % (label, len(sub), len({r["video_id"] for r in sub})))
        print("%-10s %8s %9s %11s" % ("条件", "准确率", "宏F1", "提示词字数中位"))
        for c in conds:
            pl = sorted(plen[c]) if plen[c] else [0]
            print("%-10s %7.1f%% %9.3f %11d"
                  % (c, 100 * acc(sub, c), macro_f1(sub, c), pl[len(pl) // 2]))

        units = collections.defaultdict(list)
        for r in sub:
            units[r["video_id"]].append(r)
        keys = sorted(units)
        if len(keys) < 2:
            print()
            continue
        rng = random.Random(SEED)
        reps = []
        for _ in range(BOOT):
            pick = [units[rng.choice(keys)] for _ in keys]
            rr = [r for g in pick for r in g]
            reps.append({c: (acc(rr, c), macro_f1(rr, c)) for c in conds})
        print("  计划内对比（配对 bootstrap，按录像聚类）")
        for hi, lo, why in CONTRASTS:
            if hi not in conds or lo not in conds:
                continue
            for idx, nm in ((0, "Δ准确率"), (1, "Δ宏F1")):
                obs = ([acc(sub, hi), macro_f1(sub, hi)][idx]
                       - [acc(sub, lo), macro_f1(sub, lo)][idx])
                d = sorted(r[hi][idx] - r[lo][idx] for r in reps)
                l, h = d[int(.025 * len(d))], d[int(.975 * len(d))]
                sig = "  ← CI 不跨 0" if not (l < 0 < h) else ""
                s = 100 if idx == 0 else 1
                f = "%+6.1f%%" if idx == 0 else "%+6.3f"
                cf = "[%+.1f%%, %+.1f%%]" if idx == 0 else "[%+.3f, %+.3f]"
                head = "%s − %s %s" % (hi, lo, nm) if idx == 0 else ""
                print("    %-34s %9s %22s%s"
                      % (head, f % (obs * s), cf % (l * s, h * s), sig))
            print("        （%s）" % why)
        print()

    print("提醒：M1canon / M2* / M3 的规范化与节点身份都来自人工 gold，")
    print("      是 oracle 控制的表示消融，不是可部署基线。E4 才换成预测的状态跟踪。")


if __name__ == "__main__":
    main()
