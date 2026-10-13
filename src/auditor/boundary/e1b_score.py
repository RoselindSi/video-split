"""E1b's scores, with the baseline that any arm has to beat printed beside them.

MACRO-F1 FIRST, ACCURACY NEVER ALONE. The state arm is 174/133, so accuracy
carries a 56.7% floor for free; the task arm is 302/3/2, where accuracy is
98.4% for answering one word. Both would be read as competence. So state is
reported as macro-F1 with per-class recall and a confusion matrix, and task
is reported as per-class recall with its n printed -- never as a single
number.

THE STRING BASELINE IS NOT DECORATION. "Are the two sides' descriptions
byte-identical" scores 65.8% on E1a's per-occurrence text, above the 56.7%
majority floor. An arm that does not clear 65.8% has not shown it is reading
anything the string comparison was not already reading.

CLUSTERED ON VIDEO, because 307 candidates come from 49 recordings and
boundaries inside one recording share a scene, an operator and a camera.

STRATIFIED BY `window_clean` AND BY FRAME COUNT, because the window rule had
to put a 3 s floor under 26% of candidates, and because whether accuracy
tracks frame count is the evidence that would justify changing the frame
rate -- as opposed to changing it and hoping.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

STATES = ["SAME_STATE", "DIFFERENT_STATE", "UNCERTAIN"]
TASKS = ["SAME_TASK", "RETURN_TASK", "DIFFERENT_TASK", "UNCERTAIN"]
BOOT, SEED = 2000, 20260927


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        p = d.get("parsed") or {}
        d["pred_state"] = p.get("state_relation")
        d["pred_task"] = p.get("task_relation")
        out.append(d)
    return out


def macro_f1(rows, gold_key, pred_key, labels):
    """Macro over the labels that actually occur in gold."""
    present = [l for l in labels if any(r[gold_key] == l for r in rows)]
    fs = []
    for l in present:
        tp = sum(1 for r in rows if r[gold_key] == l and r[pred_key] == l)
        fp = sum(1 for r in rows if r[gold_key] != l and r[pred_key] == l)
        fn = sum(1 for r in rows if r[gold_key] == l and r[pred_key] != l)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(fs) / len(fs) if fs else None


def cluster_ci(rows, stat, key="video_id"):
    units = collections.defaultdict(list)
    for r in rows:
        units[r.get(key, r["cand_id"])].append(r)
    keys = sorted(units)
    if len(keys) < 2:
        return None
    rng = random.Random(SEED)
    vals = []
    for _ in range(BOOT):
        pick = [units[rng.choice(keys)] for _ in keys]
        v = stat([r for g in pick for r in g])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def report(name, rows):
    n = len(rows)
    bad = sum(1 for r in rows if not r.get("parsed"))
    print("\n=== %s（%d 个候选 / %d 条视频，解析失败 %d）==="
          % (name, n, len({r["video_id"] for r in rows}), bad))

    acc = sum(1 for r in rows if r["pred_state"] == r["gold_state"]) / n
    f1 = macro_f1(rows, "gold_state", "pred_state", STATES)
    ci = cluster_ci(rows, lambda rr: macro_f1(rr, "gold_state", "pred_state", STATES))
    print("state  宏平均 F1 %.3f%s   准确率 %.1f%%（多数类 56.7%%，字符串基线 65.8%%）"
          % (f1, "  95%% CI [%.3f, %.3f]" % ci if ci else "", 100 * acc))
    for l in ("SAME_STATE", "DIFFERENT_STATE"):
        g = [r for r in rows if r["gold_state"] == l]
        hit = sum(1 for r in g if r["pred_state"] == l)
        print("       %-16s recall %5.1f%%  (%d/%d)" % (l, 100 * hit / len(g), hit, len(g)))
    print("       混淆矩阵 gold\\pred:")
    preds = sorted({r["pred_state"] for r in rows} | set(STATES[:2]),
                   key=lambda x: (x is None, str(x)))
    print("         %-17s %s" % ("", "  ".join("%-15s" % str(p) for p in preds)))
    for l in ("SAME_STATE", "DIFFERENT_STATE"):
        c = collections.Counter(r["pred_state"] for r in rows if r["gold_state"] == l)
        print("         %-17s %s" % (l, "  ".join("%-15d" % c.get(p, 0) for p in preds)))

    print("  分层：")
    for tag, sel in (("窗口干净", lambda r: r["window_clean"]),
                     ("窗口越界", lambda r: not r["window_clean"])):
        g = [r for r in rows if sel(r)]
        if g:
            a = sum(1 for r in g if r["pred_state"] == r["gold_state"]) / len(g)
            print("    %-10s n=%3d  state 准确率 %.1f%%" % (tag, len(g), 100 * a))
    for lo, hi in ((6, 8), (9, 12), (13, 16)):
        g = [r for r in rows if lo <= r["n_frames_1fps"] <= hi]
        if g:
            a = sum(1 for r in g if r["pred_state"] == r["gold_state"]) / len(g)
            print("    %2d-%2d 帧    n=%3d  state 准确率 %.1f%%" % (lo, hi, len(g), 100 * a))

    print("  task（探针，不报总体准确率——302/307 是 SAME_TASK）：")
    for l in TASKS:
        g = [r for r in rows if r["gold_task"] == l]
        if not g:
            continue
        hit = sum(1 for r in g if r["pred_task"] == l)
        print("    %-16s recall %5.1f%%  (n=%d)%s"
              % (l, 100 * hit / len(g), len(g),
                 "   ← 样本太少，只作定性" if len(g) <= 5 else ""))
    c = collections.Counter(r["pred_task"] for r in rows)
    print("    模型的 task 回答分布：%s" % dict(c))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--selftext", required=True)
    a = ap.parse_args()
    v, t = load(a.video), load(a.selftext)
    report("E1b-video", v)
    report("E1b-selftext", t)

    # 同一批候选上的配对比较：两条臂用的是同一个 307
    common = {r["cand_id"] for r in v} & {r["cand_id"] for r in t}
    vd = {r["cand_id"]: r for r in v}
    td = {r["cand_id"]: r for r in t}
    both = sum(1 for c in common
               if vd[c]["pred_state"] == vd[c]["gold_state"]
               and td[c]["pred_state"] == td[c]["gold_state"])
    vonly = sum(1 for c in common
                if vd[c]["pred_state"] == vd[c]["gold_state"]
                and td[c]["pred_state"] != td[c]["gold_state"])
    tonly = sum(1 for c in common
                if vd[c]["pred_state"] != vd[c]["gold_state"]
                and td[c]["pred_state"] == td[c]["gold_state"])
    neither = len(common) - both - vonly - tonly
    print("\n=== 配对比较（同一批 %d 个候选）===" % len(common))
    print("  两条都对 %d   只有 video 对 %d   只有 selftext 对 %d   都错 %d"
          % (both, vonly, tonly, neither))
    print("  video 独有的优势 = %d - %d = %+d" % (vonly, tonly, vonly - tonly))
    print("  若这个差明显为正，说明把感知与推理拆成两步会丢信息——")
    print("  而 E2 的 M1/M2/M3 全是文本表示，这是对 E2 设计的直接预警。")


if __name__ == "__main__":
    main()
