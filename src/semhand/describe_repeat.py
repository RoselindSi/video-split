"""Run the describe pass more than once on the same views and see if it agrees with itself.

WHY THIS IS THE QUESTION NOW. Swapping the scoring engine moved seven tracks
from owner to other, and reading the descriptions showed why: for the same
frames the two engines wrote contradictory facts -- left hand against right
hand, toward the table centre against toward the upper left, releasing a wire
against holding nothing, resting against holding a pink tool. The scoring pass
is already known to be equivalent when both sides are handed the same
description, so the divergence is entirely in generation.

WHICH MEANS "WHICH ENGINE IS RIGHT" MAY BE THE WRONG QUESTION. Greedy decoding
of a hundred and sixty tokens is chaotic: wherever two tokens are near-tied,
an arithmetic difference far below any tolerance decides the word, and the
rest of the sentence follows from that word. Continuous batching makes the
arithmetic depend on which other requests happen to share the batch, so the
same server can disagree with itself between runs. If it does, then the stored
descriptions were never a fixed reference and the gate has a variance nobody
has measured -- a larger finding than anything about vLLM, and one that has to
be settled before any engine is called wrong.

WHAT IS COMPARED. Descriptions are compared as the formatted one-line strings
the second pass actually consumes, because that is the only part the verdict
can see; two raw replies that format identically are the same input. Per view
it reports whether every run produced the same line, and per field which ones
moved -- hand_side flipping is a different failure from motion_path being
reworded, and only the fields are diagnostic.

INSTABILITY AND ITS COST ARE REPORTED SEPARATELY, and the first is printed
whether or not --score is given, because a stable verdict does not make an
unstable input acceptable -- it means this sample did not happen to catch the
cost. With --score, each run's description is put through the scoring pass and
the spread of P(wearer) is reported per view, against the gate's own
thresholds rather than against 0.5, since 0.60/0.40 is where the decision
actually changes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import time

FIELDS = ("hand_side", "motion_path", "interaction", "object", "occlusion",
          "ambiguous_regions")


def split_fields(line):
    """把格式化好的那一行拆回字段，用来分辨是哪一处变了。"""
    out = {}
    head, _, rest = line.partition(";")
    out["hand_side"] = head.replace("hand", "").strip()
    for chunk in rest.split(";"):
        key, _, value = chunk.partition(":")
        key = key.strip().replace("motion path", "motion_path").replace(
            "ambiguous", "ambiguous_regions")
        if key in FIELDS:
            out[key] = value.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18997")
    ap.add_argument("--model", default="qwen-bf16")
    ap.add_argument("--views", required=True)
    ap.add_argument("--ids", default="",
                    help="只跑这些 id 的文件，一行一个；留空按 --n 取前若干条")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--score", action="store_true",
                    help="把每一遍的描述都过一次打分，测这点不稳定值多少")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from src.semhand.owner_gate import CONTEXT_DESCRIBE
    from src.semhand.vllm_probe import describe, format_desc, run_phase

    items = list(csv.DictReader(open(os.path.join(a.views, "manifest.csv"),
                                     encoding="utf-8")))
    if a.ids:
        want = {line.strip() for line in open(a.ids) if line.strip()}
        items = [i for i in items if i["id"] in want]
    else:
        items = items[:a.n]
    print("%d 个视图 x %d 遍（同一个服务、同样的提示、temperature 0）"
          % (len(items), a.repeat))

    runs = []
    for r in range(a.repeat):
        t0 = time.time()
        raws = run_phase(items, lambda it: describe(
            a.url, a.model, it.get("context") or it["full"], it["crop"],
            CONTEXT_DESCRIBE), a.conc)
        runs.append([format_desc(x)[0] for x in raws])
        print("  第 %d 遍 %.1fs" % (r + 1, time.time() - t0))

    same = 0
    moved = collections.Counter()
    unstable = []
    for idx, it in enumerate(items):
        lines = [run[idx] for run in runs]
        if len(set(lines)) == 1:
            same += 1
            continue
        unstable.append((it["id"], lines))
        parsed = [split_fields(x) for x in lines]
        for f in FIELDS:
            if len({p.get(f, "") for p in parsed}) > 1:
                moved[f] += 1

    print("\n=== %d 个视图，%d 遍 ===" % (len(items), a.repeat))
    print("  每遍完全一致的 %d/%d = %.1f%%"
          % (same, len(items), 100 * same / len(items)))
    print("  有变化的 %d 个" % len(unstable))
    if moved:
        print("\n  变的是哪个字段（一个视图可能多处变）：")
        for f, n in moved.most_common():
            print("    %-18s %d 个视图 (%.1f%%)" % (f, n, 100 * n / len(items)))
        if moved.get("hand_side"):
            print("\n  注意 hand_side 变了 %d 个 —— 左右手是事实不是措辞，"
                  "同一张图给出两个答案说明这一步不是在读图" % moved["hand_side"])

    for vid, lines in unstable[:3]:
        print("\n  %s" % vid)
        for i, line in enumerate(lines):
            print("    第%d遍 %s" % (i + 1, line))

    scores = None
    if a.score:
        from src.semhand.owner_gate import CONTEXT_INTRO, CONTEXT_QUESTION
        from src.semhand.vllm_probe import ask, two_way
        ADMIT, REJECT = 0.60, 0.40
        scores = []
        for r, run in enumerate(runs):
            t0 = time.time()
            resps = run_phase(list(zip(items, run)), lambda pair: ask(
                a.url, a.model, pair[0].get("context") or pair[0]["full"],
                pair[0]["crop"], CONTEXT_INTRO, CONTEXT_QUESTION,
                description=pair[1]), a.conc)
            scores.append([two_way(x)[0] for x in resps])
            print("  打分第 %d 遍 %.1fs" % (r + 1, time.time() - t0))

        def band(p):
            return ("admit" if p >= ADMIT else
                    "reject" if p <= REJECT else "contested")

        spreads, flipped, banded = [], 0, 0
        for idx in range(len(items)):
            ps = [s[idx] for s in scores if s[idx] is not None]
            if len(ps) < 2:
                continue
            spreads.append(max(ps) - min(ps))
            if len({p >= .5 for p in ps}) > 1:
                flipped += 1
            if len({band(p) for p in ps}) > 1:
                banded += 1
        spreads.sort()
        print("\n=== 这点不稳定值多少（同样 %d 个视图，%d 遍打分）==="
              % (len(spreads), a.repeat))
        print("  P(wearer) 跨遍极差：中位 %.4f  90%% 分位 %.4f  最大 %.4f"
              % (spreads[len(spreads) // 2], spreads[int(.9 * len(spreads))],
                 spreads[-1]))
        print("  跨过 0.5 的 %d/%d = %.1f%%"
              % (flipped, len(spreads), 100 * flipped / len(spreads)))
        print("  换了闸门档位（admit>=%.2f / reject<=%.2f / contested）的 %d/%d = %.1f%%"
              % (ADMIT, REJECT, banded, len(spreads), 100 * banded / len(spreads)))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for idx, it in enumerate(items):
                fh.write(json.dumps({
                    "id": it["id"],
                    "lines": [run[idx] for run in runs],
                    "stable": len({run[idx] for run in runs}) == 1,
                    "p": [s[idx] for s in scores] if scores else None,
                }) + "\n")
        print("\n  逐条 -> %s" % a.out)


if __name__ == "__main__":
    main()
