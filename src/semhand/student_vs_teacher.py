"""Is the distilled student as good as the teacher it copied, judged by people?

THE NUMBER IN THE TRAINING LOG DOES NOT ANSWER THIS. `auc_target_hard` scores
the student against the teacher's own hard labels, so it measures imitation.
A student that reproduced every one of the teacher's mistakes would score 1.0
there. The only comparison that decides whether the student can replace the
teacher puts both of them against the human labels, on the same rows.

THE SAME ROWS IS THE HARD PART. The teacher was run over the pool of candidate
negatives, not over the hands, so its coverage of the two classes is wildly
uneven -- on this manifest it has a score for 98.7% of the rows a person called
`nothand` and 2.8% of the rows a person called `hand`. Any head-to-head is
therefore restricted to a subset that was selected for looking like a
non-hand, and the handful of positives in it are the hardest hands in the set,
not typical ones. So this prints three things and keeps them apart:

    student, all human-labelled rows     what the student is actually worth
    both, teacher                        the teacher where it has an opinion
    both, student                        the student on exactly those rows

Only the second and third are comparable to each other, and the counts are
printed next to them because a few dozen rows drawn from two tracks is not a
population.

AUC IS NOT THE DEPLOYMENT NUMBER. What the gate does is refuse boxes, and the
cost of refusing a real hand is that a hand goes unblurred. So the operating
point is quoted the way the decision is made: fix how many real hands may be
lost, and report how many non-hands that buys. Ranking quality shows up as
which curve gives more at the same cost.

ROWS ARE NOT INDEPENDENT. Views come from tracks and tracks from recordings,
so the distinct-track count is printed with every n. Treat it, not the row
count, as the sample size.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json


def auc(labels, scores):
    """Rank AUC with ties averaged. Ties matter here: a saturated student
    hands back long runs of 0.9999, and counting those as wins would invent
    a separation the scores do not have."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    won = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (won - positives * (positives + 1) / 2) / (positives * negatives)


def tracks(rows):
    return len({(r["rec"], r.get("canonical_tid") or r.get("raw_tid") or "")
                for r in rows})


def report(rows, score, name, allow=(0, 1, 3, 5)):
    hands = [score(r) for r in rows if r["human_label"] == "hand"]
    others = [score(r) for r in rows if r["human_label"] == "nothand"]
    if not hands or not others:
        print("  %-24s 某一类为空，测不了" % name)
        return
    labels = [1] * len(hands) + [0] * len(others)
    print("  %-24s n=%d（手 %d / 非手 %d，%d 条轨迹）  AUC %.3f"
          % (name, len(rows), len(hands), len(others), tracks(rows),
             auc(labels, hands + others)))
    ordered = sorted(hands)
    for a in allow:
        if a >= len(ordered):
            continue
        # 阈值卡在第 a+1 低的真手上：低于它的真手被拒，正好 a 只。
        threshold = ordered[a]
        blocked = sum(1 for x in others if x < threshold)
        print("      误糊 %d 只真手 -> 阈值 %.4f，挡掉非手 %3d/%d = %.1f%%"
              % (a, threshold, blocked, len(others), 100 * blocked / len(others)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True,
                    help="spatial_student score 写的 csv，列 item_id,p_hand")
    ap.add_argument("--split", default="test")
    ap.add_argument("--teacher", default="",
                    help="handness_teacher_fill 写的 jsonl，补上 manifest 里缺的老师分。"
                         "给了它就用它，这样两边覆盖同一批行")
    a = ap.parse_args()

    pred = {r["item_id"]: float(r["p_hand"])
            for r in csv.DictReader(open(a.predictions, encoding="utf-8"))}
    manifest = {r["item_id"]: r for r in
                csv.DictReader(open(a.manifest, encoding="utf-8"))}
    rows = [manifest[k] for k in pred
            if k in manifest
            and manifest[k]["split"] == a.split
            and manifest[k].get("human_label") in ("hand", "nothand")]

    filled = {}
    if a.teacher:
        for line in open(a.teacher, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                filled[d["item_id"]] = float(d["p"])
        for r in rows:
            if r["item_id"] in filled:
                r["teacher_p"] = "%.6f" % filled[r["item_id"]]
        print("补入老师分 %d 行\n" % len(filled))

    cover = collections.Counter(
        (r["human_label"], bool(r.get("teacher_p"))) for r in rows)
    print("%s 里带人工标签的 %d 条，%d 条轨迹" % (a.split, len(rows), tracks(rows)))
    for label in ("hand", "nothand"):
        yes, no = cover[(label, True)], cover[(label, False)]
        print("  %-8s %4d 条，其中老师有分 %4d（%.1f%%）"
              % (label, yes + no, yes, 100 * yes / max(1, yes + no)))

    print("\n=== 学生：全部人工标签行 ===")
    report(rows, lambda r: pred[r["item_id"]], "学生（集成）")

    both = [r for r in rows if r.get("teacher_p")]
    print("\n=== 只在老师也有分的行上（这才是能对比的）===")
    report(both, lambda r: float(r["teacher_p"]), "老师")
    report(both, lambda r: pred[r["item_id"]], "学生")

    if both:
        agree = sum(1 for r in both
                    if (float(r["teacher_p"]) >= .5) == (pred[r["item_id"]] >= .5))
        print("\n  两者硬判定一致 %d/%d = %.1f%%"
              % (agree, len(both), 100 * agree / len(both)))


if __name__ == "__main__":
    main()
