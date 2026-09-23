"""What a raw-domain training pool could be made of, counted before anything is built.

THE QUESTION THIS ANSWERS. The next student must see the same pixels in
training, validation and deployment, and deployment is a 1920x1520 raw camera
half. So: how many OWNER, OTHER and NOT_HAND examples exist in that domain
today, who labelled them, and which recordings are they in.

THE ANSWER IS NOT "NONE, START OVER", AND THAT IS THE POINT OF COUNTING.
Ownership labels in the raw domain already exist and were not recognised as
such: the labelling line's images are the native frame scaled by exactly 1/3
with two rows of letterbox, and the frame index is recoverable -- checked
against the stored images at correlation 0.93 to 0.98 on six recordings. So a
box at (x, y) there is a box at (3x, 3(y-2)) in the raw frame, and 89,367
hands with ownership become raw-domain rows without anyone labelling anything
again.

AND THERE IS A SECOND CONFOUND WAITING, WHICH IS WHY THIS FILE EXISTS RATHER
THAN A ONE-LINE COUNT. Those boxes were drawn by GPT-6; the NOT_HAND boxes
were drawn by our detector. Box provenance can carry a class the same way
image provenance just did -- a tighter or differently-shaped box is as
learnable as a different projection. The fix is to keep OUR detector's box and
GPT-6's label, matched by overlap at native resolution, so all three classes
arrive through one detector. `det_recall` already measured how often that
match exists: 96.0% for the wearer's hands and 70.5% for everyone else's.

That asymmetry is itself a cost to write down. Taking only matched boxes keeps
96% of the owner labels and 70% of the other ones, so the pool's class balance
moves, and the boxes it loses are the small distant ones -- the population
that matters most.
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
    ap.add_argument("--snapshot",
                    default="/shared/ownership_labels/direct_hand_set_20260922_v1"
                            "/snapshot_v1/train.jsonl")
    ap.add_argument("--labels",
                    default="/shared/ownership_labels/boxes_gpt6_qwen50958_v1/labels.jsonl")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    a = ap.parse_args()

    evals = set()
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            evals.add("databag-" + f[0][1:] if not f[0].startswith("databag") else f[0])
    print("cam3 评测录像 %d 条" % len(evals))

    src = {}
    for line in open(a.labels):
        d = json.loads(line)
        if d.get("status") == "complete":
            src[os.path.basename(d["image"])] = d
    roots = glob.glob("/shared/datasets/.incoming/video_1520p_raw/*/raw_databag")
    have_video = {}

    def video_ok(dev, rec):
        k = (dev, rec)
        if k not in have_video:
            have_video[k] = any(os.path.isdir(os.path.join(r, dev, rec))
                                for r in roots)
        return have_video[k]

    own = oth = 0
    by_rec = collections.Counter()
    by_day = collections.Counter()
    no_video = 0
    overlap = 0
    for line in open(a.snapshot):
        d = json.loads(line)
        if d.get("label_basis") != "DS weak supervision":
            continue
        s = src.get(os.path.basename(d["image"]))
        if not s:
            continue
        if not video_ok(s["device"], s["recording"]):
            no_video += 1
            continue
        if s["recording"] in evals:
            overlap += 1
            continue
        n_o = sum(1 for h in d["hands"]
                  if h.get("status") == "hand" and h.get("ownership") == "owner")
        n_t = sum(1 for h in d["hands"]
                  if h.get("status") == "hand" and h.get("ownership") == "other")
        own += n_o
        oth += n_t
        by_rec[s["recording"]] += n_o + n_t
        # `databag-26_0826_130452` -> `26_0826`. Slicing the first ten
        # characters gave "databag-26" for every row and reported one date.
        by_day[s["recording"].replace("databag-", "")[:7]] += n_o + n_t

    print()
    print("== 原始域 OWNER / OTHER：来自标注线的图，可映射回原生帧 ==")
    print("   OWNER %d   OTHER %d   合计 %d" % (own, oth, own + oth))
    print("   分布在 %d 条录像、%d 个日期上" % (len(by_rec), len(by_day)))
    print("   与 cam3 评测录像重叠的图：%d（已剔除）" % overlap)
    print("   原始视频不在盘上的图：%d" % no_video)
    print("   最集中的 3 条录像占 %.1f%%"
          % (100.0 * sum(n for _, n in by_rec.most_common(3)) / max(1, own + oth)))

    print()
    print("== 原始域 NOT_HAND：探针产的，已在原生帧上 ==")
    for name, p in (("第一批 100 条录像", "/workspace/neg_probe.jsonl"),
                    ("第二批 400 条录像", "/workspace/neg2_probe.jsonl")):
        # DEDUPED BY STEM. The glob matches both the merged file and the
        # shards that were concatenated into it, which counted every box
        # twice -- the same mistake, in a third place.
        seen = {}
        for q in sorted(glob.glob(p + "*")):
            for line in open(q):
                d = json.loads(line)
                seen[d["stem"]] = d["p"]
        n = sum(1 for v in seen.values() if v <= 0.10)
        print("   %-18s %d（共打分 %d）" % (name, n, len(seen)))
    idx = "/workspace/nothand_bank/pkg/index_nothand.csv"
    if os.path.exists(idx):
        rows = list(csv.DictReader(open(idx)))
        c = collections.Counter(r["y"] for r in rows)
        print("   已写成 bank 的：y=2 %s 行，y=1（对照，不训练）%s 行"
              % (c["2"], c["1"]))
        print("   分布在 %d 条录像上" % len({r["databag"] for r in rows}))

    print()
    print("== 人工确认的原始域标签（这些是验证集的来源）==")
    sheets = [("negpkg", "非手", "/workspace/negpkg/hands.csv"),
              ("neg2pkg", "非手", "/workspace/neg2pkg/hands.csv"),
              ("heldoutpkg", "非手（含 <150px）", "/workspace/heldoutpkg/hands.csv"),
              ("probepkg", "非手", "/workspace/probepkg/hands.csv"),
              ("claimpkg", "几只手", "/workspace/claimpkg/hands.csv")]
    for name, what, p in sheets:
        if os.path.exists(p):
            print("   %-12s %-18s %d 项" % (name, what,
                                            len(list(csv.DictReader(open(p))))))

    print()
    print("== 缺口 ==")
    print("   原始域 OWNER/OTHER 的**人工**标签：几乎没有。")
    print("   GPT-6 的归属标签盲审过（owner 99.8% / other 97.9%），但那是 300 框、")
    print("   在 640x512 的图上判的，不是在原生帧上，也不是这一批。")
    print("   → 验证集需要在原生帧上重新人工标一批归属，训练集可以用 GPT-6 标签。")


if __name__ == "__main__":
    main()
