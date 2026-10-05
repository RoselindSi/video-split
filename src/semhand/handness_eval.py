"""Score a hand-ness model on the frozen 20 recordings, against human labels.

THE POPULATION IS TRACKS, NOT BOXES. The decision this feeds is track-level --
`post_pass` asks "is this track a hand" once and the answer applies to the
whole track -- so a box-level number would be measuring something the system
never asks. The 291 gold tracks were labelled blind by a person: 137 owner,
48 other, 100 not_hand, 6 unsure.

THE VIEW SET IS TAKEN FROM THE EXISTING RENDER, NOT RESAMPLED. `fresh20_views`
holds the five frames per track that the teacher and the v7 student were both
scored on, and their filenames are the record of which frames those were.
Drawing a fresh sample would make the new number incomparable to the two it
has to be read against, and picking fewer frames per track is not available
either -- view count has cost us before and is not an optimisation target.

BASELINES COME OUT OF THE STORED SCORES, NOT OUT OF MEMORY. Both are recomputed
here from `teacher_handness_fresh20.json` and `fresh20_handness.json` on
whatever population is being reported, because the teacher was only ever
scored on 200 of the 291 tracks (100 not_hand + 100 hands) while the student
has all 291. Reading the teacher's recall off 100 tracks against the student's
off 185 would flatter whichever one happened to draw the easier set, so the
headline table is the 200 tracks all three share.

WHAT COUNTS AS PASSING. Non-hand false positive rate at the deployment
threshold, against the teacher's 1%. Accuracy is not reported: 65% of this set
is a real hand, and 93.8% of the training corpus was, so "always hand" scores
well while hiding nothing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics

GOLD = "/storage/fresh20_gold_v2.csv"
TEACHER = "/storage/teacher_handness_fresh20.json"
STUDENT_V7 = "/storage/fresh20_handness.json"
VIEWS = "/storage/fresh20_views"
ARM = "/storage/fresh60"
THRESHOLDS = (0.5, 0.9201)
STEM = re.compile(r"^(.+?)__tid(\d+)__f(\d+)_crop\.jpg$")


def view_set(views):
    """-> {(rec, tid): [frame]}，就是老师和 v7 当初打分的那批帧。"""
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(views, "*_crop.jpg"))):
        m = STEM.match(os.path.basename(path))
        if m:
            out[(m.group(1), m.group(2))].append(int(m.group(3)))
    return out


def boxes_for(arm, rec, wanted):
    """-> {(tid, frame): box} from the run that produced those tracks."""
    path = os.path.join(arm, "%s.csv" % rec)
    if not os.path.exists(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        key = (str(row.get("tid", "")), int(row["frame"]))
        if key in wanted:
            try:
                out[key] = [int(float(row[c]))
                            for c in ("x0", "y0", "x1", "y1")]
            except (KeyError, TypeError, ValueError):
                continue
    return out


def rates(scores, truth, keys, thr):
    """-> (非手误判, 真手召回, n_neg, n_pos) over `keys` only."""
    neg = [scores[k] for k in keys
           if truth.get(k) == "not_hand" and k in scores]
    pos = [scores[k] for k in keys
           if truth.get(k) in ("owner", "other") and k in scores]
    return (sum(1 for p in neg if p >= thr) / max(1, len(neg)),
            sum(1 for p in pos if p >= thr) / max(1, len(pos)),
            len(neg), len(pos))


def table(name, scores, truth, keys):
    lines = []
    for thr in THRESHOLDS:
        fp, rec, n_neg, n_pos = rates(scores, truth, keys, thr)
        lines.append({"model": name, "thr": thr,
                      "nonhand_false_positive": round(fp, 4),
                      "hand_recall": round(rec, 4),
                      "n_neg": n_neg, "n_pos": n_pos})
        print("  %-12s thr %.4f   非手误判 %5.1f%% (%d/%d)   真手召回 %5.1f%% (%d/%d)"
              % (name, thr, 100 * fp, round(fp * n_neg), n_neg,
                 100 * rec, round(rec * n_pos), n_pos), flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="glob，多个就取均值集成")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--views", default=VIEWS)
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--agg", choices=("mean", "median"), default="mean")
    a = ap.parse_args()

    import numpy as np
    import torch
    from src.rig import own_ctx
    from src.semhand.student import views as student_views
    from src.semhand.crossv1 import norm_rgb
    from src.rig.seam_fix import RawCameraReader

    truth = {}
    for row in csv.DictReader(open(a.gold, encoding="utf-8-sig")):
        truth[(row["rec"], row["tid"])] = row["track_truth"]
    frames = view_set(a.views)
    print("gold %d 轨 / 有视图的 %d 轨" % (len(truth), len(frames)), flush=True)

    paths = sorted(glob.glob(a.models))
    if not paths:
        raise SystemExit("没有模型：%s" % a.models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in paths:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = own_ctx.build(ck.get("arm", "both"),
                              n_out=int(ck.get("n_out", 2))).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        models.append(model)
    print("集成 %d 个模型：%s" % (len(models), [os.path.basename(p) for p in paths]),
          flush=True)

    jobs = {}
    for line in open("/storage/fresh20_jobs.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 4:
            jobs[parts[0]] = parts[1]

    by_rec = collections.defaultdict(dict)
    for (rec, tid), fs in frames.items():
        for f in fs:
            by_rec[rec][(tid, f)] = None

    new = {}
    for rec in sorted(by_rec):
        if rec not in jobs:
            continue
        box_of = boxes_for(a.arm, rec, set(by_rec[rec]))
        if not box_of:
            print("  %-22s 运行记录里找不到这些框，跳过" % rec, flush=True)
            continue
        bag = jobs[rec]
        videos = {k: os.path.join(bag, "%s.mp4" % k)
                  for k in ("cam12", "cam34", "cam56")}
        per_frame = collections.defaultdict(list)
        for (tid, f), box in box_of.items():
            per_frame[f].append((tid, box))
        order = sorted(per_frame)
        reader = RawCameraReader(videos, "cam3", order[0])
        current, image = order[0] - 1, None
        got = collections.defaultdict(list)
        for f in order:
            while current < f:
                image = reader.next()
                current += 1
                if image is None:
                    break
            if image is None:
                break
            hs, cs, tids = [], [], []
            for tid, box in per_frame[f]:
                h, c = student_views(image, box)
                hs.append(norm_rgb(h))
                cs.append(norm_rgb(c))
                tids.append(tid)
            with torch.no_grad():
                h = torch.stack(hs).to(device)
                c = torch.stack(cs).to(device)
                g = torch.zeros(len(hs), 14, device=device)
                p = np.mean([torch.softmax(m(h, c, g), 1)[:, 1].cpu().numpy()
                             for m in models], 0)
            for tid, score in zip(tids, p):
                got[tid].append(float(score))
        reader.close()
        for tid, scores in got.items():
            agg = statistics.mean if a.agg == "mean" else statistics.median
            new[(rec, tid)] = agg(scores)
        print("  %-22s %d 轨 / %d 视图" % (rec, len(got), sum(len(v) for v in got.values())),
              flush=True)

    teacher = {tuple(k.split("|")): float(v)
               for k, v in json.load(open(TEACHER)).items()} \
        if os.path.exists(TEACHER) else {}
    v7 = {tuple(k.split("|")): float(v)
          for k, v in json.load(open(STUDENT_V7)).items()} \
        if os.path.exists(STUDENT_V7) else {}

    labelled = {k for k, v in truth.items()
                if v in ("not_hand", "owner", "other")}
    shared = sorted(labelled & set(new) & set(teacher) & set(v7))
    print("\n=== 三方都打过分的 %d 轨（这是标题数字）===" % len(shared), flush=True)
    rows = []
    rows += table("老师", teacher, truth, shared)
    rows += table("学生 v7", v7, truth, shared)
    rows += table("新二分类", new, truth, shared)

    mine = sorted(labelled & set(new))
    print("\n=== 新模型打到的全部 %d 轨（老师只覆盖 200 轨，所以这栏不与老师比）==="
          % len(mine), flush=True)
    wide = table("新二分类", new, truth, mine)
    wide += table("学生 v7", v7, truth, mine)

    json.dump({"shared": rows, "all_scored": wide, "agg": a.agg,
               "n_shared": len(shared), "n_all": len(mine),
               "scores": {"%s|%s" % k: round(v, 6) for k, v in new.items()}},
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
