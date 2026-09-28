"""Merge the harvested negatives into the handness manifest.

WHY A MERGE STEP AT ALL. The harvest writes its own index and its own score
file; the student reads one manifest with `crop_path`, `full_path`, a target
and a split. Nothing else stands between them, but that nothing has to be
written down or the two halves never meet.

THE GEOMETRY ALREADY MATCHES AND THAT IS NOT LUCK. `neg_harvest` renders at
FRAME_W=1280 and CROP=256, which is exactly the geometry of `crowd10_v2`, the
largest source in the existing manifest. The last attempt at a third class
died on this: negatives rendered from 1920x1520 raw frames against a teacher
pool of 1600x900 panorama renders, and the control scored 94.5% against a
ceiling of 12% because the student was reading the border, not the hand. The
merge checks the sizes rather than trusting the constants.

SPLIT BY RECORDING, AND NOT ALL INTO TRAIN. The obvious move is to pour every
new negative into training and leave the benchmark alone, which keeps the old
numbers comparable and leaves the original hole exactly where it was: the test
split had no multi-view negative track, so nothing could say whether a whole
track is safe to delete. So recordings are split, most to train, some to val
and test, and the old and new test rows stay distinguishable through `sources`
so the comparison against earlier runs can still be made on the old rows
alone.

THE TARGET IS THE TEACHER'S, NOT AN ASSUMPTION. A harvested box is a candidate,
not a negative; the teacher decides, and a box it calls a hand joins as a
positive. Writing them all in as negatives because that is what the harvest was
for would poison the class it was meant to fix.

EVERY HARVESTED BOX IS AT LEAST 150px WIDE and that limit has to be carried
forward in the reading. The probe's `not a hand` verdict was right 40 times out
of 40 on boxes around 246px and 3 times out of 55 at 23-28px, so the harvest
stays inside the regime that was measured. The negatives this produces are all
large, other people's hands are median 87px, and the class will therefore be
weakest exactly where it matters most. Scoring has to be broken out by box
size or it will report a number that does not exist.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import subprocess

LEFT = {"cam1", "cam3", "cam5"}
PAIR = {"cam1": "cam12", "cam2": "cam12", "cam3": "cam34",
        "cam4": "cam34", "cam5": "cam56", "cam6": "cam56"}


def half_size(views_dir, rec, cam, cache, roots):
    """-> (w, h) of the half-frame the boxes were measured in, probed not assumed."""
    key = (rec, cam)
    if key in cache:
        return cache[key]
    wh = None
    name = PAIR.get(cam)
    for root in roots:
        for path in glob.glob(os.path.join(root, "*", rec, name + ".mp4")) + \
                glob.glob(os.path.join(root, rec, name + ".mp4")):
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height", "-of", "csv=p=0",
                     path], capture_output=True, text=True).stdout.strip()
                w, h = [int(x) for x in out.split(",")[:2]]
                wh = (w // 2, h)
            except Exception:
                wh = None
            break
        if wh:
            break
    cache[key] = wh
    return wh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="现有的 handness manifest")
    ap.add_argument("--views", required=True, help="neg_harvest 的 views 目录")
    ap.add_argument("--scores", required=True, help="打分 jsonl 的 glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bag_roots", nargs="+",
                    default=["/shared/datasets/.incoming/video_1520p_raw"])
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    base = list(csv.DictReader(open(a.base, encoding="utf-8-sig")))
    fields = list(base[0].keys())
    print("原 manifest %d 行" % len(base))

    idx = {r["stem"]: r for r in csv.DictReader(
        open(os.path.join(a.views, "index_full.csv"
                          if os.path.exists(os.path.join(a.views, "index_full.csv"))
                          else "index.csv"), encoding="utf-8"))}
    scored = {}
    for path in glob.glob(a.scores):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                scored[r["stem"]] = r
    print("采集视图 %d 个，已打分 %d 个" % (len(idx), len(scored)))
    if not scored:
        raise SystemExit("没有读到打分结果")

    # 几何自检：上一次三分类就死在这里，所以先量再说
    from PIL import Image
    s0 = next(iter(scored))
    with Image.open(os.path.join(a.views, s0 + "_full.jpg")) as im:
        nf = im.size
    with Image.open(os.path.join(a.views, s0 + "_crop.jpg")) as im:
        nc = im.size
    ref = [r for r in base if r.get("sources") == "crowd10_v2"]
    if ref:
        with Image.open(ref[0]["full_path"]) as im:
            bf = im.size
        with Image.open(ref[0]["crop_path"]) as im:
            bc = im.size
        print("几何：新 full %s / crop %s   ；crowd10_v2 full %s / crop %s"
              % (nf, nc, bf, bc))
        if nf[0] != bf[0] or nc != bc:
            print("  警告：几何与 crowd10_v2 不一致，学生可能学到的是画幅而不是手")
    else:
        print("几何：新 full %s / crop %s（base 里没有 crowd10_v2 可比）" % (nf, nc))

    recs = sorted({r["rec"] for r in scored.values()})
    import random
    rng = random.Random(a.seed)
    rng.shuffle(recs)
    n_val = max(1, round(a.val_frac * len(recs)))
    n_test = max(1, round(a.test_frac * len(recs)))
    split_of = {rec: ("val" if i < n_val else "test" if i < n_val + n_test
                      else "train") for i, rec in enumerate(recs)}

    cache, rows, missing = {}, [], 0
    for stem, sc in scored.items():
        meta = idx.get(stem)
        if not meta:
            missing += 1
            continue
        cam = meta["cam"]
        wh = half_size(a.views, meta["rec"], cam, cache, a.bag_roots)
        box = {}
        if wh:
            W, H = wh
            x0, y0 = float(meta["x0"]), float(meta["y0"])
            x1, y1 = float(meta["x1"]), float(meta["y1"])
            box = {"box_cx": "%.6f" % ((x0 + x1) / 2 / W),
                   "box_cy": "%.6f" % ((y0 + y1) / 2 / H),
                   "box_w": "%.6f" % ((x1 - x0) / W),
                   "box_h": "%.6f" % ((y1 - y0) / H)}
        row = {k: "" for k in fields}
        row.update({
            "item_id": stem,
            "rec": meta["rec"],
            "frame": meta["frame"],
            "canonical_tid": "neg%s" % meta.get("track_id", ""),
            "raw_tid": meta.get("track_id", ""),
            "sources": "neghar_v2",
            "stem": stem,
            "full_path": os.path.join(a.views, stem + "_full.jpg"),
            "crop_path": os.path.join(a.views, stem + "_crop.jpg"),
            # 目标由教师给，不是「采来的就是负例」——教师判成手的就当正例进来
            "teacher_p": "%.6f" % float(sc["p"]),
            "human_label": "",
            "split": split_of[meta["rec"]],
            "review_status": "neghar_v2",
        })
        row.update({k: v for k, v in box.items() if k in fields})
        rows.append(row)

    if missing:
        print("打分里有 %d 个 stem 在 index 里找不到，已跳过" % missing)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(base + rows)

    per = collections.Counter(r["canonical_tid"] for r in rows)
    ps = [float(r["teacher_p"]) for r in rows]
    print("\n新增 %d 行 / %d 条轨迹" % (len(rows), len(per)))
    print("  教师判非手（p<0.5）的 %d (%.0f%%)；p<0.2 的 %d"
          % (sum(1 for p in ps if p < 0.5), 100 * sum(1 for p in ps if p < 0.5) / len(ps),
             sum(1 for p in ps if p < 0.2)))
    negt = [t for t, _ in per.items()
            if all(float(r["teacher_p"]) < 0.5 for r in rows
                   if r["canonical_tid"] == t)]
    print("  整条都被判非手的轨迹 %d 条，其中 >=3 视图的 %d 条 ← 这是原来完全没有的"
          % (len(negt), sum(1 for t in negt if per[t] >= 3)))

    print("\n合并后每个 split 的负例（目标 <0.5 或人工 nothand）：")
    allr = base + rows
    print("%-7s %9s %9s %9s %11s" % ("split", "总行", "负例", "负例占比", "新增负例"))
    for s in ("train", "val", "test"):
        g = [r for r in allr if r["split"] == s]
        def isneg(r):
            h = (r.get("human_label") or "").strip()
            if h == "nothand":
                return True
            if h == "hand" or h == "unsure":
                return False
            t = (r.get("teacher_p") or "").strip()
            return bool(t) and float(t) < 0.5
        neg = [r for r in g if isneg(r)]
        new = [r for r in neg if r.get("sources") == "neghar_v2"]
        print("%-7s %9d %9d %8.1f%% %11d"
              % (s, len(g), len(neg), 100 * len(neg) / max(1, len(g)), len(new)))
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
