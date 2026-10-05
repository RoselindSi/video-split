"""Re-render the teacher's scored boxes in the shape the student is deployed in.

WHY THE TEACHER'S OWN IMAGES CANNOT BE REUSED. They were drawn for a vision
model reading pictures, and they differ from the student's inputs in three
ways that all matter:

    crop    teacher draws the green box before cropping; the student crops
            from the clean image and never sees a box in this branch
    frame   teacher writes 1280 x (1280*H/W), preserving aspect; the student
            resizes to a fixed 1280x704 and then 448x246
    size    teacher's crop is 256, the student's is 224

Train on the teacher's pictures and the model learns a rendering that nothing
at inference time produces. That is not a hypothetical: the previous
three-class attempt was invalidated by exactly this, a 1600x900 panorama pool
scored against 1920x1520 raw frames, and it looked healthy on a validation set
drawn the same wrong way. So the views are rebuilt here by calling
`student.views` -- the same function the pipeline calls -- which makes the
training pool and the deployment path the same code rather than two
descriptions of the same intent.

LABELS COME FROM THE TEACHER'S CONFIDENT ENDS ONLY. Below 0.10 is a negative,
0.90 and above a positive, and the band between is not rendered at all. That
band is where the teacher is unsure, and a student fitted on it learns to
reproduce an uncertainty instead of a decision.

Decoding is the cost here, not the rendering, so boxes are grouped by
recording and each recording's frames are read once in order. Shard with
`--shard i --of n` to run several at once; shards split by recording so no two
processes decode the same video.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

NEG_THR = 0.10
POS_THR = 0.90


def read_scores(patterns):
    """-> {stem: p}. The stem is unique across the corpus and present in both
    the index and the score file, so it is the join key."""
    out = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    out[row["stem"]] = float(row["p"])
    return out


def read_jobs(patterns):
    out = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in open(path, encoding="utf-8"):
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 4:
                    out[parts[0]] = parts[1]
    return out


def rec_of(stem):
    return stem.rsplit("_f", 1)[0]


def collect(indexes, scores, bags, exclude):
    """-> {rec: [(stem, box, frame, y, p)]}, only the confident ends."""
    wanted = collections.defaultdict(list)
    for pattern in indexes:
        for path in sorted(glob.glob(pattern)):
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                stem = row["stem"]
                p = scores.get(stem)
                if p is None:
                    continue
                if p < NEG_THR:
                    y = 0
                elif p >= POS_THR:
                    y = 1
                else:
                    continue
                rec = rec_of(stem)
                if rec in exclude or rec not in bags:
                    continue
                box = [int(float(row[c])) for c in ("x0", "y0", "x1", "y1")]
                wanted[rec].append((stem, box, int(row["frame"]), y, p))
    return wanted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--scores", action="append", required=True)
    ap.add_argument("--jobs", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-jobs", dest="exclude_jobs", action="append",
                    default=[], help="这些录像一条都不渲 —— 冻结验证集")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()

    import cv2
    from src.semhand.student import views
    from src.rig.seam_fix import RawCameraReader

    scores = read_scores(a.scores)
    bags = read_jobs(a.jobs)
    exclude = set(read_jobs(a.exclude_jobs)) if a.exclude_jobs else set()
    wanted = collect(a.index, scores, bags, exclude)

    recs = sorted(wanted)
    mine = [r for i, r in enumerate(recs) if i % a.of == a.shard]
    total = sum(len(wanted[r]) for r in mine)
    print("老师打分 %d 框；两端内 %d 录像 / %d 框；本分片 %d 录像 / %d 框"
          % (len(scores), len(recs), sum(len(v) for v in wanted.values()),
             len(mine), total), flush=True)
    if exclude:
        print("  排除 %d 条冻结验证录像" % len(exclude), flush=True)

    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "index_%d.csv" % a.shard)
    done = 0
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "rec", "frame", "y",
                                                "p", "w_px", "h_px"])
        writer.writeheader()
        for rec in mine:
            items = sorted(wanted[rec], key=lambda it: it[2])
            bag = bags[rec]
            videos = {k: os.path.join(bag, "%s.mp4" % k)
                      for k in ("cam12", "cam34", "cam56")}
            by_frame = collections.defaultdict(list)
            for item in items:
                by_frame[item[2]].append(item)
            frames = sorted(by_frame)
            reader = RawCameraReader(videos, "cam3", frames[0])
            current, image, made = frames[0] - 1, None, 0
            for frame in frames:
                while current < frame:
                    image = reader.next()
                    current += 1
                    if image is None:
                        break
                if image is None:
                    break
                for stem, box, _f, y, p in by_frame[frame]:
                    hand_path = os.path.join(a.out, stem + "_h.jpg")
                    ctx_path = os.path.join(a.out, stem + "_c.jpg")
                    if not (os.path.exists(hand_path)
                            and os.path.exists(ctx_path)):
                        # 同一个函数，流水线推理时调的就是它
                        hand, ctx = views(image, box)
                        cv2.imwrite(hand_path, hand[:, :, ::-1],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        cv2.imwrite(ctx_path, ctx[:, :, ::-1],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    writer.writerow({"stem": stem, "rec": rec, "frame": frame,
                                     "y": y, "p": round(p, 6),
                                     "w_px": box[2] - box[0],
                                     "h_px": box[3] - box[1]})
                    made += 1
            reader.close()
            fh.flush()
            done += made
            print("  %-22s %5d / %5d 框（累计 %d/%d）"
                  % (rec, made, len(items), done, total), flush=True)
    print("-> %s（%d 行）" % (index_path, done), flush=True)


if __name__ == "__main__":
    main()
