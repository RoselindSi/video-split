"""Build the third class: boxes the detector produced that are not hands.

THE ONLY SCARCE THING. A student that can say `not a hand` needs examples of
not-a-hand from the distribution it will see, which is boxes OUR detector
produces -- not a curated set of objects. We have about 75 such boxes judged
by a person. The obvious large source, 2,508 boxes DS labelled `not_hand`,
was checked and 63 of 65 are real hands, so it is not a source at all.

WHAT IS LEFT IS THE PROBE, AND ONLY ON BOXES IT IS GOOD AT. On our own
population -- boxes a median 246 px wide -- the probe's `not a hand` verdict
was right on 40 of 40 judged blind. On the 23-28 px boxes in the labelling
set it was right on 3 of 55, and no threshold recovered it. So the harvest
takes a hard size floor and stays inside the regime that was measured, rather
than trusting a number across the boundary that broke it.

AND THE SIZE FLOOR IS NOT FREE, WHICH HAS TO BE SAID OUT LOUD. The negatives
this produces are all large, so a student trained on them learns what a large
non-hand looks like and nothing about a small one. Small boxes are exactly
where other people's hands live. The class will have to be scored separately
by box size and it will be weakest where it matters most; harvesting small
negatives needs an instrument that works there and we do not have one.

FRAMES FROM EVERYWHERE, NOT FROM OUR SIX. The evaluation recordings are six.
The labelling line's index carries device, recording, camera and a frame for
50,958 frames over 795 databags, and the raw video is on disk, so the harvest
reads native 1920x1520 frames from hundreds of recordings the pipeline has
never been run on. Those 795 have no overlap with the six.

NOTHING IS LABELLED BY THE DETECTOR'S SCORE. It was tried for this exact job
and reached 53.3% balanced accuracy on the band where the decision is made,
with the non-hands scoring HIGHER. The score is recorded and not used.

WHAT IT PRODUCED, AND WHAT A PERSON SAID ABOUT IT. 100 recordings, 9,832
boxes, 756 called not-a-hand. 140 of them went back blind -- 100 negatives
stratified over four width bands, 40 accepted boxes shuffled in -- and 98 of
the 100 are not hands, with the controls 40 for 40. By band the precision is
96, 100, 96, 100, which is flat, so the higher non-hand RATE above 340 px
(10.6% against 5.4%) is benches and parts bins being large rather than the
probe losing its footing there. Both errors sit at the threshold, p 0.042 and
0.047.

TWO THINGS THAT CAME OUT BACKWARDS. Non-hands are WIDER than hands, 299 px
against 272, so "large means hand" is false and there is no geometric
shortcut here any more than there was in the detector score. And only 87
boxes of 9,832 land between 0.10 and 0.60: the model is almost never
undecided, which is not the same as being right and is why the 140 went out.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random

PAIR = {"cam1": "cam12", "cam2": "cam12", "cam3": "cam34",
        "cam4": "cam34", "cam5": "cam56", "cam6": "cam56"}
LEFT = {"cam1", "cam3", "cam5"}
FRAME_W = 1280
CROP = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels",
                    default="/shared/ownership_labels/boxes_gpt6_qwen50958_v1/labels.jsonl")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--recordings", type=int, default=100)
    ap.add_argument("--per_rec", type=int, default=60,
                    help="frames kept per recording: one scene must not become the class")
    ap.add_argument("--stride", type=int, default=3,
                    help="frames are read CONSECUTIVELY from one seek and thinned by "
                         "this, because seeking into a 28,000-frame file costs seconds "
                         "and decoding forward costs milliseconds")
    ap.add_argument("--min_px", type=int, default=150,
                    help="box width floor: the regime the probe was measured in")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--link_iou", type=float, default=0.3,
                    help="相邻采样帧之间连成同一条轨迹的 IoU 下限")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    # SHARDED BECAUSE THE MODEL IS THE COST, NOT THE WORK. Ten thousand boxes
    # at a second and a half is four hours on one card and eighty minutes on
    # three, and the cards are idle. Each shard writes its own file; merging
    # is a cat, and a shared file would interleave half-written lines.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    if a.prep:
        import cv2
        from ultralytics import YOLO
        from src.rig.hand_detect import detect

        rows = []
        for line in open(a.labels):
            d = json.loads(line)
            if d.get("status") == "complete":
                rows.append(d)
        print("索引里 %d 帧" % len(rows))
        rnd = random.Random(a.seed)
        by_rec = collections.defaultdict(list)
        for d in rows:
            by_rec[d["recording"]].append(d)
        recs = sorted(by_rec)
        rnd.shuffle(recs)
        recs = recs[:a.recordings]
        pick = [rnd.choice(by_rec[r]) for r in recs]   # one anchor per recording
        print("抽了 %d 条录像，每条从一个锚点往后连读 %d 帧（步长 %d）"
              % (len(pick), a.per_rec * a.stride, a.stride))

        roots = glob.glob("/shared/datasets/.incoming/video_1520p_raw/*/raw_databag")

        def bagpath(dev, rec):
            for r in roots:
                p = os.path.join(r, dev, rec)
                if os.path.isdir(p):
                    return p
            return None

        anchors = []
        for d in pick:
            p = bagpath(d["device"], d["recording"])
            if not p:
                continue
            v = os.path.join(p, PAIR[d["camera"]] + ".mp4")
            if not os.path.exists(v):
                continue
            ms = int(d["window_id"].rsplit("__", 1)[1].split("_")[0])
            f = int(round(ms / 1000.0 * 30)) + int(d["source_frame_idx"])
            anchors.append((v, f, d))
        print("%d 个锚点" % len(anchors))

        os.makedirs(a.views, exist_ok=True)
        model = YOLO(a.weights)
        idx, nframe, nbox = [], 0, 0
        next_track = [0]

        def iou(p, q):
            ax0, ay0, ax1, ay1 = p
            bx0, by0, bx1, by1 = q
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0)
            return inter / float((ax1 - ax0) * (ay1 - ay0)
                                 + (bx1 - bx0) * (by1 - by0) - inter)

        for v, f0, d in anchors:
            cap = cv2.VideoCapture(v)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            # THE NEGATIVES HAVE TO COME IN TRACKS, NOT FRAMES. The 267
            # negatives in the training split sit on 258 distinct tracks --
            # about one view each -- so nothing in the set can say whether a
            # whole track is safe to delete, and the test split had no
            # multi-view negative track at all. Detections in consecutive
            # sampled frames are linked greedily by IoU, which is enough:
            # most false positives here are furniture and table edges, which
            # do not move, and a head that does move still overlaps itself
            # across a few frames at this stride.
            prev = []
            for j in range(a.per_rec * a.stride):
                ok, fr = cap.read()
                if not ok:
                    break
                if j % a.stride:
                    continue
                f = f0 + j
                W = fr.shape[1] // 2
                half = fr[:, :W] if d["camera"] in LEFT else fr[:, W:]
                rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
                dets = detect(model, rgb, min_conf=a.conf)
                nframe += 1
                current = []
                for k, det in enumerate(dets):
                    b = [int(x) for x in det["box"]]
                    if b[2] - b[0] < a.min_px:
                        continue
                    best, best_iou = None, a.link_iou
                    for pb, ptid in prev:
                        v_iou = iou(b, pb)
                        if v_iou >= best_iou:
                            best, best_iou = ptid, v_iou
                    if best is None:
                        best = next_track[0]
                        next_track[0] += 1
                    current.append((b, best))
                    stem = "%s_%s_f%06d_h%d" % (d["recording"].replace("databag-", "R"),
                                                d["camera"], f, k)
                    vis = half.copy()
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 230, 0), 4)
                    sc = FRAME_W / float(vis.shape[1])
                    cv2.imwrite(os.path.join(a.views, stem + "_full.jpg"),
                                cv2.resize(vis, (FRAME_W, int(vis.shape[0] * sc))),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    side = int(max(b[2] - b[0], b[3] - b[1]) * 1.6)
                    mx, my = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                    x0, y0 = max(0, mx - side // 2), max(0, my - side // 2)
                    x1, y1 = min(vis.shape[1], mx + side // 2), \
                        min(vis.shape[0], my + side // 2)
                    cv2.imwrite(os.path.join(a.views, stem + "_crop.jpg"),
                                cv2.resize(vis[y0:y1, x0:x1], (CROP, CROP)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    idx.append({"stem": stem, "rec": d["recording"],
                                "cam": d["camera"], "frame": f,
                                "track_id": best,
                                "conf": round(float(det.get("conf", 0)), 3),
                                "w_px": b[2] - b[0], "h_px": b[3] - b[1],
                                "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]})
                    nbox += 1
                prev = current
            cap.release()
            print("  %-30s %d 帧 / %d 个框" % (d["recording"], nframe, nbox), flush=True)
        with open(os.path.join(a.views, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(idx[0]))
            w.writeheader()
            w.writerows(idx)
        import collections as _c
        per = _c.Counter(r["track_id"] for r in idx)
        print("-> %s：%d 帧、%d 个 >=%dpx 的框、%d 条轨迹"
              % (a.views, nframe, nbox, a.min_px, len(per)))
        print("   其中 >=3 个视图的轨迹 %d 条，>=5 个的 %d 条 —— 这才是缺的东西"
              % (sum(1 for n in per.values() if n >= 3),
                 sum(1 for n in per.values() if n >= 5)))
        return

    from PIL import Image
    from src.semhand.qwen import Qwen
    from src.semhand.handness import QUESTION, PREFIX

    idx = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    if a.nshard > 1:
        idx = [r for i, r in enumerate(idx) if i % a.nshard == a.shard]
        a.out = "%s.%d" % (a.out, a.shard)
        print("分片 %d/%d：%d 个框 -> %s" % (a.shard, a.nshard, len(idx), a.out))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["stem"] for l in open(a.out)}
        print("已有 %d，跳过" % len(done))
    q = Qwen(a.model, answer_prefix=PREFIX, question=QUESTION)
    with open(a.out, "a") as fh:
        for i, r in enumerate(idx):
            if r["stem"] in done:
                continue
            full = Image.open(os.path.join(a.views, r["stem"] + "_full.jpg"))
            crop = Image.open(os.path.join(a.views, r["stem"] + "_crop.jpg"))
            conv = [{"role": "user", "content": [
                {"type": "image", "image": full},
                {"type": "image", "image": crop},
                {"type": "text", "text": QUESTION}]}]
            s = q.score(conv)
            fh.write(json.dumps({k: r[k] for k in
                                 ("stem", "rec", "cam", "frame", "conf", "w_px")}
                                | s) + "\n")
            fh.flush()
            if i % 200 == 0:
                print("  %d/%d" % (i, len(idx)), flush=True)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
