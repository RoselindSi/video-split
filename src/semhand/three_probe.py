"""Which of the three claimed hands is not a hand -- and therefore whose.

THE ARITHMETIC GIVES THE FLOOR, THIS GIVES THE SPLIT. A frame delivering
three mutually disjoint own-hand boxes is over-claiming by construction, but
over-claiming has two causes with opposite consequences: a box on a machine
part is a wasted blur, a box on a colleague's hand is a privacy leak and a
stranger's hand in the training stream. The count cannot tell them apart and
neither can the ownership label, which is the thing being audited.

SO THE SPLIT COMES FROM THE ONE QUESTION THE SEMANTIC LAYER HAS ALREADY BEEN
VALIDATED ON: is the boxed thing a hand. AUC 0.950 on 88 blind-judged boxes,
92.0% balanced accuracy at 0.60, and the errors are refusals of real hands
rather than acceptances of parts -- 34 of 35 non-hands are caught. That error
direction is the safe one here: it makes the foreign-hand count an
underestimate, not an overestimate.

THE PROMPT IS IMPORTED, NOT RETYPED. The measured AUC belongs to that exact
question and prefix; a reworded copy is an unvalidated instrument.

WHAT A POSITIVE MEANS. Three boxes on three real hands in one frame means at
least one hand in that frame is not the wearer's and was delivered as the
wearer's. It does not say which one, and finding that needs a person -- but
the frame is already convicted without one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from src.rig.post_pass import MIN_PX

FRAME_W = 1280
CROP = 256


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def disjoint(boxes, thr):
    boxes = sorted(boxes, key=lambda b: -(b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
    keep = []
    for b in boxes:
        if all(iou(b[0], k[0]) < thr for k in keep):
            keep.append(b)
    return keep


def representative_rows(rows, n):
    """Even temporal samples, including both ends of a canonical track."""
    rows = sorted(rows, key=lambda row: int(row["frame"]))
    if n <= 0 or len(rows) <= n:
        return rows
    if n == 1:
        return [rows[len(rows) // 2]]
    indexes = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[index] for index in dict.fromkeys(indexes)]


def items_for(arm, jobs, thr, min_hands, track_maps=None,
              frames_per_track=5, min_px=MIN_PX):
    """Return representative owner boxes grouped by canonical track.

    The old probe asked Qwen about every eligible frame even though the
    verdict is aggregated per track. Sampling after disjoint ranking preserves
    the exact score keys consumed by :mod:`src.rig.post_pass` while bounding
    semantic work by tracks instead of video length.
    """
    out = []
    track_maps = track_maps or {}
    for rec, (bag, s, n) in sorted(jobs.items()):
        p = os.path.join(arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        own = collections.defaultdict(list)
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["own"] == "1":
                    own[int(r["frame"])].append(
                        ([float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                         float(r.get("conf") or 0), r.get("tid") or ""))
        eligible = []
        mapping = track_maps.get(rec, {})
        for f in range(s, s + n):
            b = own.get(f)
            if not b:
                continue
            keep = disjoint(b, thr)
            if len(keep) < min_hands:
                continue
            for k, (box, cf, tid) in enumerate(keep):
                if box[2] - box[0] < min_px or not tid:
                    continue
                eligible.append({
                    "stem": "%s_f%06d_b%d" % (rec, f, k), "rec": rec,
                    "frame": f, "box": box, "conf": cf, "tid": tid,
                    "canonical_tid": str(mapping.get(str(tid), tid)),
                    "n_claimed": len(keep)})
        by_track = collections.defaultdict(list)
        for item in eligible:
            by_track[item["canonical_tid"]].append(item)
        for rows in by_track.values():
            out.extend(representative_rows(rows, frames_per_track))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_c1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--min_hands", type=int, default=3)
    ap.add_argument("--min_px", type=int, default=MIN_PX)
    ap.add_argument("--frames_per_track", type=int, default=11)
    ap.add_argument("--track_maps", default="",
                    help="directory of <rec>.track_map.csv files")
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    track_maps = {}
    if a.track_maps:
        from src.rig.track_fusion import load_track_map
        for rec in jobs:
            path = os.path.join(a.track_maps, rec + ".track_map.csv")
            if os.path.exists(path):
                track_maps[rec] = load_track_map(path)
    items = items_for(
        a.arm, jobs, a.iou, a.min_hands, track_maps,
        a.frames_per_track, a.min_px)
    nfr = len({(x["rec"], x["frame"]) for x in items})
    print("%d 个框 / %d 帧（每帧 >=%d 只互不重叠的自己的手）" % (len(items), nfr, a.min_hands))

    if a.prep:
        import cv2
        from src.rig.seam_fix import RawCameraReader
        os.makedirs(a.views, exist_ok=True)
        idx = []
        by_rec = collections.defaultdict(list)
        for it in items:
            by_rec[it["rec"]].append(it)
        for rec in sorted(by_rec):
            bag = jobs[rec][0]
            vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
            want = sorted({x["frame"] for x in by_rec[rec]})
            rd = RawCameraReader(vids, "cam3", want[0])
            cur, cache = want[0] - 1, {}
            for f in want:
                img = None
                while cur < f:
                    img = rd.next()
                    cur += 1
                    if img is None:
                        break
                if img is None:
                    break
                cache[f] = img.copy()
            rd.close()
            got = 0
            for it in by_rec[rec]:
                img = cache.get(it["frame"])
                if img is None:
                    continue
                H, W = img.shape[:2]
                b = [int(v) for v in it["box"]]
                vis = img.copy()
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 230, 0), 4)
                sc = FRAME_W / float(W)
                full = cv2.resize(vis, (FRAME_W, int(H * sc)))
                bw, bh = b[2] - b[0], b[3] - b[1]
                side = int(max(bw, bh) * 1.6)
                mx, my = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                x0, y0 = max(0, mx - side // 2), max(0, my - side // 2)
                x1, y1 = min(W, mx + side // 2), min(H, my + side // 2)
                crop = cv2.resize(vis[y0:y1, x0:x1], (CROP, CROP))
                cv2.imwrite(os.path.join(a.views, it["stem"] + "_full.jpg"), full,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                cv2.imwrite(os.path.join(a.views, it["stem"] + "_crop.jpg"), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                idx.append({k: it[k] for k in (
                    "stem", "rec", "frame", "conf", "tid",
                    "canonical_tid", "n_claimed")})
                got += 1
            print("  %-16s %d/%d" % (rec, got, len(by_rec[rec])), flush=True)
        with open(os.path.join(a.views, "index.csv"), "w", newline="") as fh:
            fields = ["stem", "rec", "frame", "conf", "tid",
                      "canonical_tid", "n_claimed"]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(idx)
        print("-> %s (%d)" % (a.views, len(idx)))
        return

    from PIL import Image
    from src.semhand.qwen import Qwen
    from src.semhand.handness import QUESTION, PREFIX

    idx = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    if a.nshard < 1 or not 0 <= a.shard < a.nshard:
        ap.error("need 0 <= shard < nshard")
    if a.nshard > 1:
        idx = [row for index, row in enumerate(idx)
               if index % a.nshard == a.shard]
        print("  shard %d/%d -> %d" % (a.shard, a.nshard, len(idx)),
              flush=True)
    out_path = a.out if a.nshard == 1 else "%s.%d" % (a.out, a.shard)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["stem"] for l in open(out_path)}
        print("  已有 %d 个，跳过" % len(done))
    q = Qwen(a.model, tiny=a.tiny, answer_prefix=PREFIX, question=QUESTION)
    with open(out_path, "a") as fh:
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
            fh.write(json.dumps({"stem": r["stem"], "rec": r["rec"],
                                 "frame": int(r["frame"]), "conf": float(r["conf"]),
                                 "tid": r["tid"],
                                 "canonical_tid": r.get("canonical_tid", r["tid"]),
                                 "n_claimed": int(r["n_claimed"]),
                                 **s}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print("  %d/%d" % (i, len(idx)), flush=True)
    print("-> %s" % out_path)


if __name__ == "__main__":
    main()
