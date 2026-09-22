"""Is the thing under the mosaic a face -- asked of the boxes a size rule guesses about.

WHY A SECOND OPINION AND NOT A BETTER THRESHOLD. The size cap is a proxy. It
says a box wider than 0.18 of the frame is not a face at this range, which was
fitted on ten recordings containing no colleague near the camera and is right
twelve times out of thirteen on cam3 -- the thirteenth is a colleague leaning
in at width 0.27 with the highest detector score of the thirteen, and the cap
uncovers them. Every rule of this shape has the same defect: it decides by how
big the box is, and "large" is exactly what a nearby face and a workbench have
in common. Shape does not separate them either; that was tried.

WHAT MAKES THIS AFFORDABLE. Only the boxes the cap would refuse need asking,
and a box held across twelve frames is one thing seen twelve times, so eighty
seconds of cam3 comes to thirteen questions. The model that cannot run per
frame can easily run per episode.

AND THE BOXES IN QUESTION ARE LARGE, WHICH IS THE ONE REGIME THIS INSTRUMENT
IS GOOD IN. The hand-ness probe -- same model, same scoring, same view
geometry -- was right on 40 of 40 non-hand verdicts at a median 246 px and on
3 of 55 at 25 px. The cap selects boxes above 345 px by construction, so the
failure that sinks it elsewhere cannot reach here. That is not luck: "large"
is the cap's own criterion.

THE ERROR DIRECTIONS ARE NOT SYMMETRIC AND THE DEFAULT REFLECTS IT. Calling a
wall a face wastes a patch of bench. Calling a face a wall leaves a
recognisable person in the delivered video. So the threshold for KEEPING the
mosaic is deliberately low: a box is dropped only when the model is confident
it is not a face, and everything uncertain stays covered.

NOTHING HERE IS VALIDATED YET. This file asks the question and records the
answer; the sheet that decides whether the answer can be trusted is a separate
step, and until it comes back the verdicts are a measurement and not a gate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

QUESTION = ("Look at the red box. Is the thing inside it a human face or head "
            "(including the side or back of a head, and a face partly turned "
            "away)? Walls, tables, food, machine parts, bins and bare "
            "background are not faces. Reply with JSON only: "
            '{"face": true} or {"face": false}')
PREFIX = '{"face":'
FRAME_W = 1280
CROP = 256


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def episodes(rows, gap=2, min_iou=0.3):
    """Frame instances of one held box -> one item. -> [{frames, box, w_frac}]

    A box that survives twelve frames of hold is one decision seen twelve
    times. Asking twelve times would multiply both the cost and the apparent
    weight of a single detection."""
    by_f = collections.defaultdict(list)
    for r in rows:
        by_f[int(r["frame"])].append(r)
    live, out = [], []
    for f in sorted(by_f):
        for r in by_f[f]:
            box = [int(r[c]) for c in ("x0", "y0", "x1", "y1")]
            for L in live:
                if L["last"] >= f - gap and iou(L["box"], box) > min_iou:
                    L["last"], L["box"] = f, box
                    L["frames"].append(f)
                    L["w_frac"] = max(L["w_frac"], float(r["w_frac"]))
                    if r.get("conf"):
                        L["confs"].append(float(r["conf"]))
                    break
            else:
                live.append({"last": f, "box": box, "frames": [f],
                             "w_frac": float(r["w_frac"]),
                             "confs": [float(r["conf"])] if r.get("conf") else []})
    out = live
    for e in out:
        e["frame"] = e["frames"][len(e["frames"]) // 2]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", default="/workspace/cam3_faces",
                    help="a run directory holding the per-face CSVs")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--min_w", type=float, default=0.0,
                    help="only ask about boxes at least this wide (0 = all)")
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    if a.prep:
        import cv2
        from src.rig.seam_fix import RawCameraReader
        os.makedirs(a.views, exist_ok=True)
        idx = []
        for rec in sorted(jobs):
            p = os.path.join(a.faces, rec + ".faces.csv")
            if not os.path.exists(p):
                continue
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                    if float(r["w_frac"]) >= a.min_w]
            eps = episodes(rows)
            if not eps:
                continue
            bag = jobs[rec][0]
            vids = {k: os.path.join(bag, f"{k}.mp4")
                    for k in ("cam12", "cam34", "cam56")}
            want = sorted({e["frame"] for e in eps})
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
            for j, e in enumerate(sorted(eps, key=lambda x: x["frame"])):
                img = cache.get(e["frame"])
                if img is None:
                    continue
                H, W = img.shape[:2]
                b = e["box"]
                vis = img.copy()
                # RED, matching the sheet the reader already saw for these,
                # and different from the green the hand question uses so the
                # two can never be confused in a merged package.
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 5)
                sc = FRAME_W / float(W)
                cv2.imwrite(os.path.join(a.views, "%s_f%06d_h%d_full.jpg"
                                         % (rec, e["frame"], j)),
                            cv2.resize(vis, (FRAME_W, int(H * sc))),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                side = int(max(b[2] - b[0], b[3] - b[1]) * 1.6)
                mx, my = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                x0, y0 = max(0, mx - side // 2), max(0, my - side // 2)
                x1, y1 = min(W, mx + side // 2), min(H, my + side // 2)
                cv2.imwrite(os.path.join(a.views, "%s_f%06d_h%d_crop.jpg"
                                         % (rec, e["frame"], j)),
                            cv2.resize(vis[y0:y1, x0:x1], (CROP, CROP)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                idx.append({"stem": "%s_f%06d_h%d" % (rec, e["frame"], j),
                            "rec": rec, "frame": e["frame"],
                            "n_frames": len(e["frames"]),
                            "w_frac": round(e["w_frac"], 4),
                            "conf": round(max(e["confs"]), 3) if e["confs"] else "",
                            "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]})
            print("  %-16s %d 段" % (rec, len(eps)), flush=True)
        with open(os.path.join(a.views, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(idx[0]))
            w.writeheader()
            w.writerows(idx)
        print("-> %s (%d 段)" % (a.views, len(idx)))
        return

    from PIL import Image
    from src.semhand.qwen import Qwen

    idx = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["stem"] for l in open(a.out)}
    q = Qwen(a.model, tiny=a.tiny, answer_prefix=PREFIX, question=QUESTION)
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
            fh.write(json.dumps({"stem": r["stem"], "rec": r["rec"],
                                 "frame": int(r["frame"]),
                                 "n_frames": int(r["n_frames"]),
                                 "w_frac": float(r["w_frac"]), **s}) + "\n")
            fh.flush()
            if i % 25 == 0:
                print("  %d/%d" % (i, len(idx)), flush=True)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
