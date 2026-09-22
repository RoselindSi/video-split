"""Is the boxed thing a hand at all -- the one job C1 leaves for a semantic layer.

WHY THIS QUESTION AND NOT IDENTITY. C1's only established defect is that 39%
of the weak detections it admits are not hands: zero of 90 judged pairs made
the ID switch the ownership layer exists to prevent at a rate distinguishable
from the control, but 35 of 90 were a machine part, a tool or a patch of
background carried along as the wearer's hand. So the semantic layer is asked
what it is, not whose it is.

AND NOT BECAUSE A VLM IS THE FIRST IDEA. On those same 90 items the
detector's own score is useless -- the non-hands score HIGHER, median 0.181
against 0.169, and the best threshold reaches 53.3% balanced accuracy against
a 50% floor. Width and position overlap as well. Nothing cheap separates
them, which is what earns the 27B model a look.

THIS IS A NEW QUESTION FOR IT. Q1's evidence -- no OTHER read as OWNER across
1,085 decisions -- is about ownership; hand-ness has never been asked, so
nothing transfers and this is a first measurement rather than a confirmation.
It has gold to be measured against: 53 hands and 35 non-hands, judged blind.

Scored the way everything else here is scored: the assistant turn is opened
at `{"hand":` and P(hand) is the softmax over the next-token logits for
` true` against ` false`, so nothing is generated and nothing can fail to
parse.

WHAT IT MEASURED, on 88 items, 53 hands and 35 not:

    P(hand) on real hands    median 0.994, p25 0.971
    P(hand) on non-hands     median 0.000, p75 0.053
    AUC                      0.950
    at 0.60                  46 of 53 kept, 34 of 35 blocked, 92.0% balanced

Every one of the seven errors is a real hand rejected, five with p below
0.14 -- confident mistakes that moving the threshold does not recover. As a
gate this turns C1's addition from 61% real hands into roughly 98%, costing
about four of the thirty real frames it gains. The answer tokens hold
essentially all the probability mass (median 1.000), so the model is not
trying to say something else.

TWO STEPS BECAUSE TWO ENVIRONMENTS. The frames come out of the rig venv,
which has OpenCV and the readers; the model lives in the Qwen venv, which has
PIL and numpy and no cv2. So `--prep` writes the two views per item as files
and the scoring pass reads them back. Running it as one process fails on the
import, which is how this was first found out.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

QUESTION = ("Look at the green box. Is the thing inside it a human hand "
            "(or a hand in a glove)? Machine parts, tools, cloth, food and "
            "bare background are not hands. Reply with JSON only: "
            '{"hand": true} or {"hand": false}')
PREFIX = '{"hand":'
FRAME_W = 1280
CROP = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="/workspace/idswitchpkg")
    ap.add_argument("--group", default="C1_added")
    ap.add_argument("--arm", default="/workspace/cam3_c1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default="/workspace/handness.jsonl")
    ap.add_argument("--views", default="/workspace/handness_views",
                    help="where --prep writes the two views per item")
    ap.add_argument("--prep", action="store_true",
                    help="build the views (needs the rig venv) and stop")
    a = ap.parse_args()
    from PIL import Image

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    items = []
    for r in csv.DictReader(open(os.path.join(a.pkg, "hands.csv"))):
        if r["model"] != a.group or r["label"] not in ("owner", "other", "nothand"):
            continue
        rec, rest = r["stem"].rsplit("_f", 1)
        items.append({"stem": r["stem"], "rec": rec,
                      "frame": int(rest.split("_h")[0]),
                      "cx": float(r["cx_frac"]), "cy": float(r["cy_frac"]),
                      "w": float(r["w_frac"]), "h": float(r["h_frac"]),
                      "conf": float(r["conf"]), "gold": r["label"]})
    print("%d 个待判（gold: %s）" % (
        len(items), dict(collections.Counter(x["gold"] for x in items))))

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["stem"] for l in open(a.out)}
        print("  已有 %d 个，跳过" % len(done))

    if a.prep:
        import cv2
        from src.rig.seam_fix import RawCameraReader
        os.makedirs(a.views, exist_ok=True)
        by_rec = collections.defaultdict(list)
        for it in items:
            by_rec[it["rec"]].append(it)
        for rec in sorted(by_rec):
            bag, start, n = jobs[rec]
            vids = {k: os.path.join(bag, f"{k}.mp4")
                    for k in ("cam12", "cam34", "cam56")}
            want = sorted({x["frame"] for x in by_rec[rec]})
            rd = RawCameraReader(vids, "cam3", want[0])
            cur = want[0] - 1
            cache = {}
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
            for it in by_rec[rec]:
                img = cache.get(it["frame"])
                if img is None:
                    continue
                H, W = img.shape[:2]
                cx, cy = it["cx"] * W, it["cy"] * H
                bw, bh = it["w"] * W, it["h"] * H
                box = [int(cx - bw / 2), int(cy - bh / 2),
                       int(cx + bw / 2), int(cy + bh / 2)]
                vis = img.copy()
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]),
                              (0, 230, 0), 4)
                sc = FRAME_W / float(W)
                cv2.imwrite(os.path.join(a.views, it["stem"] + "_full.jpg"),
                            cv2.resize(vis, (FRAME_W, int(H * sc))),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                side = int(max(bw, bh) * 1.6)
                mx, my = int(cx), int(cy)
                x0, y0 = max(0, mx - side // 2), max(0, my - side // 2)
                x1, y1 = min(W, mx + side // 2), min(H, my + side // 2)
                cv2.imwrite(os.path.join(a.views, it["stem"] + "_crop.jpg"),
                            cv2.resize(vis[y0:y1, x0:x1], (CROP, CROP)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print("  %-16s %d 个视图" % (rec, len(by_rec[rec])), flush=True)
        print("-> %s" % a.views)
        return

    from src.semhand.qwen import Qwen
    q = Qwen(a.model, tiny=a.tiny, answer_prefix=PREFIX, question=QUESTION)
    with open(a.out, "a") as fh:
        for it in items:
            if it["stem"] in done:
                continue
            fp = os.path.join(a.views, it["stem"] + "_full.jpg")
            cp = os.path.join(a.views, it["stem"] + "_crop.jpg")
            if not (os.path.exists(fp) and os.path.exists(cp)):
                continue
            full, crop = Image.open(fp), Image.open(cp)
            conv = [{"role": "user", "content": [
                {"type": "image", "image": full},
                {"type": "image", "image": crop},
                {"type": "text", "text": QUESTION}]}]
            s = q.score(conv)
            fh.write(json.dumps({"stem": it["stem"], "gold": it["gold"],
                                 "conf": it["conf"], **s}) + "\n")
            fh.flush()
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
