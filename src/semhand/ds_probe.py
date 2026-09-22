"""Gate 2: what the hand-ness probe does to OTHER PEOPLE'S hands.

THE RISK THIS MEASURES. Today a colleague's hand is covered because the
ownership head calls it `other`. Put hand-ness in front of that head as a veto
and a box it rejects never reaches the head at all -- so every foreign hand
the probe wrongly calls "not a hand" walks out of the pipeline uncovered.
That failure does not exist today and the probe's 88-box validation cannot
see it: those boxes were drawn from C1's own-hand admissions, so the
foreign-hand arm of the question has never been measured.

WHY THIS DATASET AND NOT OURS. Our six cam3 recordings contain a handful of
labelled foreign hands. This snapshot has 45,483 boxes called `other` across
795 databags, none of them ours -- the recordings were checked for overlap
and there is none, so nothing here is leaking into the evaluation set.

THE LABELS ARE MACHINE LABELS AND ARE USED ONLY TO CHOOSE THE SAMPLE. Every
row carries `human_verified: false`; the factory half is `DS weak
supervision`, its own manifest saying DS classification "does not establish
exhaustive coverage" and its release state `pending_visual_review`. Scoring
one model against another model's unreviewed output is the mistake this
project keeps making under a new name. So DS picks which boxes to look at,
the probe scores them, and where the two disagree a person decides -- the
disagreements are exactly where a cheap gold set has to be spent.

THE VIEW IS THE PROBE'S OWN, prompt and geometry unchanged, because the
0.950 belongs to that exact instrument. These images are 640x512 against the
1920x1520 the pipeline reads, which is a real difference and a caveat on
transfer, not something to correct for here.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random

FRAME_W = 1280
CROP = 256
CELLS = (("hand", "other"), ("hand", "owner"),
         ("not_hand", None), ("ambiguous", None))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot",
                    default="/shared/ownership_labels/direct_hand_set_20260922_v1"
                            "/snapshot_v1/train.jsonl")
    ap.add_argument("--basis", default="DS weak supervision")
    ap.add_argument("--n", type=int, action="append", default=[],
                    help="one per cell, in order: hand/other, hand/owner, not_hand, ambiguous")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    ns = a.n or [200, 200, 200, 100]

    if a.prep:
        import cv2
        pool = collections.defaultdict(list)
        for line in open(a.snapshot):
            d = json.loads(line)
            if d.get("label_basis") != a.basis:
                continue
            for h in d.get("hands", []):
                key = (h.get("status"),
                       h.get("ownership") if h.get("status") == "hand" else None)
                if key in CELLS:
                    pool[key].append((d, h))
        rnd = random.Random(a.seed)
        pick = []
        for cell, n in zip(CELLS, ns):
            have = pool.get(cell, [])
            take = rnd.sample(have, min(n, len(have)))
            print("%-22s 总体 %6d，抽 %d" % ("%s/%s" % cell, len(have), len(take)))
            pick += [(cell, d, h) for d, h in take]
        rnd.shuffle(pick)

        os.makedirs(a.views, exist_ok=True)
        idx, key = [], []
        seen = collections.Counter()
        for cell, d, h in pick:
            img = cv2.imread(d["image"])
            if img is None:
                continue
            H, W = img.shape[:2]
            b = [int(v) for v in h["box_xyxy"]]
            bag = d["recordings"][0].split("/")[-1].replace("databag-", "R")
            i = seen[bag]
            seen[bag] += 1
            stem = "%s_f%06d_h%d" % (bag, i, 0)
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
            cv2.imwrite(os.path.join(a.views, stem + "_full.jpg"), full,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            cv2.imwrite(os.path.join(a.views, stem + "_crop.jpg"), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            idx.append({"stem": stem, "rec": bag, "frame": i, "conf": 0.0,
                        "w_frac": round(bw / float(W), 5),
                        "h_frac": round(bh / float(H), 5)})
            # DS's answer goes in a SEPARATE file: the sheet a person labels
            # must not carry the label being tested.
            key.append({"stem": stem, "ds_status": cell[0],
                        "ds_ownership": cell[1] or "",
                        "image": d["image"], "box": json.dumps(h["box_xyxy"])})
        with open(os.path.join(a.views, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(idx[0]))
            w.writeheader()
            w.writerows(idx)
        with open(a.views.rstrip("/") + "_key.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(key[0]))
            w.writeheader()
            w.writerows(key)
        print("-> %s (%d)，DS 的答案另存 %s_key.csv"
              % (a.views, len(idx), a.views.rstrip("/")))
        return

    from PIL import Image
    from src.semhand.qwen import Qwen
    from src.semhand.handness import QUESTION, PREFIX

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
                                 "w_frac": float(r["w_frac"]),
                                 "h_frac": float(r["h_frac"]), **s}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print("  %d/%d" % (i, len(idx)), flush=True)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
