"""Clean native frames for a views directory that only has drawn ones.

WHY THIS EXISTS AND WHAT IT FIXES. `neg_harvest --prep` writes two pictures
per box for the PROBE: a 1280-wide frame with the green box drawn on it, and
a crop cut from that drawn copy. Those are the probe's validated input and
they are not the student's. The student's transform resizes first and draws
at 1280x704, and cuts its crop from the UNDRAWN image.

Feeding the probe's jpgs to the student through `qwen.views` does something
worse than a small mismatch. The index carries native 1920x1520 coordinates
and the jpg is 1280 wide, so `views` computes a scale of 1.0 and draws the box
at native coordinates on a smaller picture -- off the edge for anything past
x=1280 -- and cuts the crop from the same wrong place. Every box comes back
looking like nothing at all, which is how a held-out run reported that 100% of
6,703 boxes were not hands, in every width band, from a model whose dev
precision was 97%.

So the frames are extracted again, clean, at native size, and the index gains
the path. Nothing about the boxes changes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

PAIR = {"cam1": "cam12", "cam2": "cam12", "cam3": "cam34",
        "cam4": "cam34", "cam5": "cam56", "cam6": "cam56"}
LEFT = {"cam1", "cam3", "cam5"}
JPEG_Q = 92


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True, help="where the clean frames go")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    import cv2

    rows = list(csv.DictReader(open(os.path.join(a.views, "index.csv"))))
    want = {(r["rec"], r["cam"], int(r["frame"])) for r in rows}
    print("%d 个框 -> %d 张不同的帧" % (len(rows), len(want)))

    roots = glob.glob("/shared/datasets/.incoming/video_1520p_raw/*/raw_databag")

    def bagpath(rec):
        for r in roots:
            for p in glob.glob(os.path.join(r, "*", rec)):
                if os.path.isdir(p):
                    return p
        return None

    by_vid = collections.defaultdict(list)
    for rec, cam, f in sorted(want):
        p = bagpath(rec)
        if p:
            by_vid[(p, cam, rec)].append(f)
    keys = sorted(by_vid)
    if a.nshard > 1:
        keys = [k for i, k in enumerate(keys) if i % a.nshard == a.shard]
        print("分片 %d/%d：%d 条录像" % (a.shard, a.nshard, len(keys)))

    n = 0
    for (p, cam, rec) in keys:
        v = os.path.join(p, PAIR[cam] + ".mp4")
        if not os.path.exists(v):
            continue
        d = os.path.join(a.out, rec, cam)
        os.makedirs(d, exist_ok=True)
        fs = sorted(by_vid[(p, cam, rec)])
        todo = [f for f in fs
                if not os.path.exists(os.path.join(d, "%06d.jpg" % f))]
        if not todo:
            continue
        cap = cv2.VideoCapture(v)
        at = -1
        for f in todo:
            if 0 <= f - at <= 240:
                fr = None
                while at < f:
                    ok, fr = cap.read()
                    at += 1
                    if not ok:
                        break
                if fr is None:
                    continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                at = f
                if not ok:
                    continue
            W = fr.shape[1] // 2
            half = fr[:, :W] if cam in LEFT else fr[:, W:]
            cv2.imwrite(os.path.join(d, "%06d.jpg" % f), half,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            n += 1
        cap.release()
        print("  %-30s %s  %d 帧" % (rec, cam, len(todo)), flush=True)
    print("-> %s（写了 %d 张）" % (a.out, n))


if __name__ == "__main__":
    main()
