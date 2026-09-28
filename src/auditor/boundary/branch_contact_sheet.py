"""One strip of frames per recording, with the detector's boxes drawn on them.

WHY THE BOXES ARE DRAWN AND NOT JUST COUNTED. The number that rejected a
recording -- what fraction of frames hold a hand wide enough to see -- is only
as good as the boxes it counts, and those boxes are exactly what a table
cannot show. A recording can score well on small boxes that are not hands at
all, which is how the rejected recording came top of the first ranking. With
the boxes drawn, a person can see in one pass whether the detector is finding
hands doing something or finding furniture.

FRAMES ARE SPREAD OVER THE WHOLE RECORDING and labelled with their timestamp,
so a strip that is empty in the middle reads as "nothing happens there"
rather than as a sampling accident.

THE WIDTH FLOOR IS DRAWN DIFFERENTLY from the rest. Boxes at or above it are
solid, the smaller ones dashed, because the whole question is which of the two
a recording is made of.
"""
from __future__ import annotations

import argparse
import os

DETECTOR = "/shared/models/HaWoR/weights/external/detector.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--weights", default=DETECTOR)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--thumb_w", type=int, default=240)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    import cv2
    import numpy as np
    from decord import VideoReader
    from ultralytics import YOLO

    os.makedirs(a.outdir, exist_ok=True)
    model = YOLO(a.weights)
    for p in a.videos:
        rid = os.path.splitext(os.path.basename(p))[0]
        vr = VideoReader(p)
        fps = vr.get_avg_fps() or 10.0
        idx = np.linspace(0, len(vr) - 1, a.cols).astype(int).tolist()
        tiles = []
        for i in idx:
            f = vr[i].asnumpy()[:, :, ::-1].copy()      # -> BGR
            r = model.predict(f, conf=a.min_conf, verbose=False)[0]
            b = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                 else np.zeros((0, 4)))
            for x1, y1, x2, y2 in b.astype(int):
                big = (x2 - x1) >= a.floor
                col = (0, 230, 0) if big else (0, 170, 255)
                if big:
                    cv2.rectangle(f, (x1, y1), (x2, y2), col, 3)
                else:                                   # 小框画成虚线
                    for k in range(x1, x2, 10):
                        cv2.line(f, (k, y1), (min(k + 5, x2), y1), col, 2)
                        cv2.line(f, (k, y2), (min(k + 5, x2), y2), col, 2)
                    for k in range(y1, y2, 10):
                        cv2.line(f, (x1, k), (x1, min(k + 5, y2)), col, 2)
                        cv2.line(f, (x2, k), (x2, min(k + 5, y2)), col, 2)
            t = a.thumb_w / f.shape[1]
            f = cv2.resize(f, (a.thumb_w, int(f.shape[0] * t)))
            cv2.putText(f, "%d:%02d" % (int(i / fps) // 60, int(i / fps) % 60),
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(f, "%d:%02d" % (int(i / fps) // 60, int(i / fps) % 60),
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(f)
        h = tiles[0].shape[0]
        sheet = np.hstack([t if t.shape[0] == h else cv2.resize(t, (a.thumb_w, h))
                           for t in tiles])
        bar = np.full((26, sheet.shape[1], 3), 245, np.uint8)
        cv2.putText(bar, "%s   实线=手宽>=%dpx  虚线=更小" % (rid, a.floor),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        out = os.path.join(a.outdir, rid + ".jpg")
        cv2.imwrite(out, np.vstack([bar, sheet]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("-> %s" % out, flush=True)


if __name__ == "__main__":
    main()
