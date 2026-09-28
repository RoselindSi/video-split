"""Are the hands actually visible in these recordings.

WHY THIS RUNS AFTER THE PACKET WAS BUILT. The batch was picked on interaction
structure -- how often the person switches between object families -- and
structure says nothing about whether a hand is on screen. The reviewer opened
the fourth recording and found almost no visible hand action, which is not a
property the object-family screen can see. One such recording getting through
means the screen never checked, so all of them are checked here rather than
just the one that was caught.

IT MEASURES PRESENCE, NOT OWNERSHIP OR CORRECTNESS. A frame counts when the
detector puts a box on it above the usual confidence floor; whose hand it is,
and whether the box is right, are different questions with their own
instruments. What is being ruled out is a recording where the task happens
mostly off-camera, and for that a presence rate over uniformly sampled frames
is enough.

FRAMES ARE SAMPLED UNIFORMLY ACROSS THE WHOLE RECORDING, not from the middle
or the first minute. A person sets up, works, then tidies; any fixed window
would measure one of those phases and call it the recording.

THE BARE DETECTION RATE IS THE WRONG NUMBER, which the first run of this
module demonstrated: the recording the reviewer rejected for having no visible
hand action scored the *highest* rate of all twenty-three, 55%, on boxes whose
median width was 30px in a 640x480 frame. Thirty pixels is the band where
deciding whether a box is even a hand runs at a few percent, so that rate is
mostly false positives, and it moves in the opposite direction from what it
claims to measure. What matters for annotating a manipulation is a hand close
enough to see what it is doing, so the rate is reported again at width floors
of 60 and 100 pixels and the selection uses those.
"""
from __future__ import annotations

import argparse
import json
import os

DETECTOR = "/shared/models/HaWoR/weights/external/detector.pt"
MIN_CONF = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--weights", default=DETECTOR)
    ap.add_argument("--n", type=int, default=80, help="每条均匀抽多少帧")
    ap.add_argument("--min_conf", type=float, default=MIN_CONF)
    ap.add_argument("--floors", type=int, nargs="+", default=[60, 100],
                    help="框宽下限，用来分开「远处的小框」和「在干活的手」")
    ap.add_argument("--out")
    a = ap.parse_args()

    import numpy as np
    from decord import VideoReader
    from ultralytics import YOLO

    model = YOLO(a.weights)
    rows = []
    for p in a.videos:
        rid = os.path.splitext(os.path.basename(p))[0]
        vr = VideoReader(p)
        idx = np.linspace(0, len(vr) - 1, min(a.n, len(vr))).astype(int).tolist()
        frames = vr.get_batch(idx).asnumpy()
        hit, widths, nbox = 0, [], []
        big = {w0: 0 for w0 in a.floors}
        for f in frames:
            r = model.predict(f[:, :, ::-1], conf=a.min_conf, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            nbox.append(len(b))
            if len(b):
                hit += 1
                ws = (b[:, 2] - b[:, 0])
                widths.extend(ws.tolist())
                for w0 in a.floors:
                    if (ws >= w0).any():
                        big[w0] += 1
        w = sorted(widths)
        rows.append({"rid": rid, "n": len(idx),
                     "rate": round(hit / max(1, len(idx)), 3),
                     "med_w": round(w[len(w) // 2], 1) if w else 0.0,
                     "boxes_per_frame": round(sum(nbox) / max(1, len(nbox)), 2),
                     **{"rate_%d" % w0: round(big[w0] / max(1, len(idx)), 3)
                        for w0 in a.floors}})
        print("%-22s 有手 %5.1f%%  >=60px %5.1f%%  >=100px %5.1f%%  框宽中位 %5.1f"
              % (rid, 100 * rows[-1]["rate"], 100 * rows[-1]["rate_60"],
                 100 * rows[-1]["rate_100"], rows[-1]["med_w"]), flush=True)

    rows.sort(key=lambda r: -r["rate_100"])
    print("\n按「>=100px 的手出现帧率」排序 —— 这才是能看清在做什么的比例")
    print("  %-22s %8s %8s %9s %8s" % ("recording", ">=100px", ">=60px", "任意框", "框宽中位"))
    for r in rows:
        print("  %-22s %7.1f%% %7.1f%% %8.1f%% %8.1f"
              % (r["rid"], 100 * r["rate_100"], 100 * r["rate_60"],
                 100 * r["rate"], r["med_w"]))
    if a.out:
        json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
