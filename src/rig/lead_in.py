"""The frames BEFORE a switch, with every own-hand box lettered.

WHY THIS IS A DIFFERENT QUESTION FROM THE SWITCH SHEET. That sheet asked who
the newly attached box belongs to, and the answer was: the wearer, six times
out of six. It could not ask what the track was carrying beforehand, because
only three frames were drawn and the box under audit was the new one. A track
that has been holding a colleague's hand for a second and then grafts the
wearer's hand onto it passes that test and is still a privacy leak.

SO THE ANSWER IS ON THE PICTURE HERE, AND THAT IS DELIBERATE. The flip sheet
hid the label because the reader was being asked to reproduce it. Here the
label IS the thing under audit -- the reader is being asked which green box
the system should not have called the wearer's -- so it has to be visible,
and each box carries a letter for its track id so it can be named.

EVERY BOX, NOT ONE. Green is what the system delivered as the wearer's, red
is what it called someone else's. The red boxes are what makes a green box
on a stranger legible: with only the green ones drawn, a foreign hand looks
like any other hand.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_tid")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cell_w", type=int, default=560)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    import numpy as np
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    rows = collections.defaultdict(list)
    tids = []
    for r in csv.DictReader(open(os.path.join(a.arm, a.rec + ".csv"), encoding="utf-8")):
        f = int(r["frame"])
        if not (a.lo <= f <= a.hi):
            continue
        t = r.get("tid") or ""
        rows[f].append(([float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                        r["own"] == "1", t, float(r.get("conf") or 0)))
        if r["own"] == "1" and t and t not in tids:
            tids.append(t)
    letter = {t: chr(65 + i) for i, t in enumerate(tids)}
    print("own tids ->", letter)

    want = list(range(a.lo, a.hi + 1, a.stride))
    bag = jobs[a.rec][0]
    vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
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

    tiles = []
    for f in want:
        img = cache.get(f)
        if img is None:
            continue
        v = img.copy()
        for box, own, t, cf in rows.get(f, []):
            b = [int(x) for x in box]
            col = (0, 230, 0) if own else (0, 0, 230)
            cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), col, 5)
            if own:
                cv2.putText(v, letter.get(t, "?"), (b[0] + 6, b[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4, cv2.LINE_AA)
        cv2.putText(v, "f%d" % f, (24, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 255, 255), 3, cv2.LINE_AA)
        sc = a.cell_w / float(v.shape[1])
        tiles.append(cv2.resize(v, (a.cell_w, int(v.shape[0] * sc))))
    if not tiles:
        raise SystemExit("no frames read")
    h, w = tiles[0].shape[:2]
    nr = (len(tiles) + a.cols - 1) // a.cols
    sheet = np.zeros((nr * h, a.cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        r_, c_ = divmod(k, a.cols)
        sheet[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    print("%d tiles -> %s  (%dx%d)" % (len(tiles), a.out, sheet.shape[1], sheet.shape[0]))

    # how long each own track has been alive inside the window
    span = collections.defaultdict(list)
    for f, rr in rows.items():
        for box, own, t, cf in rr:
            if own and t:
                span[t].append(f)
    for t in tids:
        fs = sorted(span[t])
        print("  %s tid=%s  %d 帧  f%d..f%d" % (letter[t], t, len(fs), fs[0], fs[-1]))


if __name__ == "__main__":
    main()
