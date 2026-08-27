"""A brush, so the gold masks get drawn instead of discussed.

The annotation task is small -- a handful of frames -- and the tooling around
it was the whole cost: exporting a layer from a general image editor at
exactly the right size, with exactly two values, saved back over the right
filename. This does that and nothing else.

It runs where the package is, which is a laptop, and needs only OpenCV.

    drag                paint the wearer's arm
    right-drag / alt    erase
    [ ]                 brush size
    v                   toggle the overlay so the photo is unobstructed
    s                   save and mark this frame done
    n / p               next / previous frame
    q                   quit

WHAT IS BEING DRAWN is class 1 only: the wearer's own hand and bare forearm,
stopping at the sleeve cuff. A colleague's hand is not class 1 no matter how
close it is, and neither is anything merely skin-COLOURED -- the wooden
turntable and the beige machine strap are the automatic labeller's two most
frequent mistakes on this corpus, and getting them right here is most of the
value of the exercise.

BLIND FRAMES OPEN EMPTY AND THAT IS THE POINT. Half the package ships without
a starting mask so that half the gold is drawn without being anchored to the
machine's guess. The overlay key shows the photo, not the automatic result;
there is deliberately nothing to accept.
"""
from __future__ import annotations

import csv
import os

import numpy as np

BRUSH0 = 28
MAXW = 1500          # window width; the mask is always saved at full size


def load(gold_dir, stem):
    import cv2
    img = cv2.imread(os.path.join(gold_dir, stem + "_image.png"))
    m = cv2.imread(os.path.join(gold_dir, stem + "_start.png"),
                   cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"missing {stem}_image.png")
    if m is None:
        m = np.zeros(img.shape[:2], np.uint8)
    return img, (m > 127).astype(np.uint8)


def save(gold_dir, stem, mask, rows):
    import cv2
    cv2.imwrite(os.path.join(gold_dir, stem + "_start.png"),
                np.where(mask > 0, 255, 0).astype(np.uint8))
    for r in rows:
        if r["stem"] == stem:
            r["done"] = "y"
    p = os.path.join(gold_dir, "manifest.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(gold_dir, only_blind=False, only_todo=False):
    import cv2
    rows = list(csv.DictReader(open(os.path.join(gold_dir, "manifest.csv"),
                                    encoding="utf-8-sig")))
    work = rows
    if only_blind:
        work = [r for r in work if r["seed_mode"] == "blind"]
    if only_todo:
        work = [r for r in work if not str(r.get("done", "")).strip()]
    if not work:
        print("  nothing to do with those filters")
        return
    print(f"  {len(work)} frames\n"
          "  drag paint | right-drag erase | [ ] brush | v photo only | "
          "s save | n/p | q quit")

    i = 0
    while 0 <= i < len(work):
        stem = work[i]["stem"]
        img, mask = load(gold_dir, stem)
        H, W = mask.shape
        scale = min(1.0, MAXW / W)
        st = {"draw": 0, "brush": BRUSH0, "show": True, "dirty": False}

        def on_mouse(ev, x, y, flags, _):
            px, py = int(x / scale), int(y / scale)
            if ev == cv2.EVENT_LBUTTONDOWN:
                st["draw"] = 1
            elif ev == cv2.EVENT_RBUTTONDOWN:
                st["draw"] = -1
            elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                st["draw"] = 0
            if st["draw"]:
                alt = bool(flags & cv2.EVENT_FLAG_ALTKEY)
                v = 0 if (st["draw"] < 0 or alt) else 1
                cv2.circle(mask, (px, py), int(st["brush"] / scale), v, -1)
                st["dirty"] = True

        win = f"gold: {stem}"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            vis = img.copy()
            if st["show"] and mask.any():
                sel = mask > 0
                vis[sel] = (0.4 * vis[sel]
                            + 0.6 * np.array([0, 230, 0])).astype(np.uint8)
            vis = cv2.resize(vis, (int(W * scale), int(H * scale)))
            tag = (f"[{i+1}/{len(work)}] {work[i]['seed_mode']} "
                   f"{work[i].get('regime','')}  brush {st['brush']}  "
                   f"{'*' if st['dirty'] else ''}")
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(vis, tag, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("["), ord("]")):
                st["brush"] = max(3, st["brush"] + (6 if k == ord("]") else -6))
            elif k == ord("v"):
                st["show"] = not st["show"]
            elif k == ord("s"):
                save(gold_dir, stem, mask, rows)
                st["dirty"] = False
                print(f"  saved {stem}  ({mask.mean():.2%} of the frame)")
            elif k in (ord("n"), ord("p"), ord("q")):
                if st["dirty"]:
                    print("  ! unsaved changes -- press s first, or press "
                          "the key again to discard")
                    st["dirty"] = False
                    continue
                cv2.destroyWindow(win)
                if k == ord("q"):
                    return
                i += 1 if k == ord("n") else -1
                break
    cv2.destroyAllWindows()


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_dir")
    ap.add_argument("--blind", action="store_true",
                    help="only the frames with no starting mask -- the ones "
                         "that carry the information")
    ap.add_argument("--todo", action="store_true",
                    help="only frames not yet marked done")
    a = ap.parse_args()
    run(a.gold_dir, a.blind, a.todo)


if __name__ == "__main__":
    main()
