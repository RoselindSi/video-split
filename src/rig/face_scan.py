"""What score does the box that mosaicked half the frame actually have?

RAISING THE FLOOR ONLY HELPS IF THE FALSE POSITIVES ARE THE LOW-SCORING ONES,
and nothing in this project has ever recorded the score of the boxes that do
the damage. The labelled sweep in `face_mask` reports precision and recall by
threshold over a harvest of candidates; it says nothing about SIZE, and the
failure that destroys the wearer's hands is a box big enough to swallow the
frame, not a box that is merely wrong.

This renders a clip, runs the face detector at a floor low enough to see the
whole distribution, runs the hand detector beside it so the veto can be
applied as it ships, and writes one row per proposal:

    frame, x0, y0, x1, y1, score, frac (of the frame), veto (would be dropped),
    n_hands

Nothing is covered and no video is written. The question it answers is narrow:
if the floor moved from 0.35 to 0.5, or to 0.7, which boxes would go -- the
huge ones, or the small ones that are mostly real faces.
"""
from __future__ import annotations

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--floor", type=float, default=0.10,
                    help="score floor for the scan, well under the shipped 0.35")
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    from src.rig import face_mask
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.hand_detect import detect
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    rig = RigCalibration(os.path.join(a.databag, "calibration.yaml"))
    vids = {k: os.path.join(a.databag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    vcam = VirtualWideCamera.from_rig(rig)
    fdet = face_mask.load_detector(face_mask.MODEL, a.floor)
    yolo = YOLO(a.weights)
    rd = Prefetch(ClipReader(rig, vids, a.start), skip=max(0, a.stride - 1))
    rows, mc = [], {}
    for k in range(a.n):
        src = rd.next()
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        H, W = rgb.shape[:2]
        hands = detect(yolo, rgb)
        faces = face_mask.detect_faces(fdet, rgb)
        keep, dropped = face_mask.split_on_hands(faces, hands)
        for f in faces:
            rows.append({"frame": a.start + k * a.stride,
                         "x0": int(f[0]), "y0": int(f[1]), "x1": int(f[2]), "y1": int(f[3]),
                         "score": round(float(f[4]), 4),
                         "frac": round((f[2] - f[0]) * (f[3] - f[1]) / float(W * H), 5),
                         "veto": int(any(f is d for d in dropped)),
                         "n_hands": len(hands)})
        if (k + 1) % 50 == 0:
            print(f"    [{k + 1}/{a.n}] {len(rows)} 个候选", flush=True)
    rd.close()
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                           ["frame", "x0", "y0", "x1", "y1", "score", "frac", "veto", "n_hands"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 个人脸候选 -> {a.out}")
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)):
        g = [r for r in rows if lo <= r["frac"] < hi]
        if not g:
            continue
        sc = sorted(r["score"] for r in g)
        print(f"  占画面 {lo:.0%}-{hi:.0%}: {len(g):>5} 个，分数 中位 {sc[len(sc)//2]:.2f} "
              f"最小 {sc[0]:.2f} 最大 {sc[-1]:.2f}；"
              f"分数 >=0.35 的 {sum(1 for s in sc if s >= 0.35)}，"
              f">=0.50 的 {sum(1 for s in sc if s >= 0.50)}，"
              f">=0.70 的 {sum(1 for s in sc if s >= 0.70)}")


if __name__ == "__main__":
    main()
