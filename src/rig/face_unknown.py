"""Everything a candidate detector proposes that nobody has ever judged.

THE LABELS BELONG TO THE OLD DETECTORS. Every face crop labelled so far is a
box that YOLOv8-face or YuNet proposed, so a candidate can be scored for
RECALL against them -- a face is a face whoever found it -- and cannot be
scored for precision at all. Its own proposals that no labelled box overlaps
are the unknown pile, and that pile holds both of the things the comparison
turns on: faces the shipped detector never found, and false positives that are
the candidate's alone.

Measured on SCRFD at its 0.35 working point, the known part already says
something: 4 confirmed false positives against YOLOv8-face's 24, at 95% of the
true positives. But 2,124 of its boxes are in the unknown pile, and a
precision computed without them is a precision over the errors of a different
model.

This writes that pile as a labelling package in the layout `label_tool`
serves: the crop, the whole frame with the box drawn on it, and one row per
box. Labelling it closes the only gap left in the comparison.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pkg", action="append", default=None,
                    help="packages whose labelled boxes define what is already known")
    ap.add_argument("--thresh", type=float, default=0.35,
                    help="the working point being argued about, not the scan floor")
    ap.add_argument("--match_iou", type=float, default=0.3)
    ap.add_argument("--crop_px", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    from src.rig import face_mask
    from src.rig.face_compare import box_of, iou, labelled

    pkgs = a.pkg or sorted(glob.glob("/workspace/facepkg_*"))
    frames = labelled(pkgs)
    det = face_mask.load_detector(a.model, a.thresh)
    if det is None:
        raise SystemExit(f"打不开 {a.model}")
    tag = os.path.basename(a.model)
    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows = []
    keys = sorted(frames)
    for k, ctx in enumerate(keys):
        img = cv2.imread(ctx)
        if img is None:
            continue
        H, W = img.shape[:2]
        known = [box_of(it, W, H) for it in frames[ctx]]
        # The context images already carry the OLD detector's box drawn on
        # them. Drawing another one on top would ask the labeller about two
        # boxes at once, so the source for the crop and the new context is the
        # image as it is, and the new box is the only one in a different
        # colour -- stated in the sheet's legend rather than assumed.
        rec = os.path.basename(ctx)[:-4].rsplit("_f", 1)[0]
        j = 0
        for p in face_mask.detect_faces(det, img):
            if any(iou(g, list(p[:4])) >= a.match_iou for g in known):
                continue                       # somebody has judged this one
            x0, y0, x1, y1 = (int(v) for v in p[:4])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            # `label_tool` identifies a crop entirely from its name and its
            # pattern ends in _h<n>; a suffix of any other shape makes the
            # package load as zero crops. The index starts at 900 so it cannot
            # collide with a real hand index from the frame it came from.
            base = os.path.basename(ctx)[:-4].rsplit("_h", 1)[0]
            stem = f"{base}_h{900 + j}"
            j += 1
            side = max(x1 - x0, y1 - y0) * 2
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cx0, cy0 = max(0, cx - side // 2), max(0, cy - side // 2)
            cx1, cy1 = min(W, cx + side // 2), min(H, cy + side // 2)
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size:
                cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                            cv2.resize(crop, (a.crop_px, a.crop_px)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            vis = img.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"), vis,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            rows.append({"stem": stem, "frame": 0, "hand": j - 1,
                         "conf": round(float(p[4]), 4),
                         "w_frac": round((x1 - x0) / float(W), 5),
                         "h_frac": round((y1 - y0) / float(H), 5),
                         "cx_frac": round(cx / float(W), 5),
                         "cy_frac": round(cy / float(H), 5),
                         "model": tag, "label": "", "label_mode": ""})
        if (k + 1) % 300 == 0:
            print(f"    [{k + 1}/{len(keys)}] 已收 {len(rows)}", flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 个无人判过的框 -> {a.out}")
    c = sorted(r["conf"] for r in rows)
    print(f"  分数 中位 {c[len(c) // 2]:.2f}  最小 {c[0]:.2f}  最大 {c[-1]:.2f}")
    wf = sorted(r["w_frac"] for r in rows)
    print(f"  宽占画面 中位 {wf[len(wf) // 2]:.1%}  p95 {wf[int(len(wf) * .95)]:.1%}  "
          f"最大 {wf[-1]:.1%}")


if __name__ == "__main__":
    main()
