"""Does the deployed path reproduce the student's training-time predictions?

`distil_train --predict_only` reads each hand through `crossv1.Views`; the
pipeline reads it through `semhand.student.predict`, from the frame the
render just produced rather than from a file. Same weights, same transforms,
so the probabilities have to match. This compares them on a test batch and
fails loudly if they do not.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/workspace/testb10")
    ap.add_argument("--arm", default="S_wide_g6")
    ap.add_argument("--ckpt", default="/workspace/distil/student/S_wide_g6_seed*.pt")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    import cv2
    from src.semhand import student
    cv2.setNumThreads(4)
    pred = {r["id"]: float(r[a.arm]) for r in csv.DictReader(open(os.path.join(a.root, "student", "pred.csv")))
            if r.get(a.arm)}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "index.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            ident = f"{r['rec']}|{r['frame']}|{r['tid']}"
            if r["status"] == "ok" and ident in pred:
                rows.append(dict(r, id=ident))
    rows = rows[:: max(1, len(rows) // a.n)][:a.n]
    models, device = student.load(a.ckpt)
    print(f"{len(models)} 个种子，比对 {len(rows)} 只手")
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    diffs = []
    for img, rs in by_img.items():
        rgb = cv2.imread(img)
        dets = [{"box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")]} for r in rs]
        for r, (_, p) in zip(rs, student.predict(models, device, rgb, dets)):
            diffs.append(abs(p - pred[r["id"]]))
    d = np.array(diffs)
    print(f"|部署路径 P - 训练时 P|  中位 {np.median(d):.6f}  p99 {np.percentile(d, 99):.6f}  最大 {d.max():.6f}")
    bad = int((d > 1e-3).sum())
    print(f"差异 > 1e-3 的手: {bad}")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
