"""Hands in metres, from the two eyes of one module.

THE PIPELINE THROWS THE GEOMETRY AWAY BEFORE IT USES IT. Six calibrated,
hardware-synchronised cameras are stitched into one wide image and every
ownership feature is then read off that image -- box centre, exit height,
forearm direction. Those are picture coordinates, and the same physical place
produces different ones depending on where the wearer's head is pointing and
how the panorama warped that region. The rig can say where a hand is in
metres; nothing downstream has ever asked it.

THIS IS THE GEOMETRY ONLY. No ownership label is read, written or predicted
here, so running it cannot contaminate the probe it is being built for. What
it answers is narrower and has to be settled first: does the calibration
triangulate, and can the two eyes of one module be matched at all when several
hands are in view.

WITHIN ONE MODULE, NOT ACROSS THEM. The three modules fan out at 0, 31 and 60
degrees and a hand usually falls in one of them; the two eyes inside a module
sit 6 cm apart and see almost the same picture, which makes their association
an easy problem rather than a research one. Cross-module association is a
different problem and is deliberately not attempted.

WHAT 6 CM BUYS. Depth error goes as Z^2/(f*B): about 5 mm at half a metre,
4 cm at a metre and a half, 17 cm at three. That is precisely the wrong way
round from most stereo applications and precisely the right way round for this
question, because a colleague's hand does not need to be located accurately --
it needs to be known to be far.

THE KNOWN COUNTEREXAMPLE. The bench edge sits at 0.21-0.25 m and the wearer's
forearm at 0.23-0.30 m, so depth alone already failed once at separating the
arm from the surface it rests on. That is why the probe this feeds keeps
`Z alone` and `trajectory` in separate arms: a scalar depth that cannot
separate them says nothing about whether a 3D trajectory can.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

# A triangulated pair is accepted when it reprojects into both eyes within
# this many pixels. Loose on purpose: the question here is whether the
# association is unique, not whether the corner is sub-pixel.
REPROJ_PX = 12.0
# Hands nearer or further than this are not physically plausible for a
# bench-scale scene and mark a bad match rather than a distant hand.
Z_RANGE = (0.10, 6.0)

FIELDS = ("rec", "frame", "module", "tid",
          "x", "y", "z", "reproj_error", "stereo_valid", "n_views",
          "vx", "vy", "vz", "speed",
          "start_x", "start_y", "start_z",
          "entry_dir_x", "entry_dir_y", "entry_dir_z", "distance_to_rig",
          "left_x0", "left_y0", "left_x1", "left_y1", "left_conf", "left_side",
          "right_x0", "right_y0", "right_x1", "right_y1", "right_conf",
          "right_side")


def undistort(pts, cam, cv2, np):
    """Fisheye pixels -> normalised rays in that camera's frame."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    K = np.asarray(cam.K, np.float64)
    D = np.asarray(cam.D, np.float64).reshape(4, 1)
    out = cv2.fisheye.undistortPoints(p, K, D)
    return out.reshape(-1, 2)


def proj_matrix(cam, np):
    """cam1-frame point -> normalised image of `cam`. -> 3x4

    The calibration stores (R, t) as camX -> cam1, so a point given in cam1's
    frame reaches camX through the inverse."""
    R = np.asarray(cam.R, np.float64)
    t = np.asarray(cam.t, np.float64).reshape(3)
    Rt = R.T
    return np.hstack([Rt, (-Rt @ t).reshape(3, 1)])


def triangulate(pL, pR, camL, camR, cv2, np):
    """One normalised point in each eye -> (XYZ in cam1 frame, reproj px)"""
    PL, PR = proj_matrix(camL, np), proj_matrix(camR, np)
    X = cv2.triangulatePoints(PL, PR,
                              np.asarray(pL, np.float64).reshape(2, 1),
                              np.asarray(pR, np.float64).reshape(2, 1))
    X = (X[:3] / X[3]).reshape(3)
    err = 0.0
    for cam, p in ((camL, pL), (camR, pR)):
        P = proj_matrix(cam, np)
        q = P @ np.append(X, 1.0)
        if abs(q[2]) < 1e-9:
            return X, float("inf")
        q = q[:2] / q[2]
        # Back to pixels so the tolerance means something a person can read.
        f = float(np.asarray(cam.K)[0, 0])
        err = max(err, f * float(np.linalg.norm(q - np.asarray(p))))
    return X, err


def eyes(frame, np):
    """A 3840x1520 side-by-side frame -> (left eye, right eye)"""
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


def probe(a):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.hand_detect import detect

    model = YOLO(a.weights)
    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        tag = os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")
        clips[tag] = (p[0], int(p[1]))

    want = a.rec or sorted(clips)[:a.n_rec]
    rows, stats = [], collections.Counter()
    pair_hist = collections.Counter()
    for tag in want:
        if tag not in clips:
            continue
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        mods = rig.modules() if callable(rig.modules) else rig.modules
        vids = {"module_A": "cam12", "module_B": "cam34", "module_C": "cam56"}
        print(f"\n  {tag}  from {start}, {a.n} frames", flush=True)
        for m in mods:
            path = os.path.join(databag, vids.get(m.name, "") + ".mp4")
            if not os.path.exists(path):
                continue
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for k in range(a.n):
                ok, frame = cap.read()
                if not ok:
                    break
                if k % a.stride:
                    continue
                imL, imR = eyes(frame, np)
                dL = detect(model, imL, min_conf=a.conf)
                dR = detect(model, imR, min_conf=a.conf)
                stats["frames"] += 1
                stats["det_left"] += len(dL)
                stats["det_right"] += len(dR)
                if not dL or not dR:
                    stats["one_eye_only" if (dL or dR) else "no_det"] += 1
                    continue
                # Every candidate pairing, scored by how well it triangulates.
                cand = []
                for i, di in enumerate(dL):
                    cx = (di["box"][0] + di["box"][2]) / 2.0
                    cy = (di["box"][1] + di["box"][3]) / 2.0
                    pL = undistort([(cx, cy)], m.left, cv2, np)[0]
                    for j, dj in enumerate(dR):
                        qx = (dj["box"][0] + dj["box"][2]) / 2.0
                        qy = (dj["box"][1] + dj["box"][3]) / 2.0
                        pR = undistort([(qx, qy)], m.right, cv2, np)[0]
                        X, err = triangulate(pL, pR, m.left, m.right, cv2, np)
                        good = (err <= REPROJ_PX
                                and Z_RANGE[0] <= X[2] <= Z_RANGE[1])
                        cand.append((err, i, j, X, good, di, dj))
                good = [c for c in cand if c[4]]
                pair_hist[(len(dL), len(dR), len(good))] += 1
                stats["pairs_tried"] += len(cand)
                stats["pairs_good"] += len(good)
                if not good:
                    stats["no_valid_pair"] += 1
                    continue
                # Greedy by reprojection error, one detection used once. With
                # a 6 cm baseline the right answer is usually far better than
                # the alternatives, which is what makes greedy adequate here.
                usedL, usedR = set(), set()
                for err, i, j, X, _, di, dj in sorted(good):
                    if i in usedL or j in usedR:
                        continue
                    usedL.add(i)
                    usedR.add(j)
                    stats["matched"] += 1
                    rows.append({
                        "rec": tag, "frame": start + k, "module": m.name,
                        "tid": "", "x": round(float(X[0]), 4),
                        "y": round(float(X[1]), 4), "z": round(float(X[2]), 4),
                        "reproj_error": round(err, 2), "stereo_valid": 1,
                        "n_views": 2, "vx": "", "vy": "", "vz": "",
                        "speed": "", "start_x": "", "start_y": "",
                        "start_z": "", "entry_dir_x": "", "entry_dir_y": "",
                        "entry_dir_z": "",
                        "distance_to_rig": round(float(np.linalg.norm(X)), 4),
                        "left_x0": int(di["box"][0]), "left_y0": int(di["box"][1]),
                        "left_x1": int(di["box"][2]), "left_y1": int(di["box"][3]),
                        "left_conf": round(float(di["conf"]), 3),
                        "left_side": di.get("side", ""),
                        "right_x0": int(dj["box"][0]), "right_y0": int(dj["box"][1]),
                        "right_x1": int(dj["box"][2]), "right_y1": int(dj["box"][3]),
                        "right_conf": round(float(dj["conf"]), 3),
                        "right_side": dj.get("side", "")})
            cap.release()

    print(f"\n  === 检测（每只眼单独跑，原始鱼眼图）===")
    print(f"  帧-模块 {stats['frames']}   左眼检测 {stats['det_left']}   "
          f"右眼检测 {stats['det_right']}")
    print(f"  两眼都没有 {stats['no_det']}   只有一只眼有 {stats['one_eye_only']}")
    print(f"\n  === 三角化 ===")
    print(f"  候选配对 {stats['pairs_tried']}   通过 "
          f"(重投影<={REPROJ_PX:g}px 且 {Z_RANGE[0]}<Z<{Z_RANGE[1]}m) "
          f"{stats['pairs_good']}")
    print(f"  配上的手 {stats['matched']}   一个有效配对都没有的帧 "
          f"{stats['no_valid_pair']}")
    if rows:
        zs = sorted(float(r["z"]) for r in rows)
        es = sorted(float(r["reproj_error"]) for r in rows)
        print(f"  Z 中位 {zs[len(zs)//2]:.2f} m   四分位 "
              f"{zs[len(zs)//4]:.2f}-{zs[3*len(zs)//4]:.2f}   "
              f"范围 {zs[0]:.2f}-{zs[-1]:.2f}")
        print(f"  重投影误差 中位 {es[len(es)//2]:.1f}px  p90 "
              f"{es[int(0.9*len(es))]:.1f}px")

    print(f"\n  === 模块内匹配是否唯一（左眼数, 右眼数, 有效配对数）===")
    amb = tot = 0
    for (nl, nr, ng), c in pair_hist.most_common(8):
        flag = ""
        if ng > min(nl, nr):
            flag = "  ← 有效配对多于手数，存在歧义"
            amb += c
        tot += c
        print(f"    {nl} x {nr} -> {ng:>2} 个有效   {c:>5} 次{flag}")
    print(f"\n  存在歧义的帧-模块 {amb}/{tot} = {amb/max(1,tot):.1%}")
    print("  歧义高就先别谈 3D ownership —— 模块内关联都不稳，3D 位置不可信。")

    if a.out and rows:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(FIELDS))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  {len(rows)} 行 -> {a.out}")
        print("  轨迹字段(vx/vy/vz/start/entry)留空：要先有 3D 轨迹关联才能填，")
        print("  这一轮只验证几何与单帧匹配，不做跟踪。")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--rec", action="append")
    ap.add_argument("--n_rec", type=int, default=3)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out")
    probe(ap.parse_args())


if __name__ == "__main__":
    main()
