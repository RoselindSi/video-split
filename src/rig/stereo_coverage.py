"""Which hands can this rig actually measure in 3D, and which does it drop?

A QUARTER OF THE TRIANGULATED PAIRS WERE GEOMETRICALLY UNUSABLE AND THE
REPROJECTION ERROR DID NOT SAY SO -- the rejected ones had a MEDIAN
reprojection of 1.56 px against 2.00 px for the survivors. Near-parallel rays
reproject beautifully and put the point anywhere; the diagnostic that sees it
is the triangulation angle, which a 6 cm baseline drives from about 11 degrees
at 0.3 m to 3.4 at 1 m and 2.3 at 1.5 m. That is not a threshold anyone chose.
It is the geometry, and it means coverage falls off with distance.

WHY THAT IS A GATE AND NOT A DETAIL. If a colleague's hand tends to be further
away than the wearer's, the missing 3D measurements are not missing at random
-- they are missing in a way correlated with the label. Any later claim that
`3D separates ownership well` would then be measured on whichever foreign
hands happened to come close enough, which is the easy half of the problem.
So this counts, per hand, how far down the funnel it gets:

    a hand the pipeline sees          the population, from the panorama
      -> found in the left eye         raw fisheye, where the detector was
      -> found in the right eye        never trained to work
      -> the two matched
      -> triangulation angle usable
      -> physically plausible
      -> a 3D measurement exists

DISTANCE IS NOT MEASURED WITH THE THING BEING TESTED. Stratifying by the
triangulated Z would be circular, because a bad triangulation invents its own
depth: the 24.8% that failed had a median Z of 0.46 m against 0.31 for the
survivors, entirely manufactured. Apparent size in the panorama is used
instead -- it comes from the detector, not from the geometry under test.

OWNERSHIP COMES FROM THE FROZEN EXIT-HEIGHT RULE, on the panorama where that
rule was validated, and it is a second opinion rather than a verdict. For a
coverage question that is enough: what is being compared is whether the funnel
behaves differently for the two classes, not whether either label is right.

ENTRY MATTERS MORE THAN THE AVERAGE. The hypothesis this feeds is about where
a hand comes from, so a track whose 3D only becomes usable after the hand has
arrived is useless to it even at 60% overall coverage. The first fifth of each
track is therefore counted separately.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

# How far from the predicted pixel an eye detection may sit and still be
# called the same hand. Generous on purpose: the prediction assumes a single
# render depth, so its parallax error is real and is not what is being tested.
SEED_RADIUS = 180.0
# Below this angle the depth is conditioned badly enough that the point is not
# a measurement. 3 degrees is roughly one metre at a 6 cm baseline.
MIN_ANGLE_DEG = 3.0
# Physical sanity, not a geometry criterion.
MAX_LATERAL_M = 1.5
Z_RANGE = (0.10, 6.0)

STAGES = ("seen", "left", "right", "matched", "angle_ok", "phys_ok", "usable")


def size_band(w_frac):
    """Apparent width as a stand-in for distance. Detector-derived, so it does
    not depend on the triangulation being tested."""
    if w_frac >= 0.12:
        return "大 >=12%"
    if w_frac >= 0.06:
        return "中 6-12%"
    return "小 <6%"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="/workspace/e2e_main2.txt")
    ap.add_argument("--rec", action="append", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--new_track_conf", type=float, default=0.60)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera, source_maps
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.hand_detect import detect
    from src.rig.hand_track import Tracker, MAX_LOST
    from src.rig import demo_video
    from src.rig.stereo3d import (undistort, triangulate, tri_angle_deg,
                                  epipolar_px)

    clips = {}
    for line in open(a.clips):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(":", 2)
        tag = os.path.basename(p[0].rstrip("/")).replace("databag-26_", "R")
        clips[tag] = (p[0], int(p[1]))

    model = YOLO(a.weights)
    rows = []
    for tag in a.rec:
        if tag not in clips:
            print(f"  !! {tag} 不在 clips 里")
            continue
        databag, start = clips[tag]
        rig = RigCalibration(os.path.join(databag, "calibration.yaml"))
        vcam = VirtualWideCamera.from_rig(rig)
        mods = rig.modules() if callable(rig.modules) else rig.modules
        vids = {k: os.path.join(databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        rd = ClipReader(rig, vids, start)
        tracker = Tracker(max_lost=max(MAX_LOST,
                                       demo_video.MAX_PREDICTION_AGE))
        mc, maps = {}, {}
        print(f"  {tag}: {start}-{start + a.n - 1}", flush=True)
        for k in range(a.n):
            src = rd.next()
            if not src:
                break
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
            H, W = rgb.shape[:2]
            pano = detect(model, rgb, min_conf=a.conf)
            ids = tracker.update(pano, rgb.shape,
                                 new_track_conf=a.new_track_conf,
                                 continue_conf=a.conf)
            if k % a.stride:
                continue
            # The six raw eyes, detected on directly. The detector was trained
            # on ordinary images and these are uncorrected fisheye, so how much
            # it loses here is part of what the funnel is measuring.
            eye = {}
            for m in mods:
                for cam in (m.left, m.right):
                    if cam.name in src:
                        eye[cam.name] = detect(model, src[cam.name],
                                               min_conf=a.conf)
            for cam in eye:
                if cam not in maps:
                    maps[cam] = source_maps(rig, cam, vcam, 0.6)

            for di, tid in zip(pano, ids):
                x0, y0, x1, y1 = [int(v) for v in di["box"]]
                u, v = (x0 + x1) // 2, (y0 + y1) // 2
                u = min(max(u, 0), W - 1)
                v = min(max(v, 0), H - 1)
                rec = {"rec": tag, "frame": start + k,
                       "tid": "" if tid is None else tid,
                       "reference_owner": int(bool(di.get("rule_owner"))),
                       "w_frac": round((x1 - x0) / float(W), 4),
                       "u_frac": round(u / float(W), 4),
                       "v_frac": round(v / float(H), 4),
                       "conf": round(float(di["conf"]), 3),
                       "module": "", "left_found": 0, "right_found": 0,
                       "matched": 0, "angle_deg": "", "disparity_px": "",
                       "reproj_error": "", "x": "", "y": "", "z": "",
                       "angle_ok": 0, "phys_ok": 0, "usable": 0}
                best = None
                for m in mods:
                    got = {}
                    for side, cam in (("left", m.left), ("right", m.right)):
                        mp = maps.get(cam.name)
                        if mp is None or not mp[2][v, u]:
                            continue
                        sx, sy = float(mp[0][v, u]), float(mp[1][v, u])
                        cands = eye.get(cam.name, [])
                        near, nd = None, SEED_RADIUS
                        for dj in cands:
                            cx = (dj["box"][0] + dj["box"][2]) / 2.0
                            cy = (dj["box"][1] + dj["box"][3]) / 2.0
                            d = math.hypot(cx - sx, cy - sy)
                            if d < nd:
                                near, nd = dj, d
                        if near is not None:
                            got[side] = near
                    if not got:
                        continue
                    if best is None or len(got) > len(best[1]):
                        best = (m, got)
                if best is None:
                    rows.append(rec)
                    continue
                m, got = best
                rec["module"] = m.name
                rec["left_found"] = int("left" in got)
                rec["right_found"] = int("right" in got)
                if len(got) < 2:
                    rows.append(rec)
                    continue
                dl, dr = got["left"], got["right"]
                cl = ((dl["box"][0] + dl["box"][2]) / 2.0,
                      (dl["box"][1] + dl["box"][3]) / 2.0)
                cr = ((dr["box"][0] + dr["box"][2]) / 2.0,
                      (dr["box"][1] + dr["box"][3]) / 2.0)
                pL = undistort([cl], m.left, cv2, np)[0]
                pR = undistort([cr], m.right, cv2, np)[0]
                X, err = triangulate(pL, pR, m.left, m.right, cv2, np)
                ang = tri_angle_deg(X, m.left, m.right, np)
                rec["matched"] = 1
                rec["angle_deg"] = round(ang, 2)
                rec["disparity_px"] = round(cl[0] - cr[0], 1)
                rec["reproj_error"] = round(err, 2)
                rec["x"], rec["y"], rec["z"] = (round(float(X[0]), 4),
                                                round(float(X[1]), 4),
                                                round(float(X[2]), 4))
                rec["angle_ok"] = int(ang >= MIN_ANGLE_DEG)
                rec["phys_ok"] = int(
                    Z_RANGE[0] <= X[2] <= Z_RANGE[1]
                    and max(abs(X[0]), abs(X[1])) <= MAX_LATERAL_M)
                rec["usable"] = int(rec["angle_ok"] and rec["phys_ok"])
                rows.append(rec)
        rd.close()

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} 个全景检测 -> {a.out}")
    report(rows)


def stage_flags(r):
    ok_l = int(r["left_found"])
    ok_r = int(r["right_found"])
    return {"seen": 1, "left": ok_l, "right": ok_r,
            "matched": int(r["matched"]),
            "angle_ok": int(r["angle_ok"]),
            "phys_ok": int(r["phys_ok"]),
            "usable": int(r["usable"])}


def funnel(rows, title):
    n = len(rows)
    if not n:
        print(f"  {title}: 无样本")
        return
    tot = collections.Counter()
    for r in rows:
        for k, v in stage_flags(r).items():
            tot[k] += v
    print(f"  {title:<22} n={n}")
    # Only the chain seen -> left -> right -> matched -> usable is nested.
    # `angle_ok` and `phys_ok` are two independent checks on a matched pair,
    # so showing them as a share of each other would invent a funnel that is
    # not there -- one can pass while the other fails.
    prev = n
    for st in ("seen", "left", "right", "matched"):
        c = tot[st]
        print(f"      {st:<10} {c:>6}  {c/n:>7.1%}"
              + (f"   (上一层的 {c/prev:.0%})" if prev and st != "seen" else ""))
        prev = c if c else prev
    mt = tot["matched"] or 1
    for st in ("angle_ok", "phys_ok"):
        c = tot[st]
        print(f"      {st:<10} {c:>6}  {c/n:>7.1%}   (配上的里 {c/mt:.0%})")
    c = tot["usable"]
    print(f"      {'usable':<10} {c:>6}  {c/n:>7.1%}   (配上的里 {c/mt:.0%})")


def report(rows):
    own = [r for r in rows if int(r["reference_owner"])]
    oth = [r for r in rows if not int(r["reference_owner"])]
    print(f"\n=== 逐帧漏斗（参照：冻结出口规则，不是人工真值）===")
    funnel(rows, "全部")
    funnel(own, "参照=自己的手")
    funnel(oth, "参照=别人的手")

    print(f"\n=== Gate C：3D 可得性是否和标签相关 ===")
    for name, s in (("自己的手", own), ("别人的手", oth)):
        u = sum(int(r["usable"]) for r in s)
        print(f"  P(usable 3D | {name}) = {u}/{len(s)} = "
              f"{u/max(1,len(s)):.1%}")
    if own and oth:
        pa = sum(int(r["usable"]) for r in own) / len(own)
        pb = sum(int(r["usable"]) for r in oth) / len(oth)
        print(f"  差距 {pa-pb:+.1%}  —— 差距大意味着后面任何只在 usable 子集上")
        print("  得到的『3D 分得开』都可疑：那是丢掉了最难的 other 之后的结果。")

    print(f"\n=== 按表观大小（不用三角化的 Z，避免循环）===")
    for band in ("大 >=12%", "中 6-12%", "小 <6%"):
        s = [r for r in rows if size_band(float(r["w_frac"])) == band]
        if s:
            u = sum(int(r["usable"]) for r in s)
            o = sum(1 for r in s if not int(r["reference_owner"]))
            print(f"  {band:<10} n={len(s):>5}  usable {u/len(s):>6.1%}   "
                  f"其中参照=别人的手 {o/len(s):>6.1%}")

    print(f"\n=== 轨迹级（P3 只需要够估轨迹，不需要每帧都有）===")
    by = collections.defaultdict(list)
    for r in rows:
        if r["tid"] not in ("", None):
            by[(r["rec"], r["tid"])].append(r)
    for name, want in (("自己的手", 1), ("别人的手", 0)):
        # A track is described by what it mostly was; one frame's verdict is
        # not the track's, and the reference rule is per-frame.
        ts = [v for v in by.values()
              if (sum(int(x["reference_owner"]) for x in v) * 2 >= len(v))
              == bool(want)]
        if not ts:
            continue
        any_valid = sum(1 for v in ts if any(int(x["usable"]) for x in v))
        fracs, entry = [], 0
        for v in ts:
            v.sort(key=lambda r: int(r["frame"]))
            fracs.append(sum(int(x["usable"]) for x in v) / len(v))
            head = v[:max(1, len(v) // 5)]
            entry += any(int(x["usable"]) for x in head)
        fracs.sort()
        print(f"  {name}: {len(ts)} 条轨迹")
        print(f"      至少一帧可用 3D      {any_valid}/{len(ts)} = "
              f"{any_valid/len(ts):.1%}")
        print(f"      可用帧占比 中位       {fracs[len(fracs)//2]:.1%}")
        print(f"      进入段(前 20%)有 3D  {entry}/{len(ts)} = "
              f"{entry/len(ts):.1%}   ← Gate B")

    print(f"\n=== 视差符号约定（先看，不当硬门）===")
    for mod in sorted({r["module"] for r in rows if r["module"]}):
        s = [float(r["disparity_px"]) for r in rows
             if r["module"] == mod and r["disparity_px"] != ""]
        if s:
            pos = sum(1 for d in s if d > 0)
            print(f"  {mod}: n={len(s)}  正 {pos/len(s):.1%}  "
                  f"中位 {sorted(s)[len(s)//2]:+.1f}px")
    print("  符号若不是每个 module 都一致，就不能把『视差为负』写成全局硬门。")


if __name__ == "__main__":
    main()
