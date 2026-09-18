"""Does the cover land on the WEARER's hand, and how often does it blink?

WHY THIS EXISTS. The ownership label is not the only thing that decides what
the viewer sees. After the label come the segmentation, the dilation, the
owner-mask veto and the coasted boxes of tracks the detector lost, and any of
them can paint the wearer's own hand for a frame or two while the label stays
"self" the whole time. A flip count read off the labels cannot see that, so a
video can look worse than the metric while the metric is right.

WHAT IS MEASURED. The pipeline runs unchanged; a hook compares the delivered
frame with the clean one and reports, for every detected hand, the fraction of
its box that was altered. Per hand per frame:

    frame, box, own (what ownership said), p, covered (0..1)

`own=1` with `covered` high is the cover on the wearer's hand: the thing the
eye calls a flip. Boxes are followed across frames by overlap, not by track
id, because an id change is one of the ways the cover blinks.
"""
from __future__ import annotations

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", required=True)
    ap.add_argument("--out", required=True, help="the .mp4 the run writes")
    ap.add_argument("--csv", required=True, help="per-hand coverage")
    ap.add_argument("--start", type=int, default=78)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--clf_student", default=None)
    ap.add_argument("--clf_ctx", default=None)
    ap.add_argument("--geom", default=None)
    ap.add_argument("--geom_w", type=float, default=None)
    ap.add_argument("--no_cap", action="store_true")
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    a = ap.parse_args()
    import numpy as np
    from ultralytics import YOLO
    from src.rig import demo_video, face_mask, geom_prior, own_cnn
    from src.rig.calibration import RigCalibration

    vids = {k: os.path.join(a.databag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    rig = RigCalibration(os.path.join(a.databag, "calibration.yaml"))
    student = ctx = None
    geom_w = 0.5 if a.geom_w is None else a.geom_w
    if a.clf_student:
        from src.semhand import student as student_mod
        models, sdev = student_mod.load(a.clf_student)
        if not models:
            raise SystemExit("no checkpoint")
        student = (models, sdev)
        geom_w = 0.0 if a.geom_w is None else a.geom_w
        a.no_cap = True if a.geom_w is None else a.no_cap
    if a.clf_ctx:
        from src.rig import own_ctx
        m, dev, arm = own_ctx.load_model(a.clf_ctx)
        if m is None:
            raise SystemExit(f"--clf_ctx {a.clf_ctx} not found")
        ctx = (m, dev)
    rows = []

    def hook(k, clean, sup, info=None):
        if info is None:
            return
        diff = np.abs(sup.astype(np.int16) - clean.astype(np.int16)).max(2) > 8
        H, W = diff.shape
        for i, d in enumerate(info["dets"]):
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            sub = diff[y0:y1, x0:x1]
            rows.append({"frame": info["frame"], "box": i,
                         "own": int(info["own"][i]), "p": info["p_owner"][i],
                         "covered": round(float(sub.mean()) if sub.size else 0.0, 4),
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "conf": round(float(d["conf"]), 3),
                         "n_det": len(info["dets"]), "oth_px": info["oth_px"],
                         "veto_px": info["veto_px"]})

    n, dis, nf, _ = demo_video.run(
        rig, vids, a.out, a.start, a.n, a.stride, YOLO(a.weights),
        *own_cnn.load_model(None), 10, 14.0, a.fps,
        face_model=face_mask.MODEL, face_conf=face_mask.MIN_CONF,
        geom=geom_prior.load_model(a.geom), geom_w=geom_w, student=student, ctx=ctx,
        max_owner=None if a.no_cap else 2, panorama_mode="baseline", frame_hook=hook)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    own = [r for r in rows if r["own"]]
    hit = [r for r in own if r["covered"] >= 0.15]
    print(f"\n{n} 帧、{len(rows)} 只手；被判为自己的手 {len(own)} 只，"
          f"其中被盖住 >=15% 的 {len(hit)} 只（{len(hit) / max(1, len(own)):.1%}），"
          f"涉及 {len({r['frame'] for r in hit})} 帧")
    print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
