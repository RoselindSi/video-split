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
import collections
import csv
import os


def read(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["frame"], r["own"] = int(r["frame"]), int(r["own"])
        r["covered"], r["oth_px"] = float(r["covered"]), int(r["oth_px"])
        for c in ("x0", "y0", "x1", "y1"):
            r[c] = float(r[c])
    return rows


def gold_of(rows, dump_path, gold_paths, rec):
    """Attach human track gold to each measured hand, by frame and overlap.

    The run's own `own` flag cannot answer "was the WEARER's hand covered":
    on the frame a verdict flips, that hand's flag already says foreign. The
    dump carries the track ids for the same detections, and the gold is on
    those ids."""
    from src.selfother.train import read_gold
    gold = read_gold(gold_paths)
    dump = collections.defaultdict(list)
    for r in csv.DictReader(open(dump_path, encoding="utf-8")):
        if r["rec"] == rec:
            dump[int(r["frame"])].append(r)
    n = 0
    for r in rows:
        box = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
        best, bg = 0.0, None
        for q in dump[r["frame"]]:
            v = _iou(box, [float(q[c]) for c in ("x0", "y0", "x1", "y1")])
            if v > best:
                best, bg = v, q
        r["gold"] = gold.get((rec, str(bg["tid"]))) if bg is not None and best >= 0.5 else None
        r["tid"] = bg["tid"] if bg is not None and best >= 0.5 else None
        n += r["gold"] is not None
    return n


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def compare(paths, thr=0.15, dump=None, gold_paths=(), rec=None):
    """What the two runs put on screen, hand by hand and frame by frame."""
    for path in paths:
        rows = read(path)
        if dump:
            n = gold_of(rows, dump, list(gold_paths), rec)
            g = [r for r in rows if r["gold"] == "owner"]
            hit = [r for r in g if r["covered"] >= thr]
            frames = sorted({r["frame"] for r in hit})
            runs = 1 + sum(1 for i in range(1, len(frames)) if frames[i] != frames[i - 1] + 1)
            print(f"\n{os.path.basename(path)}  真的盖到主人的手："
                  f"{len(hit)} / {len(g)} 手帧（{len(hit) / max(1, len(g)):.2%}），"
                  f"{len(frames)} 帧、{runs if frames else 0} 段"
                  f"（对上 gold 的手帧 {n}）")
            # The wearer's hands are what the downstream model is meant to
            # learn from, so ANY damage to those pixels counts, not only the
            # frames a viewer would call blurred.
            any_hit = [r for r in g if r["covered"] > 0.01]
            px = sum(r["covered"] * (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in g)
            tot = sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in g)
            print(f"    任何破坏（盖住 >1%）：{len(any_hit)} 手帧（{len(any_hit) / max(1, len(g)):.2%}）；"
                  f"主人手像素被破坏 {px / max(1, tot):.3%}")
            for r in hit[:6]:
                print(f"    frame {r['frame']} tid {r['tid']}  own={r['own']} "
                      f"p={r['p']}  盖住 {r['covered']:.0%}")
        frames = sorted({r["frame"] for r in rows})
        own = [r for r in rows if r["own"]]
        hit = [r for r in own if r["covered"] >= thr]
        # The cover appearing and disappearing on screen at all: the thing a
        # viewer reads as flicker even when no label moved.
        px = {f: max(r["oth_px"] for r in rows if r["frame"] == f) for f in frames}
        on = [px[f] > 0 for f in frames]
        sw = sum(1 for i in range(1, len(on)) if on[i] != on[i - 1])
        runs = []
        cur = 0
        for v in on:
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        print(f"\n{os.path.basename(path)}")
        print(f"  {len(frames)} 帧、{len(rows)} 只手；判为自己 {len(own)}、判为别人 "
              f"{len(rows) - len(own)}")
        print(f"  自己的手被盖住 >= {thr:.0%} 的：{len(hit)} 只手帧"
              f"（{len(hit) / max(1, len(own)):.1%}），涉及 {len({r['frame'] for r in hit})} 帧")
        print(f"  画面上有遮挡的帧 {sum(on)}/{len(frames)}；开关切换 {sw} 次；"
              f"连续遮挡段 {len(runs)} 段，最短 {min(runs) if runs else 0} 帧、"
              f"中位 {sorted(runs)[len(runs) // 2] if runs else 0} 帧")
        short = [r for r in runs if r <= 3]
        print(f"  其中 <=3 帧的遮挡闪现 {len(short)} 段  <- 看起来最像闪烁的东西")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs="+", help="read CSVs written earlier and stop")
    ap.add_argument("--dump", help="with --compare: the dump whose track ids carry the gold")
    ap.add_argument("--gold", action="append", default=[])
    ap.add_argument("--rec", default="R0824_160752")
    ap.add_argument("--databag")
    ap.add_argument("--out", help="the .mp4 the run writes")
    ap.add_argument("--csv", help="per-hand coverage")
    ap.add_argument("--start", type=int, default=78)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--clf_student", default=None)
    ap.add_argument("--clf_ctx", default=None)
    ap.add_argument("--geom", default=None)
    ap.add_argument("--geom_w", type=float, default=None)
    ap.add_argument("--no_cap", action="store_true")
    ap.add_argument("--inherit_self_on_reacquire", action="store_true",
                    help="ablation: do not reset a reacquired self track (the flicker's source)")
    ap.add_argument("--assoc_log", help="why a live track got no box: the "
                                        "cheapest candidate and the rule that refused it")
    ap.add_argument("--grace_log", help="per-event provenance of every box "
                                        "the new-hand grace spared, with its anchor")
    ap.add_argument("--new_hand_grace", type=int, default=0,
                    help="spare a box this new when it touches a hand already called self")
    ap.add_argument("--self_reconfirm", type=int, default=2,
                    help="frames a reacquired hand must support `self`; 2 is shipped, "
                         "1 judges the frame it returns on its own evidence")
    ap.add_argument("--face_model", default=None,
                    help="a different face/head detector; default is the shipped one")
    ap.add_argument("--face_conf", type=float, default=None)
    ap.add_argument("--max_face_frac", type=float, default=None,
                    help="drop a face box wider than this fraction of the frame")
    ap.add_argument("--camera", help="read one camera raw (e.g. cam3) instead "
                                    "of the stitched wide render")
    ap.add_argument("--reacquire_log", help="one row per reacquisition")
    ap.add_argument("--reacquire_edge", type=float, default=None,
                    help="1.50 ships; 0.40 is the never-lost value")
    ap.add_argument("--veto_held", action="store_true",
                    help="apply the hand veto to held face boxes too")
    ap.add_argument("--new_track_conf", type=float, default=None,
                    help="score a detection needs to START a track; 0.60 ships, and 56.6% of a "
                         "colleague's hands never reach it")
    ap.add_argument("--continue_conf", type=float, default=None)
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare, dump=a.dump, gold_paths=a.gold, rec=a.rec)
    if not (a.databag and a.out and a.csv):
        ap.error("--databag, --out 和 --csv 一起给，或者只给 --compare")
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
            # The face masker is the other thing that changes these pixels, and
            # on one recording it was the ONLY thing: a false face box large
            # enough to swallow the frame is not vetoed, because the veto
            # measures how much of the FACE a hand covers, not how much of the
            # hand the face covers.
            # MEASURED ON THE MASK, NOT ON THE BOXES. `cover` grows each box
            # by PAD of its own size, so the mosaic reaches a third further
            # than the box in every direction. Against the boxes this read
            # 0.00 on five frames of R0825_102941 where the mosaic had in fact
            # destroyed 67-99% of the wearer's hand -- the measurement said
            # the face masker was innocent of the exact damage it was doing.
            fpx = info.get("face_px")
            if fpx is not None:
                fsub = fpx[y0:y1, x0:x1]
                fa = float(fsub.mean()) if fsub.size else 0.0
            else:
                fa = 0.0
                for f in info.get("faces_covered") or info.get("faces") or []:
                    ix = max(0, min(x1, f[2]) - max(x0, f[0]))
                    iy = max(0, min(y1, f[3]) - max(y0, f[1]))
                    fa = max(fa, ix * iy / max(1, (x1 - x0) * (y1 - y0)))
            # How much of the WHOLE FRAME the face masker painted. Hand-box
            # damage misses the rest of the picture, and a detector that
            # mosaics twice the area for the same faces is worse for the
            # downstream model even on frames where no hand is touched.
            fpxf = float(fpx.mean()) if fpx is not None else 0.0
            cov = info.get("faces_covered") or info.get("faces") or []
            big = max([(f[2] - f[0]) * (f[3] - f[1]) / float(W * H) for f in cov],
                      default=0.0)
            rows.append({"face_on_hand": round(fa, 4), "biggest_face": round(big, 4),
                         "face_px_frac": round(fpxf, 5),
                         "n_faces": len(info.get("faces_covered") or info.get("faces") or []),
                         "frame": info["frame"], "box": i,
                         "own": int(info["own"][i]), "p": info["p_owner"][i],
                         "covered": round(float(sub.mean()) if sub.size else 0.0, 4),
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "conf": round(float(d["conf"]), 3),
                         "n_det": len(info["dets"]), "oth_px": info["oth_px"],
                         "veto_px": info["veto_px"]})

    n, dis, nf, _ = demo_video.run(
        rig, vids, a.out, a.start, a.n, a.stride, YOLO(a.weights),
        *own_cnn.load_model(None), 10, 14.0, a.fps,
        face_model=a.face_model or face_mask.MODEL,
        face_conf=face_mask.MIN_CONF if a.face_conf is None else a.face_conf,
        geom=geom_prior.load_model(a.geom), geom_w=geom_w, student=student, ctx=ctx,
        max_owner=None if a.no_cap else 2, panorama_mode="baseline", frame_hook=hook,
        safe_reacquire=not a.inherit_self_on_reacquire, self_reconfirm=a.self_reconfirm,
        new_hand_grace=a.new_hand_grace, max_face_frac=a.max_face_frac,
        grace_log=a.grace_log, assoc_log=a.assoc_log, veto_held=a.veto_held,
        reacquire_edge=a.reacquire_edge, reacquire_log=a.reacquire_log,
        camera=a.camera,
        **({} if a.new_track_conf is None else {"new_track_conf": a.new_track_conf}),
        **({} if a.continue_conf is None else {"continue_conf": a.continue_conf}))
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
