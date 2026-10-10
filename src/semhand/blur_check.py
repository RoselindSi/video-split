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
import json
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


def _load_handness_boxes(path, thr=0.10):
    """-> {frame: {"x0,y0,x1,y1": is_hand}} or None。阈值沿用 post_pass 的 0.10。"""
    if not path:
        return None
    out = collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["frame"])][str(row["box"])] = int(float(row["p"]) >= thr)
    print("  hand-ness（按框）：%d 帧" % len(out))
    return dict(out)


def _load_handness(path):
    """-> {frame: {tid: is_hand}} or None。只取 is_hand，不碰归属。"""
    if not path:
        return None
    out = collections.defaultdict(dict)
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        try:
            out[int(row["frame"])][str(row["tid"])] = int(float(row["is_hand"]))
        except (KeyError, TypeError, ValueError):
            continue
    print("  hand-ness 输入：%d 帧" % len(out))
    return dict(out)


def _load_decisions(path):
    """-> {frame: {tid: (own, side, is_hand)}} or None.

    Keyed on the track id because that is what the pass decided over and what
    the render has in hand; keying on geometry would re-introduce the
    box-matching that has gone wrong here before."""
    if not path:
        return None
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(int(r["frame"]), {})[r["tid"]] = (
            int(r["own"]), r.get("side") or "", int(r.get("is_hand") or 1))
    print(f"  轨迹级决策：{sum(len(v) for v in out.values())} 条，"
          f"覆盖 {len(out)} 帧")
    return out


def _load_owner_gate(path):
    """-> {tid: owner|other|unsure} or None."""
    if not path:
        return None
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        verdict = (r.get("verdict") or "").strip().lower()
        if verdict not in ("owner", "other", "unsure"):
            raise ValueError(
                f"{path}: invalid semantic verdict {verdict!r} for tid "
                f"{r.get('tid')!r}")
        out[str(r["tid"])] = verdict
    print(f"  第二只 owner 语义闸门：{len(out)} 条轨迹")
    return out


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
    ap.add_argument("--no_faces", action="store_true",
                    help="完全不加载人脸检测器。此前没有任何开关能关掉它："
                         "--face_model 默认 None，但下面那行会 `or` 回 "
                         "face_mask.MODEL，于是每一帧都在跑人脸检测 —— 即使"
                         "整条流水线已经不打码。CSV 的 face_* 列会变成 0，"
                         "下游没有任何一处读它们（post_pass / handness_* / "
                         "track_join 都不读），所以这只省时间不改判定。")
    ap.add_argument("--face_verdicts",
                    help="a <rec>.faceverdict.csv: the size cap becomes a "
                         "question instead of a refusal")
    ap.add_argument("--face_pad", type=float, default=None,
                    help="grow each face box by this fraction before mosaicking "
                         "(default 0.35 -> 2.89x the area)")
    ap.add_argument("--face_csv", help="one row per MOSAICKED face box: the "
                    "population the size cap would be applied to")
    ap.add_argument("--handness_models",
                    help="hand-ness 权重的 glob。给了就在这一遍里顺手给每个"
                         "检测框打分，省掉单独一个 stage 重解一次视频")
    ap.add_argument("--handness_out",
                    help="打分写到这个 jsonl，格式和 handness_student_score "
                         "一致，可直接喂下一遍的 --handness_boxes")
    ap.add_argument("--handness_boxes", help="a jsonl with frame,box,p: "
                    "hand-ness keyed on the box, applied before the tracker "
                    "so a non-hand never gets an id at all")
    ap.add_argument("--handness", help="a csv with frame,tid,is_hand: only the "
                    "hand-ness is taken, the ownership is left to this run. "
                    "Use it to give a first pass what --decisions can only "
                    "give a replay -- non-hands out of the two-owner cap "
                    "without replaying the ownership that cap produced")
    ap.add_argument("--decisions", help="a <rec>.decisions.csv from "
                    "`rig.post_pass`: the track-level verdicts replayed into "
                    "this render, so the delivered pixels carry them")
    ap.add_argument("--owner_gate", help="track-level Qwen verdicts from "
                    "`semhand.owner_gate`; every second owner must pass")
    ap.add_argument("--no_mosaic", action="store_true",
                    help="review render only: keep both panels completely raw")
    ap.add_argument("--metadata_only", action="store_true",
                    help="collector mode: write cache/CSV only; skip masks and video")
    ap.add_argument("--detector_batch_size", type=int, default=1,
                    help="number of future frames passed to the detector together")
    ap.add_argument("--max_face_frac", type=float, default=None,
                    help="drop a face box wider than this fraction of the frame")
    ap.add_argument("--gate_frac", type=float, default=None,
                    help="max per-frame hand motion as a fraction of the diagonal")
    ap.add_argument("--owner_detector", help="detector whose classes are "
                                             "owner_hand/other_hand")
    ap.add_argument("--camera", help="read one camera raw (e.g. cam3) instead "
                                    "of the stitched wide render")
    ap.add_argument("--reacquire_log", help="one row per reacquisition")
    ap.add_argument("--reacquire_edge", type=float, default=None,
                    help="1.50 ships; 0.40 is the never-lost value")
    ap.add_argument("--veto_held", action="store_true",
                    help="apply the hand veto to held face boxes too")
    ap.add_argument("--new_track_conf", type=float, default=None,
                    help="score a detection needs to START a track; 0.60 ships, and 56.6%% of a "
                         "colleague's hands never reach it")
    ap.add_argument("--continue_conf", type=float, default=None)
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    cache = ap.add_mutually_exclusive_group()
    cache.add_argument("--track_cache_out",
                       help="collect detection/tracking results into this "
                            "versioned JSONL cache")
    cache.add_argument("--track_cache_in",
                       help="replay this cache without loading or rerunning "
                            "the detector, tracker, or frame owner model")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare, dump=a.dump, gold_paths=a.gold, rec=a.rec)
    if not (a.databag and a.csv and (a.out or a.metadata_only)):
        ap.error("需要 --databag、--csv 和 --out；--metadata_only 可省略 --out")
    if a.metadata_only and not a.track_cache_out:
        ap.error("--metadata_only 只用于 --track_cache_out 采集")
    if a.detector_batch_size < 1:
        ap.error("--detector_batch_size 必须 >= 1")
    import numpy as np
    from src.rig import demo_video, face_mask, track_cache
    from src.rig.calibration import RigCalibration

    vids = {k: os.path.join(a.databag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    calibration = os.path.join(a.databag, "calibration.yaml")
    rig = RigCalibration(calibration)
    replaying = bool(a.track_cache_in)
    student = ctx = None
    geom_w = 0.5 if a.geom_w is None else a.geom_w
    if a.clf_student:
        geom_w = 0.0 if a.geom_w is None else a.geom_w
        if not replaying:
            from src.semhand import student as student_mod
            models, sdev = student_mod.load(a.clf_student)
            if not models:
                raise SystemExit("no checkpoint")
            student = (models, sdev)
        # THE CAP IS NO LONGER TIED TO THE PRIOR. This line used to also set
        # `no_cap`, so choosing the student silently switched off `max_owner`
        # as well, and every delivered render since then has run with no limit
        # on how many hands in one frame could be called the wearer's. On the
        # ten new recordings that is 636 of 3850 frames -- 16.5% -- and on one
        # of them, where a colleague works beside the wearer for the whole
        # clip, 399 of 400. A person has two hands, so at least one box is
        # wrong on every one of those frames.
        #
        # The two were never one decision. `geom_w=0` says the student does
        # not need the fitted geometric prior, which the ablation established.
        # It says nothing about anatomy. Rendered both ways on the recording
        # where the cap buys most and the one where it buys nothing, it moved
        # 587 boxes and 44 boxes out of `owner` respectively, and cost one box
        # of the wearer's own hand. `--no_cap` still turns it off, explicitly.
    if a.clf_ctx and not replaying:
        from src.rig import own_ctx
        m, dev, arm = own_ctx.load_model(a.clf_ctx)
        if m is None:
            raise SystemExit(f"--clf_ctx {a.clf_ctx} not found")
        ctx = (m, dev)
    if replaying:
        detector = None
        cnn, device = None, None
        geom = None
    else:
        from ultralytics import YOLO
        from src.rig import geom_prior, own_cnn
        detector = YOLO(a.weights)
        cnn, device = own_cnn.load_model(None)
        geom = geom_prior.load_model(a.geom)

    import glob

    model_files = {}
    for label, pattern in (("detector", a.weights),
                           ("student", a.clf_student),
                           ("context", a.clf_ctx),
                           ("geometry", a.geom),
                           ("owner_detector", a.owner_detector),
                           ("face", a.face_model or face_mask.MODEL),
                           ("face_fallback", face_mask.MODEL_FALLBACK)):
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern)) or [pattern]
        for index, path in enumerate(matches):
            model_files[f"{label}:{index}"] = path
    cache_meta = {
        "sources": track_cache.source_manifest(
            {**vids, "calibration": calibration}),
        "model_arguments": {
            "weights": a.weights,
            "clf_student": a.clf_student,
            "clf_ctx": a.clf_ctx,
            "geom": a.geom,
            "owner_detector": a.owner_detector,
        },
    }
    # Record the concrete model files in the collector for provenance, but do
    # not require them to remain installed during replay. The cache contract
    # still checks the requested model arguments; replay never imports or
    # opens the detector/classifier checkpoints.
    if not replaying:
        cache_meta["models"] = track_cache.source_manifest(model_files)
    rows = []
    face_rows = []

    def hook(k, clean, sup, info=None):
        if info is None:
            return
        diff = np.abs(sup.astype(np.int16) - clean.astype(np.int16)).max(2) > 8
        H, W = diff.shape
        # ONE ROW PER BOX THAT GOT A MOSAIC, hands or no hands in the frame.
        # The per-hand rows below are written only where a hand was delivered,
        # so a frame mosaicked from edge to edge with no hand in it leaves no
        # trace in them at all -- and the size cap has to be decided on every
        # box it would refuse, not on the ones that happened to share a frame
        # with a hand.
        det_now = list(zip(info.get("faces") or [], info.get("faces_conf") or []))
        for f in info.get("faces_covered") or []:
            best, bc = 0.0, -1.0
            for g, c in det_now:
                ix = max(0, min(f[2], g[2]) - max(f[0], g[0]))
                iy = max(0, min(f[3], g[3]) - max(f[1], g[1]))
                inter = ix * iy
                union = ((f[2] - f[0]) * (f[3] - f[1])
                         + (g[2] - g[0]) * (g[3] - g[1]) - inter)
                v = inter / union if union > 0 else 0.0
                if v > best:
                    best, bc = v, c
            face_rows.append({
                "frame": info["frame"],
                "x0": f[0], "y0": f[1], "x1": f[2], "y1": f[3],
                "w_frac": round((f[2] - f[0]) / float(W), 5),
                "h_frac": round((f[3] - f[1]) / float(H), 5),
                "area_frac": round((f[2] - f[0]) * (f[3] - f[1]) / float(W * H), 5),
                # A held box has no detection behind it this frame, which is
                # the difference between "the detector says so now" and "the
                # detector said so up to twelve frames ago".
                "from_det": int(best >= 0.5),
                "conf": round(bc, 3) if best >= 0.5 else "",
                "n_covered": len(info.get("faces_covered") or []),
                "face_px_frac": round(float(info["face_px"].mean()), 5)
                                if info.get("face_px") is not None else ""})
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
                         "not_hand": int((info.get("not_hand") or [False] * len(info["dets"]))[i]),
                         "covered": round(float(sub.mean()) if sub.size else 0.0, 4),
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "conf": round(float(d["conf"]), 3),
                         "side": d.get("side") or "",
                         "tid": "" if d.get("tid") is None else d["tid"],
                         "n_det": len(info["dets"]), "oth_px": info["oth_px"],
                         "veto_px": info["veto_px"]})

    # FUSED HAND-NESS. Loaded here rather than in its own stage because the
    # expensive part of that stage was decoding frames this pass decodes
    # anyway -- 744 of 2326 seconds on ten recordings. Skipped when replaying
    # a track cache: there are no fresh detections to score, and the cache was
    # built from a pass that already scored them.
    handness_models = handness_sink = None
    handness_rec = os.path.splitext(os.path.basename(a.csv or "rec"))[0]
    if a.handness_models and a.handness_out and not replaying:
        from src.semhand import student as student_mod
        hmodels, hdev = student_mod.load(a.handness_models)
        if not hmodels:
            raise SystemExit("no hand-ness checkpoint: %s" % a.handness_models)
        handness_models = (hmodels, hdev)
        handness_sink = open(a.handness_out, "w", encoding="utf-8")

    try:
        n, dis, nf, _ = demo_video.run(
            rig, vids, None if a.metadata_only else a.out,
            a.start, a.n, a.stride, detector,
            cnn, device, 10, 14.0, a.fps,
            face_model=(None if a.no_faces
                        else (a.face_model or face_mask.MODEL)),
            face_conf=(face_mask.MIN_CONF if a.face_conf is None
                       else a.face_conf),
            geom=geom, geom_w=geom_w, student=student, ctx=ctx,
            max_owner=None if a.no_cap else 2, panorama_mode="baseline",
            frame_hook=hook,
            safe_reacquire=not a.inherit_self_on_reacquire,
            self_reconfirm=a.self_reconfirm,
            new_hand_grace=a.new_hand_grace, max_face_frac=a.max_face_frac,
            face_pad=a.face_pad,
            face_verdicts=demo_video.load_face_verdicts(a.face_verdicts),
            grace_log=a.grace_log, assoc_log=a.assoc_log,
            veto_held=a.veto_held,
            reacquire_edge=a.reacquire_edge, reacquire_log=a.reacquire_log,
            camera=a.camera, owner_detector=a.owner_detector,
            decisions=_load_decisions(a.decisions),
            handness=_load_handness(a.handness),
            handness_boxes=_load_handness_boxes(a.handness_boxes),
            owner_gate=_load_owner_gate(a.owner_gate),
            no_mosaic=a.no_mosaic,
            track_cache_in=a.track_cache_in,
            track_cache_out=a.track_cache_out,
            track_cache_meta=cache_meta,
            metadata_only=a.metadata_only,
            detector_batch_size=a.detector_batch_size,
            gate_frac=a.gate_frac,
            handness_models=handness_models,
            handness_sink=handness_sink,
            handness_rec=handness_rec,
            **({} if a.new_track_conf is None
               else {"new_track_conf": a.new_track_conf}),
            **({} if a.continue_conf is None
               else {"continue_conf": a.continue_conf}))
    finally:
        if handness_sink is not None:
            handness_sink.close()
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.face_csv and face_rows:
        with open(a.face_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(face_rows[0]))
            w.writeheader()
            w.writerows(face_rows)
        print(f"-> {a.face_csv} ({len(face_rows)} 个打了码的人脸框)")
    own = [r for r in rows if r["own"]]
    hit = [r for r in own if r["covered"] >= 0.15]
    print(f"\n{n} 帧、{len(rows)} 只手；被判为自己的手 {len(own)} 只，"
          f"其中被盖住 >=15% 的 {len(hit)} 只（{len(hit) / max(1, len(own)):.1%}），"
          f"涉及 {len({r['frame'] for r in hit})} 帧")
    print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
