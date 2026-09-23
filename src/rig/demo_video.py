"""A before-and-after clip of the hand filter, straight from a databag.

Made to be watched, not measured. The panels are stacked rather than placed
side by side because the wide render is already three times wider than it is
tall; two of them abreast would be six-to-one and unreadable on any screen.

WHAT THE TOP PANEL SHOWS is every detection, boxed and named: green for the
wearer's, red for a colleague's, with the classifier's confidence. That is the
decision being demonstrated, and it has to be visible for the bottom panel to
mean anything -- otherwise a viewer cannot tell a correct blur from a lucky
one.

WHAT THE BOTTOM PANEL SHOWS is the frame that leaves the pipeline. No boxes,
no overlay: exactly the pixels a downstream model would receive.

OWNERSHIP COMES FROM THE CNN WHEN A CHECKPOINT IS GIVEN, and the rule's
verdict is drawn beside it whenever they differ. A demo that quietly falls
back to the rule would show the incumbent's behaviour while claiming to show
the replacement, and on a well-behaved clip the two are indistinguishable --
the whole difference lives in the cases the rule gets wrong.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BAR_H = 34
GREEN = (60, 220, 60)
RED = (60, 60, 240)
AMBER = (40, 190, 250)
NEW_TRACK_CONF = 0.60
CONTINUE_TRACK_CONF = 0.25
MAX_PREDICTION_AGE = 2
# Frames a REACQUIRED hand must support `self` before the cover comes off it.
# It was 2, which covers the frame a dropped hand returns on WHATEVER the
# classifier says -- and the detector drops a hand for a frame constantly, 63
# times in one 400-frame clip, so the wearer's hands blinked under the cover
# about one and a half times a second. Over 145 recordings that rule kept a
# colleague's hand covered on 103 frames and covered the wearer's own on 1,173.
#
# At 1 the history is still thrown away on a reacquisition -- a returning box
# inherits no verdict -- but the frame it returns on is judged on its own
# evidence. On four held-out recordings that takes own-hand flips from 45 to 2
# and leaves 2 more of a colleague's frames uncovered; on the clip this was
# first seen on, from 20 to 0 and none.
SELF_RECONFIRM_FRAMES = 1

# Track age up to which a box touching a hand already called the wearer's is
# left uncovered: the detector splitting one hand into a palm and a set of
# fingers, where the second box cannot have the hand's id and so has no
# history to be judged by. Over 145 recordings this keeps 144 own hand-frames
# and exposes 17 of a colleague's; without the adjacency half of the test it
# would be 144 against 1,130, which is why it is not simply "new tracks are
# not covered".
NEW_HAND_GRACE = 2


def load_face_verdicts(path):
    """-> {(x0,y0,x1,y1): bool} or None.

    Keyed on the box because a held box keeps the coordinates of the detection
    that created it, so one entry covers every frame the hold lasts."""
    if not path:
        return None
    import csv
    if not os.path.exists(path):
        # `face_verdicts` writes a file for every recording, so a missing one
        # is a wrong path and not an empty answer. Guessing between the two is
        # how a privacy switch ends up off without anyone deciding it.
        raise FileNotFoundError(
            "%s: no verdicts here. `semhand.face_verdicts` writes one file "
            "per recording, empty when nothing exceeds the cap." % path)
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[tuple(int(r[c]) for c in ("x0", "y0", "x1", "y1"))] = \
            r["is_face"] == "1"
    n_drop = sum(1 for v in out.values() if not v)
    print(f"  人脸第二意见：{len(out)} 个框，其中判为不是脸的 {n_drop} 个")
    return out


def _bar(width, text, height=BAR_H, bg=(28, 28, 30), fg=(235, 235, 235)):
    import cv2
    b = np.full((height, width, 3), bg, np.uint8)
    cv2.putText(b, text, (14, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, fg, 1, cv2.LINE_AA)
    return b


def annotate(rgb, dets, own_flags, m_oth=None, kept=()):
    """Top panel: the decision, drawn. -> BGR image

    The region bound for suppression is tinted here rather than only blurred
    below. A blur is a weak thing to look for on a small dark sleeve, and a
    viewer who cannot find it has no way to tell a hand that was cleared from
    a hand that was missed. The tint is on the DECISION panel only; the output
    panel stays the real output, so nothing on screen claims the downstream
    model receives a red arm."""
    import cv2
    vis = rgb.copy()
    if m_oth is not None and m_oth.any():
        vis[m_oth] = (0.45 * vis[m_oth]
                      + 0.55 * np.array(RED, np.float32)).astype(np.uint8)
    for d, (is_own, p) in zip(dets, own_flags):
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        col = GREEN if is_own else RED
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
        det_conf = d.get("conf")
        lab = f"{'self' if is_own else 'other'} own {p:.2f}"
        if det_conf is not None:
            lab += f" det {float(det_conf):.2f}"
        if any(d is k for k in kept):
            # Called foreign and NOT covered, because it is a new box against a
            # hand already called the wearer's. Named on the panel: a spared box
            # and a box nobody proposed look identical in the output otherwise.
            lab += "  kept (new, beside self)"
            col = AMBER
        elif bool(d.get("rule_owner")) != bool(is_own):
            # The only frames worth arguing about. Marked so a viewer can
            # find them instead of taking the agreement on trust.
            lab += "  != rule"
            col = AMBER
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x0, max(0, y0 - th - 8)), (x0 + tw + 8, y0), col,
                      -1)
        cv2.putText(vis, lab, (x0 + 4, max(11, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)
    return vis


def compose(rgb, vis, out, n_own, n_oth, frame, disagreed, frac=None,
            cfg=None):
    """`frac` is the share of pixels actually suppressed.

    A frame where nothing was blurred is the failure that hides best: it looks
    exactly like a frame that never had a colleague in it. Saying so on the
    bar costs nothing and makes the two readable apart."""
    import cv2
    W = rgb.shape[1]
    # THE CONFIGURATION IS STAMPED ON EVERY FRAME. Two renders of the same
    # clip under different switches are indistinguishable once the files are
    # copied off the machine, and an A/B nobody can tell apart is not an A/B.
    top = _bar(W, (f"{cfg}   " if cfg else "")
               + f"frame {frame}      self {n_own}   other {n_oth}"
               + ("   [classifier disagrees with the rule]" if disagreed
                  else ""))
    # The status leads. Appended to a label it is the first thing a narrow
    # frame truncates, and it is the only part that changes.
    if frac is None:
        note = "output to the downstream model"
    elif frac <= 0:
        note = "NOTHING SUPPRESSED - no hand here was called foreign"
    else:
        note = f"{frac:.2%} of pixels suppressed - output to the model"
    return np.vstack([top, vis, _bar(W, note), out])


def _report_trace(rows, path):
    """Say which of the three things dropped the cover, per frame.

    A frame counts as a DROP when the previous frame suppressed something and
    this one did not. Each drop is attributed to the first stage that came up
    empty, because they are in series: no `other` label means the cut is never
    asked, and an empty cut means the veto has nothing to cancel.

    THE FIRST STAGE IS THE DETECTOR, AND LEAVING IT OUT MISREAD THE ANSWER.
    With only label/cut/veto to choose from, a frame where the detector simply
    stopped finding the hand fell into `label` -- there was no `other` label,
    after all, because there was no detection to carry one. That reported the
    classifier as the cause of drops it had no part in: on the run that
    prompted this, all three surviving drops had `n_det` FALL at the same
    frame, and one of them had no detections at all. A stage that can be empty
    has to be a category, or its failures are charged to the next one down."""
    drops = {"detector": [], "label": [], "cut": [], "veto": [],
             "unexplained": []}
    covered = [r for r in rows if r["alpha_frac"] > 0]
    for a, b in zip(rows, rows[1:]):
        if not (a["alpha_frac"] > 0 and b["alpha_frac"] <= 0):
            continue
        if b["n_oth"] == 0 and b["n_det"] < a["n_det"]:
            drops["detector"].append(b["frame"])
        elif b["n_oth"] == 0:
            drops["label"].append(b["frame"])
        elif b["oth_px"] == 0:
            drops["cut"].append(b["frame"])
        elif b["veto_px"] >= b["oth_px"]:
            drops["veto"].append(b["frame"])
        else:
            drops["unexplained"].append(b["frame"])
    blind = [r["frame"] for r in rows if r["n_det"] == 0]
    print(f"\n  trace -> {path}")
    print(f"  {len(covered)} of {len(rows)} frames suppressed something; "
          f"{sum(len(v) for v in drops.values())} drops")
    for k, v in drops.items():
        if v:
            print(f"    {k:<12} {len(v):3d}   frames {v[:10]}")
    if blind:
        print(f"    ({len(blind)} of {len(rows)} frames had NO detections at "
              f"all: {blind[:12]})")
    print("  detector = the hand stopped being found, so nothing reached the "
          "classifier.\n  label = it was found and called the wearer's.  "
          "cut = GrabCut returned nothing\n  for a hand still called foreign."
          "  veto = the owner mask covered the whole of\n  it. These are in "
          "series, so each drop is charged to the first one that was\n  "
          "empty -- and the detector is first.")
    return drops


NEAR_SELF = 0.25


def _touches(a, b, slack):
    """True when box `a` overlaps `b` or comes within `slack` pixels of it."""
    dx = max(a[0], b[0]) - min(a[2], b[2])
    dy = max(a[1], b[1]) - min(a[3], b[3])
    return max(dx, dy) <= slack


def _to_h264(path, verbose=True):
    """Re-encode the finished file in place, if ffmpeg is here. -> True if done."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not os.path.exists(path):
        return False
    tmp = path + ".h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-movflags", "+faststart", tmp])
    if r.returncode != 0 or not os.path.exists(tmp):
        if verbose:
            print("  (ffmpeg failed; leaving the MPEG-4 Part 2 file as it is)")
        return False
    os.replace(tmp, path)
    return True


def run(rig, videos, out_path, start, n, stride, model, cnn, device,
        dilate, sigma, fps, verbose=True, face_model=None, face_conf=None,
        trace_path=None, geom=None, geom_w=0.5, max_owner=None, student=None,
        max_prediction_age=MAX_PREDICTION_AGE,
        new_track_conf=NEW_TRACK_CONF,
        continue_conf=CONTINUE_TRACK_CONF,
        predict_motion=True, safe_association=True, safe_reacquire=True,
        min_conf=None, bridge=None, panorama_mode="baseline",
        panorama_fit_frames=0, panorama_depth=True, panorama_flow=False,
        ctx=None, frame_hook=None, self_reconfirm=SELF_RECONFIRM_FRAMES,
        new_hand_grace=NEW_HAND_GRACE, max_face_frac=None, grace_log=None,
        assoc_log=None, veto_held=False, reacquire_edge=None,
        reacquire_log=None, camera=None, owner_detector=None,
        gate_frac=None, face_pad=None, decisions=None,
        face_verdicts=None):
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch, RawCameraReader
    from src.rig.hand_detect import (detect, owner_detect, masks_from,
                                     OwnHold)
    from src.rig.hand_track import (Tracker, FlipCount, MAX_LOST,
                                    duplicate_pairs, LowConfRuns,
                                    Fragmentation,
                                    MAX_ASSOC_COST, UNMATCHED_COST)
    from src.rig.suppress_other import suppress
    from src.rig import own_cnn

    from src.rig import face_mask

    if bridge is not None:
        # Compatibility with commands written before the policy acquired its
        # real name. Prediction says where; this age says for how long.
        max_prediction_age = int(bridge)
    if min_conf is not None:
        # Deprecated single-floor alias. ``is not None`` is intentional:
        # min_conf=0 must reach the detector rather than restoring 0.60.
        continue_conf = float(min_conf)
    cfg = ("[" + ("motion" if predict_motion else "no-motion")
           + ("+assoc" if safe_association else "")
           + ("+reacq" if safe_reacquire else "")
           + f"+pred{int(max_prediction_age)}"
           + f"+T{float(new_track_conf):.2f}/{float(continue_conf):.2f}"
           + (f"+cap{max_owner}" if max_owner else "")
           + "]")
    max_prediction_age = max(0, int(max_prediction_age))
    new_track_conf, continue_conf = (float(new_track_conf),
                                     float(continue_conf))
    if not 0.0 <= continue_conf <= new_track_conf <= 1.0:
        raise ValueError("need 0 <= continue_conf <= new_track_conf <= 1")

    fdet = face_mask.load_detector(face_model, face_conf) if face_model \
        else None
    hold = face_mask.Hold(max_frac=max_face_frac
                          if max_face_frac is not None else face_mask.MAX_FACE_FRAC,
                          verdicts=face_verdicts)
    # Without a fitted prior this falls back to the single exit-height rule,
    # which is what every render before this one used.
    tracker = Tracker(
        max_lost=max(MAX_LOST, max_prediction_age),
        predict_motion=predict_motion,
        rich_association=safe_association,
        max_assoc_cost=MAX_ASSOC_COST if safe_association else None,
        unmatched_cost=UNMATCHED_COST if safe_association else None,
        **({} if gate_frac is None else {"gate_frac": gate_frac}),
        **({} if reacquire_edge is None else {"reacquire_edge": reacquire_edge}))
    # How many frames a REACQUIRED hand must support `self` before the cover
    # comes off it. The shipped 2 means the frame a dropped hand returns is
    # covered whatever the classifier says; 1 keeps the history reset and
    # judges that frame on its own evidence.
    ownhold = OwnHold(geom=geom, geom_w=geom_w, max_owner=max_owner,
                      state_ttl=tracker.max_lost,
                      self_reconfirm_frames=self_reconfirm)
    flips = FlipCount()
    # Three quantities the architecture argument turns on and
    # nothing has ever recorded. They decide nothing.
    runs = LowConfRuns()
    frag = Fragmentation()
    n_dup = n_dropped = n_demoted = n_kept_new = 0
    owner_model = None
    if owner_detector:
        # A detector that names ownership itself replaces four stages: the
        # hand detector, the distilled student, OwnHold's smoothing and the
        # owner cap all exist to turn a box into a verdict, and this arrives
        # with one. The smoothing still runs -- a per-frame verdict still
        # flickers -- but it is now smoothing the detector's opinion rather
        # than a separate classifier's.
        from ultralytics import YOLO as _YOLO
        owner_model = _YOLO(owner_detector)
        if verbose:
            print(f"  归属来自检测器本身：{os.path.basename(owner_detector)} "
                  f"{owner_model.names}")
    # ONE CAMERA, UNRENDERED, when a camera is named. The wide render is what
    # every constant downstream was fitted to -- the association gate is a
    # fraction of its diagonal, the face size cap a fraction of its width, the
    # ownership student was distilled on its field of view -- so reading a
    # camera raw is a DIFFERENT INPUT, not a different resolution. It gets its
    # own reader rather than a flag inside the renderer, so the two can be run
    # on the same frames and the difference attributed.
    vcam = None if camera else VirtualWideCamera.from_rig(rig)
    panorama = None
    panorama_fit = {}
    if panorama_mode == "depth":
        from src.rig.panorama import DepthAwarePanorama
        panorama = DepthAwarePanorama(
            rig, vcam, use_depth=panorama_depth,
            use_residual_flow=panorama_flow)
        if panorama_fit_frames > 0:
            sample_reader = ClipReader(rig, videos, start)

            def _samples():
                try:
                    for j in range(int(panorama_fit_frames)):
                        got = sample_reader.next(
                            skip=0 if j == 0 else max(0, stride - 1))
                        if not got:
                            break
                        yield got
                finally:
                    sample_reader.close()

            panorama_fit = panorama.fit(_samples())
    elif panorama_mode != "baseline":
        raise ValueError("panorama_mode must be 'baseline' or 'depth'")
    rd = Prefetch(RawCameraReader(videos, camera, start) if camera
                  else ClipReader(rig, videos, start),
                  skip=max(0, stride - 1))
    if camera and verbose:
        print(f"  输入：{camera} 原始帧，不做拼接渲染")
    mc, writer = {}, None
    n_predicted = 0
    trace = [] if trace_path else None
    grace_rows = [] if grace_log else None
    assoc_rows = [] if assoc_log else None
    reacq_rows = [] if reacquire_log else None
    n_dis = n_written = n_face = n_blank = n_overridden = 0
    t0 = time.time()
    for k in range(n):
        # The stride is the prefetcher's now: it applies skip=0 to the first
        # frame and the stride thereafter, on its own thread.
        src = rd.next()
        if src is None:      # end of file. A frame is not falsy; an array has
            break            # no truth value at all, and a dict could be empty

        pano_stats = {}
        if camera:
            rgb = src                       # the camera's own frame, untouched
        elif panorama is not None:
            rgb, _, pano_stats, _ = panorama.render(src)
        else:
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
        # A FRAME THAT DID NOT DECODE IS NOT A FRAME WITH NOTHING IN IT.
        # Under load ffmpeg's scaler fails to allocate -- "Failed
        # initializing scaling graph (Resource temporarily unavailable)" --
        # and hands back a blank picture. The pipeline then finds no hands and
        # no faces in it and records that as fact, so a run that lost 95 of
        # 400 frames to thread exhaustion reported 400 frames, no dropouts,
        # and a lower rate of everything. Counted here and refused at the end:
        # a measurement taken on frames that were never decoded is not a
        # measurement.
        if float(rgb.std()) < 1.0:
            n_blank += 1
        # EVERY DECISION IS MADE ON THE UNTOUCHED FRAME. The detector and the
        # ownership classifier both read `clean`; only the panels are drawn on
        # the covered copy. Classifying on the mosaic would let the privacy
        # step corrupt the crop the verdict is read from -- and a colleague's
        # face sits directly above a colleague's hands, so that is precisely
        # where the two would collide.
        clean = rgb
        face_mask_px = None
        raw_dets = (owner_detect(owner_model, clean, min_conf=continue_conf)
                    if owner_model is not None
                    else detect(model, clean, min_conf=continue_conf))
        raw_ids = tracker.update(raw_dets, rgb.shape,
                                 new_track_conf=new_track_conf,
                                 continue_conf=continue_conf)
        if reacq_rows is not None:
            for e in tracker.reacquire_events:
                reacq_rows.append({
                    "frame": start + k * stride, "tid": e["tid"],
                    "lost": e["lost"], "cost": e["cost"],
                    "fx0": e["from"][0], "fy0": e["from"][1],
                    "fx1": e["from"][2], "fy1": e["from"][3],
                    "tx0": e["to"][0], "ty0": e["to"][1],
                    "tx1": e["to"][2], "ty1": e["to"][3],
                    "terms": ";".join(f"{a}={round(b, 3)}"
                                      for a, b in sorted(e["terms"].items()))})
        if assoc_rows is not None:
            for u in tracker.unmatched:
                assoc_rows.append({
                    "frame": start + k * stride, "tid": u["tid"],
                    "lost": u["lost"], "why": u["why"],
                    "conf": u.get("conf", ""), "iou": u.get("iou", ""),
                    "cost": ("" if not np.isfinite(u.get("cost", np.inf))
                             else round(float(u["cost"]), 4)),
                    "stage": u.get("stage", ""),
                    "taken_by": "" if u.get("taken_by") is None else u["taken_by"],
                    "terms": ";".join(f"{k}={v}" for k, v in
                                      sorted(u.get("terms", {}).items()))})
        keep_i = [i for i, tid in enumerate(raw_ids) if tid is not None]
        # A detection with no id is not passed on: no box, no classification,
        # no cover. Measure the wait that creates before changing it.
        dup = duplicate_pairs(raw_dets)
        dropped_boxes = [raw_dets[i]["box"] for i, tid in enumerate(raw_ids)
                         if tid is None]
        runs.update(k, raw_dets, raw_ids, tracker.new_ids)
        frag.update(k, raw_dets, raw_ids)
        n_dup += len(dup)
        n_dropped += len(dropped_boxes)
        dets = [raw_dets[i] for i in keep_i]
        tids = [raw_ids[i] for i in keep_i]
        provenance = [tracker.provenance[i] for i in keep_i]
        not_hand = set()
        if fdet is not None:
            faces = face_mask.detect_faces(fdet, clean)
            # The hand detector runs first for a reason: it is the better
            # instrument for deciding whether a patch of skin is a hand, and
            # the face detector fires on skin. Watched back, this is what was
            # mosaicking the wearer's own hands.
            # Low-score unmatched candidates still veto a face false positive:
            # failure to start a hand track does not turn that patch into a
            # plausible face.
            faces, faces_vetoed = face_mask.split_on_hands(faces, raw_dets)
            n_face += len(faces)
            # WHAT IS COVERED IS THE HELD LIST, NOT THIS FRAME'S DETECTIONS. A
            # face keeps its mosaic for HOLD_FRAMES after the detector stops
            # proposing it, so a frame with no detection at all can still have
            # a large region mosaicked -- and an audit that records the
            # proposals reads those frames as "nothing was covered here",
            # which is how a face box was first mistaken for not being the
            # thing that destroyed the wearer's hand.
            faces_covered = hold.update(faces, shape=clean.shape)
            if veto_held:
                # THE HOLD OUTLIVES THE VETO. `split_on_hands` filters this
                # frame's proposals, but a box already in the hold keeps its
                # mosaic for HOLD_FRAMES whatever arrives underneath it, so a
                # hand moving into a face box that was admitted while the
                # bench was clear is mosaicked and nothing stops it. That is
                # how one false box held a 59%-of-frame mosaic over the
                # wearer's hand for 17 frames. Off by default: dropping a
                # held box also uncovers a REAL face for as long as a hand
                # passes in front of it, which is a privacy cost and has to be
                # measured, not assumed.
                faces_covered, _ = face_mask.split_on_hands(
                    [tuple(f) + (1.0,) for f in faces_covered], raw_dets)
                faces_covered = [tuple(f[:4]) for f in faces_covered]
            # THE PAD IS THE MASK'S AREA, SQUARED. Every box is grown by this
            # fraction of its own size on each side before the mosaic goes on,
            # so 0.35 multiplies the covered area by 1.7^2 = 2.89 -- and on a
            # frame with a dozen faces that factor is the difference between a
            # dozen patches and a mosaicked picture. 120 judged frames put the
            # wearer's forearm under the mosaic on 9, and all 9 sit above 10%
            # of the frame while none of the 111 clean ones do, so this is the
            # dial the arm damage is on. Per-run rather than edited in place:
            # shrinking it uncovers the jaw and hairline the pad exists for,
            # which is a privacy cost that has to be watched, not assumed.
            rgb, face_mask_px = face_mask.cover(
                clean, faces_covered,
                pad=face_mask.PAD if face_pad is None else float(face_pad))
        if student is not None:
            # THE DISTILLED STUDENT. It reads the whole frame with the box and
            # a zoom on it, so it gets the same `clean` frame the render
            # produced and nothing is cut for it here. Its prior weight is 0
            # and its cap is off (see `semhand.distil_ablate`); both are set
            # where the run is configured, not here.
            from src.semhand import student as student_mod
            flags = student_mod.predict(student[0], student[1], clean, dets)
        elif ctx is not None:
            # HAND + SURROUNDING WINDOW + GEOMETRY, one head. The hand-only
            # classifier recalls 0.644 of foreign hands on the frozen set and
            # this recalls 0.83 at equal or better precision, having trained
            # on 1014 hands against its 2533: what was missing was the
            # forearm and where it goes, not more examples of hands.
            from src.rig import own_ctx
            flags = own_ctx.predict(ctx[0], ctx[1], clean, dets)
        elif cnn is not None:
            flags = own_cnn.predict(cnn, device, clean, dets)
        else:
            flags = [(bool(d.get("owner")), float(d.get("owner_p", 1.0)))
                     for d in dets]
        # One frame's doubt is a flicker; the trace charged every dropped
        # cover to the label and none to the cut or the veto.
        # Identity first, ownership second. Anything accumulated per hand is
        # keyed on the track id, so a hand the tracker calls new starts from
        # its own score instead of inheriting a departed hand's verdict.
        flags = ownhold.update(
            dets, flags, shape=rgb.shape, ids=tids,
            reacquired=tracker.reacquired if safe_reacquire else ())
        if decisions is not None:
            # THE TRACK-LEVEL DECISIONS, APPLIED WHERE THE FRAME-LEVEL ONES
            # END. Whether a box is a hand, whose it is and which hand it is
            # are three questions that need the whole track, so they are
            # decided by a pass over a finished run and replayed here. The
            # render is deterministic given the configuration, so the track
            # ids match the ones the pass read; a frame whose ids have moved
            # means the two runs are not the same run, and that is worth a
            # crash rather than a silently mismatched mask.
            key = start + k * stride
            row = decisions.get(key)
            if row is not None:
                # Track ids are ints here and text in the CSV. Comparing them
                # raw made the guard below fire on every frame of a run that
                # matched perfectly -- a type mismatch wearing the costume of
                # a real one.
                ids = [str(t) for t in tids]
                missing = [t for t in ids if t not in row]
                if missing:
                    raise RuntimeError(
                        "frame %d: track ids %r are not in the decisions; "
                        "the run that produced them is not this run"
                        % (key, missing))
                # CHANGES, not states, and counted BEFORE the assignment.
                # The first version counted every box the decisions call
                # foreign -- mostly boxes that were already foreign -- and
                # reported 3,166 overrides on a recording where 314 flags
                # actually moved. Counting after the assignment would have
                # reported zero, which is the same mistake facing the other
                # way.
                n_overridden += sum(1 for t, (o, _p) in zip(ids, flags)
                                    if bool(row[t][0]) != o)
                flags = [(bool(row[t][0]), p) for t, (_o, p) in zip(ids, flags)]
                for d, t in zip(dets, ids):
                    if row[t][1]:
                        d["side"] = row[t][1]
                # THE THIRD STATE, WHICH `own` CANNOT HOLD. A box the pass
                # judged not to be a hand is not the wearer's and is not a
                # person either, so it belongs in neither list: delivering it
                # puts bench clutter in the training stream, and covering it
                # destroys a patch of bench to hide a machine part. Collapsing
                # it into `own=0` on the first attempt mosaicked 287 boxes of
                # parts bin on one recording, which is what a reader watching
                # the output noticed and no counter did.
                not_hand = {i for i, t in enumerate(ids) if row[t][2] == 0}
        flips.update(tids, [o for o, _ in flags])
        demoted = list(ownhold.last_demoted)
        n_demoted += len(demoted)
        own = [d for i, (d, (o, _)) in enumerate(zip(dets, flags))
               if o and i not in not_hand]
        oth = [d for i, (d, (o, _)) in enumerate(zip(dets, flags))
               if not o and i not in not_hand]
        # A SECOND BOX ON A HAND ALREADY FOUND. The detector sometimes puts one
        # box on a palm and another on the fingers above it. The tracker cannot
        # give the extra box the hand's id -- the hand has it -- so it starts a
        # track of its own with no history, and `OwnHold` has one frame's
        # probability to judge it by. On the clip that prompted this it read
        # 0.32 and covered the wearer's own fingers while the palm below stayed
        # sharp. Sparing a box that is BOTH new AND touching a hand already
        # called the wearer's is measured over 145 recordings as 144 own
        # hand-frames kept against 17 frames of a colleague's hand exposed; the
        # same rule without the adjacency test is 70 against 91, which is why
        # it is not "new tracks are not covered".
        # WHO THE ANCHOR WAS, not merely that there was one. The rule leaves a
        # box uncovered because a hand BESIDE it was called the wearer's, so
        # the exemption is worth exactly what that verdict is worth -- and 29
        # audited tracks that are not hands at all (knees, rags, machine
        # parts, blank floor) include several called the wearer's on every
        # frame they lived. An anchor like that would hand out exemptions.
        # Counting spared frames cannot tell the two apart, so each event is
        # logged with BOTH track ids and the anchor can be sent to the same
        # audit as the thing it exempted.
        kept_new = []
        if new_hand_grace and own:
            anchors = [(d, tid, p) for d, tid, (o, p) in zip(dets, tids, flags) if o]
            spare = []
            for d, tid, (o_, p_) in zip(dets, tids, flags):
                if tid is None or o_:
                    continue
                age = tracker.tracks.get(tid, {}).get("age", 99)
                if age > new_hand_grace:
                    continue
                b = [float(v) for v in d["box"]]
                side = max(b[2] - b[0], b[3] - b[1])
                hit = next((a for a in anchors
                            if _touches(b, [float(v) for v in a[0]["box"]],
                                        NEAR_SELF * side)), None)
                if hit is None:
                    continue
                spare.append(d)
                if grace_rows is not None:
                    ad, atid, ap = hit
                    ab = [float(v) for v in ad["box"]]
                    grace_rows.append({
                        "frame": start + k * stride, "tid": tid, "age": age,
                        "p": round(float(p_), 4),
                        "conf": round(float(d.get("score", d.get("conf", 0)) or 0), 4),
                        "x0": int(b[0]), "y0": int(b[1]),
                        "x1": int(b[2]), "y1": int(b[3]),
                        "anchor_tid": atid, "anchor_p": round(float(ap), 4),
                        "ax0": int(ab[0]), "ay0": int(ab[1]),
                        "ax1": int(ab[2]), "ay1": int(ab[3])})
            if spare:
                oth = [d for d in oth if not any(d is s for s in spare)]
                kept_new = spare
                n_kept_new += len(spare)
        # A track the detector lost for a few frames is still a hand. Its box
        # is advanced by the track velocity, then bounded by the prediction
        # horizon. Covering it is the difference between a cover that survives
        # a dropout and one that blinks off:
        # every drop on the clip that prompted this was the detector losing
        # ONE hand for ONE frame while the wearer's two stayed put.
        n_predicted_frame = 0
        for tid, d in tracker.coasting(max_prediction_age):
            st = ownhold.state.get(tid)
            if st is not None and not st[1]:
                oth.append(d)
                n_predicted += 1
                n_predicted_frame += 1
        dis = any(bool(d.get("rule_owner")) != bool(o)
                  for d, (o, _) in zip(dets, flags))
        n_dis += bool(dis)

        # The owner mask exists only to veto overlap with the other mask, so
        # with nothing to suppress it is a segmentation computed and thrown
        # away. On a clip where a colleague is rare that is most frames.
        m_oth = masks_from(clean, oth) if oth \
            else np.zeros(clean.shape[:2], bool)
        m_own = masks_from(clean, own) if (own and m_oth.any()) \
            else np.zeros(clean.shape[:2], bool)
        sup, alpha = suppress(rgb, m_oth, dilate, 4, sigma, protect=m_own)
        if frame_hook is not None:
            # THE SOURCE FRAME AND WHAT THE PIPELINE MADE OF IT, and nothing
            # else. `clean` is untouched; `sup` is the delivered picture with
            # faces and foreign hands covered. No boxes, no scores, no panel:
            # an auditor asked to find what the system MISSED must not be
            # shown what it believes, or they will only check its work.
            #
            # `info` carries what the pipeline believed, for the OTHER job.
            # Attribution is not detection: once a person has named a frame as
            # wrong, the question becomes which stage lost it, and answering
            # that needs the boxes the audit deliberately withheld. A hook
            # that wants only the pictures ignores the third argument.
            try:
                frame_hook(k, clean, sup, {
                    "frame": start + k * stride,
                    # THE DETECTOR ALREADY SAYS WHICH HAND IT IS. Its two
                    # classes are `left` and `right`, the tracker has been
                    # charging SIDE_MISMATCH on them all along, and nothing
                    # downstream recorded it -- so the wearer's two hands were
                    # indistinguishable in every measurement taken so far.
                    # AND THE TRACK ID, without which a chain built by
                    # overlap afterwards cannot be told from the tracker's own
                    # identity: "the post-processing merged two hands" and
                    # "the tracker swapped them" look the same in the output
                    # and are repaired in different places.
                    "dets": [{"box": [int(v) for v in d["box"]],
                              "conf": float(d.get("conf", 1.0)),
                              "side": d.get("side"), "tid": t}
                             for d, t in zip(dets, tids)],
                    "raw_dets": [{"box": [int(v) for v in d["box"]],
                                  "conf": float(d.get("conf", 1.0))}
                                 for d in raw_dets],
                    "own": [bool(o) for o, _ in flags],
                    # Neither delivered nor covered: the third state, so a
                    # reader of the CSV can tell a box that was dropped from
                    # one that was mosaicked.
                    "not_hand": [i in not_hand for i in range(len(dets))],
                    "p_owner": [round(float(p), 3) for _, p in flags],
                    "faces": [[int(v) for v in f[:4]] for f in faces]
                              if fdet is not None else [],
                    # The score each surviving proposal carried. Without it a
                    # covered box cannot be told from the detection that put
                    # it there, and the question "is this big box a face"
                    # cannot be asked of the score at all.
                    "faces_conf": [float(f[4]) if len(f) > 4 else -1.0
                                   for f in faces] if fdet is not None else [],
                    # The boxes the mosaic actually went on, held ones included.
                    "faces_covered": [[int(v) for v in f[:4]] for f in faces_covered]
                                     if fdet is not None else [],
                    # THE PIXELS, NOT THE BOXES. `cover` grows every box by
                    # PAD of its own size before mosaicking, so a measurement
                    # taken against the boxes misses the third of the mask
                    # that lies outside them -- and on the worst run in the
                    # corpus it reported a face mask as touching 0% of the
                    # wearer's hand on five frames where that mask had in
                    # fact destroyed between 67% and 99% of it.
                    "face_px": face_mask_px,
                    # Proposed and then discarded by the hand veto. Without
                    # this a vetoed face looks exactly like a face the
                    # detector never found, and the two need opposite fixes.
                    "faces_vetoed": [[int(v) for v in f[:4]]
                                     for f in faces_vetoed]
                                    if fdet is not None else [],
                    "oth_px": int(m_oth.sum()),
                    "own_px": int(m_own.sum()),
                    "veto_px": int((m_oth & m_own).sum())})
            except TypeError:
                frame_hook(k, clean, sup)
        if trace is not None:
            # Every quantity between "a hand was called foreign" and "pixels
            # were suppressed", so a frame where the cover drops can be
            # attributed instead of guessed at. A stable box with no cover is
            # one of: the label flipped, the cut returned nothing, or the
            # owner mask vetoed it -- and these three columns separate them.
            trace.append({
                "frame": start + k * stride,
                "n_raw_det": len(raw_dets), "n_det": len(dets),
                "panorama": pano_stats.get("renderer", "baseline"),
                "pano_views": int(pano_stats.get("n_views", 3)),
                "pano_depth_coverage": round(float(
                    pano_stats.get("depth_coverage", float("nan"))), 6),
                "pano_gated_frac": round(float(
                    pano_stats.get("gated_frac", float("nan"))), 6),
                "n_dup_pairs": len(dup),
                "n_dropped_no_id": len(dropped_boxes),
                "n_demoted": len(demoted),
                "demoted_p_max": round(max([p for _, p in demoted],
                                           default=float("nan")), 4),
                "n_low_match": sum(p == "matched_low" for p in provenance),
                "n_new_track": sum(p == "new_high" for p in provenance),
                "n_predicted": n_predicted_frame,
                "n_own": len(own), "n_oth": len(oth),
                "oth_px": int(m_oth.sum()), "own_px": int(m_own.sum()),
                "veto_px": int((m_oth & m_own).sum()),
                "alpha_frac": round(float((alpha > 0.5).mean()), 6),
                "det_conf_min": round(min(
                    [float(d.get("conf", 1.0)) for d in dets],
                    default=float("nan")), 4),
                "p_min_oth": round(min([p for (o, p) in flags if not o],
                                       default=float("nan")), 4),
                "p_max_own": round(max([p for (o, p) in flags if o],
                                       default=float("nan")), 4)})
        if out_path is None:
            # A caller that only wants the frames -- the end-to-end auditor --
            # passes no path. Composing and encoding a demo panel it will
            # never look at is the most expensive part of the loop.
            n_written += 1
            if verbose and ((k + 1) % 20 == 0 or k + 1 == n):
                el = time.time() - t0
                print(f"    [{k+1}/{n}] {el:.0f}s, "
                      f"{el/(k+1)*(n-k-1):.0f}s left", flush=True)
            continue
        panel = compose(rgb, annotate(rgb, dets, flags, m_oth, kept=kept_new), sup,
                        len(own), len(oth), start + k * stride, dis,
                        float((alpha > 0.5).mean()), cfg=cfg)
        if writer is None:
            h, w = panel.shape[:2]
            writer = cv2.VideoWriter(out_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(fps), (int(w), int(h)))
            if not writer.isOpened():
                raise SystemExit(f"cannot open {out_path} for writing")
        writer.write(panel)
        n_written += 1
        if verbose and ((k + 1) % 20 == 0 or k + 1 == n):
            el = time.time() - t0
            print(f"    [{k+1}/{n}] {el:.0f}s, "
                  f"{el/(k+1)*(n-k-1):.0f}s left", flush=True)
    rd.close()
    if verbose:
        if new_hand_grace:
            print(f"\n  NEW BOXES SPARED beside a hand already called the wearer's: "
                  f"{n_kept_new} hand-frames (age <= {new_hand_grace}, gap <= "
                  f"{NEAR_SELF:.2f} of the box). These are the detector splitting "
                  f"one hand in two.")
        fr = flips.report()
        print(f"\n  TRACK-LEVEL FLIPS: {fr['flips']} over {fr['tracks']} "
              f"tracks and {fr['hand_frames']} hand-frames "
              f"({fr['flips_per_100_hand_frames']:.1f} per 100)")
        if fr["worst"]:
            print(f"    worst tracks: {fr['worst']}")
        if panorama is not None:
            print(f"    panorama: depth-aware six-view; fit {panorama_fit}")
        print(f"    tracker: {tracker.n_new} tracks started, "
              f"{tracker.n_lost} ended")
        print(f"    predicted covers: {n_predicted} hand-frames, horizon "
              f"{max_prediction_age}")
        print(f"\n  SAME-FRAME DUPLICATE PAIRS: {n_dup} over {n} frames "
              f"(IoU >= 0.70).\n    An upper bound on how much of the flicker "
              f"one hand found twice could\n    explain. Two hands really do "
              f"overlap, so this is not a duplicate count.")
        print(f"\n  CAP DEMOTIONS: {n_demoted} hand-frames were called "
              f"foreign by the two-hand\n    cap rather than by the "
              f"classifier or the prior. Against "
              f"{fr['hand_frames']} hand-frames\n    seen. If this is most "
              f"of the `other` verdicts then the cap is the decision maker.")
        print(f"\n  DETECTIONS DROPPED FOR HAVING NO ID: {n_dropped} over "
              f"{n} frames.")
        for block in (runs.report(fps=fps), frag.report()):
            if block:
                print(block)
        print("  A flip is one hand changing its own verdict between two "
              "frames it was seen\n  in. The frame-level `drop` count above "
              "cannot see a flip that happens\n  while another hand keeps "
              "the frame's suppression non-empty.")
    if writer is not None:
        writer.release()
        # MPEG-4 Part 2 is what OpenCV can write here, and several players --
        # including the one the person reviewing these clips uses -- will not
        # open it. The picture is finished at this point; this only changes
        # the container and codec, in place, and is skipped if ffmpeg is
        # missing rather than failing a render that already succeeded.
        _to_h264(out_path, verbose)
    if trace:
        import csv
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)
        _report_trace(trace, trace_path)
    if reacq_rows is not None:
        import csv
        with open(reacquire_log, "w", newline="") as f:
            cols = ["frame", "tid", "lost", "cost", "fx0", "fy0", "fx1", "fy1",
                    "tx0", "ty0", "tx1", "ty1", "terms"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(reacq_rows)
        if verbose:
            print(f"\n  重新接回 {len(reacq_rows)} 次 -> {reacquire_log}")
    if assoc_rows is not None:
        import csv
        with open(assoc_log, "w", newline="") as f:
            cols = ["frame", "tid", "lost", "why", "conf", "iou", "cost",
                    "stage", "taken_by", "terms"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(assoc_rows)
        if verbose:
            import collections as _c
            why = _c.Counter(r["why"] for r in assoc_rows)
            print(f"\n  活着却没拿到框的 {len(assoc_rows)} 轨迹帧，原因: "
                  f"{dict(why.most_common())} -> {assoc_log}")
    if grace_rows is not None:
        import csv
        with open(grace_log, "w", newline="") as f:
            cols = ["frame", "tid", "age", "p", "conf", "x0", "y0", "x1", "y1",
                    "anchor_tid", "anchor_p", "ax0", "ay0", "ax1", "ay1"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(grace_rows)
        if verbose:
            anc = {r["anchor_tid"] for r in grace_rows}
            print(f"\n  GRACE 放行 {len(grace_rows)} 帧，来自 "
                  f"{len({r['tid'] for r in grace_rows})} 条被放行的轨迹，"
                  f"锚点是 {len(anc)} 条轨迹 -> {grace_log}")
    if n_blank:
        msg = (f"!! {n_blank} / {n_written} 帧解码后是空白（多半是并发太高，"
               f"ffmpeg 分不到线程）。这些帧会被当成『没有手也没有脸』计入，"
               f"任何比率都会被压低 —— 降低并发后重跑。")
        print("\n  " + msg, flush=True)
        if n_blank > 0.02 * max(1, n_written):
            raise RuntimeError(msg)
    # The flip report is computed either way; returning it lets a
    # caller that runs quietly still measure flicker, which is the
    # other end of every trade this pipeline makes against over-blur.
    if decisions is not None:
        print(f"  轨迹级决策：{n_overridden} 个框的归属被改判")
    if face_verdicts is not None:
        # UNJUDGED IS A FAILURE COUNT, NOT A DETAIL. An over-cap box with no
        # verdict is covered, so a verdict table that misses them turns the
        # size cap off without changing a line of it -- which is exactly what
        # happened, and nothing said so until the delivered mask was measured.
        print(f"  人脸第二意见：超过 cap 但没有判读的框 {hold.unjudged} 个"
              + ("（这些被盖住了；不为 0 就说明两次运行对不上）"
                 if hold.unjudged else ""))
    return n_written, n_dis, n_face, flips.report()


def _self_test():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok += bool(cond)

    rgb = np.full((60, 300, 3), 120, np.uint8)
    dets = [{"box": (10, 10, 40, 40), "rule_owner": 1},
            {"box": (200, 15, 230, 45), "rule_owner": 1}]
    flags = [(True, 0.97), (False, 0.03)]
    vis = annotate(rgb, dets, flags)
    chk("annotating does not modify the input", np.array_equal(
        rgb, np.full((60, 300, 3), 120, np.uint8)))
    chk("something was drawn", not np.array_equal(vis, rgb))

    panel = compose(rgb, vis, rgb, 1, 1, 42, True)
    chk("the panel is two frames plus two bars",
        panel.shape == (60 * 2 + BAR_H * 2, 300, 3))
    chk("the output panel is the unannotated frame",
        np.array_equal(panel[-60:], rgb))
    chk("the input panel is the annotated one",
        np.array_equal(panel[BAR_H:BAR_H + 60], vis))

    # The second hand is `other` while the rule called it the wearer's, so the
    # disagreement marker must be present -- that is the only thing on screen
    # that separates the two systems.
    v2 = annotate(rgb, [dets[1]], [(False, 0.03)])
    v3 = annotate(rgb, [{"box": (200, 15, 230, 45), "rule_owner": 0}],
                  [(False, 0.03)])
    chk("a disagreeing hand is drawn differently from an agreeing one",
        not np.array_equal(v2, v3))

    m = np.zeros((60, 300), bool)
    m[20:30, 200:220] = True
    chk("the suppressed region is tinted on the decision panel",
        not np.array_equal(annotate(rgb, dets, flags, m), vis))
    # A frame with nothing suppressed has to read differently from a frame
    # with something suppressed, or the demo cannot show a miss.
    chk("an empty suppression says so",
        not np.array_equal(compose(rgb, vis, rgb, 1, 0, 42, False, 0.0),
                           compose(rgb, vis, rgb, 1, 1, 42, False, 0.05)))
    # The prefetcher stands in for the reader, so it has to hand back exactly
    # the same sequence and apply the stride the same way: nothing skipped
    # before the first frame, the stride before every one after it.
    from src.rig.seam_fix import Prefetch

    class FakeReader:
        def __init__(self, n):
            self.n, self.i, self.skips, self.closed = n, 0, [], False

        def next(self, skip=0):
            self.skips.append(skip)
            if self.i >= self.n:
                return None
            self.i += 1
            return {"f": self.i}

        def close(self):
            self.closed = True

    fr = FakeReader(4)
    pf = Prefetch(fr, skip=3)
    got = [pf.next() for _ in range(5)]
    chk("the prefetcher yields the reader's frames in order",
        [g["f"] for g in got[:4]] == [1, 2, 3, 4])
    chk("and passes end of file through", got[4] is None)
    chk("asking past the end returns None instead of hanging",
        pf.next() is None)
    chk("the first frame skips nothing and the rest skip the stride",
        fr.skips[0] == 0 and set(fr.skips[1:]) == {3})
    pf.close()
    chk("closing it closes the reader underneath", fr.closed)

    print(f"\n  {ok}/13")
    return ok == 13


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    from src.rig import face_mask
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", help="a databag directory; supplies "
                                      "calibration and all three videos")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--out", help="an .mp4 path on the SAN")
    ap.add_argument("--geom", help="a geom_prior JSON. Without it the "
                                   "ownership prior is the single "
                                   "exit-height rule, which the cue scan put "
                                   "ninth of sixteen.")
    ap.add_argument("--geom_w", type=float, default=0.5,
                    help="weight on the geometric prior against the CNN. "
                         "1.0 is geometry alone, which beat the 0.5 blend on "
                         "two of nineteen held-out recordings and tied on "
                         "the rest.")
    ap.add_argument("--max_prediction_age", type=int,
                    default=MAX_PREDICTION_AGE,
                    help="maximum missed frames for which a confirmed other "
                         "track's motion prediction may cover output pixels")
    ap.add_argument("--bridge", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--new_track_conf", type=float,
                    default=NEW_TRACK_CONF,
                    help="high detector threshold: unmatched detections at or "
                         "above it may start tracks")
    ap.add_argument("--continue_conf", type=float,
                    default=CONTINUE_TRACK_CONF,
                    help="low detector threshold: detections below the new "
                         "track threshold may only continue existing tracks")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="deprecated alias for --continue_conf")
    ap.add_argument("--no_motion_prediction", action="store_true")
    ap.add_argument("--legacy_association", action="store_true",
                    help="ablation only: disable rich costs and explicit "
                         "unmatched assignments")
    ap.add_argument("--inherit_self_on_reacquire", action="store_true",
                    help="ablation only: restore the unsafe old ownership hold")
    ap.add_argument("--max_face_frac", type=float, default=None,
                    help="refuse a face box wider than this fraction of the frame. 0.18 "
                         "refuses all 25 boxes inspected over ten recordings, every one a "
                         "false positive, and would also refuse a face 37.5% wide")
    ap.add_argument("--face_verdicts",
                    help="a <rec>.faceverdict.csv from `semhand.face_verdicts`: "
                         "the size cap becomes a question instead of a refusal")
    ap.add_argument("--face_pad", type=float, default=None,
                    help="grow each face box by this fraction of its own size before "
                         "mosaicking (default 0.35, which triples the covered area)")
    ap.add_argument("--new_hand_grace", type=int, default=NEW_HAND_GRACE,
                    help="do not cover a box this new (track age) when it touches a hand "
                         "already called the wearer's: the detector splitting one hand")
    ap.add_argument("--self_reconfirm", type=int, default=SELF_RECONFIRM_FRAMES,
                    help="frames a reacquired hand must support `self` before the "
                         "cover comes off it; 2 covers the frame it returns")
    ap.add_argument("--max_owner", type=int, default=2,
                    help="most hands one frame may call the wearer's")
    ap.add_argument("--no_cap", action="store_true",
                    help="lift the two-hand cap")
    ap.add_argument("--clf_student", metavar="GLOB",
                    help="the distilled student's checkpoints (a glob over its "
                         "seeds). Reads the whole frame plus the hand's box; "
                         "sets --geom_w 0 and --no_cap unless you pass them.")
    ap.add_argument("--clf_ctx",
                    help="an own_ctx checkpoint: ownership from the hand, "
                         "the surrounding window and the geometry together. "
                         "Takes precedence over --clf. On the frozen test "
                         "set this lifted foreign-hand recall from 0.644 to "
                         "0.83 at equal or better precision.")
    ap.add_argument("--clf", help="own_cnn.pt. Without it the demo shows the "
                                  "geometric RULE, which is not the thing "
                                  "being demonstrated.")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=14.0)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    # BASELINE IS THE DEFAULT AGAIN, ON THE STRENGTH OF WATCHING THE OUTPUT.
    # The six-view depth renderer wins the numbers it was built to win: it
    # cut near-black hole area from 7.4-9.7% to 3.0-3.9% by actually reaching
    # all six cameras. It did not fix the thing anyone looks at. Seam ratio
    # stayed above 1.20 through the depth densifier, the guided filter, the
    # two-view blend and the hole fade, and the rendered result was called
    # worse than the old renderer by the person it is being built for.
    #
    # A per-pixel range map is a warp field, and every defect in it is a
    # geometric one -- a ripple, a tear, a speckle -- which reads as broken
    # in a way a visible straight seam does not. The old renderer's constant
    # plane is wrong everywhere by a smooth amount, and smooth-and-wrong
    # survives viewing better than sharp-and-nearly-right.
    #
    # The depth path is kept, not deleted: the black-hole measurement is real
    # and the mode still runs under `--panorama depth`. What is withdrawn is
    # its claim on being the default.
    ap.add_argument("--panorama", choices=("depth", "baseline"),
                    default="baseline",
                    help="baseline is the three-left-eye constant-depth "
                         "renderer and is what ships; depth uses per-pixel "
                         "range and all six RGB views, which closes the black "
                         "holes but has never got the seam ratio under 1.20")
    ap.add_argument("--pano_fit_frames", type=int, default=6,
                    help="synchronized frames used once to fit frozen colour "
                         "and residual-flow corrections")
    ap.add_argument("--no_pano_depth", action="store_true",
                    help="ablation: use one depth plane in the new six-view "
                         "renderer")
    flow = ap.add_mutually_exclusive_group()
    flow.add_argument("--pano_flow", action="store_true",
                      help="experimental: enable content-fitted residual "
                           "alignment (off by default; current ablation "
                           "worsens seam ratio)")
    flow.add_argument("--no_pano_flow", action="store_true",
                      help="compatibility alias; residual flow is already off")
    # On by default. A demo that leaks a colleague's face is not a demo that
    # can be sent anywhere, and defaulting the privacy step off would make
    # that failure the quiet one.
    ap.add_argument("--no_faces", action="store_true",
                    help="do NOT cover faces (they are covered by default)")
    ap.add_argument("--face_model", default=face_mask.MODEL)
    ap.add_argument("--face_conf", type=float, default=face_mask.MIN_CONF)
    ap.add_argument("--trace", help="write a per-frame CSV of every quantity "
                                    "between the label and the suppressed "
                                    "pixels, and attribute each dropout")
    ap.add_argument("--gate_frac", type=float, default=None,
                    help="how far a hand may move between frames, as a "
                         "fraction of the frame diagonal. 0.15 was fitted to "
                         "the wide render; measured on 7,028 real cam3 "
                         "movements the p99 is 0.031 and the largest seen is "
                         "0.088")
    ap.add_argument("--owner_detector", help="a detector whose classes are "
                                            "owner_hand/other_hand; replaces "
                                            "the hand detector AND the "
                                            "ownership classifier")
    ap.add_argument("--camera", help="read ONE camera's raw frames (e.g. cam3) "
                                     "instead of the stitched wide render. "
                                     "Every constant downstream was fitted to "
                                     "the render, so this is a different input")
    ap.add_argument("--reacquire_log", help="one row per reacquisition, with "
                                            "the box the hand was last seen "
                                            "in and the box it was given")
    ap.add_argument("--reacquire_edge", type=float, default=None,
                    help="penalty for a forearm-exit label that changed while "
                         "the track was lost. 1.50 ships; 0.40 is the value "
                         "used when the track was never lost")
    ap.add_argument("--veto_held", action="store_true",
                    help="apply the hand veto to HELD face boxes too, not "
                         "only to this frame's proposals")
    ap.add_argument("--assoc_log", help="write one row per LIVE track that got "
                                        "no detection, with the cheapest candidate "
                                        "and the rule that refused it")
    ap.add_argument("--grace_log", help="write one row per box the new-hand "
                                        "grace left uncovered, WITH the track "
                                        "that anchored the exemption. Counting "
                                        "spared frames cannot say whether the "
                                        "anchor was a hand at all")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig import own_cnn

    if not a.out:
        ap.error("--out is required")
    if a.databag:
        cal = os.path.join(a.databag, "calibration.yaml")
        vids = {k: os.path.join(a.databag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
    else:
        if not a.calibration or not a.video:
            ap.error("give --databag, or --calibration and --video")
        cal, vids = a.calibration, dict(s.split("=", 1) for s in a.video)
    rig = RigCalibration(cal)
    from src.rig import geom_prior
    geom = geom_prior.load_model(a.geom)
    if a.geom and geom is None:
        raise SystemExit(f"--geom {a.geom} not found")
    student = None
    if a.clf_student:
        from src.semhand import student as student_mod
        models, sdev = student_mod.load(a.clf_student)
        if not models:
            raise SystemExit(f"--clf_student {a.clf_student} matched no checkpoint")
        student = (models, sdev)
        # The prior and the cap were fitted for V1's classifier and measured
        # to hurt this one; they go off unless the caller insisted.
        argv = " ".join(sys.argv)
        if "--geom_w" not in argv:
            a.geom_w = 0.0
        if "--max_owner" not in argv:
            a.no_cap = True
        print(f"  ownership by the distilled student ({len(models)} seeds)  "
              f"<- geom_w {a.geom_w}, cap {'off' if a.no_cap else a.max_owner}")
    ctx_model = ctx_arm = None
    if a.clf_ctx:
        from src.rig import own_ctx
        ctx_model, ctx_device, ctx_arm = own_ctx.load_model(a.clf_ctx)
        if ctx_model is None:
            raise SystemExit(f"--clf_ctx {a.clf_ctx} not found")
    cnn, device = own_cnn.load_model(a.clf)
    if a.clf and cnn is None:
        raise SystemExit(f"--clf {a.clf} not found. Refusing to fall back to "
                         f"the rule\n  silently: the demo would show the "
                         f"incumbent under the replacement's name.")
    print(f"  {a.n} frames from {a.start}, stride {a.stride}, {a.fps} fps")
    print(f"  ownership by "
          f"{('hand+context+geometry (' + str(ctx_arm) + ')  <- V1') if ctx_model else ('the hand-only CNN  <- V1 baseline' if cnn else 'the geometric RULE')}"
          f"{'' if cnn else ''}")
    print(f"  prior: {'fitted geometry, ' + str(len(geom['cues'])) + ' cues'
                    if geom else 'the single exit-height rule'}")
    print(f"  detector: new>={a.new_track_conf:.2f}, "
          f"continue>={a.continue_conf if a.min_conf is None else a.min_conf:.2f}; "
          f"prediction age {a.max_prediction_age}")
    print(f"  panorama: {a.panorama}"
          + (f", fit {a.pano_fit_frames} frames, "
             f"depth {'off' if a.no_pano_depth else 'on'}, "
             f"residual flow {'on' if a.pano_flow else 'off'}"
             if a.panorama == "depth" else ""))
    print(f"  faces {'NOT covered' if a.no_faces else 'covered'}"
          f"{'   <- do not send this anywhere' if a.no_faces else ''}")
    n, dis, nf, _fl = run(rig, vids, a.out, a.start, a.n, a.stride,
                     YOLO(a.weights), cnn, device, a.dilate, a.sigma, a.fps,
                     face_model=None if a.no_faces else a.face_model,
                     face_conf=a.face_conf, trace_path=a.trace,
                     geom=geom, geom_w=a.geom_w, student=student,
                     max_owner=None if a.no_cap else a.max_owner,
                     max_prediction_age=a.max_prediction_age,
                     new_track_conf=a.new_track_conf,
                     continue_conf=a.continue_conf,
                     predict_motion=not a.no_motion_prediction,
                     safe_association=not a.legacy_association,
                     safe_reacquire=not a.inherit_self_on_reacquire,
                     self_reconfirm=a.self_reconfirm,
                     new_hand_grace=a.new_hand_grace,
                     max_face_frac=a.max_face_frac, face_pad=a.face_pad,
                     face_verdicts=load_face_verdicts(a.face_verdicts),
                     grace_log=a.grace_log,
                     assoc_log=a.assoc_log, veto_held=a.veto_held,
                     reacquire_edge=a.reacquire_edge,
                     reacquire_log=a.reacquire_log, camera=a.camera,
                     owner_detector=a.owner_detector, gate_frac=a.gate_frac,
                     bridge=a.bridge, min_conf=a.min_conf,
                     panorama_mode=a.panorama,
                     panorama_fit_frames=a.pano_fit_frames,
                     panorama_depth=not a.no_pano_depth,
                     panorama_flow=a.pano_flow and not a.no_pano_flow,
                     ctx=None if ctx_model is None
                     else (ctx_model, ctx_device))
    if not n:
        raise SystemExit("no frames written")
    mb = os.path.getsize(a.out) / 1e6
    print(f"\n  {n} frames -> {a.out}  ({mb:.1f} MB, "
          f"{n/a.fps:.0f}s of video)")
    print(f"  the classifier disagreed with the rule on {dis} of {n} frames "
          f"({dis/n:.1%})")
    if not a.no_faces:
        print(f"  {nf} face detections over {n} frames "
              f"({nf/n:.2f} per frame), held {face_mask.HOLD_FRAMES} frames "
              f"each")
        if nf == 0:
            print("  Zero faces is not the same as nobody present. Check one "
                  "frame that has a\n  person in it before treating this "
                  "clip as safe to send.")
    if dis == 0:
        print("  Zero disagreement means this clip does not distinguish them. "
              "It is a fine\n  demo of the pipeline and no evidence at all "
              "about the classifier -- pick a\n  segment with a colleague in "
              "frame if that is what needs showing.")


if __name__ == "__main__":
    main()
