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

import numpy as np

BAR_H = 34
GREEN = (60, 220, 60)
RED = (60, 60, 240)
AMBER = (40, 190, 250)
NEW_TRACK_CONF = 0.60
CONTINUE_TRACK_CONF = 0.25
MAX_PREDICTION_AGE = 2


def _bar(width, text, height=BAR_H, bg=(28, 28, 30), fg=(235, 235, 235)):
    import cv2
    b = np.full((height, width, 3), bg, np.uint8)
    cv2.putText(b, text, (14, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, fg, 1, cv2.LINE_AA)
    return b


def annotate(rgb, dets, own_flags, m_oth=None):
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
        if bool(d.get("rule_owner")) != bool(is_own):
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


def run(rig, videos, out_path, start, n, stride, model, cnn, device,
        dilate, sigma, fps, verbose=True, face_model=None, face_conf=None,
        trace_path=None, geom=None, geom_w=0.5, max_owner=None,
        max_prediction_age=MAX_PREDICTION_AGE,
        new_track_conf=NEW_TRACK_CONF,
        continue_conf=CONTINUE_TRACK_CONF,
        predict_motion=True, safe_association=True, safe_reacquire=True,
        min_conf=None, bridge=None, panorama_mode="baseline",
        panorama_fit_frames=0, panorama_depth=True, panorama_flow=False,
        ctx=None, frame_hook=None):
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect, masks_from, OwnHold
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
    hold = face_mask.Hold()
    # Without a fitted prior this falls back to the single exit-height rule,
    # which is what every render before this one used.
    tracker = Tracker(
        max_lost=max(MAX_LOST, max_prediction_age),
        predict_motion=predict_motion,
        rich_association=safe_association,
        max_assoc_cost=MAX_ASSOC_COST if safe_association else None,
        unmatched_cost=UNMATCHED_COST if safe_association else None)
    ownhold = OwnHold(geom=geom, geom_w=geom_w, max_owner=max_owner,
                      state_ttl=tracker.max_lost)
    flips = FlipCount()
    # Three quantities the architecture argument turns on and
    # nothing has ever recorded. They decide nothing.
    runs = LowConfRuns()
    frag = Fragmentation()
    n_dup = n_dropped = n_demoted = 0
    vcam = VirtualWideCamera.from_rig(rig)
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
    rd = Prefetch(ClipReader(rig, videos, start), skip=max(0, stride - 1))
    mc, writer = {}, None
    n_predicted = 0
    trace = [] if trace_path else None
    n_dis = n_written = n_face = 0
    t0 = time.time()
    for k in range(n):
        # The stride is the prefetcher's now: it applies skip=0 to the first
        # frame and the stride thereafter, on its own thread.
        src = rd.next()
        if not src:
            break
        pano_stats = {}
        if panorama is not None:
            rgb, _, pano_stats, _ = panorama.render(src)
        else:
            try:
                rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
            except TypeError:
                rgb, _, _, _ = render(rig, vcam, src, 0.6)
        # EVERY DECISION IS MADE ON THE UNTOUCHED FRAME. The detector and the
        # ownership classifier both read `clean`; only the panels are drawn on
        # the covered copy. Classifying on the mosaic would let the privacy
        # step corrupt the crop the verdict is read from -- and a colleague's
        # face sits directly above a colleague's hands, so that is precisely
        # where the two would collide.
        clean = rgb
        raw_dets = detect(model, clean, min_conf=continue_conf)
        raw_ids = tracker.update(raw_dets, rgb.shape,
                                 new_track_conf=new_track_conf,
                                 continue_conf=continue_conf)
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
        if fdet is not None:
            faces = face_mask.detect_faces(fdet, clean)
            # The hand detector runs first for a reason: it is the better
            # instrument for deciding whether a patch of skin is a hand, and
            # the face detector fires on skin. Watched back, this is what was
            # mosaicking the wearer's own hands.
            # Low-score unmatched candidates still veto a face false positive:
            # failure to start a hand track does not turn that patch into a
            # plausible face.
            faces = face_mask.drop_on_hands(faces, raw_dets)
            n_face += len(faces)
            rgb, _ = face_mask.cover(clean, hold.update(faces,
                                                        shape=clean.shape))
        if ctx is not None:
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
        flips.update(tids, [o for o, _ in flags])
        demoted = list(ownhold.last_demoted)
        n_demoted += len(demoted)
        own = [d for d, (o, _) in zip(dets, flags) if o]
        oth = [d for d, (o, _) in zip(dets, flags) if not o]
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
                    "dets": [{"box": [int(v) for v in d["box"]],
                              "conf": float(d.get("conf", 1.0))}
                             for d in dets],
                    "raw_dets": [{"box": [int(v) for v in d["box"]],
                                  "conf": float(d.get("conf", 1.0))}
                                 for d in raw_dets],
                    "own": [bool(o) for o, _ in flags],
                    "p_owner": [round(float(p), 3) for _, p in flags],
                    "faces": [[int(v) for v in f[:4]] for f in faces]
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
        panel = compose(rgb, annotate(rgb, dets, flags, m_oth), sup,
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
    if trace:
        import csv
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)
        _report_trace(trace, trace_path)
    return n_written, n_dis, n_face


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
    ap.add_argument("--max_owner", type=int, default=2,
                    help="most hands one frame may call the wearer's")
    ap.add_argument("--no_cap", action="store_true",
                    help="lift the two-hand cap")
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
    n, dis, nf = run(rig, vids, a.out, a.start, a.n, a.stride,
                     YOLO(a.weights), cnn, device, a.dilate, a.sigma, a.fps,
                     face_model=None if a.no_faces else a.face_model,
                     face_conf=a.face_conf, trace_path=a.trace,
                     geom=geom, geom_w=a.geom_w,
                     max_owner=None if a.no_cap else a.max_owner,
                     max_prediction_age=a.max_prediction_age,
                     new_track_conf=a.new_track_conf,
                     continue_conf=a.continue_conf,
                     predict_motion=not a.no_motion_prediction,
                     safe_association=not a.legacy_association,
                     safe_reacquire=not a.inherit_self_on_reacquire,
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
