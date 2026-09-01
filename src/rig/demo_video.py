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


def _bar(width, text, height=BAR_H, bg=(28, 28, 30), fg=(235, 235, 235)):
    import cv2
    b = np.full((height, width, 3), bg, np.uint8)
    cv2.putText(b, text, (14, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, fg, 1, cv2.LINE_AA)
    return b


def annotate(rgb, dets, own_flags):
    """Top panel: the decision, drawn. -> BGR image"""
    import cv2
    vis = rgb.copy()
    for d, (is_own, p) in zip(dets, own_flags):
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        col = GREEN if is_own else RED
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
        lab = f"{'self' if is_own else 'other'} {p:.2f}"
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


def compose(rgb, vis, out, n_own, n_oth, frame, disagreed):
    import cv2
    W = rgb.shape[1]
    top = _bar(W, f"input + decision      frame {frame}      "
                  f"self {n_own}   other {n_oth}"
                  + ("   [classifier disagrees with the rule]" if disagreed
                     else ""))
    bot = _bar(W, "output to the downstream model      "
                  "other arms blurred, the wearer's untouched")
    return np.vstack([top, vis, bot, out])


def run(rig, videos, out_path, start, n, stride, model, cnn, device,
        dilate, sigma, fps, verbose=True):
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.hand_detect import detect, masks_from
    from src.rig.suppress_other import suppress
    from src.rig import own_cnn

    vcam = VirtualWideCamera.from_rig(rig)
    rd = ClipReader(rig, videos, start)
    mc, writer = {}, None
    n_dis = n_written = 0
    t0 = time.time()
    for k in range(n):
        src = rd.next(skip=(stride - 1) if k else 0)
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        dets = detect(model, rgb)
        if cnn is not None:
            flags = own_cnn.predict(cnn, device, rgb, dets)
        else:
            flags = [(bool(d.get("owner")), float(d.get("owner_p", 1.0)))
                     for d in dets]
        own = [d for d, (o, _) in zip(dets, flags) if o]
        oth = [d for d, (o, _) in zip(dets, flags) if not o]
        dis = any(bool(d.get("rule_owner")) != bool(o)
                  for d, (o, _) in zip(dets, flags))
        n_dis += bool(dis)

        m_own = masks_from(rgb, own) if own else np.zeros(rgb.shape[:2], bool)
        m_oth = masks_from(rgb, oth) if oth else np.zeros(rgb.shape[:2], bool)
        sup, _ = suppress(rgb, m_oth, dilate, 4, sigma, protect=m_own)
        panel = compose(rgb, annotate(rgb, dets, flags), sup,
                        len(own), len(oth), start + k * stride, dis)
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
    if writer is not None:
        writer.release()
    return n_written, n_dis


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
    print(f"\n  {ok}/6")
    return ok == 6


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", help="a databag directory; supplies "
                                      "calibration and all three videos")
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--out", help="an .mp4 path on the SAN")
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
    cnn, device = own_cnn.load_model(a.clf)
    if a.clf and cnn is None:
        raise SystemExit(f"--clf {a.clf} not found. Refusing to fall back to "
                         f"the rule\n  silently: the demo would show the "
                         f"incumbent under the replacement's name.")
    print(f"  {a.n} frames from {a.start}, stride {a.stride}, {a.fps} fps")
    print(f"  ownership by {'the CNN' if cnn else 'the geometric RULE'}"
          f"{'' if cnn else '   <- not the shipped path'}")
    n, dis = run(rig, vids, a.out, a.start, a.n, a.stride, YOLO(a.weights),
                 cnn, device, a.dilate, a.sigma, a.fps)
    if not n:
        raise SystemExit("no frames written")
    mb = os.path.getsize(a.out) / 1e6
    print(f"\n  {n} frames -> {a.out}  ({mb:.1f} MB, "
          f"{n/a.fps:.0f}s of video)")
    print(f"  the classifier disagreed with the rule on {dis} of {n} frames "
          f"({dis/n:.1%})")
    if dis == 0:
        print("  Zero disagreement means this clip does not distinguish them. "
              "It is a fine\n  demo of the pipeline and no evidence at all "
              "about the classifier -- pick a\n  segment with a colleague in "
              "frame if that is what needs showing.")


if __name__ == "__main__":
    main()
