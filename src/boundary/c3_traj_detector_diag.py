"""Why does the trajectory extractor detect almost nothing?

The event-centred extractor reported a median per-event detection coverage of
0.000 over its candidate windows. That contradicts what is already known about
the same detector on the same footage: c3_crop_coverage measured 0.87 at
candidate moments and extract_features_local a median of 0.93 per recording.
So the new path is broken, not the data, and the extractor differs from the
working one in exactly TWO ways -- it runs mediapipe in VIDEO mode instead of
IMAGE mode, and it upscales the eye frame before detection.

Two changes, four combinations, one window. Each is run on the SAME decoded
frames so nothing but the named difference varies, and the per-frame hit
pattern is printed rather than only a rate: a detector that finds hands in the
first few frames and then stops has a different problem from one that never
starts, and VIDEO mode's tracking makes that distinction the informative one.

Exceptions are counted and shown. The extractor's first version swallowed them
into a per-frame flag nothing printed, so a detector raising on every call
looked exactly like a detector finding no hands.

Usage:
    python -m src.boundary.c3_traj_detector_diag \
        --video /shared/datasets/human_ego_recording_segmentation_10fps_r01_part_01/recordings/recording_000102/mid.mp4 \
        --t 95.0 --hand_model /workspace/tr1/ckpts/hand_landmarker.task \
        --dump_dir /workspace/tr1/results/hal/c3/trajdiag
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.boundary.extract_hand_trajectory import eye_slice, upscale_frame


def run_variant(frames, model, mode, upscale, fps, max_hands, conf):
    from src.boundary.hand_detect import HandDetector
    det = HandDetector(model, max_hands=max_hands, min_confidence=conf,
                       running_mode=mode)
    hits, exc, first = [], 0, None
    for fi, f in enumerate(frames):
        img = upscale_frame(np.ascontiguousarray(f), upscale)
        try:
            r = det.detect_full(img, timestamp_ms=int(fi * 1000 / fps))
            hits.append(len(r["landmarks"]))
        except Exception as e:
            exc += 1
            hits.append(-1)
            if first is None:
                first = f"{type(e).__name__}: {e}"[:160]
    det.close()
    return hits, exc, first


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--hand_model", required=True)
    ap.add_argument("--eye", default="left")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--max_hands", type=int, default=2)
    ap.add_argument("--min_confidence", type=float, default=0.3)
    ap.add_argument("--dump_dir")
    a = ap.parse_args()

    from decord import VideoReader
    vr = VideoReader(a.video, num_threads=1)
    vfps, n = vr.get_avg_fps(), len(vr)
    nf = int(round(2 * a.window * a.fps)) + 1
    want = a.t + np.linspace(-a.window, a.window, nf)
    idx = np.clip(np.round(want * vfps).astype(int), 0, n - 1)
    frames = eye_slice(vr.get_batch(idx.tolist()).asnumpy(), a.eye)
    print(f"{nf} frames around t={a.t}s, eye frame {frames.shape[2]}x{frames.shape[1]}, "
          f"video {vfps:.1f} fps")

    if a.dump_dir:
        from PIL import Image
        os.makedirs(a.dump_dir, exist_ok=True)
        for k in (0, nf // 2, nf - 1):
            Image.fromarray(np.ascontiguousarray(frames[k])).save(
                os.path.join(a.dump_dir, f"frame_{k:02d}.png"))
        print(f"  wrote 3 sample frames to {a.dump_dir} -- if every variant "
              f"fails, look at these before touching the code again")

    print(f"\n{'variant':<34} {'hits':>5} {'rate':>6} {'exc':>4}  per-frame")
    results = {}
    for mode in ("image", "video"):
        for up in ("none", "auto"):
            name = f"{mode} mode, upscale={up}"
            hits, exc, first = run_variant(frames, a.hand_model, mode, up,
                                           a.fps, a.max_hands, a.min_confidence)
            got = sum(1 for h in hits if h > 0)
            results[name] = {"hits": got, "rate": got / len(hits), "exc": exc,
                             "first_exception": first, "per_frame": hits}
            pat = "".join("." if h == 0 else ("!" if h < 0 else str(min(h, 9)))
                          for h in hits)
            print(f"{name:<34} {got:>5} {got / len(hits):>6.2f} {exc:>4}  {pat}")
            if first:
                print(f"{'':>34}  first exception: {first}")
    print("  per-frame key: digit = hands found, '.' = none, '!' = exception")

    base = results["image mode, upscale=none"]
    cur = results["video mode, upscale=auto"]
    print()
    if base["rate"] > 0.5 and cur["rate"] < 0.2:
        vid = results["video mode, upscale=none"]
        img_up = results["image mode, upscale=auto"]
        culprit = ("VIDEO MODE" if vid["rate"] < 0.2 <= img_up["rate"]
                   else "UPSCALING" if img_up["rate"] < 0.2 <= vid["rate"]
                   else "the COMBINATION (each alone is survivable)")
        print(f"  The known-good configuration works here ({base['rate']:.2f}) and "
              f"the extractor's does not ({cur['rate']:.2f}). Culprit: {culprit}.")
        print("  Change the extractor to the configuration that works rather than "
              "lowering min_confidence, which would trade a broken call for a "
              "noisier one.")
    elif base["rate"] <= 0.5:
        print(f"  Even the known-good configuration only reaches {base['rate']:.2f} "
              f"on this window, so this window is not the place to diagnose a "
              f"pipeline difference. Pick an event whose 2 fps local extraction "
              f"recorded high coverage and rerun.")
    else:
        print(f"  Both configurations behave similarly here "
              f"({base['rate']:.2f} vs {cur['rate']:.2f}), so the extractor's "
              f"low coverage is not explained by mode or upscaling on this "
              f"window.")


if __name__ == "__main__":
    main()
