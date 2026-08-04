"""Why did the hand detector return ZERO boxes on every sampled frame?

c3_crop_coverage.py detected 0 hands across 183 frames from 25 recordings,
on footage where a sample frame plainly shows two hands filling the lower
third. A complete miss on obvious hands is far more likely to be a defect in
how the frame is handed to the detector than a genuine limitation of the
model, so this isolates the cause instead of guessing at fixes.

The leading suspect is memory layout. eye_slice() returns
`frames[:, :, :W//2]`, a slice along axis 2, which is NOT C-contiguous:

    np.zeros((4,480,1280,3))[:, :, :640].flags['C_CONTIGUOUS']  ->  False

mediapipe's Image wraps the buffer it is given and assumes a contiguous
row-major layout, so a non-contiguous view is read with the wrong row stride
-- the model sees a scrambled image and legitimately finds no hands, with no
error raised anywhere.

Other candidates worth separating rather than fixing blind, since each has a
different remedy: channel order (decord yields RGB, mediapipe SRGB expects
RGB, but a mismatch would silently halve skin-tone plausibility), the packed
stereo half versus the whole frame, detection confidence, and image scale.

Each variant is run on THE SAME frame, so exactly one thing differs at a
time and the reported hand count attributes the failure to one cause.

Usage (server):
    python -m src.boundary.c3_detector_diag \
        --video /shared/datasets/human_ego_recording_segmentation_10fps_r01_part_01/recordings/recording_000022/mid.mp4 \
        --t 30 --hand_model /workspace/tr1/ckpts/hand_landmarker.task \
        --dump_dir /workspace/tr1/results/hal/c3/diag
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def variants(frame_full):
    """(name, image, note) for one decoded full stereo frame."""
    W = frame_full.shape[1]
    left_view = frame_full[:, :W // 2]              # non-contiguous, as shipped
    left_cont = np.ascontiguousarray(left_view)
    out = [
        ("left eye, AS SHIPPED (non-contiguous view)", left_view,
         "what c3_crop_coverage actually passed"),
        ("left eye, ascontiguousarray", left_cont,
         "identical pixels, contiguous buffer -- isolates memory layout"),
        ("left eye, contiguous + BGR->RGB swap", left_cont[:, :, ::-1].copy(),
         "isolates channel order"),
        ("full packed frame, contiguous", np.ascontiguousarray(frame_full),
         "isolates the stereo split"),
        ("left eye, contiguous, 2x upscaled", None, "isolates image scale"),
        ("left eye, contiguous, lower half only", np.ascontiguousarray(
            left_cont[left_cont.shape[0] // 3:]), "hands occupy the lower part"),
    ]
    from PIL import Image
    h, w = left_cont.shape[:2]
    up = np.asarray(Image.fromarray(left_cont).resize((w * 2, h * 2), Image.BICUBIC))
    out[4] = (out[4][0], np.ascontiguousarray(up), out[4][2])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, default=30.0)
    ap.add_argument("--hand_model")
    ap.add_argument("--min_conf", type=float, nargs="*", default=[0.3, 0.1, 0.05])
    ap.add_argument("--dump_dir", help="save each variant as PNG -- if every "
                                       "variant finds nothing, looking at what "
                                       "was actually fed in is the next step")
    a = ap.parse_args()

    from decord import VideoReader
    vr = VideoReader(a.video, num_threads=1)
    fps = vr.get_avg_fps()
    j = min(len(vr) - 1, max(0, int(round(a.t * fps))))
    frame = vr.get_batch([j]).asnumpy()[0]
    print(f"frame {j} at t={j / fps:.1f}s  shape {frame.shape}  dtype {frame.dtype}  "
          f"contiguous {frame.flags['C_CONTIGUOUS']}")
    print(f"pixel stats: min {frame.min()} max {frame.max()} mean {frame.mean():.1f}")

    vs = variants(frame)
    if a.dump_dir:
        from PIL import Image
        os.makedirs(a.dump_dir, exist_ok=True)
        for i, (name, img, _) in enumerate(vs):
            Image.fromarray(np.ascontiguousarray(img)).save(
                os.path.join(a.dump_dir, f"{i}_{name.split(',')[0].replace(' ', '_')}.png"))
        print(f"wrote {len(vs)} variant PNGs -> {a.dump_dir}")

    from src.boundary.hand_detect import HandDetector
    print(f"\n{'variant':<46} {'shape':>14} {'contig':>7} " +
          "".join(f"{'conf=' + str(c):>10}" for c in a.min_conf))
    any_hit = False
    for name, img, note in vs:
        counts = []
        for c in a.min_conf:
            try:
                det = HandDetector(a.hand_model, min_confidence=c)
                n = len(det.boxes(img))
            except SystemExit:
                raise
            except Exception as e:
                n = f"ERR:{type(e).__name__}"
            counts.append(n)
            if isinstance(n, int) and n > 0:
                any_hit = True
        print(f"{name:<46} {str(img.shape):>14} "
              f"{str(img.flags['C_CONTIGUOUS']):>7} " +
              "".join(f"{str(c):>10}" for c in counts))
        print(f"{'  ^ ' + note:<46}")

    print()
    if not any_hit:
        print("NO variant found a hand. The frame handling is then NOT the cause, "
              "and the remaining candidates are the model bundle itself (a "
              "truncated or wrong .task download would still load) and the "
              "footage genuinely defeating this model. Check the dumped PNGs "
              "look like normal images, then verify the bundle's size and hash.")
    else:
        print("At least one variant found hands -- the difference between the "
              "working and failing rows IS the bug. Apply that fix in "
              "hand_detect.py rather than lowering the confidence threshold, "
              "which would only trade a silent failure for a noisy one.")


if __name__ == "__main__":
    main()
