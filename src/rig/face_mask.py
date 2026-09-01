"""Cover other people's faces without touching the rest of the frame.

WHY THIS IS NOT THE HAND PROBLEM WITH A DIFFERENT DETECTOR. Ownership does not
arise. The camera is head-mounted, so the wearer's own face is never in view
and every face found is somebody else's. There is no classifier here and so
none of the flip risk that comes from one -- the only instability left is the
detector's own, frame to frame.

THE ERROR ASYMMETRY RUNS THE OPPOSITE WAY, TOO. A false positive costs a
mosaicked patch of bench, which the downstream model reads as a fixed texture
and learns to ignore. A false negative is a frame in which a real person's
face left the building recognisable. The two are not comparable, so the
threshold belongs low and the hold below belongs long. This is the reason the
`min_conf` default here is well under the 0.5 a detector ships with, and it is
a deliberate choice rather than a tuning oversight.

MOSAIC, NOT BLUR. A Gaussian blur on a 60-pixel face leaves the arrangement of
eyes, hairline and jaw intact at low frequency, and low frequency is most of
what a face recognizer uses; people identify blurred faces of colleagues they
know. Downscaling to blocks and scaling back up throws the information away
instead of smearing it. The block size is a fraction of the face, not a
constant, so a face near the camera is destroyed as thoroughly as a far one.

THE HOLD IS THE POINT, NOT A REFINEMENT. A detector that finds a face on one
frame and misses it on the next produces a mosaic that switches off for a
frame. That is both a leak and, worse for what this pipeline feeds, a
flicker -- a local region changing sharply in time is exactly the signature
the downstream segmenter reads as an event. So a face, once seen, keeps its
cover for HOLD_FRAMES afterwards even with no detection to support it, and the
held box is the union of where it has recently been.

RUNNING IT. MediaPipe's bindings need libglvnd, which this container does not
carry and which cannot be apt-installed without root. The .deb was unpacked to
/workspace/glvnd, so callers need

    export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

before the interpreter starts -- the dynamic loader reads it at exec time, so
setting it from inside Python is too late to help.

THE MODEL IS THE FULL-RANGE ONE, AND HAS TO BE. `blaze_face_short_range` is
built for a face that fills much of a 128px input. Measured on this corpus a
colleague's face is about 60px in a 1100px frame, which survives the resize as
roughly seven pixels, and that model returns nothing at all on frames where a
face is plainly visible. The full-range model finds the same face at 0.43-0.56.
"""
from __future__ import annotations

import os

import numpy as np

MODEL = "/workspace/models/face_detection_full_range.tflite"

# Well below a detector's usual 0.5. See the error asymmetry above: the cost of
# being wrong is a mosaicked patch of bench in one direction and a recognisable
# face in the other.
MIN_CONF = 0.30

# Frames a face keeps its cover after the last detection supporting it. At the
# 10-15 fps these demos render, this is roughly a second.
HOLD_FRAMES = 12

# A detection wider than this fraction of the frame is refused. Nobody's face
# but the wearer's could be that large here, and the wearer's is behind the
# camera. This is a statement about the rig, not a tuning knob.
MAX_FACE_FRAC = 0.12

# The detector bounds a face from brow to chin. Ears, hairline and jaw are
# outside that and carry identity, so the box is grown before it is used.
PAD = 0.35

# Mosaic cell as a fraction of the covered box's short side. A fraction rather
# than a pixel count, so a face close to the camera is destroyed as completely
# as a distant one.
BLOCK_FRAC = 0.18


def load_detector(model_path=MODEL, min_conf=MIN_CONF):
    """-> a MediaPipe FaceDetector, or None if the model file is absent.

    Absent is not an error at import time; it is an error at the point a caller
    asks for faces to be covered and cannot have them covered, and that is
    where it should be raised."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except OSError as e:                       # libglvnd missing, not absence
        raise SystemExit(
            f"MediaPipe could not load its native library ({e}).\n"
            "  export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/"
            "x86_64-linux-gnu:$LD_LIBRARY_PATH  before starting python.")
    return vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=float(min_conf)))


def detect_faces(det, rgb):
    """-> [(x0, y0, x1, y1, score)] in pixels, clipped to the frame."""
    import cv2
    import mediapipe as mp
    H, W = rgb.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    out = []
    for d in det.detect(img).detections:
        b = d.bounding_box
        out.append((max(0, int(b.origin_x)), max(0, int(b.origin_y)),
                    min(W, int(b.origin_x + b.width)),
                    min(H, int(b.origin_y + b.height)),
                    float(d.categories[0].score)))
    return out


class Hold:
    """Keeps a face covered for a while after the detector stops finding it.

    A TRACK IS ITS LATEST BOX, NOT THE UNION OF ITS BOXES. The first version
    grew a held box to the union of everything that overlapped it, so that a
    face moving during a hold stayed covered. That is a runaway: the grown box
    overlaps more of the next frame's detections, absorbing them extends it
    further, and the box that does the matching is the one growth has already
    inflated. Measured on a rendered clip it swallowed a face of 84x84 into a
    region of roughly 350x420 over fifty frames -- the colleague, the shelving
    and a third of the bench. Replacing the box on each match instead bounds
    the region by what the detector actually saw, and the hold still covers the
    frames where it saw nothing.

    Matching is by IoU rather than by any overlap, for the same reason: a box
    that merely touches another is not evidence they are the same face, and
    accepting it is what let one track eat the frame."""

    def __init__(self, frames=HOLD_FRAMES, min_iou=0.2, max_frac=MAX_FACE_FRAC):
        self.frames = int(frames)
        self.min_iou = float(min_iou)
        self.max_frac = max_frac
        self.items = []                        # [[x0, y0, x1, y1, ttl]]

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1])
        ub = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(ua + ub - inter)

    def update(self, boxes, shape=None):
        """boxes: [(x0,y0,x1,y1,score)] -> [(x0,y0,x1,y1)] to cover now.

        `shape` enables the size sanity check: the wearer's own face is never
        in view, so every face here belongs to somebody across a bench and is
        small. A detection wider than `max_frac` of the frame is not a face at
        this range, and covering it destroys the bench this pipeline exists to
        keep."""
        if self.max_frac and shape is not None:
            W = shape[1]
            boxes = [b for b in boxes
                     if (b[2] - b[0]) <= self.max_frac * W]
        for it in self.items:
            it[4] -= 1
        for b in boxes:
            best, best_iou = None, self.min_iou
            for it in self.items:
                v = self._iou(b, it)
                if v >= best_iou:
                    best, best_iou = it, v
            if best is None:
                self.items.append([b[0], b[1], b[2], b[3], self.frames])
            else:
                best[0], best[1], best[2], best[3] = b[0], b[1], b[2], b[3]
                best[4] = self.frames
        self.items = [it for it in self.items if it[4] > 0]
        return [tuple(it[:4]) for it in self.items]


def pad_box(box, shape, pad=PAD):
    """Grow a box by `pad` of its own size, clipped. -> (x0, y0, x1, y1)"""
    H, W = shape[:2]
    x0, y0, x1, y1 = box[:4]
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - dx), max(0, y0 - dy),
            min(W, x1 + dx), min(H, y1 + dy))


def cover(rgb, boxes, pad=PAD, block_frac=BLOCK_FRAC):
    """Mosaic every box. -> (frame, mask)

    Each box is resampled on its own rather than the whole frame at once, so
    the cell size follows the face's size. Pixels outside every box are
    untouched, which is checked in the self test rather than assumed."""
    import cv2
    out = np.asarray(rgb).copy()
    mask = np.zeros(out.shape[:2], bool)
    for b in boxes:
        x0, y0, x1, y1 = pad_box(b, out.shape, pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patch = out[y0:y1, x0:x1]
        cell = max(2, int(min(y1 - y0, x1 - x0) * block_frac))
        small = cv2.resize(patch, (max(1, (x1 - x0) // cell),
                                   max(1, (y1 - y0) // cell)),
                           interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                       interpolation=cv2.INTER_NEAREST)
        mask[y0:y1, x0:x1] = True
    return out, mask


def _self_test():
    ok = n = 0

    def chk(name, cond):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    out, mask = cover(img, [(100, 60, 140, 110, 0.9)])
    chk("pixels outside the box are bit-identical",
        np.array_equal(out[~mask], img[~mask]))
    chk("pixels inside the box changed", not np.array_equal(out[mask],
                                                            img[mask]))
    # The pad is what puts hairline and jaw inside the cover; without it the
    # mask would be exactly the detector's box.
    chk("the covered area is larger than the detector's box",
        int(mask.sum()) > (140 - 100) * (110 - 60))

    # Mosaic and not blur: a mosaic has flat cells, so the count of distinct
    # rows inside the patch collapses. A blur would leave them all distinct.
    patch = out[60:110, 100:140].reshape(-1, 3)
    chk("the cover is blocky rather than smoothed",
        len(np.unique(patch, axis=0)) < 0.5 * len(patch))

    h = Hold(frames=3)
    b = [(10, 10, 30, 30, 0.9)]
    chk("a detected face is covered", len(h.update(b)) == 1)
    chk("it survives a frame with no detection", len(h.update([])) == 1)
    h.update([])
    chk("and is dropped once the hold runs out", len(h.update([])) == 0)

    h2 = Hold(frames=5)
    h2.update([(10, 10, 30, 30, 0.9)])
    kept = h2.update([(16, 16, 36, 36, 0.9)])
    chk("a matching detection refreshes one face, not two", len(kept) == 1)
    chk("and the held box FOLLOWS it instead of growing to the union",
        kept[0] == (16, 16, 36, 36))
    h2.update([(200, 200, 220, 220, 0.9)])
    chk("a disjoint detection starts its own", len(h2.items) == 2)

    # The runaway that made the first version cover a third of the frame: a
    # chain of detections each barely touching the last. Held boxes must not
    # accumulate across it.
    h3 = Hold(frames=30)
    for i in range(40):
        h3.update([(10 + 5 * i, 10, 90 + 5 * i, 90, 0.9)])
    widest = max(b[2] - b[0] for b in h3.update([]))
    chk(f"a drifting face never inflates the held box (widest {widest})",
        widest <= 80)

    # Size sanity: the wearer's face is behind the camera, so a face filling a
    # quarter of the width is not a face.
    h4 = Hold(frames=5, max_frac=0.12)
    chk("an implausibly large detection is refused",
        h4.update([(0, 0, 400, 300, 0.9)], shape=(600, 1000)) == [])
    chk("a plausible one at the same threshold is kept",
        len(h4.update([(0, 0, 90, 90, 0.9)], shape=(600, 1000))) == 1)

    print(f"\n  {ok}/{n}")
    return ok == n


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", required=True, help="a frame to try it on")
    ap.add_argument("--out", help="write the covered frame here")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min_conf", type=float, default=MIN_CONF)
    a = ap.parse_args()

    import cv2
    det = load_detector(a.model, a.min_conf)
    if det is None:
        raise SystemExit(f"no face model at {a.model}")
    rgb = cv2.imread(a.img)
    if rgb is None:
        raise SystemExit(f"cannot read {a.img}")
    faces = detect_faces(det, rgb)
    out, mask = cover(rgb, faces)
    print(f"  {len(faces)} faces in {os.path.basename(a.img)}, "
          f"{mask.mean():.2%} of pixels covered")
    for x0, y0, x1, y1, s in sorted(faces, key=lambda f: -f[4]):
        print(f"    conf {s:.2f}   {x1-x0:3d}x{y1-y0:3d} at {x0},{y0}")
    if a.out:
        cv2.imwrite(a.out, out)
        print(f"  -> {a.out}")
    print("\n  A low threshold is correct here: a false positive mosaics a "
          "patch of bench,\n  a false negative leaves a face. Read the "
          "confidences above before raising it.")


if __name__ == "__main__":
    main()
