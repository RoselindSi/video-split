"""Cover other people's faces without touching the rest of the frame.

WHY THIS IS NOT THE HAND PROBLEM WITH A DIFFERENT DETECTOR. Ownership does not
arise. The camera is head-mounted, so the wearer's own face is never in view
and every face found is somebody else's. There is no classifier here and so
none of the flip risk that comes from one -- the only instability left is the
detector's own, frame to frame.

THE ERROR ASYMMETRY IS REAL AND WAS ONCE PUSHED TOO FAR. A false negative
leaves a real person recognisable; a false positive mosaics something that is
not a face. That argued for a low threshold, and 0.30 was chosen from it --
which turned out to cover the wearer's own hands, because a detector this
permissive fires on skin and on a bench most skin is a hand. Destroying the
subject is not the conservative choice. The threshold now sits at 0.60, where
the scores actually separate, and the hand-overlap veto below removes what is
left.

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
cover for HOLD_FRAMES afterwards even with no detection to support it, at the
last position the detector reported. It is NOT the union of where the face has
been: growing a held box by everything that overlapped it made one 84x84 face
into a 350x420 region over fifty frames, swallowing the colleague, the shelving
and a third of the bench.

TWO BACKENDS, AND THE FALLBACK IS THE ONE TO AVOID. YuNet runs through OpenCV
and needs nothing beyond it. BlazeFace runs through MediaPipe, whose native
bindings need libglvnd, which this container lacks and which cannot be
apt-installed without root; the .deb was unpacked to /workspace/glvnd, so a
caller falling back to it needs

    export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

set before the interpreter starts -- the loader reads it at exec time, so
setting it from Python is too late.

AND IF IT DOES FALL BACK, USE THE FULL-RANGE WEIGHTS. `blaze_face_short_range`
is built for a face filling much of a 128px input. A colleague's face here is
about 60px in a 1100px frame, seven pixels after the resize, and that model
returns nothing at all on frames where a face is plainly visible.
"""
from __future__ import annotations

import os

import numpy as np

# YUNET IS THE DEFAULT AND BLAZEFACE IS THE FALLBACK, WHICH IS THE REVERSE OF
# HOW THIS STARTED. BlazeFace was chosen because it was the only detector whose
# weights the SERVER could reach -- GitHub, gitee, gitcode and HuggingFace are
# all blocked there. That reasoning had a hole: the laptop's network is not the
# server's, and the same ssh pipe that carries code patches carries a 230 KB
# model. YuNet's weights live behind Git LFS, so they come from
# media.githubusercontent.com rather than raw.githubusercontent.com, which
# returns a 131-byte pointer file instead.
#
# ON THE SAME FRAME, THE SCORES SEPARATE AND BLAZEFACE'S DO NOT. YuNet puts the
# real faces at 0.82 and 0.86 and its first false positive at 0.39, a gap of
# 0.43 to place a threshold in. BlazeFace put real faces at 0.43-0.56 and a
# false one at 0.32: a gap of 0.11, so any threshold cuts through the overlap.
# That gap is the whole reason a threshold can be set at all, and it is why the
# earlier version needed the hand-overlap veto to be usable.
MODEL = "/workspace/models/face_detection_yunet_2023mar.onnx"
MODEL_FALLBACK = "/workspace/models/face_detection_full_range.tflite"

# THE ASYMMETRY ARGUMENT ABOVE IS TRUE AND WAS APPLIED TOO FAR. At 0.30 this
# detector fires on skin, and on a bench full of hands most skin is a hand.
# The cost of a false positive was described as "a mosaicked patch of bench",
# which understated it: watched back, the wearer's own hands and forearms were
# being covered, and those are the pixels the entire pipeline exists to
# deliver. A privacy cover that destroys the subject is not a conservative
# choice. 0.5 is the detector's own default and the floor a real face at this
# range clears; the overlap veto below removes the rest.
MIN_CONF = 0.60

# A "face" covering this much of a detected hand is that hand. The hand
# detector is the better instrument here -- it was trained to find hands, it
# reports 0.83-0.87 on real ones, and a face detector has no business firing
# inside its boxes. This is a structural veto, not a threshold: it does not
# get weaker as the face score rises.
HAND_OVERLAP_VETO = 0.45

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


class _YuNet:
    """cv2.FaceDetectorYN behind the same two calls as the MediaPipe one.

    setInputSize IS NOT OPTIONAL AND IS THE CLASSIC WAY TO GET GARBAGE FROM
    THIS MODEL. The network is built for a fixed input and the detector rescales
    boxes by whatever size it was last told about, so a size that does not match
    the frame returns boxes in the wrong coordinates -- which looks exactly like
    a detector hallucinating. It is set on construction and again whenever the
    frame shape changes."""

    kind = "yunet"

    def __init__(self, path, min_conf):
        import cv2
        self.cv2 = cv2
        self.min_conf = float(min_conf)
        self.size = (320, 320)
        self.det = cv2.FaceDetectorYN.create(path, "", self.size,
                                             float(min_conf), 0.3, 5000)

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        if (W, H) != self.size:
            self.size = (W, H)
            self.det.setInputSize(self.size)
        _, faces = self.det.detect(bgr)
        out = []
        for f in (faces if faces is not None else []):
            x, y, w, h = [float(v) for v in f[:4]]
            out.append((max(0, int(x)), max(0, int(y)),
                        min(W, int(x + w)), min(H, int(y + h)), float(f[-1])))
        return out

    def close(self):
        pass


class _YoloFace:
    """A YOLOv8 face head through cv2.dnn, behind the same two calls.

    The export is a single-class detect head: (1, 5, 8400), four box
    parameters and one score per candidate, no keypoints. Decoding is the
    plain YOLO one -- centre/size to corners, then NMS -- and the resize is a
    straight one rather than a letterbox, so the two axes scale back
    independently.

    IT IS HERE AS AN ALTERNATIVE, NOT AN UPGRADE. It is fifty times the size
    of YuNet for the same job, and nothing has yet been measured that says it
    is better on this footage; the point of having two is that the choice can
    be made from a comparison rather than from a release date."""

    kind = "yolo"
    SIDE = 640

    def __init__(self, path, min_conf, nms=0.45):
        import cv2
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromONNX(path)
        self.min_conf = float(min_conf)
        self.nms = float(nms)

    def detect(self, bgr):
        cv2 = self.cv2
        H, W = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (self.SIDE, self.SIDE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        a = self.net.forward()
        a = a[0] if a.ndim == 3 else a
        if a.shape[0] < a.shape[1]:
            a = a.T                       # -> (candidates, 5)
        sx, sy = W / float(self.SIDE), H / float(self.SIDE)
        boxes, scores = [], []
        for cx, cy, w, h, sc in a:
            if sc < self.min_conf:
                continue
            boxes.append([int((cx - w / 2) * sx), int((cy - h / 2) * sy),
                          int(w * sx), int(h * sy)])
            scores.append(float(sc))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.min_conf, self.nms)
        out = []
        for i in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(i)]
            out.append((max(0, x), max(0, y), min(W, x + w), min(H, y + h),
                        scores[int(i)]))
        return out

    def close(self):
        pass


def load_detector(model_path=MODEL, min_conf=MIN_CONF):
    """-> a MediaPipe FaceDetector, or None if the model file is absent.

    Absent is not an error at import time; it is an error at the point a caller
    asks for faces to be covered and cannot have them covered, and that is
    where it should be raised."""
    if not model_path or not os.path.exists(model_path):
        # Named model missing: fall back rather than silently covering nothing.
        if model_path == MODEL and os.path.exists(MODEL_FALLBACK):
            model_path = MODEL_FALLBACK
        else:
            return None
    if str(model_path).endswith(".onnx"):
        # Both are ONNX; the file name says which head it is. Guessing from
        # the graph would be cleverer and would fail silently on a rename.
        if "yolo" in os.path.basename(model_path).lower():
            return _YoloFace(model_path, min_conf)
        return _YuNet(model_path, min_conf)
    # THE GUARD HAS TO COVER THE CONSTRUCTION, NOT ONLY THE IMPORT. MediaPipe
    # loads its native library lazily, inside create_from_options, so wrapping
    # the import alone let the real failure through as a bare OSError about
    # libEGL -- accurate and useless. The message a caller needs is which
    # environment variable is missing.
    try:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=float(min_conf)))
    except OSError as e:                       # libglvnd missing, not absence
        raise SystemExit(
            f"MediaPipe could not load its native library ({e}).\n"
            "  export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/"
            "x86_64-linux-gnu:$LD_LIBRARY_PATH  before starting python.\n"
            "  YuNet needs none of this and is the default; only the "
            "BlazeFace fallback does.")


def detect_faces(det, rgb):
    """-> [(x0, y0, x1, y1, score)] in pixels, clipped to the frame."""
    if getattr(det, "kind", None) in ("yunet", "yolo"):
        return det.detect(rgb)
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


def drop_on_hands(faces, dets, frac=HAND_OVERLAP_VETO):
    """Remove face detections that sit on a detected hand. -> [face]

    THE WEARER'S OWN FACE IS NEVER IN VIEW, so a face found on the wearer's
    own hand is wrong twice over: it is not a face, and even if it were, it
    could not be anyone whose privacy this protects. The same holds for a
    colleague's hand -- a hand is not a head.

    Overlap is measured against the FACE's area, not the union. A small false
    face sitting inside a large hand box has an IoU near zero and would
    survive a symmetric test, and small false faces on hands are exactly what
    this is for."""
    if not dets:
        return list(faces)
    keep = []
    for f in faces:
        fa = max(1, (f[2] - f[0]) * (f[3] - f[1]))
        worst = 0.0
        for d in dets:
            x0, y0, x1, y1 = [int(v) for v in d["box"]]
            ix = max(0, min(f[2], x1) - max(f[0], x0))
            iy = max(0, min(f[3], y1) - max(f[1], y0))
            worst = max(worst, ix * iy / float(fa))
        if worst < frac:
            keep.append(f)
    return keep


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



def harvest(pkgs, out_dir, model_path=MODEL, min_conf=0.50, crop_px=192,
            verbose=True):
    """Save every surviving face detection as a labellable crop. -> n

    THE FACE LINE HAS NEVER BEEN MEASURED AND IT IS NOW THE ONLY ONE THAT HAS
    NOT. The hand side has 322 held-out hands, a false-positive rate and a
    confidence interval. The face side has a threshold picked from the score
    gap on ONE frame, and that frame turned out not to be representative: on
    48 frames of a car-repair recording the same detector produced 1280
    proposals, 90% of them below 0.35, and called an engine cover a face at
    0.57.

    A threshold cannot be chosen from proposals alone, because the question is
    not how many there are but how many are faces. This writes the survivors
    -- what passes the threshold and the size cap, which is exactly what would
    be mosaicked -- in the layout `label_tool` already serves, so the same
    workflow that produced the hand numbers produces these.

    THE CONTEXT FRAME MATTERS MORE HERE THAN FOR HANDS. A 192px crop of an
    engine cover and a 192px crop of a face across a workshop are both blurry
    beige rectangles; whether it is a face is often only answerable from what
    surrounds it."""
    import cv2
    import glob
    import re
    stem_re = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")
    det = load_detector(model_path, min_conf)
    if det is None:
        raise SystemExit(f"no face model at {model_path}")
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "context"), exist_ok=True)
    rows, seen = [], set()
    for pkg in pkgs:
        for f in sorted(glob.glob(os.path.join(pkg, "context", "*.jpg"))):
            m = stem_re.match(os.path.basename(f)[:-4])
            if not m:
                continue
            tag, frame = m.group(1), m.group(2)
            if (tag, frame) in seen:
                continue                  # one pass per frame, not per hand
            seen.add((tag, frame))
            img = cv2.imread(f)
            if img is None:
                continue
            H, W = img.shape[:2]
            faces = [x for x in detect_faces(det, img)
                     if (x[2] - x[0]) <= MAX_FACE_FRAC * W]
            for j, (x0, y0, x1, y1, sc) in enumerate(faces):
                stem = f"{tag}f{int(frame):06d}_h{j}"
                px = int(max(x1 - x0, y1 - y0) * 0.6)
                cx0, cy0 = max(0, x0 - px), max(0, y0 - px)
                cx1, cy1 = min(W, x1 + px), min(H, y1 + px)
                cv2.imwrite(os.path.join(out_dir, "crops", stem + ".jpg"),
                            cv2.resize(img[cy0:cy1, cx0:cx1],
                                       (crop_px, crop_px)),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                ctx = img.copy()
                cv2.rectangle(ctx, (x0, y0), (x1, y1), (0, 255, 255), 3)
                cv2.imwrite(os.path.join(out_dir, "context", stem + ".jpg"),
                            ctx, [cv2.IMWRITE_JPEG_QUALITY, 82])
                rows.append({"stem": stem, "frame": int(frame), "hand": j,
                             "conf": round(float(sc), 4),
                             "w_frac": round((x1 - x0) / W, 4),
                             "h_frac": round((y1 - y0) / H, 4),
                             "cx_frac": round((x0 + x1) / 2 / W, 4),
                             "cy_frac": round((y0 + y1) / 2 / H, 4),
                             "model": os.path.basename(model_path),
                             "label": ""})
    if rows:
        import csv as _csv
        with open(os.path.join(out_dir, "hands.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if verbose:
        import numpy as _np
        sc = _np.array([r["conf"] for r in rows]) if rows else _np.zeros(1)
        print(f"  {len(rows)} surviving detections over {len(seen)} frames "
              f"-> {out_dir}")
        print(f"  scores 10/50/90: {_np.percentile(sc, 10):.2f} / "
              f"{_np.percentile(sc, 50):.2f} / {_np.percentile(sc, 90):.2f}")
        print("\n  These are the regions that WOULD be mosaicked. Label them "
              "face/not-face with\n  label_tool and the threshold stops being "
              "a guess -- the hand line's numbers\n  came from exactly this "
              "loop.")
    return len(rows)


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

    # A face landing on a hand is a hand. This is what was covering the
    # wearer's own fingers in the first batch of renders.
    hand = [{"box": (100, 100, 200, 200)}]
    on_hand = (120, 120, 170, 170, 0.9)
    off_hand = (400, 400, 450, 450, 0.9)
    chk("a face inside a hand box is dropped",
        drop_on_hands([on_hand], hand) == [])
    chk("...and one away from every hand is kept",
        drop_on_hands([off_hand], hand) == [off_hand])
    # Overlap is against the FACE's area: a small box inside a big one has a
    # tiny IoU and would survive a symmetric test.
    big_hand = [{"box": (0, 0, 900, 900)}]
    chk("a small face inside a large hand is still dropped",
        drop_on_hands([on_hand], big_hand) == [])
    chk("with no hands detected nothing is vetoed",
        drop_on_hands([on_hand], []) == [on_hand])

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
    ap.add_argument("--img", help="a frame to try it on")
    ap.add_argument("--harvest", action="append", default=[],
                    metavar="PKG", help="packages whose context/ frames to "
                                        "scan for faces")
    ap.add_argument("--out_pkg", help="where to write the harvested crops")
    ap.add_argument("--out", help="write the covered frame here")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min_conf", type=float, default=MIN_CONF)
    a = ap.parse_args()

    if a.harvest:
        if not a.out_pkg:
            raise SystemExit("--harvest needs --out_pkg")
        harvest(a.harvest, a.out_pkg, a.model, a.min_conf)
        raise SystemExit(0)
    if not a.img:
        raise SystemExit("give --img, or --harvest with --out_pkg")

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
