"""One hand detector wrapper covering both mediapipe APIs.

mediapipe 1.0.0 removed the legacy `mp.solutions.hands` interface entirely --
`AttributeError: module 'mediapipe' has no attribute 'solutions'` -- and
replaced it with the Tasks API, which needs an explicitly supplied model
bundle instead of one packaged inside the wheel. Both APIs exist in the wild
(0.10.x is still widely pinned), and both c3_crop_coverage.py and
extract_features_local.py need hand boxes, so the version handling lives here
once rather than being duplicated and drifting.

The two APIs differ in more than construction: legacy returns
`res.multi_hand_landmarks`, each element having a `.landmark` list, while
Tasks returns `res.hand_landmarks`, each element already BEING the list. A
wrapper that only abstracted construction would still hand back two different
shapes, so `boxes()` normalises the output too.

Getting the Tasks model (about 7 MB, from Google's official mediapipe model
store; nothing else in this repo downloads anything, so this is deliberately
a manual step rather than an automatic fetch):

    curl -L -o /workspace/tr1/ckpts/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""
from __future__ import annotations

import os


class HandDetector:
    """detector(frame_rgb_uint8) -> list of (x0, y0, x1, y1) pixel boxes, one
    per detected hand. Empty list when nothing is found."""

    def __init__(self, model_path=None, max_hands=2, min_confidence=0.3):
        import mediapipe as mp
        self.mp = mp
        self.api = None
        self.version = getattr(mp, "__version__", "?")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.det = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=max_hands,
                min_detection_confidence=min_confidence)
            self.api = "legacy"
            return

        model_path = model_path or os.environ.get("HAND_LANDMARKER_TASK")
        if not model_path or not os.path.exists(model_path):
            raise SystemExit(
                f"mediapipe {self.version} exposes only the Tasks API, which needs a "
                f"model bundle, and none was found"
                + (f" at {model_path!r}" if model_path else " (--hand_model / "
                   "$HAND_LANDMARKER_TASK not set)") + ".\n"
                "Fetch it (~7 MB, official mediapipe model store):\n"
                "  curl -L -o /workspace/tr1/ckpts/hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/hand_landmarker"
                "/hand_landmarker/float16/1/hand_landmarker.task\n"
                "then pass --hand_model /workspace/tr1/ckpts/hand_landmarker.task\n"
                "Alternatively pin the older API: python -m pip install "
                "'mediapipe<1' (bundles the model, but changes packages in a "
                "venv that is otherwise working).")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=min_confidence))
        self.api = "tasks"

    def boxes(self, frame):
        """frame: HxWx3 uint8 RGB.

        The array is forced C-contiguous and uint8 before it reaches
        mediapipe. mp.Image wraps the buffer it is handed and assumes a
        row-major contiguous layout, so a non-contiguous VIEW is read with the
        wrong stride and the model sees a scrambled image -- returning zero
        hands with no error raised anywhere. That is exactly what a
        packed-stereo half is: frames[:, :, :W//2] slices axis 2 and is not
        contiguous. Copying here rather than at every call site means a future
        caller cannot reintroduce the failure."""
        import numpy as np
        if not frame.flags["C_CONTIGUOUS"] or frame.dtype != np.uint8:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        if self.api == "legacy":
            res = self.det.process(frame)
            hands = [lm.landmark for lm in (res.multi_hand_landmarks or [])]
        else:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)
            hands = self.det.detect(img).hand_landmarks or []
        out = []
        for lms in hands:
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            if xs and ys:
                out.append((min(xs), min(ys), max(xs), max(ys)))
        return out

    def close(self):
        """mediapipe 1.0's HandLandmarker.__del__ raises
        "TypeError: 'NoneType' object is not callable" during interpreter
        shutdown when it is left to the garbage collector. Harmless, but it
        prints a traceback after every run, which trains you to ignore
        tracebacks."""
        try:
            self.det.close()
        except Exception:
            pass

    def union_box(self, frame):
        """Single box covering every detected hand, or None.

        The union rather than the largest hand: a crop around one hand can
        miss what the other is doing, and two-handed interaction is exactly
        the case the crop has to survive."""
        bs = self.boxes(frame)
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs),
                max(b[2] for b in bs), max(b[3] for b in bs))
