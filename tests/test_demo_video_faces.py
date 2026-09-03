"""Regression coverage for the face cover, end to end through the renderer.

This runs ``demo_video.run`` with faces enabled and asserts on the frames the
video writer actually receives -- not on the helpers in isolation. The privacy
claim is about the pixels that leave the pipeline, so that is where it is
checked. Only the camera, the projection and the two networks are replaced by
fixtures; the veto, the hold, the mosaic and the compositing are real.

Every case here is a bug this pipeline actually shipped:

  * the cover reaching the DECISION panel but not the output the model gets,
  * a face detector firing on skin and mosaicking the wearer's own hands,
  * a held box growing to the union of everything it touched until it covered
    a third of the frame,
  * a one-frame detector dropout switching the mosaic off and back on, which
    a downstream segmenter reads as an event.

Run:
    python3 tests/test_demo_video_faces.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig import demo_video  # noqa: E402
from src.rig import face_mask   # noqa: E402

HEIGHT = 96
WIDTH = 160
# face_mask.MAX_FACE_FRAC is a fraction of the frame width. It was 0.12 and is
# now 0.055, fitted against labelled detections rather than guessed, so on this
# 160px fixture a plausible face is under 9px wide. These boxes were 14px and
# the tightening refused them -- which is the cap working, and the reason the
# fixture moves rather than the constant.
FACE = (100, 8, 108, 18)
FACE_ON_HAND = (62, 58, 70, 68)
FACE_TOO_BIG = (10, 5, 70, 65)


def _detection(box, *, owner, score, exit_y):
    x0, y0, x1, y1 = box
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return {"box": np.array(box, dtype=int),
            "kp": np.array([centre] * 21, dtype=float),
            "owner": bool(owner), "owner_p": float(score), "conf": 0.90,
            "rule_owner": bool(owner),
            "exit": np.array([centre[0], exit_y], dtype=float)}


def _frame(seed=0):
    """A frame with texture everywhere, so a mosaic is detectable as a LOSS
    of detail rather than merely a change of colour."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


class _StubYuNet:
    """Stands in for cv2.FaceDetectorYN behind the same interface.

    `kind` is what `detect_faces` dispatches on, so the real dispatch, the
    real hand veto, the real hold and the real mosaic all run."""

    kind = "yunet"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def detect(self, bgr):
        del bgr
        out = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [tuple(b) for b in out]

    def close(self):
        pass


class _ClipReader:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def next(self, skip=0):
        del skip
        return next(self.frames, None)

    def close(self):
        self.closed = True


class _Writer:
    instances = []

    def __init__(self, *args):
        del args
        self.frames = []
        self.released = False
        self.__class__.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def release(self):
        self.released = True


def _run(face_script, detections, n):
    """-> (written panels, faces counted, the clean input frames)"""
    clean = [_frame(i) for i in range(n)]
    reader = _ClipReader([{"rgb": c.copy()} for c in clean])
    det_iter = iter(detections)
    _Writer.instances.clear()

    def fake_render(rig, vcam, sources, depth, map_cache=None):
        del rig, vcam, depth, map_cache
        return sources["rgb"].copy(), None, None, None

    def fake_detect(model, image, **kwargs):
        del model, image, kwargs
        return copy.deepcopy(next(det_iter))

    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "demo.mp4")
        with mock.patch("src.rig.geometry.VirtualWideCamera.from_rig",
                        return_value=object()), \
             mock.patch("src.rig.seam_fix.ClipReader", return_value=reader), \
             mock.patch("src.rig.render_wide.render",
                        side_effect=fake_render), \
             mock.patch("src.rig.hand_detect.detect",
                        side_effect=fake_detect), \
             mock.patch("src.rig.face_mask.load_detector",
                        return_value=_StubYuNet(face_script)), \
             mock.patch("cv2.VideoWriter", side_effect=_Writer):
            written, _dis, faces = demo_video.run(
                rig=object(), videos={}, out_path=out_path, start=0, n=n,
                stride=1, model=object(), cnn=None, device="cpu", dilate=10,
                sigma=14.0, fps=12.0, verbose=False,
                face_model="stub.onnx", max_owner=2)
    panels = _Writer.instances[0].frames if _Writer.instances else []
    return written, faces, clean, panels


def _output_panel(panel):
    """The bottom half of the composed panel: what the model receives."""
    return panel[-HEIGHT:]


def _detail(patch):
    """Local variation. A mosaic flattens cells, so this collapses."""
    p = np.asarray(patch, np.int16)
    return float(np.abs(np.diff(p, axis=0)).mean()
                 + np.abs(np.diff(p, axis=1)).mean())


class FaceCoverRenderTest(unittest.TestCase):
    def setUp(self):
        self.own = [_detection((20, 55, 50, 90), owner=True, score=0.95,
                               exit_y=HEIGHT),
                    _detection((58, 55, 88, 90), owner=True, score=0.96,
                               exit_y=HEIGHT)]

    def test_face_is_destroyed_in_the_frame_the_model_receives(self):
        n = 3
        written, faces, clean, panels = _run(
            [[FACE + (0.90,)]], [list(self.own) for _ in range(n)], n)
        self.assertEqual(written, n)
        self.assertEqual(faces, n)
        x0, y0, x1, y1 = FACE
        for panel, src in zip(panels, clean):
            out = _output_panel(panel)
            # The cover has to reach the OUTPUT panel. An earlier version
            # tinted the decision panel and left the output untouched, which
            # looks correct on screen and ships the face downstream.
            self.assertLess(_detail(out[y0:y1, x0:x1]),
                            0.5 * _detail(src[y0:y1, x0:x1]),
                            "face region still carries its detail")
            # And nothing else may be touched: a privacy cover that repaints
            # the bench destroys the pixels the pipeline exists to deliver.
            self.assertTrue(np.array_equal(out[:, 130:], src[:, 130:]),
                            "pixels far from any face were modified")

    def test_cover_survives_a_frame_the_detector_missed(self):
        """The hold, end to end. A mosaic that switches off for one frame is
        both a leak and a sharp local change in time, which is the signature
        the downstream segmenter reads as an event."""
        n = 3
        script = [[FACE + (0.90,)], [], []]
        _w, faces, clean, panels = _run(script,
                                        [list(self.own) for _ in range(n)], n)
        self.assertEqual(faces, 1, "only the first frame detected a face")
        x0, y0, x1, y1 = FACE
        for i, (panel, src) in enumerate(zip(panels, clean)):
            self.assertLess(_detail(_output_panel(panel)[y0:y1, x0:x1]),
                            0.5 * _detail(src[y0:y1, x0:x1]),
                            f"cover dropped on frame {i}")

    def test_a_face_landing_on_a_detected_hand_is_refused(self):
        """The veto that stopped the wearer's own hands being mosaicked. The
        hand detector is the better instrument for whether a patch of skin is
        a hand, and a face detector has no business firing inside its box."""
        n = 2
        _w, faces, clean, panels = _run(
            [[FACE_ON_HAND + (0.95,)]],
            [list(self.own) for _ in range(n)], n)
        self.assertEqual(faces, 0, "a face on a hand was counted")
        x0, y0, x1, y1 = FACE_ON_HAND
        for panel, src in zip(panels, clean):
            self.assertAlmostEqual(
                _detail(_output_panel(panel)[y0:y1, x0:x1]),
                _detail(src[y0:y1, x0:x1]), delta=1e-6,
                msg="the wearer's own hand was mosaicked")

    def test_an_implausibly_large_detection_is_refused(self):
        """Nobody's face but the wearer's could fill this much of the frame,
        and the wearer's is behind the camera."""
        n = 2
        _w, _faces, clean, panels = _run(
            [[FACE_TOO_BIG + (0.99,)]],
            [list(self.own) for _ in range(n)], n)
        x0, y0, x1, y1 = FACE_TOO_BIG
        for panel, src in zip(panels, clean):
            self.assertAlmostEqual(
                _detail(_output_panel(panel)[y0:y1, x0:x1]),
                _detail(src[y0:y1, x0:x1]), delta=1e-6,
                msg="an oversized detection was covered")

    def test_a_drifting_face_does_not_inflate_the_covered_region(self):
        """One 84x84 face once grew into a 350x420 region over fifty frames,
        because the held box took the union of everything that overlapped it.
        The held box follows the detector instead."""
        n = 6
        script = [[(100 + 3 * i, 8, 114 + 3 * i, 24, 0.90)] for i in range(n)]
        _w, _faces, clean, panels = _run(script,
                                         [list(self.own) for _ in range(n)], n)
        covered = []
        for panel, src in zip(panels, clean):
            out = _output_panel(panel)
            covered.append(int((out != src).any(axis=2).sum()))
        # The face is 14x16 and the pad grows it; a region several times that
        # would mean the box is accumulating rather than following.
        self.assertLess(max(covered), 4 * 14 * 16,
                        f"covered area grew to {max(covered)} px")

    def test_yunet_is_the_default_model_and_blazeface_the_fallback(self):
        self.assertTrue(face_mask.MODEL.endswith(".onnx"))
        self.assertTrue(face_mask.MODEL_FALLBACK.endswith(".tflite"))


MODELS = {
    "yunet": "/workspace/models/face_detection_yunet_2023mar.onnx",
    "yolo": "/workspace/models/yolov8n-face-lindevs.onnx",
    "blazeface": "/workspace/models/face_detection_full_range.tflite",
}


class RealBackendTest(unittest.TestCase):
    """The backends themselves, on the machine that has their weights.

    Skipped where the files are absent, which is every machine but the one
    that renders. What it checks is not accuracy -- there is no ground truth
    here -- but that each backend honours the contract the rest of the module
    assumes: boxes inside the frame, scores above the floor, and coordinates
    that survive a change of frame size."""

    def _frames(self):
        return [_frame(1), np.repeat(np.repeat(_frame(2), 2, 0), 2, 1)]

    def test_every_available_backend_honours_the_contract(self):
        checked, skipped = [], []
        for name, path in MODELS.items():
            if not Path(path).exists():
                continue
            try:
                det = face_mask.load_detector(path, face_mask.MIN_CONF)
            except SystemExit as e:
                # A backend whose native library is missing is not a failure
                # of the cover logic. It is recorded rather than swallowed:
                # a silently skipped backend is how one stops being tested.
                skipped.append(f"{name} ({str(e).splitlines()[0]})")
                continue
            self.assertIsNotNone(det, f"{name} failed to load")
            for img in self._frames():
                H, W = img.shape[:2]
                for box in face_mask.detect_faces(det, img):
                    x0, y0, x1, y1, sc = box
                    # THE FRAME-SIZE BUG THIS GUARDS. YuNet rescales its
                    # output by whatever size it was last told about, so a
                    # stale setInputSize returns boxes in another frame's
                    # coordinates -- which looks exactly like hallucination.
                    self.assertTrue(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H,
                                    f"{name} box {box} outside {W}x{H}")
                    self.assertGreaterEqual(sc, face_mask.MIN_CONF - 1e-6)
            checked.append(name)
        print(f"\n    backends exercised: {checked or 'none'}"
              + (f"   unavailable: {skipped}" if skipped else ""))
        if not checked:
            self.skipTest("no model weights on this machine")

    def test_the_two_reachable_onnx_backends_dispatch_differently(self):
        for name in ("yunet", "yolo"):
            if not Path(MODELS[name]).exists():
                self.skipTest("weights absent")
        a = face_mask.load_detector(MODELS["yunet"], face_mask.MIN_CONF)
        b = face_mask.load_detector(MODELS["yolo"], face_mask.MIN_CONF)
        self.assertEqual((a.kind, b.kind), ("yunet", "yolo"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
