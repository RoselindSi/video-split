"""Regression coverage for detector dropouts in the rendered demo pipeline.

This exercises ``demo_video.run`` from rendered wide frames through tracking,
ownership hold, coasting detections, GrabCut, suppression, panel composition,
and the video-writer boundary. Camera I/O, projection, and model inference are
replaced with deterministic fixtures; the privacy path itself is real.

Run:
    python3 tests/test_demo_video_dropout.py
"""
from __future__ import annotations

import copy
import csv
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


HEIGHT = 96
WIDTH = 160


def _detection(box, *, owner, score, exit_y):
    x0, y0, x1, y1 = box
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return {
        "box": np.array(box, dtype=int),
        "kp": np.array([centre] * 21, dtype=float),
        "owner": bool(owner),
        "owner_p": float(score),
        "conf": 0.90,
        "rule_owner": bool(owner),
        "exit": np.array([centre[0], exit_y], dtype=float),
    }


def _frame():
    img = np.full((HEIGHT, WIDTH, 3), (35, 45, 55), dtype=np.uint8)
    yy, xx = np.indices((45, 32))
    pattern = ((xx + yy) % 2)[..., None]
    img[20:65, 8:40] = np.where(
        pattern, np.array([40, 150, 230]), np.array([220, 80, 30]))
    img[55:90, 58:88] = (90, 170, 210)
    img[55:90, 112:142] = (100, 180, 220)
    return img


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


class DemoVideoDropoutRegressionTest(unittest.TestCase):
    def test_foreign_hand_stays_suppressed_through_two_missed_detections(self):
        clean = _frame()
        frames = [{"rgb": clean.copy()} for _ in range(4)]

        foreign = _detection((8, 20, 40, 65), owner=False, score=0.05,
                             exit_y=8)
        own_left = _detection((58, 55, 88, 90), owner=True, score=0.95,
                              exit_y=HEIGHT)
        own_right = _detection((112, 55, 142, 90), owner=True, score=0.96,
                               exit_y=HEIGHT)
        detections = [
            [own_left, own_right, foreign],
            [own_left, own_right],
            [own_left, own_right],
            [own_left, own_right, foreign],
        ]

        reader = _ClipReader(frames)
        detect_calls = iter(detections)
        detector_floors = []
        _Writer.instances.clear()

        def fake_render(rig, vcam, sources, depth, map_cache=None):
            del rig, vcam, depth, map_cache
            return sources["rgb"].copy(), None, None, None

        def fake_detect(model, image, **kwargs):
            del model, image
            detector_floors.append(kwargs.get("min_conf"))
            return copy.deepcopy(next(detect_calls))

        with tempfile.TemporaryDirectory() as td:
            trace_path = str(Path(td) / "trace.csv")
            out_path = str(Path(td) / "demo.mp4")
            with mock.patch(
                    "src.rig.geometry.VirtualWideCamera.from_rig",
                    return_value=object()), mock.patch(
                    "src.rig.seam_fix.ClipReader", return_value=reader), \
                    mock.patch("src.rig.render_wide.render",
                               side_effect=fake_render), mock.patch(
                    "src.rig.hand_detect.detect",
                    side_effect=fake_detect), mock.patch(
                    "cv2.VideoWriter", side_effect=_Writer):
                written, disagreements, faces, _flips = demo_video.run(
                    rig=object(), videos={}, out_path=out_path, start=100,
                    n=4, stride=1, model=object(), cnn=None, device="cpu",
                    dilate=10, sigma=14.0, fps=12.0, verbose=False,
                    face_model=None, trace_path=trace_path, max_owner=2,
                    bridge=2, min_conf=0)

            with open(trace_path, newline="") as f:
                trace = list(csv.DictReader(f))

        self.assertEqual(written, 4)
        self.assertEqual(disagreements, 0)
        self.assertEqual(faces, 0)
        self.assertEqual(detector_floors, [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(reader.closed)
        self.assertEqual([int(row["n_det"]) for row in trace], [3, 2, 2, 3])
        self.assertEqual([int(row["n_oth"]) for row in trace], [1, 1, 1, 1])
        self.assertTrue(all(float(row["alpha_frac"]) > 0 for row in trace))

        writer = _Writer.instances[-1]
        self.assertTrue(writer.released)
        self.assertEqual(len(writer.frames), 4)
        for panel in writer.frames:
            rendered_output = panel[-HEIGHT:]
            difference = np.abs(rendered_output.astype(np.int16)
                                - clean.astype(np.int16))
            self.assertGreater(
                int(difference[15:70, 3:45].sum()), 0,
                "the foreign-hand region passed through unchanged")
            self.assertTrue(np.array_equal(rendered_output[55:90, 58:88],
                                           clean[55:90, 58:88]))
            self.assertTrue(np.array_equal(rendered_output[55:90, 112:142],
                                           clean[55:90, 112:142]))


if __name__ == "__main__":
    unittest.main()
