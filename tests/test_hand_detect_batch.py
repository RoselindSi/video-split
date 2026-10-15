from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig.hand_detect import detect_batch, owner_detect_batch  # noqa: E402


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, box, conf=0.8, cls=0):
        self.xyxy = _Array([box])
        self.conf = _Array([conf])
        self.cls = _Array([cls])

    def __len__(self):
        return 1


class _Keypoints:
    def __init__(self):
        points = np.tile([[50.0, 30.0]], (21, 1))
        points[0] = [50.0, 80.0]
        self.xy = _Array([points])


class _Result:
    def __init__(self, index, keypoints=True):
        self.boxes = _Boxes([index, 2, index + 10, 20])
        self.keypoints = _Keypoints() if keypoints else None


class _Model:
    names = {0: "owner_hand"}

    def __init__(self, keypoints=True):
        self.keypoints = keypoints
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append((len(images), kwargs))
        return [_Result(index, self.keypoints)
                for index, _image in enumerate(images)]


class BatchDetectorTests(unittest.TestCase):
    def test_hand_detector_invokes_model_once_for_ordered_batch(self):
        model = _Model()
        images = [np.zeros((100, 120, 3), np.uint8) for _ in range(3)]

        got = detect_batch(model, images, min_conf=0.1)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], 3)
        self.assertEqual([int(rows[0]["box"][0]) for rows in got], [0, 1, 2])

    def test_owner_detector_invokes_model_once_for_batch(self):
        model = _Model(keypoints=False)
        images = [np.zeros((20, 20, 3), np.uint8) for _ in range(2)]

        got = owner_detect_batch(model, images, min_conf=0.1)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(got), 2)
        self.assertTrue(all(rows[0]["owner"] for rows in got))


if __name__ == "__main__":
    unittest.main()
