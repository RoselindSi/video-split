import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.rig.stabstitch_pair import (find_module, pair_key, run_official,
                                     valid_fraction)
from src.rig.stabstitch_pair_eval import edge_chamfer


class _Camera:
    def __init__(self, name):
        self.name = name


class _Module:
    def __init__(self, left, right):
        self.left = _Camera(left)
        self.right = _Camera(right)


class StabStitchPairTest(unittest.TestCase):
    def test_pair_lookup_uses_both_eyes_of_one_module(self):
        modules = [_Module("cam1", "cam2"), _Module("cam3", "cam4")]
        rig = mock.Mock(modules=modules)
        self.assertEqual(pair_key(modules[0]), "cam12")
        self.assertIs(find_module(rig, "cam34"), modules[1])
        with self.assertRaisesRegex(ValueError, "calibrated pairs: cam12, cam34"):
            find_module(rig, "cam13")

    def test_valid_fraction_rejects_out_of_sensor_samples(self):
        mx = np.array([[0.0, 9.0], [-1.0, 5.0]], np.float32)
        my = np.array([[0.0, 5.0], [2.0, 7.0]], np.float32)
        self.assertEqual(valid_fraction((mx, my), (10, 8)), 0.25)

    def test_edge_chamfer_increases_when_a_hard_edge_is_displaced(self):
        import cv2
        image = np.zeros((100, 120, 3), np.uint8)
        cv2.line(image, (40, 10), (40, 90), (255, 255, 255), 3)
        shifted = np.zeros_like(image)
        cv2.line(shifted, (48, 10), (48, 90), (255, 255, 255), 3)
        aligned = edge_chamfer(image, image)
        displaced = edge_chamfer(image, shifted)
        self.assertLess(aligned["chamfer_p90_px"], 0.1)
        self.assertEqual(aligned["ghost_risk_gt3_fraction"], 0.0)
        self.assertGreater(displaced["chamfer_p90_px"], 5.0)
        self.assertGreater(displaced["ghost_risk_gt3_fraction"], 0.5)

    @mock.patch("src.rig.stabstitch_pair.subprocess.run")
    def test_official_runner_contains_upstream_path_quirks(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "StabStitch2")
            codes = os.path.join(root, "Full_model_inference", "Codes")
            models = os.path.join(root, "Full_model_inference", "full_model_tra")
            os.makedirs(codes)
            os.makedirs(models)
            open(os.path.join(codes, "test_online_tra.py"), "w").close()
            for name in ("spatial_warp.pth", "temporal_warp.pth",
                         "smooth_warp.pth"):
                open(os.path.join(models, name), "w").close()
            input_root = os.path.join(directory, "input")
            output_root = os.path.join(directory, "output")
            command = run_official(
                root, input_root, output_root, python="python", gpu="3")

        run.assert_called_once_with(command, cwd=os.fspath(Path(codes).resolve()),
                                    check=True)
        self.assertEqual(command[:4],
                         ["python", "test_online_tra.py", "--gpu", "3"])
        self.assertTrue(command[command.index("--output_path") + 1]
                        .endswith(os.sep))
        self.assertEqual(command[-4:],
                         ["--warp_mode", "NORMAL",
                          "--fusion_mode", "LINEAR"])


if __name__ == "__main__":
    unittest.main()
