import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.rig.nopo4d_validation import (
    central_camera,
    inspect_image_grid,
    resize_without_camera_parameters,
    run_command,
    split_module_frames,
    videos_from_databag,
    wide_intrinsics,
)


class NoPo4DValidationTest(unittest.TestCase):
    def test_split_preserves_hardware_camera_order(self):
        packed = {}
        for module_index, module in enumerate(("cam12", "cam34", "cam56")):
            left = np.full((2, 3, 3), module_index * 2 + 1, np.uint8)
            right = np.full((2, 3, 3), module_index * 2 + 2, np.uint8)
            packed[module] = np.concatenate((left, right), axis=1)

        cameras = split_module_frames(packed)

        self.assertEqual(list(cameras),
                         ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"])
        for index, name in enumerate(cameras, start=1):
            self.assertTrue(np.all(cameras[name] == index))

    def test_parameter_free_resize_crops_without_stretching(self):
        image = np.zeros((100, 200, 3), np.uint8)
        image[:, :30] = 17
        image[:, 170:] = 23
        resized = resize_without_camera_parameters(image, (56, 42))
        self.assertEqual(resized.shape, (42, 56, 3))
        # Cropping a 2:1 source to 4:3 removes both marked side bands.
        self.assertEqual(int(resized.max()), 0)
        with self.assertRaisesRegex(ValueError, "multiples of 14"):
            resize_without_camera_parameters(image, (55, 42))

    def test_image_grid_is_camera_major_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for camera in range(2):
                for time in range(3):
                    (root / f"cam{camera}_t{time}.png").touch()
            paths = inspect_image_grid(root, expected_cameras=2)
            self.assertEqual(
                [path.name for path in paths],
                ["cam0_t0.png", "cam0_t1.png", "cam0_t2.png",
                 "cam1_t0.png", "cam1_t1.png", "cam1_t2.png"])
            (root / "cam1_t1.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                inspect_image_grid(root, expected_cameras=2)

    def test_virtual_camera_uses_predicted_pose_not_rig_calibration(self):
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 6, axis=0)
        poses[:, 0, 3] = np.array([-3, -2, -1, 1, 2, 30], np.float32)
        poses[2, :3, :3] = np.array(
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)

        target = central_camera(poses, central_index=2)

        np.testing.assert_array_equal(target[:3, :3], poses[2, :3, :3])
        self.assertAlmostEqual(float(target[0, 3]), 0.0)
        k = wide_intrinsics(448, 336, hfov_deg=90, vfov_deg=90)
        self.assertAlmostEqual(float(k[0, 0]), 224.0, places=4)
        self.assertAlmostEqual(float(k[1, 1]), 168.0, places=4)

    def test_databag_requires_all_three_packed_videos_not_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cam12.mp4", "cam34.mp4", "cam56.mp4"):
                (root / name).touch()
            self.assertEqual(set(videos_from_databag(root)),
                             {"cam12", "cam34", "cam56"})
            (root / "cam56.mp4").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "cam56.mp4"):
                videos_from_databag(root)

    def test_runner_uses_local_checkpoints_and_all_six_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            nopo = root / "NoPo4D"
            images = root / "images"
            output = root / "output"
            model = root / "model"
            da3 = root / "da3"
            command = run_command(
                "python", runner, nopo, images, output, model, da3,
                extra=("--gpu", "3"))

        self.assertEqual(command[0], "python")
        self.assertEqual(command[command.index("--num_cameras") + 1], "6")
        self.assertEqual(command[command.index("--gpu") + 1], "3")
        self.assertEqual(command[command.index("--model") + 1],
                         os.fspath(model.resolve()))
        self.assertEqual(command[command.index("--da3_model") + 1],
                         os.fspath(da3.resolve()))


if __name__ == "__main__":
    unittest.main()
