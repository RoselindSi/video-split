from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.rig.seam360_validation import (
    factor_rigid_rig,
    inspect_rig_grid,
    mean_transform,
    parse_rig_image_name,
    resize_full_frame,
)
from third_party.self_cali_gs.prepare_rig import (
    select_reconstruction,
    six_camera_rig_config,
)


def _yaw_pose(angle, translation=(0, 0, 0)):
    radians = np.radians(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [np.cos(radians), 0, np.sin(radians)],
        [0, 1, 0],
        [-np.sin(radians), 0, np.cos(radians)],
    ])
    pose[:3, 3] = translation
    return pose


class Seam360ValidationTest(unittest.TestCase):
    def test_names_retain_physical_camera_and_time(self):
        self.assertEqual(parse_rig_image_name("cam5_t019.png"), (5, 19))
        self.assertEqual(parse_rig_image_name("cam0_t000.jpg"), (0, 0))
        self.assertEqual(parse_rig_image_name("cam3/t041.jpg"), (3, 41))
        with self.assertRaisesRegex(ValueError, "six-camera"):
            parse_rig_image_name("frame_001.png")

    def test_grid_is_complete_and_time_major(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for time in range(2):
                for camera in range(6):
                    (root / f"cam{camera}_t{time:03d}.png").touch()
            paths = inspect_rig_grid(root)
            self.assertEqual(paths[0].name, "cam0_t000.png")
            self.assertEqual(paths[5].name, "cam5_t000.png")
            self.assertEqual(paths[6].name, "cam0_t001.png")
            (root / "cam3_t001.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                inspect_rig_grid(root)

    def test_grid_accepts_native_rig_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for camera in range(6):
                camera_dir = root / f"cam{camera}"
                camera_dir.mkdir()
                (camera_dir / "t000.jpg").touch()
            paths = inspect_rig_grid(root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in paths],
                [f"cam{camera}/t000.jpg" for camera in range(6)])

    def test_colmap_rig_configuration_has_six_fixed_sensors(self):
        config = six_camera_rig_config()
        self.assertEqual(len(config), 1)
        self.assertEqual(len(config[0]["cameras"]), 6)
        self.assertEqual(config[0]["cameras"][0]["image_prefix"], "cam0/")
        self.assertEqual(
            [camera["ref_sensor"] for camera in config[0]["cameras"]],
            [True, False, False, False, False, False])

    def test_selects_the_most_complete_colmap_model(self):
        class Reconstruction:
            def __init__(self, images, points):
                self.images = images
                self.points = points

            def num_reg_images(self):
                return self.images

            def num_points3D(self):
                return self.points

        selected = select_reconstruction({
            0: Reconstruction(20, 1000),
            1: Reconstruction(30, 500),
            2: Reconstruction(30, 800),
        })
        self.assertEqual((selected.images, selected.points), (30, 800))

    def test_resize_keeps_the_complete_frame(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, :2] = 50
        image[:, -2:] = 100
        resized = resize_full_frame(image, (40, 20))
        self.assertEqual(resized.shape, (20, 40, 3))
        self.assertGreater(int(resized[:, :4].max()), 0)
        self.assertGreater(int(resized[:, -4:].max()), 0)

    def test_mean_transform_projects_back_to_a_rigid_rotation(self):
        average = mean_transform([_yaw_pose(-10), _yaw_pose(10)])
        np.testing.assert_allclose(
            average[:3, :3].T @ average[:3, :3], np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(average[:3, :3])), 1.0)

    def test_factorization_recovers_a_fixed_six_camera_rig(self):
        rig = np.stack([
            _yaw_pose(0, (0, 0, 0)),
            _yaw_pose(7, (0.1, 0, 0.2)),
            _yaw_pose(14, (0.3, 0, 0.4)),
        ])
        camera_to_rig = np.stack([
            _yaw_pose(angle, (camera * 0.01, 0, 0))
            for camera, angle in enumerate((-28, -27, 0, 1, 28, 29))
        ])
        poses = rig[:, None] @ camera_to_rig[None]

        recovered_rig, recovered_cameras, reconstructed = factor_rigid_rig(poses)

        expected_cameras = (np.linalg.inv(camera_to_rig[0])[None]
                            @ camera_to_rig)
        np.testing.assert_allclose(recovered_rig, poses[:, 0], atol=1e-8)
        np.testing.assert_allclose(recovered_cameras, expected_cameras, atol=1e-8)
        np.testing.assert_allclose(reconstructed, poses, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
