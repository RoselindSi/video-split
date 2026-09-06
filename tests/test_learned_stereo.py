"""Regression tests for image-estimated stereo foreground geometry."""
from __future__ import annotations

import unittest

import numpy as np

from src.rig.learned_stereo import (
    camera_matrix_from_fov,
    rectified_world_points,
    rectify_learned_pair,
    scale_camera_matrix,
    splat_world_points,
    world_to_equirectangular,
)


class LearnedStereoTest(unittest.TestCase):
    def test_camera_matrix_matches_requested_fov(self):
        K = camera_matrix_from_fov(200, 100, np.pi / 2, np.pi / 2)
        self.assertAlmostEqual(float(K[0, 0]), 100.0)
        self.assertAlmostEqual(float(K[1, 1]), 50.0)
        self.assertEqual(K[:2, 2].tolist(), [100.0, 50.0])

    def test_scale_camera_matrix_preserves_normalized_rays(self):
        K = np.asarray([
            [400.0, 0.0, 320.0],
            [0.0, 360.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        scaled = scale_camera_matrix(K, (640, 480), (960, 576))
        np.testing.assert_allclose(
            scaled,
            [[600.0, 0.0, 480.0],
             [0.0, 432.0, 288.0],
             [0.0, 0.0, 1.0]])

    def test_rectified_constant_disparity_recovers_positive_depth(self):
        width, height = 80, 60
        K = camera_matrix_from_fov(width, height, np.pi / 2, np.pi / 2)
        left = np.eye(4)
        right = np.eye(4)
        right[0, 3] = 0.1
        rect = rectify_learned_pair(K, K, left, right, (width, height))
        points, valid = rectified_world_points(
            np.full((height, width), 10.0, np.float32), rect)
        self.assertAlmostEqual(rect.baseline, 0.1)
        self.assertTrue(valid.all())
        self.assertGreater(float(points[height // 2, width // 2, 2]), 0.0)
        self.assertAlmostEqual(
            float(points[height // 2, width // 2, 2]), 0.4, places=2)

    def test_cardinal_world_points_project_to_erp_quadrants(self):
        points = np.asarray([
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0],
        ])
        x, y, radius = world_to_equirectangular(
            points, np.zeros(3), np.eye(3), 20, 40)
        np.testing.assert_allclose(x, [20, 30, 40, 10], atol=1e-5)
        np.testing.assert_allclose(y, np.full(4, 10), atol=1e-5)
        np.testing.assert_allclose(radius, np.ones(4), atol=1e-5)

    def test_splat_z_buffer_keeps_nearest_colour_without_averaging(self):
        near_points = np.asarray([[[0.0, 0.0, 1.0]]], np.float32)
        far_points = np.asarray([[[0.0, 0.0, 2.0]]], np.float32)
        red = np.asarray([[[1.0, 0.0, 0.0]]], np.float32)
        blue = np.asarray([[[0.0, 0.0, 1.0]]], np.float32)
        valid = np.ones((1, 1), bool)
        result = splat_world_points(
            [(1, far_points, blue, valid),
             (0, near_points, red, valid)],
            np.zeros(3), np.eye(3), 20, 40, splat_radius=0)
        self.assertEqual(result.rgb[10, 20].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(int(result.source[10, 20]), 0)
        self.assertAlmostEqual(float(result.range_m[10, 20]), 1.0)


if __name__ == "__main__":
    unittest.main()
