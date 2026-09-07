"""Regression tests for image-estimated stereo foreground geometry."""
from __future__ import annotations

import unittest

import numpy as np

from src.rig.learned_stereo import (
    camera_matrix_from_fov,
    combine_instance_masks,
    estimate_depth_scale,
    pose_guided_foreground_mask,
    rectified_camera_points,
    rectified_world_points,
    rectify_learned_pair,
    regularize_component_disparity,
    scale_camera_matrix,
    select_stereo_replacements,
    splat_world_points,
    temporal_change_mask,
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

    def test_rectified_depth_scale_scales_about_left_camera(self):
        width, height = 80, 60
        K = camera_matrix_from_fov(width, height, np.pi / 2, np.pi / 2)
        left = np.eye(4)
        right = np.eye(4)
        right[0, 3] = 0.1
        rect = rectify_learned_pair(K, K, left, right, (width, height))
        points, valid = rectified_camera_points(
            np.full((height, width), 10.0, np.float32), rect,
            depth_scale=2.5)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(
            float(points[height // 2, width // 2, 2]), 1.0, places=2)

    def test_depth_scale_is_robust_to_outliers(self):
        stereo = np.linspace(1.0, 4.0, 1000, dtype=np.float32)
        reference = stereo * 2.5
        reference[:25] *= 4.0
        estimate = estimate_depth_scale(
            stereo, reference, min_samples=100)
        self.assertAlmostEqual(estimate.scale, 2.5, places=5)
        self.assertEqual(estimate.sample_count, 1000)
        self.assertLess(estimate.median_relative_error, 1e-5)
        self.assertLess(estimate.p90_relative_error, 1e-5)

    def test_depth_scale_rejects_too_few_samples(self):
        estimate = estimate_depth_scale(
            np.ones((4, 4)), np.full((4, 4), 2.0), min_samples=32)
        self.assertEqual(estimate.scale, 1.0)
        self.assertEqual(estimate.sample_count, 16)
        self.assertTrue(np.isinf(estimate.p90_relative_error))

    def test_temporal_change_rejects_static_pixels(self):
        background = np.zeros((12, 16, 3), np.float32)
        current = background.copy()
        current[3:8, 5:11] = 1.0
        changed = temporal_change_mask(
            current, [background, background], threshold=0.2)
        self.assertEqual(int(changed.sum()), 30)
        self.assertTrue(changed[3:8, 5:11].all())

    def test_instance_masks_keep_people_and_reject_other_classes(self):
        masks = np.zeros((3, 8, 10), np.float32)
        masks[0, 1:5, 2:6] = 0.9
        masks[1, 4:7, 6:9] = 0.8
        masks[2, 0:2, 0:2] = 0.95
        combined = combine_instance_masks(
            masks, [0, 0, 24], (8, 10), threshold=0.5)
        self.assertTrue(combined[1:5, 2:6].all())
        self.assertTrue(combined[4:7, 6:9].all())
        self.assertFalse(combined[0, 0])

    def test_instance_masks_require_source_resolution(self):
        with self.assertRaisesRegex(ValueError, "match the source image"):
            combine_instance_masks(
                np.zeros((1, 4, 5), np.float32), [0], (8, 10))

    def test_pose_guided_mask_keeps_person_and_rejects_box_exterior(self):
        image = np.zeros((30, 40, 3), np.uint8)
        image[5:26, 12:29] = 220
        pose = np.zeros((1, 17, 3), np.float32)
        pose[0, 5] = [16, 10, 1]
        pose[0, 6] = [24, 10, 1]
        pose[0, 11] = [17, 20, 1]
        pose[0, 12] = [23, 20, 1]
        mask = pose_guided_foreground_mask(
            image, [[10, 3, 31, 28]], pose)
        self.assertTrue(mask[15, 20])
        self.assertGreater(int(mask[5:26, 12:29].sum()), 250)
        self.assertFalse(mask[:, :8].any())

    def test_component_disparity_is_rigid_and_drops_specks(self):
        mask = np.zeros((12, 16), bool)
        mask[2:8, 4:12] = True
        mask[10, 15] = True
        disparity = np.zeros(mask.shape, np.float32)
        disparity[2:8, 4:12] = np.linspace(8.0, 12.0, 48).reshape(6, 8)
        regularized, accepted, components = regularize_component_disparity(
            disparity, mask, min_component_px=8)
        self.assertEqual(components, 1)
        self.assertTrue(accepted[2:8, 4:12].all())
        self.assertFalse(accepted[10, 15])
        self.assertEqual(np.unique(regularized[accepted]).size, 1)

    def test_stereo_replacement_requires_component_coverage(self):
        dynamic = np.zeros((20, 30), bool)
        dynamic[2:8, 2:8] = True
        dynamic[12:18, 20:26] = True
        stereo = np.zeros_like(dynamic)
        stereo[2:8, 2:8] = True
        stereo[12:14, 20:22] = True
        erase, selected, accepted = select_stereo_replacements(
            dynamic, stereo, min_coverage=0.75,
            min_component_px=8, association_px=0)
        self.assertEqual(accepted, 1)
        self.assertTrue(erase[2:8, 2:8].all())
        self.assertFalse(erase[12:18, 20:26].any())
        self.assertEqual(int(selected.sum()), 36)

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

    def test_splat_tie_uses_preferred_camera_without_colour_average(self):
        points = np.asarray([[[0.0, 0.0, 1.0]]], np.float32)
        red = np.asarray([[[1.0, 0.0, 0.0]]], np.float32)
        blue = np.asarray([[[0.0, 0.0, 1.0]]], np.float32)
        valid = np.ones((1, 1), bool)
        preferred = np.full((20, 40), -1, np.int16)
        preferred[10, 20] = 1
        result = splat_world_points(
            [(0, points, red, valid), (1, points, blue, valid)],
            np.zeros(3), np.eye(3), 20, 40, splat_radius=0,
            preferred_source=preferred)
        self.assertEqual(result.rgb[10, 20].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(int(result.source[10, 20]), 1)


if __name__ == "__main__":
    unittest.main()
