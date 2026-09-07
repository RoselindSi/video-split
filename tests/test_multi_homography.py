import unittest

import numpy as np

from src.rig.multi_homography import (
    PairGeometryConfig,
    accepted_components,
    analyze_pair,
    assign_segments_to_models,
    fit_stable_affine,
    fit_piecewise_homographies,
    guard_segment_models,
    grid_coverage,
    regularized_match_map,
    segmented_warp,
    static_correspondence_mask,
    transformed_support_points,
)
from src.rig.segmented_panorama import (
    canvas_from_footprints,
    compose_frequency_selective_blend,
    content_segments,
    enforce_anchor_authority,
    hierarchical_six_owner,
    minimum_vertical_seam,
    overlay_dense_anchor_warp,
    overlay_regularized_match_warp,
    pair_seam_cost,
    three_band_owner,
)


class MultiHomographyTest(unittest.TestCase):
    def test_dynamic_and_low_certainty_matches_are_excluded(self):
        source = np.array([[1, 1], [4, 4], [8, 8]], np.float32)
        target = source + 2
        source_mask = np.zeros((12, 12), bool)
        source_mask[4, 4] = True
        keep = static_correspondence_mask(
            source, target, certainty=[0.9, 0.9, 0.01],
            source_exclusion=source_mask, min_certainty=0.05)
        self.assertEqual(keep.tolist(), [True, False, False])

    def test_grid_coverage_reports_spatial_support(self):
        one_cell = np.array([[1, 1], [2, 2]], np.float32)
        four_cells = np.array([[1, 1], [9, 1], [1, 9], [9, 9]], np.float32)
        self.assertEqual(grid_coverage(one_cell, (10, 10), 2, 2), 0.25)
        self.assertEqual(grid_coverage(four_cells, (10, 10), 2, 2), 1.0)

    def test_two_parallax_planes_get_two_homographies(self):
        rng = np.random.default_rng(4)
        first = rng.uniform([5, 5], [95, 95], size=(160, 2)).astype(np.float32)
        second = rng.uniform([5, 5], [95, 95], size=(160, 2)).astype(np.float32)
        first_target = first + np.array([12, -3], np.float32)
        second_target = second + np.array([-9, 7], np.float32)
        source = np.concatenate((first, second))
        target = np.concatenate((first_target, second_target))
        models, labels = fit_piecewise_homographies(
            source, target,
            PairGeometryConfig(
                homography_threshold_px=0.5,
                min_homography_inliers=80,
                max_homographies=3))
        self.assertEqual(len(models), 2)
        self.assertEqual(int(np.count_nonzero(labels >= 0)), 320)
        self.assertTrue(all(model["p90_error_px"] < 0.1 for model in models))

    def test_pair_report_accepts_piecewise_static_geometry(self):
        rng = np.random.default_rng(9)
        source = rng.uniform([2, 2], [198, 98], size=(300, 2)).astype(np.float32)
        target = source + np.array([7, 2], np.float32)
        report = analyze_pair(
            source, target, (100, 200), (100, 200),
            certainty=np.ones(300, np.float32),
            config=PairGeometryConfig(
                fundamental_threshold_px=0.5,
                homography_threshold_px=0.5,
                min_fundamental_inliers=100,
                min_fundamental_ratio=0.5,
                min_homography_inliers=80,
                min_explained_matches=200,
                min_explained_ratio=0.5,
                min_grid_coverage=0.25))
        self.assertTrue(report["accepted"], report["checks"])
        self.assertEqual(report["explained_matches"], 300)

    def test_only_accepted_pairs_connect_the_camera_graph(self):
        reports = [
            {"pair": [0, 1], "accepted": True},
            {"pair": [1, 2], "accepted": True},
            {"pair": [2, 3], "accepted": False},
            {"pair": [3, 4], "accepted": True},
        ]
        self.assertEqual(
            accepted_components(5, reports), [[0, 1, 2], [3, 4]])
        reports[2]["accepted"] = True
        self.assertEqual(accepted_components(5, reports), [[0, 1, 2, 3, 4]])

    def test_segment_assignment_never_splits_a_content_region(self):
        segments = np.zeros((12, 18), np.int16)
        segments[:, 6:12] = 1
        segments[:, 12:] = 2
        points = np.array([[2, 2], [3, 8], [14, 2], [15, 8]], np.float32)
        labels = np.array([0, 0, 1, 1])
        model_map = assign_segments_to_models(segments, points, labels, 2)
        self.assertEqual(np.unique(model_map[:, :6]).tolist(), [0])
        self.assertEqual(np.unique(model_map[:, 6:12]).size, 1)
        self.assertEqual(np.unique(model_map[:, 12:]).tolist(), [1])

    def test_unmatched_segments_use_stable_default_model(self):
        segments = np.zeros((12, 18), np.int16)
        segments[:, 6:12] = 1
        segments[:, 12:] = 2
        points = np.array([[2, 2], [2, 4], [2, 6], [2, 8]], np.float32)
        labels = np.ones(4, np.int16)
        model_map = assign_segments_to_models(
            segments, points, labels, 2, default_model=0, min_votes=4)
        self.assertEqual(np.unique(model_map[:, :6]).tolist(), [1])
        self.assertEqual(np.unique(model_map[:, 6:]).tolist(), [0])

    def test_stable_affine_does_not_create_projective_extrapolation(self):
        rng = np.random.default_rng(21)
        source = rng.uniform([0, 0], [100, 60], (100, 2)).astype(np.float32)
        target = source @ np.array([[1.1, 0.1], [-0.05, 0.9]]) \
            + np.array([20, -7])
        transform, inliers = fit_stable_affine(source, target, 0.5)
        self.assertGreater(int(inliers.sum()), 95)
        np.testing.assert_allclose(transform[2], [0, 0, 1])

    def test_regularized_match_map_is_continuous_and_rejects_outlier(self):
        yy, xx = np.meshgrid(
            np.arange(6, 35, 7), np.arange(6, 55, 8), indexing="ij")
        target = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
        source = target + np.asarray([4.0, -2.0], np.float32)
        target = np.vstack((target, [[30.0, 20.0]])).astype(np.float32)
        source = np.vstack((source, [[500.0, 500.0]])).astype(np.float32)
        mapping, active, report = regularized_match_map(
            (40, 60), (40, 60), source, target, np.eye(3),
            cell_px=6, smooth_sigma=0.5, max_correction_px=20,
            hull_feather_px=3)
        np.testing.assert_allclose(mapping[20, 30], [34.0, 18.0], atol=0.75)
        self.assertTrue(active[20, 30])
        self.assertLess(report["bounded_matches"], report["input_matches"])

    def test_wild_local_model_falls_back_for_complete_segment(self):
        segments = np.zeros((20, 40), np.int16)
        segments[:, 20:] = 1
        model_map = np.ones_like(segments)
        local = np.array([[1, 0, 1000], [0, 1, 0], [0, 0, 1]], np.float64)
        guarded = guard_segment_models(
            segments, model_map, [np.eye(3), local], max_delta_px=100)
        self.assertTrue(np.all(guarded == 0))

    def test_segmented_warp_preserves_single_source_colours(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, :10] = (10, 20, 30)
        image[:, 10:] = (100, 110, 120)
        model_map = np.zeros((10, 20), np.int16)
        model_map[:, 10:] = 1
        homographies = [
            np.eye(3),
            np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], np.float64),
        ]
        warped, valid = segmented_warp(
            image, model_map, homographies, np.eye(3), (30, 10))
        colours = np.unique(warped[valid], axis=0)
        self.assertEqual(colours.tolist(), [[10, 20, 30], [100, 110, 120]])
        self.assertTrue(valid[:, :10].all())
        self.assertTrue(valid[:, 15:25].all())

    def test_transformed_support_tracks_piecewise_extent(self):
        model_map = np.zeros((9, 21), np.int16)
        model_map[:, 10:] = 1
        homographies = [
            np.eye(3),
            np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], np.float64),
        ]
        points = transformed_support_points(model_map, homographies, step=4)
        self.assertLessEqual(float(points[:, 0].min()), 0.0)
        self.assertGreaterEqual(float(points[:, 0].max()), 40.0)

    def test_person_mask_is_one_indivisible_content_segment(self):
        image = np.zeros((40, 60, 3), np.uint8)
        image[:, 30:] = 255
        person = np.zeros((40, 60), bool)
        person[5:35, 20:40] = True
        segments = content_segments(image, person, count=12, compactness=5)
        self.assertEqual(np.unique(segments[person]).size, 1)

    def test_canvas_contains_anchor_and_translated_footprints(self):
        footprints = [
            np.array([[0, 0], [100, 50]], np.float32),
            np.array([[-80, -10], [170, 60]], np.float32),
        ]
        transform, size, stats = canvas_from_footprints(
            footprints, (50, 100), margin=0)
        self.assertGreaterEqual(size[0], 250)
        self.assertGreaterEqual(size[1], 70)
        np.testing.assert_allclose(transform[:2, 2], [80, 10], atol=2)
        self.assertEqual(stats["size"], list(size))

    def test_anchor_authority_removes_every_internal_camera_seam(self):
        owner = np.ones((12, 20), np.int16)
        anchor_valid = np.zeros((12, 20), bool)
        anchor_valid[2:10, 4:16] = True
        result = enforce_anchor_authority(owner, anchor_valid, anchor=3)
        self.assertTrue(np.all(result[anchor_valid] == 3))
        self.assertTrue(np.all(result[~anchor_valid] == 1))

    def test_vertical_seam_follows_low_cost_corridor(self):
        cost = np.ones((30, 40), np.float32)
        corridor = np.arange(30) // 3 + 12
        cost[np.arange(30), corridor] = 0.0
        seam = minimum_vertical_seam(
            cost, np.ones_like(cost, bool), nominal=15, max_step=2)
        self.assertLess(float(np.abs(seam - corridor).mean()), 1.0)

    def test_pair_seam_cost_keeps_margin_around_protected_content(self):
        shape = (30, 60)
        image = np.zeros((*shape, 3), np.uint8)
        valid = np.ones(shape, bool)
        protected = np.zeros(shape, bool)
        protected[10:20, 28:32] = True
        cost, allowed = pair_seam_cost(
            image, image, valid, valid, protected, None,
            protect_margin=12)
        self.assertTrue(allowed.all())
        self.assertGreater(float(cost[15, 24]), float(cost[15, 8]))
        self.assertGreater(float(cost[15, 30]), float(cost[15, 24]))

    def test_three_band_owner_keeps_ordered_contiguous_sources(self):
        shape = (20, 60)
        images = {camera: np.zeros((*shape, 3), np.uint8)
                  for camera in (1, 3, 5)}
        valid = {camera: np.ones(shape, bool) for camera in images}
        protected = {camera: np.zeros(shape, bool) for camera in images}
        owner, first, second = three_band_owner(
            images, valid, protected, cameras=(1, 3, 5))
        self.assertTrue(np.all(first < second))
        for row in range(shape[0]):
            changes = np.count_nonzero(owner[row, 1:] != owner[row, :-1])
            self.assertEqual(changes, 2)
            self.assertEqual(int(owner[row, 0]), 1)
            self.assertEqual(int(owner[row, -1]), 5)

    def test_hierarchical_owner_uses_ordered_stereo_modules(self):
        shape = (24, 120)
        images = {
            camera: np.full((*shape, 3), camera * 20, np.uint8)
            for camera in range(6)
        }
        valid = {camera: np.ones(shape, bool) for camera in images}
        semantic = {camera: np.zeros(shape, bool) for camera in images}
        cost = {
            camera: np.zeros(shape, np.float32) for camera in images
        }
        owner, protected, stats = hierarchical_six_owner(
            images, valid, semantic, cost, max_step=2)
        self.assertEqual(np.unique(owner).tolist(), list(range(6)))
        self.assertFalse(protected.any())
        self.assertEqual(len(stats["pair_seams"]), 3)
        for row in range(shape[0]):
            self.assertEqual(
                np.unique(owner[row], return_index=True)[0].tolist(),
                list(range(6)))

    def test_frequency_blend_smooths_colour_but_keeps_protected_pixels(self):
        shape = (32, 64)
        images = {
            0: np.full((*shape, 3), 51, np.uint8),
            1: np.full((*shape, 3), 204, np.uint8),
        }
        valid = {key: np.ones(shape, bool) for key in images}
        owner = np.zeros(shape, np.int16)
        owner[:, shape[1] // 2:] = 1
        hard = np.full((*shape, 3), 0.2, np.float32)
        hard[:, shape[1] // 2:] = 0.8
        protected = np.zeros(shape, bool)
        protected[8:16, shape[1] // 2 - 2:shape[1] // 2 + 2] = True
        result, transition = compose_frequency_selective_blend(
            images, valid, owner, hard, protected=protected,
            sigma=4.0, boundary_px=10, protect_dilate_px=0)
        seam = shape[1] // 2
        self.assertLess(
            float(np.abs(result[24, seam] - result[24, seam - 1]).max()),
            0.15)
        np.testing.assert_allclose(
            result[protected], hard[protected], atol=1e-6)
        self.assertTrue(transition[24, seam])
        self.assertFalse(transition[10, seam])

    def test_dense_refine_never_warps_protected_content(self):
        source = np.zeros((8, 10, 3), np.uint8)
        source[:] = (20, 80, 160)
        global_image = np.zeros_like(source)
        global_valid = np.ones(source.shape[:2], bool)
        yy, xx = np.indices(source.shape[:2])
        mapping = np.stack((
            2.0 * (xx + 0.5) / source.shape[1] - 1.0,
            2.0 * (yy + 0.5) / source.shape[0] - 1.0,
        ), axis=-1).astype(np.float32)
        protected = np.zeros(source.shape[:2], bool)
        protected[2:6, 3:7] = True
        result, valid, fraction = overlay_dense_anchor_warp(
            global_image, global_valid, source, protected,
            np.zeros_like(protected), mapping,
            np.ones(source.shape[:2], np.float32), np.eye(3),
            (source.shape[1], source.shape[0]),
            certainty_floor=0.1, certainty_feather=0.1)
        self.assertTrue(np.all(result[protected] == 0))
        self.assertTrue(np.all(result[~protected] == (20, 80, 160)))
        self.assertTrue(valid.all())
        self.assertGreater(fraction, 0.4)

    def test_regularized_overlay_replaces_only_active_pixels(self):
        source = np.full((8, 10, 3), (20, 80, 160), np.uint8)
        global_image = np.zeros_like(source)
        global_valid = np.zeros(source.shape[:2], bool)
        yy, xx = np.indices(source.shape[:2])
        mapping = np.stack((xx, yy), axis=-1).astype(np.float32)
        active = np.zeros(source.shape[:2], bool)
        active[2:6, 3:7] = True
        result, valid = overlay_regularized_match_warp(
            global_image, global_valid, source, mapping, active,
            np.eye(3), (source.shape[1], source.shape[0]))
        self.assertTrue(np.all(result[active] == (20, 80, 160)))
        self.assertTrue(np.all(result[~active] == 0))
        np.testing.assert_array_equal(valid, active)


if __name__ == "__main__":
    unittest.main()
