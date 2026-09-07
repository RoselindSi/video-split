import unittest

import numpy as np

from src.rig.multi_homography import (
    PairGeometryConfig,
    accepted_components,
    analyze_pair,
    fit_piecewise_homographies,
    grid_coverage,
    static_correspondence_mask,
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


if __name__ == "__main__":
    unittest.main()
