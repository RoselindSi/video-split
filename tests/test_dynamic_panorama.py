"""Regression tests for learned-depth, single-source dynamic composition."""
from __future__ import annotations

import unittest

import numpy as np

from src.rig.dynamic_panorama import (
    DynamicOwnershipConfig,
    DynamicSingleSourceCompositor,
    compose_single_source,
    disagreement_mask,
    geometric_owner,
    regularize_dynamic_owner,
)


H, W = 40, 80


def _valid():
    return {0: np.ones((H, W), bool), 1: np.ones((H, W), bool)}


def _cost(left=0.0, right=1.0):
    return {
        0: np.full((H, W), left, np.float32),
        1: np.full((H, W), right, np.float32),
    }


class DynamicPanoramaTest(unittest.TestCase):
    def test_geometric_owner_never_selects_invalid_camera(self):
        valid = _valid()
        valid[0][:, W // 2:] = False
        owner = geometric_owner(valid, _cost(0.0, 1.0))
        self.assertTrue(np.all(owner[:, :W // 2] == 0))
        self.assertTrue(np.all(owner[:, W // 2:] == 1))

    def test_displaced_object_is_one_disagreement_region(self):
        a = np.zeros((H, W, 3), np.float32)
        b = np.zeros_like(a)
        a[14:26, 20:30] = 1.0
        b[14:26, 34:44] = 1.0
        config = DynamicOwnershipConfig(
            disagreement=0.2, close_px=5, dilate_px=2,
            min_component_px=1)
        mask = disagreement_mask(
            {0: a, 1: b}, _valid(), _cost(), config)
        self.assertTrue(mask[20, 24])
        self.assertTrue(mask[20, 38])
        self.assertTrue(mask[20, 32])

    def test_unreachable_pixels_do_not_emit_numeric_warnings(self):
        valid = _valid()
        valid[0][:] = False
        valid[1][:] = False
        with np.errstate(all="raise"):
            mask = disagreement_mask(
                {0: np.zeros((H, W, 3)), 1: np.zeros((H, W, 3))},
                valid, _cost(),
                DynamicOwnershipConfig(close_px=0, dilate_px=0))
        self.assertFalse(mask.any())

    def test_connected_dynamic_region_gets_exactly_one_owner(self):
        base = np.zeros((H, W), np.int16)
        base[:, W // 2:] = 1
        dynamic = np.zeros((H, W), bool)
        dynamic[10:30, 24:56] = True
        cost = _cost(5.0, 6.0)
        config = DynamicOwnershipConfig(
            min_component_px=1, min_source_coverage=0.5)
        owner, accepted, components = regularize_dynamic_owner(
            base, dynamic, _valid(), cost, config)
        self.assertEqual(components, 1)
        self.assertTrue(accepted[dynamic].all())
        self.assertEqual(np.unique(owner[dynamic]).tolist(), [0])

    def test_previous_owner_is_kept_within_switch_margin(self):
        dynamic = np.zeros((H, W), bool)
        dynamic[10:30, 20:60] = True
        previous = np.full((H, W), -1, np.int16)
        previous[dynamic] = 1
        config = DynamicOwnershipConfig(
            min_component_px=1, min_source_coverage=0.5,
            switch_margin=4.0)
        owner, _, _ = regularize_dynamic_owner(
            geometric_owner(_valid(), _cost(4.0, 2.0)), dynamic,
            _valid(), _cost(0.0, 2.0), config,
            previous_owner=previous, previous_dynamic=dynamic)
        self.assertEqual(np.unique(owner[dynamic]).tolist(), [1])

    def test_uncovered_dynamic_pixels_fall_back_to_learned_background(self):
        red = np.zeros((H, W, 3), np.float32)
        red[..., 0] = 1.0
        blue = np.zeros_like(red)
        blue[..., 2] = 1.0
        background = np.full_like(red, 0.25)
        owner = np.zeros((H, W), np.int16)
        valid = _valid()
        valid[0][12:20, 30:40] = False
        owner[12:20, 30:40] = -1
        image, current, fallback = compose_single_source(
            {0: red, 1: blue}, valid, owner, background)
        self.assertTrue(np.all(image[12:20, 30:40] == 0.25))
        self.assertFalse(current[12:20, 30:40].any())
        self.assertTrue(fallback[12:20, 30:40].all())

    def test_end_to_end_output_contains_no_cross_camera_average(self):
        red = np.zeros((H, W, 3), np.uint8)
        red[..., 0] = 255
        blue = np.zeros_like(red)
        blue[..., 2] = 255
        compositor = DynamicSingleSourceCompositor(
            DynamicOwnershipConfig(
                disagreement=0.2, close_px=0, dilate_px=0,
                min_component_px=1))
        image, _, _, _ = compositor.render(
            {0: red, 1: blue}, _valid(), _cost(),
            np.full((H, W, 3), 0.5, np.float32))
        colours = np.unique(image.reshape(-1, 3), axis=0)
        self.assertEqual(colours.tolist(), [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
