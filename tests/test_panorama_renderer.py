"""Regression tests for the depth-aware six-view panorama renderer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig.panorama import (DepthAwarePanorama,
                              _detail_preserving_compose)  # noqa: E402
from src.rig.seam_fix import blend_weights, densify_range  # noqa: E402
from src.rig.wide_depth import composite_rgbd  # noqa: E402


H, W = 48, 120


def _rig():
    cameras = {f"cam{i}": SimpleNamespace(name=f"cam{i}")
               for i in range(1, 7)}
    modules = []
    for i, (left, right) in enumerate((("cam1", "cam2"),
                                       ("cam3", "cam4"),
                                       ("cam5", "cam6"))):
        modules.append(SimpleNamespace(
            name=f"module_{i}", left=cameras[left], right=cameras[right]))
    return SimpleNamespace(cameras=cameras, modules=tuple(modules))


VCAM = SimpleNamespace(width=W, height=H)


def _maps(rig, name, vcam, range_m):
    del rig, name, range_m
    yy, xx = np.mgrid[:vcam.height, :vcam.width]
    return xx.astype(np.float32), yy.astype(np.float32), \
        np.ones((vcam.height, vcam.width), bool)


def _cost(rig, name, vcam):
    del rig
    i = int(name[-1]) - 1
    centre = (i + 0.5) * vcam.width / 6.0
    xx = np.arange(vcam.width, dtype=np.float32)[None, :]
    return np.broadcast_to(np.abs(xx - centre),
                           (vcam.height, vcam.width)).copy()


def _sources():
    yy, xx = np.mgrid[:H, :W]
    out = {}
    for i in range(1, 7):
        image = np.empty((H, W, 3), np.uint8)
        image[..., 0] = (xx + i * 31) % 255
        image[..., 1] = (yy * 3 + i * 17) % 255
        image[..., 2] = i * 35
        out[f"cam{i}"] = image
    return out


class PanoramaRendererTest(unittest.TestCase):
    def _renderer(self, **kwargs):
        patches = (
            mock.patch("src.rig.panorama.off_axis_deg", side_effect=_cost),
            mock.patch("src.rig.panorama.source_maps_perpixel",
                       side_effect=_maps),
        )
        with patches[0], patches[1]:
            return DepthAwarePanorama(_rig(), VCAM, **kwargs)

    def test_all_six_rgb_views_participate(self):
        renderer = self._renderer(use_depth=False,
                                  use_residual_flow=False,
                                  disagreement_gate=0.0)
        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=_maps):
            image, owner, stats, _ = renderer.render(_sources())

        self.assertEqual(stats["n_views"], 6)
        self.assertEqual(set(np.unique(owner)), set(range(6)))
        self.assertGreater(int(image.sum()), 0)

    def test_stereo_right_eyes_can_be_geometry_only(self):
        renderer = self._renderer(
            use_depth=False, use_residual_flow=False,
            disagreement_gate=0.0, texture_mode="module_left")
        seen = []

        def maps(rig, name, vcam, range_m):
            seen.append(name)
            return _maps(rig, name, vcam, range_m)

        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=maps):
            _, _, stats, _ = renderer.render(_sources())

        self.assertEqual(stats["n_views"], 3)
        self.assertEqual(stats["texture_mode"], "module_left")
        self.assertEqual(set(seen), {"cam1", "cam3", "cam5"})

    def test_rgbd_composite_selects_one_colour_after_z_buffer(self):
        vcam = SimpleNamespace(
            width=20, height=10, eye=np.zeros(3), R=np.eye(3),
            hfov=np.radians(90), vfov=np.radians(60))
        point_near = np.array([[0.0, 0.0, 0.5]])
        point_far = np.array([[0.0, 0.0, 1.0]])
        red = np.array([[0, 0, 255]], np.uint8)
        blue = np.array([[255, 0, 0]], np.uint8)
        result = composite_rgbd(
            vcam,
            [("far", point_far, blue, np.array([0.0])),
             ("near", point_near, red, np.array([20.0]))],
            splat=1)
        self.assertEqual(result.rgb[5, 10].tolist(), red[0].tolist())
        self.assertEqual(int(result.module[5, 10]), 1)
        self.assertAlmostEqual(float(result.range_m[5, 10]), 0.5)

        same_surface = composite_rgbd(
            vcam,
            [("less_on_axis", point_near, red, np.array([10.0])),
             ("more_on_axis", point_near * 1.02, blue, np.array([1.0]))],
            splat=1, depth_tie_m=0.03)
        pixel = same_surface.rgb[5, 10]
        self.assertIn(pixel.tolist(), (red[0].tolist(), blue[0].tolist()))
        self.assertEqual(pixel.tolist(), blue[0].tolist())
        self.assertNotEqual(pixel.tolist(), [127, 0, 127])

    def test_depth_provider_controls_the_warp_range(self):
        calls = []

        def provider(rig, vcam, sources):
            calls.append((rig, vcam, len(sources)))
            measured = np.full((H, W), 1.25, np.float32)
            valid = np.zeros((H, W), bool)
            valid[:, :W // 2] = True
            return measured, valid

        renderer = self._renderer(depth_provider=provider,
                                  use_residual_flow=False)
        seen_ranges = []

        def maps(rig, name, vcam, range_m):
            seen_ranges.append(range_m.copy())
            return _maps(rig, name, vcam, range_m)

        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=maps):
            _, _, stats, _ = renderer.render(_sources())

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], 6)
        self.assertAlmostEqual(stats["depth_coverage"], 0.5)
        self.assertTrue(all(abs(float(np.median(x[:, :W // 3])) - 1.25)
                            < 1e-4 for x in seen_ranges))
        self.assertTrue(all(abs(float(np.median(x[:, -W // 6:])) - 0.6)
                            < 1e-4 for x in seen_ranges))

    def test_depth_is_temporally_damped_only_when_measurements_agree(self):
        values = iter((1.0, 1.06, 1.5))

        def provider(_rig, _vcam, _sources):
            value = next(values)
            return (np.full((H, W), value, np.float32),
                    np.ones((H, W), bool))

        renderer = self._renderer(depth_provider=provider,
                                  use_residual_flow=False)
        renderer._constant_depth_guide = lambda _sources: None
        seen = []

        def maps(rig, name, vcam, range_m):
            seen.append(float(range_m[H // 2, W // 2]))
            return _maps(rig, name, vcam, range_m)

        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=maps):
            renderer.render(_sources())
            renderer.render(_sources())
            renderer.render(_sources())

        per_frame = seen[::6]
        self.assertAlmostEqual(per_frame[0], 1.0, places=4)
        self.assertAlmostEqual(per_frame[1], 1.027, places=3)
        self.assertAlmostEqual(per_frame[2], 1.5, places=4)

    def test_frozen_residual_flow_is_applied_to_output(self):
        renderer = self._renderer(use_depth=False,
                                  use_residual_flow=True,
                                  disagreement_gate=0.0)
        one = {"cam1": _sources()["cam1"]}
        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=_maps):
            before = renderer.render(one)[0]
            flow = np.zeros((H, W, 2), np.float32)
            flow[..., 0] = 2.0
            renderer.flows[0] = flow
            after = renderer.render(one)[0]

        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(renderer.last_stats["flow_views"], 1)

    def test_blend_keeps_high_frequency_detail_from_one_camera(self):
        yy, xx = np.mgrid[:H, :W]
        checker = (((xx + yy) % 2) * 255).astype(np.uint8)
        sharp = np.repeat(checker[..., None], 3, axis=2)
        flat = np.full_like(sharp, 127)
        valid = {0: np.ones((H, W), bool),
                 1: np.ones((H, W), bool)}
        weights = {0: np.full((H, W), 0.5),
                   1: np.full((H, W), 0.5)}
        hard = np.zeros((H, W), np.int8)
        out, _, gated = _detail_preserving_compose(
            {0: sharp, 1: flat}, valid, weights, hard,
            np.ones((H, W), bool), gate=255)

        self.assertFalse(gated.any())
        self.assertGreater(float(out.std()), 110.0)

    def test_low_frequency_tone_correction_is_bounded(self):
        hard_image = np.full((H, W, 3), 100, np.uint8)
        bright = np.full((H, W, 3), 200, np.uint8)
        valid = {0: np.ones((H, W), bool),
                 1: np.ones((H, W), bool)}
        weights = {0: np.full((H, W), 0.5),
                   1: np.full((H, W), 0.5)}
        out, _, gated = _detail_preserving_compose(
            {0: hard_image, 1: bright}, valid, weights,
            np.zeros((H, W), np.int8), np.ones((H, W), bool), gate=255)

        self.assertFalse(gated.any())
        self.assertLessEqual(int(np.abs(out.astype(int) - 100).max()), 12)

    def test_pixelwise_gate_noise_is_not_rendered_as_speckles(self):
        yy, xx = np.mgrid[:H, :W]
        hard_image = np.full((H, W, 3), 100, np.uint8)
        alternate = hard_image.copy()
        alternate[(xx + yy) % 2 == 0] = 200
        valid = {0: np.ones((H, W), bool),
                 1: np.ones((H, W), bool)}
        weights = {0: np.full((H, W), 0.5),
                   1: np.full((H, W), 0.5)}
        out, _, gated = _detail_preserving_compose(
            {0: hard_image, 1: alternate}, valid, weights,
            np.zeros((H, W), np.int8), np.ones((H, W), bool), gate=40)

        self.assertAlmostEqual(float(gated.mean()), 0.5)
        self.assertEqual(float(out.std()), 0.0)

    def test_blend_uses_only_two_best_sources(self):
        valid = {i: np.ones((H, W), bool) for i in range(4)}
        cost = {i: np.full((H, W), float(i), np.float32)
                for i in range(4)}
        weights, hard, reach = blend_weights(
            valid, cost, mid_i=None, mid_authority_deg=0.0, temp=6.0)

        count = sum((weights[i] > 0).astype(np.uint8) for i in weights)
        self.assertTrue(reach.all())
        self.assertTrue((hard == 0).all())
        self.assertTrue((count == 2).all())
        self.assertTrue((weights[2] == 0).all())
        self.assertTrue((weights[3] == 0).all())

    def test_depth_hole_fill_stays_float_and_is_bounded(self):
        depth = np.full((41, 41), 1.013, np.float32)
        valid = np.ones(depth.shape, bool)
        valid[20, 20] = False
        dense = densify_range(depth, valid, fallback=0.6, median_px=1,
                              bilat_sigma=1e-6, max_inpaint_distance=3)
        self.assertAlmostEqual(float(dense[20, 20]), 1.013, places=2)

        valid[8:33, 8:33] = False
        dense = densify_range(depth, valid, fallback=0.6, median_px=1,
                              bilat_sigma=1e-6, max_inpaint_distance=3)
        self.assertAlmostEqual(float(dense[20, 20]), 0.6, places=3)


if __name__ == "__main__":
    unittest.main()
