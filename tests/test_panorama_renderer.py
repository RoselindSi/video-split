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
            seen_ranges.append(float(np.median(range_m)))
            return _maps(rig, name, vcam, range_m)

        with mock.patch("src.rig.panorama.source_maps_perpixel",
                        side_effect=maps):
            _, _, stats, _ = renderer.render(_sources())

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], 6)
        self.assertAlmostEqual(stats["depth_coverage"], 0.5)
        self.assertTrue(all(abs(x - 1.25) < 1e-4 for x in seen_ranges))

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


if __name__ == "__main__":
    unittest.main()
