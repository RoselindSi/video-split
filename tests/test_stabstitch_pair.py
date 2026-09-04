import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.rig.stabstitch_pair import (find_module, pair_key, run_official,
                                     valid_fraction)
from src.rig.stabstitch_pair_eval import edge_chamfer
from src.rig.fast_foundation_stereo import (FastFoundationStereo,
                                             FastFoundationStereoProvider,
                                             disparity_to_depth)
from src.rig.fast_foundation_stereo_eval import temporal_disparity_error
from src.rig.fast_foundation_stereo_sixview import (build_renderer,
                                                     videos_from_databag)


class _Camera:
    def __init__(self, name):
        self.name = name


class _Module:
    def __init__(self, left, right):
        self.left = _Camera(left)
        self.right = _Camera(right)


class _IO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _StereoSession:
    def __init__(self, providers=None):
        self.feed = None
        self._providers = providers or ["FakeExecutionProvider"]

    def get_inputs(self):
        return [_IO("left_image", [1, 3, 8, 12]),
                _IO("right_image", [1, 3, 8, 12])]

    def get_outputs(self):
        return [_IO("disparity", [1, 1, 8, 12])]

    def get_providers(self):
        return self._providers

    def run(self, names, feed):
        self.feed = feed
        return [np.full((1, 1, 8, 12), 6.0, np.float32)]


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

    def test_fast_foundation_stereo_restores_input_disparity_scale(self):
        session = _StereoSession()
        model = FastFoundationStereo(session=session)
        left = np.zeros((4, 6, 3), np.uint8)
        left[..., 2] = 255  # BGR red must become normalized RGB channel zero.
        disparity = model.disparity(left, left)
        self.assertEqual(disparity.shape, (4, 6))
        np.testing.assert_allclose(disparity, 3.0)
        expected_red = (1.0 - 0.485) / 0.229
        self.assertAlmostEqual(
            float(session.feed["left_image"][0, 0, 0, 0]), expected_red,
            places=5)
        self.assertEqual(model.providers, ["FakeExecutionProvider"])

    def test_fast_foundation_stereo_rejects_silent_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "did not load"):
            FastFoundationStereo(
                session=_StereoSession(["CPUExecutionProvider"]),
                require_cuda=True)

    def test_fast_foundation_provider_reaches_one_owner_rgbd_path(self):
        provider = FastFoundationStereoProvider(
            session=_StereoSession(["CUDAExecutionProvider"]),
            require_cuda=True)
        expected = object()
        with mock.patch("src.rig.wide_depth.wide_rgbd",
                        return_value=expected) as render:
            result = provider.rgbd("rig", "vcam", {"cam1": "image"})

        self.assertIs(result, expected)
        self.assertIs(render.call_args.kwargs["rect_cache"],
                      provider.rect_cache)
        self.assertEqual(render.call_args.kwargs["depth_tie_m"], 0.02)
        self.assertIs(render.call_args.kwargs["matcher"].__self__,
                      provider.stereo)

    def test_six_view_runner_wires_learned_depth_to_left_eye_texture(self):
        provider = object()
        renderer = object()
        with mock.patch(
                "src.rig.fast_foundation_stereo.FastFoundationStereoProvider",
                return_value=provider) as provider_class, mock.patch(
                "src.rig.panorama.DepthAwarePanorama",
                return_value=renderer) as renderer_class:
            got = build_renderer(
                "rig", "vcam", "model.onnx", rectified_size=(960, 720))

        self.assertEqual(got, (renderer, provider))
        self.assertEqual(provider_class.call_args.kwargs["rectified_size"],
                         (960, 720))
        self.assertTrue(provider_class.call_args.kwargs["owner_aligned"])
        self.assertEqual(
            provider_class.call_args.kwargs["owner_mid_authority_deg"], 72.0)
        self.assertIs(renderer_class.call_args.kwargs["depth_provider"],
                      provider)
        self.assertEqual(renderer_class.call_args.kwargs["texture_mode"],
                         "module_left")
        self.assertTrue(renderer_class.call_args.kwargs["use_depth"])
        self.assertEqual(renderer_class.call_args.kwargs["mid_authority_deg"],
                         72.0)

    def test_six_view_runner_refuses_an_incomplete_databag(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("calibration.yaml", "cam12.mp4", "cam34.mp4"):
                open(os.path.join(directory, name), "w").close()
            with self.assertRaisesRegex(FileNotFoundError, "cam56.mp4"):
                videos_from_databag(directory)

    def test_disparity_to_depth_keeps_only_calibrated_range(self):
        disparity = np.array([[0.0, 10.0, 100.0]], np.float32)
        depth, valid = disparity_to_depth(
            disparity, focal_px=100.0, baseline_m=0.1,
            min_depth=0.2, max_depth=2.0)
        self.assertEqual(valid.tolist(), [[False, True, False]])
        self.assertAlmostEqual(float(depth[0, 1]), 1.0)
        self.assertTrue(np.isnan(depth[0, 0]))
        self.assertTrue(np.isnan(depth[0, 2]))

    def test_temporal_disparity_error_is_motion_aligned(self):
        image = np.zeros((64, 96, 3), np.uint8)
        image[16:48, 24:72] = 255
        disparity = np.full((64, 96), 12.0, np.float32)
        stable = temporal_disparity_error(
            image, image, disparity, disparity)
        jumped = temporal_disparity_error(
            image, image, disparity, disparity + 4.0)
        self.assertLess(stable["p90_px"], 0.01)
        self.assertGreater(jumped["median_px"], 3.9)
        self.assertGreater(jumped["jump_gt3_fraction"], 0.99)

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
