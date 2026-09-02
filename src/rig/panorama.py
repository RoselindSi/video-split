"""Depth-aware, content-refined panorama rendering for the six-camera rig.

Calibration defines where a view starts; it does not solve parallax.  This
renderer obtains range from the three stereo pairs, reprojects every one of
the SIX RGB views to a common virtual eye, and learns the small photometric and
flow residuals from image content.  The learned corrections are frozen for a
clip so a moving hand cannot drag the panorama warp around from frame to frame.

The depth source is injectable.  The built-in backend is the repository's
SGBM reconstruction; a temporally consistent learned stereo backend can return
the same ``(range_m, valid)`` contract without changing the compositor.
"""
from __future__ import annotations

import numpy as np

from src.rig.render_wide import off_axis_deg
from src.rig.seam_fix import (apply_flow, apply_photometric, blend_weights,
                              compose, densify_range, fit_photometric,
                              fit_residual_flow, residual_flow,
                              source_maps_perpixel)


DEFAULT_DEPTH_M = 0.6
DEFAULT_FIT_FRAMES = 6
FLOW_SCALE = 0.25
LOW_FREQUENCY_SIGMA = 12.0


def _remap_mask(mask, flow):
    import cv2
    h, w = mask.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    out = cv2.remap(mask.astype(np.uint8), gx - flow[..., 0],
                    gy - flow[..., 1], cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 0


def _small_flow(src, ref, overlap, scale=FLOW_SCALE):
    """Residual flow at reduced resolution; displacement is in small pixels."""
    import cv2
    h, w = src.shape[:2]
    size = (max(32, int(round(w * scale))),
            max(24, int(round(h * scale))))
    a = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(overlap.astype(np.uint8), size,
                   interpolation=cv2.INTER_NEAREST) > 0
    return residual_flow(a, b, m), m


def _detail_preserving_compose(warped, valid, weights, hard, reach, gate,
                               low_sigma=LOW_FREQUENCY_SIGMA):
    """Blend low-frequency tone while keeping detail from one physical view."""
    import cv2
    soft, owner, gated = compose(warped, valid, weights, hard, reach,
                                 gate=gate)
    hard_image = np.zeros_like(soft)
    for i, image in warped.items():
        selected = reach & (hard == i)
        hard_image[selected] = image[selected]
    low_soft = cv2.GaussianBlur(soft, (0, 0), low_sigma)
    low_hard = cv2.GaussianBlur(hard_image, (0, 0), low_sigma)
    corrected = (hard_image.astype(np.float32)
                 + low_soft.astype(np.float32)
                 - low_hard.astype(np.float32))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    out = np.where(gated[..., None], hard_image, corrected)
    out[~reach] = 0
    return out, owner, gated


class DepthAwarePanorama:
    """Stateful all-six-view renderer.

    ``fit`` reads a handful of synchronized source dictionaries and estimates
    gain/bias plus a robust residual flow against the middle module.  ``render``
    then keeps those corrections fixed while depth and visibility remain
    frame-specific.
    """

    def __init__(self, rig, vcam, depth_m=DEFAULT_DEPTH_M, use_depth=True,
                 use_residual_flow=True, depth_provider=None, blend_temp=6.0,
                 disagreement_gate=40.0, flow_scale=FLOW_SCALE):
        self.rig = rig
        self.vcam = vcam
        self.depth_m = float(depth_m)
        self.use_depth = bool(use_depth)
        self.use_residual_flow = bool(use_residual_flow)
        self.depth_provider = depth_provider
        self.blend_temp = float(blend_temp)
        self.disagreement_gate = float(disagreement_gate)
        self.flow_scale = float(flow_scale)
        self.camera_names = tuple(sorted(rig.cameras))
        mid = rig.modules[len(rig.modules) // 2].left.name
        self.reference = self.camera_names.index(mid)
        self.photo = {self.reference: (np.ones(3), np.zeros(3))}
        self.flows = {}
        self._rect_cache = {}
        self.n_unguided = 0
        self._cost = {
            i: off_axis_deg(rig, name, vcam).astype(np.float32)
            for i, name in enumerate(self.camera_names)
        }
        self.last_stats = {}

    def _range(self, sources):
        h, w = self.vcam.height, self.vcam.width
        if not self.use_depth:
            return np.full((h, w), self.depth_m, np.float32), 0.0
        if self.depth_provider is None:
            from src.rig.wide_depth import wide_depth
            got = wide_depth(self.rig, self.vcam, sources,
                             rect_cache=self._rect_cache)
        else:
            got = self.depth_provider(self.rig, self.vcam, sources)
        if isinstance(got, tuple):
            measured, valid = got
        else:
            measured, valid = got.range_m, got.valid
        valid = np.asarray(valid, bool)
        # THE GUIDE IS NOT OPTIONAL AND LEAVING IT OUT PUT WAVES IN THE OUTPUT.
        # `densify_range` smooths the range before it is used as a warp field,
        # and with no guide it does so isotropically: depth bleeds across
        # object boundaries, the sampling coordinate follows it, and the
        # picture ripples along every edge where near meets far. Guided by the
        # image, the same filter holds the depth step where the picture has
        # one, so a hand keeps its own depth instead of being averaged into
        # the bench behind it.
        #
        # The guide has to be in the virtual camera's frame, and none exists
        # yet at this point -- the range is what the warp needs. So the middle
        # module is warped once at the constant fallback depth purely to make
        # one. That costs a single remap, about two milliseconds against the
        # stereo match already done above, and it does not have to be
        # geometrically perfect: it is being read for WHERE THE EDGES ARE.
        guide = self._constant_depth_guide(sources)
        dense = densify_range(measured, valid, fallback=self.depth_m,
                              guide=guide)
        return dense, float(valid.mean())

    def _constant_depth_guide(self, sources):
        """The middle module at constant depth, as an edge reference. -> img

        Returns None when no guide can be built, which falls back to the
        un-guided smoothing -- the behaviour that put waves in the picture.
        That is a degradation, so it is COUNTED. Silent fallbacks are how this
        pipeline shipped a no-op residual flow, a deleted two-hand cap and a
        mask that never asked the tracker where the hand was; each looked
        fine and each cost a render cycle to find."""
        import cv2
        from src.rig.geometry import source_maps
        mid = self.camera_names[len(self.camera_names) // 2]
        if mid not in sources:
            mid = next((n for n in self.camera_names if n in sources), None)
        if mid is None:
            self.n_unguided += 1
            return None
        key = ("guide", mid, round(float(self.depth_m), 6))
        try:
            if key not in self._rect_cache:
                self._rect_cache[key] = source_maps(self.rig, mid, self.vcam,
                                                    self.depth_m)
            mx, my, _ok = self._rect_cache[key]
            return cv2.remap(sources[mid], mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
        except Exception:                       # noqa: BLE001
            self.n_unguided += 1
            return None

    def _warp(self, sources, range_m):
        import cv2
        warped, valid = {}, {}
        for i, name in enumerate(self.camera_names):
            if name not in sources:
                continue
            mx, my, ok = source_maps_perpixel(
                self.rig, name, self.vcam, range_m)
            warped[i] = cv2.remap(
                sources[name], mx, my, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            valid[i] = ok
        return warped, valid, {i: self._cost[i] for i in warped}

    def fit(self, source_frames):
        """Fit clip-constant photometric and flow residuals from synchronized frames."""
        photo_est, flow_est = {}, {}
        n = 0
        for sources in source_frames:
            if not sources:
                continue
            range_m, _ = self._range(sources)
            warped, valid, _ = self._warp(sources, range_m)
            if self.reference not in warped:
                continue
            n += 1
            ref = warped[self.reference]
            for i, img in warped.items():
                if i == self.reference:
                    continue
                overlap = valid[i] & valid[self.reference]
                if overlap.sum() < 2000:
                    continue
                photo_est.setdefault(i, []).append(
                    fit_photometric([(img, ref, overlap)], min_px=1000))
                if self.use_residual_flow:
                    flow_est.setdefault(i, []).append(
                        _small_flow(img, ref, overlap, self.flow_scale))

        for i, estimates in photo_est.items():
            gains = np.stack([x[0] for x in estimates])
            biases = np.stack([x[1] for x in estimates])
            self.photo[i] = (np.median(gains, axis=0),
                             np.median(biases, axis=0))
        if self.use_residual_flow:
            import cv2
            size = (self.vcam.width, self.vcam.height)
            for i, samples in flow_est.items():
                small = fit_residual_flow(
                    samples, max_px=max(1.0, 12.0 * self.flow_scale))
                if small is None:
                    continue
                full = cv2.resize(small, size, interpolation=cv2.INTER_LINEAR)
                full /= max(self.flow_scale, 1e-6)
                self.flows[i] = full.astype(np.float32)
        return {"fit_frames": n, "photo_views": len(self.photo),
                "flow_views": len(self.flows)}

    def render(self, sources):
        """Render one synchronized six-view frame.

        Returns the same four-item shape as ``render_wide.render`` so callers
        can switch implementations without changing the downstream pipeline.
        """
        range_m, depth_coverage = self._range(sources)
        warped, valid, cost = self._warp(sources, range_m)
        if not warped:
            raise ValueError("no calibrated camera image reached the panorama")
        for i in list(warped):
            gain, bias = self.photo.get(i, (np.ones(3), np.zeros(3)))
            warped[i] = apply_photometric(warped[i], gain, bias)
            if i in self.flows:
                warped[i] = apply_flow(warped[i], self.flows[i])
                valid[i] = _remap_mask(valid[i], self.flows[i])

        # No camera receives permanent authority.  Per-pixel measured range
        # has put the views on one surface; off-axis quality now decides the
        # hard fallback, while agreeing views share a soft transition.
        weights, hard, reach = blend_weights(
            valid, cost, mid_i=None, mid_authority_deg=0.0,
            temp=self.blend_temp)
        rgb, owner, gated = _detail_preserving_compose(
            warped, valid, weights, hard, reach,
            gate=self.disagreement_gate)
        self.last_stats = {
            "renderer": "depth-aware-six-view",
            "n_views": len(warped),
            "depth_coverage": depth_coverage,
            "gated_frac": float(gated.mean()),
            "flow_views": len(self.flows),
        }
        return rgb, owner, self.last_stats, range_m
