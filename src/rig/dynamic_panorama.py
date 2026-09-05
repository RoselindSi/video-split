"""Ghost-free ownership for dynamic content in a learned panorama.

The learned scene supplies geometry and a static fallback, while synchronized
camera images supply the current frame.  A moving object must never be
averaged across cameras: parallax makes the two observations occupy different
pixels even when the static background is aligned correctly.

This module therefore detects disagreement between the two geometrically best
views, groups nearby disagreement into connected regions, and assigns every
region to one physical camera.  Ownership is sticky across frames, but only
while the previous camera remains a competitive, valid observation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicOwnershipConfig:
    disagreement: float = 40.0 / 255.0
    foreground_disagreement: float = 48.0 / 255.0
    max_pair_cost_gap: float = 35.0
    close_px: int = 20
    dilate_px: int = 6
    min_component_px: int = 24
    max_component_fraction: float = 0.10
    min_source_coverage: float = 0.55
    invalid_cost: float = 90.0
    history_overlap: float = 0.08
    switch_margin: float = 4.0


def _float_rgb(image):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("images must have shape [height, width, 3]")
    value = value.astype(np.float32)
    if np.issubdtype(np.asarray(image).dtype, np.integer) or \
            (value.size and float(np.nanmax(value)) > 1.5):
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def geometric_owner(valid, cost):
    """Choose the valid camera with the lowest geometric cost per pixel."""
    keys = sorted(valid)
    if not keys or keys != sorted(cost):
        raise ValueError("valid and cost must contain the same cameras")
    shape = np.asarray(valid[keys[0]]).shape
    scores = []
    for key in keys:
        if np.asarray(valid[key]).shape != shape or \
                np.asarray(cost[key]).shape != shape:
            raise ValueError("valid and cost maps must share one shape")
        scores.append(np.where(valid[key], cost[key], np.inf))
    stack = np.stack(scores)
    reachable = np.isfinite(stack).any(axis=0)
    owner = np.asarray(keys, np.int16)[np.argmin(stack, axis=0)]
    return np.where(reachable, owner, -1).astype(np.int16)


def _owner_image(images, owner):
    output = np.zeros((*owner.shape, 3), np.float32)
    for key, image in images.items():
        selected = owner == key
        output[selected] = _float_rgb(image)[selected]
    return output


def _owner_value(values, owner):
    output = np.zeros(owner.shape, np.float32)
    for key, value in values.items():
        selected = owner == key
        output[selected] = np.asarray(value, np.float32)[selected]
    return output


def disagreement_mask(warped, valid, cost, config=DynamicOwnershipConfig(),
                      foreground=None):
    """Find foreground pixels where the best overlapping views disagree.

    Cross-view RGB difference alone is not a motion detector.  A small depth
    error makes every textured static edge disagree, which can connect a
    complete tabletop into one component.  ``foreground`` is each current
    source's residual against the learned static scene at the same viewpoint;
    when supplied, at least one of the compared sources must also disagree
    with that learned background.
    """
    keys = sorted(warped)
    if keys != sorted(valid) or keys != sorted(cost):
        raise ValueError("warped, valid and cost must contain the same cameras")
    if len(keys) < 2:
        return np.zeros(np.asarray(valid[keys[0]]).shape, bool)

    score = np.stack([
        np.where(valid[key], cost[key], np.inf) for key in keys
    ])
    order = np.argsort(score, axis=0, kind="stable")
    best_index, second_index = order[0], order[1]
    best_cost = np.take_along_axis(score, best_index[None], axis=0)[0]
    second_cost = np.take_along_axis(score, second_index[None], axis=0)[0]
    key_array = np.asarray(keys, np.int16)
    best_owner = key_array[best_index]
    second_owner = key_array[second_index]
    best_image = _owner_image(warped, best_owner)
    second_image = _owner_image(warped, second_owner)
    difference = np.abs(best_image - second_image).mean(axis=2)
    comparable = np.isfinite(best_cost) & np.isfinite(second_cost)
    cost_gap = np.full(best_cost.shape, np.inf, np.float32)
    np.subtract(second_cost, best_cost, out=cost_gap, where=comparable)
    comparable &= cost_gap <= config.max_pair_cost_gap
    mask = comparable & (difference >= config.disagreement)
    if foreground is not None:
        if keys != sorted(foreground):
            raise ValueError(
                "foreground must contain the same cameras as warped")
        foreground_difference = np.maximum(
            _owner_value(foreground, best_owner),
            _owner_value(foreground, second_owner),
        )
        mask &= foreground_difference >= config.foreground_disagreement

    if config.close_px > 0 or config.dilate_px > 0:
        import cv2
        binary = mask.astype(np.uint8)
        if config.close_px > 0:
            radius = int(config.close_px)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if config.dilate_px > 0:
            radius = int(config.dilate_px)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            binary = cv2.dilate(binary, kernel)
        mask = binary > 0
    return mask


def _component_candidate(component, valid, cost, config):
    candidates = {}
    area = int(component.sum())
    for key in sorted(valid):
        seen = component & np.asarray(valid[key], bool)
        coverage = float(seen.sum()) / max(area, 1)
        if coverage < config.min_source_coverage:
            continue
        values = np.asarray(cost[key], np.float32)[seen]
        if not values.size:
            continue
        candidates[key] = (
            float(np.median(values))
            + (1.0 - coverage) * config.invalid_cost
        )
    return candidates


def regularize_dynamic_owner(
        base_owner, dynamic, valid, cost, config=DynamicOwnershipConfig(),
        previous_owner=None, previous_dynamic=None):
    """Give each connected dynamic region one camera owner.

    Pixels not covered by that camera become learned-background holes rather
    than falling through to a second current image.  This is deliberate: a
    small fill is preferable to rendering the same hand twice.
    """
    import cv2

    owner = np.asarray(base_owner, np.int16).copy()
    dynamic = np.asarray(dynamic, bool)
    accepted = np.zeros(dynamic.shape, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dynamic.astype(np.uint8), connectivity=8)
    kept = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_component_px:
            continue
        if area / dynamic.size > config.max_component_fraction:
            continue
        component = labels == label
        candidates = _component_candidate(component, valid, cost, config)
        if not candidates:
            owner[component] = -1
            accepted[component] = True
            kept += 1
            continue
        winner = min(candidates, key=candidates.get)

        if previous_owner is not None and previous_dynamic is not None:
            overlap = component & np.asarray(previous_dynamic, bool)
            if float(overlap.sum()) / area >= config.history_overlap:
                old = np.asarray(previous_owner, np.int16)[overlap]
                old = old[old >= 0]
                if old.size:
                    values, counts = np.unique(old, return_counts=True)
                    previous = int(values[np.argmax(counts)])
                    if previous in candidates and \
                            candidates[previous] <= \
                            candidates[winner] + config.switch_margin:
                        winner = previous

        owner[component] = -1
        selected = component & np.asarray(valid[winner], bool)
        owner[selected] = winner
        accepted[component] = True
        kept += 1
    return owner, accepted, kept


def compose_single_source(warped, valid, owner, background,
                          background_valid=None):
    """Compose without ever averaging RGB from two physical cameras."""
    background = _float_rgb(background)
    owner = np.asarray(owner, np.int16)
    if owner.shape != background.shape[:2]:
        raise ValueError("owner and background shapes differ")
    if background_valid is None:
        background_valid = np.ones(owner.shape, bool)
    else:
        background_valid = np.asarray(background_valid, bool)

    output = np.zeros_like(background)
    filled = np.zeros(owner.shape, bool)
    for key, image in warped.items():
        selected = (owner == key) & np.asarray(valid[key], bool)
        output[selected] = _float_rgb(image)[selected]
        filled[selected] = True
    fallback = ~filled & background_valid
    output[fallback] = background[fallback]
    return output, filled, fallback


class DynamicSingleSourceCompositor:
    """Stateful region ownership with bounded camera-switch hysteresis."""

    def __init__(self, config=DynamicOwnershipConfig()):
        self.config = config
        self.previous_owner = None
        self.previous_dynamic = None

    def render(self, warped, valid, cost, background, background_valid=None,
               foreground=None):
        base = geometric_owner(valid, cost)
        detected = disagreement_mask(
            warped, valid, cost, config=self.config,
            foreground=foreground)
        owner, dynamic, components = regularize_dynamic_owner(
            base, detected, valid, cost, config=self.config,
            previous_owner=self.previous_owner,
            previous_dynamic=self.previous_dynamic)
        image, current, fallback = compose_single_source(
            warped, valid, owner, background, background_valid)
        switches = 0
        comparable = np.zeros(dynamic.shape, bool)
        if self.previous_owner is not None and self.previous_dynamic is not None:
            comparable = dynamic & self.previous_dynamic
            switches = int(np.count_nonzero(
                comparable & (owner != self.previous_owner)))
        self.previous_owner = owner.copy()
        self.previous_dynamic = dynamic.copy()
        stats = {
            "detected_fraction": float(detected.mean()),
            "dynamic_fraction": float(dynamic.mean()),
            "dynamic_components": components,
            "current_fraction": float(current.mean()),
            "background_fallback_fraction": float(fallback.mean()),
            "owner_switch_pixels": switches,
            "owner_comparable_pixels": int(comparable.sum()),
        }
        return image, owner, dynamic, stats
