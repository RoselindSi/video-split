"""Focused regression tests for motion, association, ownership and thresholds.

Run:
    python3 tests/test_hand_tracking_policy.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig.hand_detect import OwnHold, detect  # noqa: E402
from src.rig.hand_track import Tracker, _centre  # noqa: E402


SHAPE = (900, 1600, 3)


def _det(cx, cy=400, size=80, *, conf=0.9, edge=None, owner=False,
         owner_p=0.1, side=None):
    box = np.array([cx - size, cy - size, cx + size, cy + size], dtype=int)
    kp = np.array([[cx, cy]] * 21, dtype=float)
    return {"box": box, "kp": kp, "conf": float(conf), "edge": edge,
            "side": side, "owner": bool(owner), "owner_p": float(owner_p),
            "rule_owner": bool(owner), "exit": None}


class HandTrackingPolicyTest(unittest.TestCase):
    def test_motion_prediction_advances_a_coasting_box(self):
        moving = Tracker(predict_motion=True)
        static = Tracker(predict_motion=False)
        for tracker in (moving, static):
            tracker.update([_det(100)], SHAPE)
            tracker.update([_det(120)], SHAPE)
            tracker.update([], SHAPE)

        moving_box = moving.coasting(max_prediction_age=2)[0][1]["box"]
        static_box = static.coasting(max_prediction_age=2)[0][1]["box"]
        self.assertAlmostEqual(float(_centre(moving_box)[0]), 140.0, places=5)
        self.assertAlmostEqual(float(_centre(static_box)[0]), 120.0, places=5)

    def test_reacquired_hand_with_incompatible_entry_starts_new_track(self):
        safe = Tracker()
        legacy = Tracker(predict_motion=False, rich_association=False,
                         max_assoc_cost=None, unmatched_cost=None)
        for tracker in (safe, legacy):
            first = tracker.update([_det(400, edge="bottom")], SHAPE)[0]
            tracker.update([], SHAPE)
            second = tracker.update([_det(405, edge="left")], SHAPE)[0]
            tracker.result = first, second

        self.assertNotEqual(*safe.result)
        self.assertEqual(*legacy.result)

    def test_low_confidence_detection_continues_but_cannot_start_track(self):
        tracker = Tracker()
        first = tracker.update([_det(300, conf=0.9)], SHAPE,
                               new_track_conf=0.6, continue_conf=0.25)[0]
        ids = tracker.update([_det(310, conf=0.3),
                              _det(1200, conf=0.3)], SHAPE,
                             new_track_conf=0.6, continue_conf=0.25)
        self.assertEqual(ids[0], first)
        self.assertIsNone(ids[1])
        self.assertEqual(tracker.n_new, 1)
        self.assertEqual(tracker.provenance,
                         ["matched_low", "dropped"])

    def test_reacquired_self_does_not_inherit_old_self_verdict(self):
        hand = _det(400, owner=True, owner_p=0.95)
        ambiguous_other = _det(405, owner=False, owner_p=0.40)

        safe = OwnHold(rule_w=0.0, self_reconfirm_frames=2)
        legacy = OwnHold(rule_w=0.0, self_reconfirm_frames=2)
        safe.update([hand], [(True, 0.95)], ids=[7])
        legacy.update([hand], [(True, 0.95)], ids=[7])
        safe.update([], [], ids=[])
        legacy.update([], [], ids=[])

        safe_flag = safe.update([ambiguous_other], [(False, 0.40)], ids=[7],
                                reacquired={7})[0][0]
        legacy_flag = legacy.update([ambiguous_other], [(False, 0.40)],
                                    ids=[7], reacquired=())[0][0]
        self.assertFalse(safe_flag)
        self.assertTrue(legacy_flag)

    def test_reacquired_self_needs_two_fresh_supporting_frames(self):
        hand = _det(400, owner=True, owner_p=0.95)
        hold = OwnHold(rule_w=0.0, self_reconfirm_frames=2)
        hold.update([hand], [(True, 0.95)], ids=[7])
        hold.update([], [], ids=[])

        first = hold.update([hand], [(True, 0.95)], ids=[7],
                            reacquired={7})[0][0]
        second = hold.update([hand], [(True, 0.95)], ids=[7])[0][0]
        self.assertFalse(first)
        self.assertTrue(second)

    def test_detector_floor_zero_reaches_model_inference(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Boxes:
            xyxy = Array(np.empty((0, 4)))
            conf = Array(np.empty((0,)))
            cls = Array(np.empty((0,)))

            def __len__(self):
                return 0

        class Result:
            boxes = Boxes()
            keypoints = None

        class Model:
            def __init__(self):
                self.kwargs = None

            def __call__(self, image, **kwargs):
                del image
                self.kwargs = kwargs
                return [Result()]

        model = Model()
        self.assertEqual(detect(model, np.zeros(SHAPE, np.uint8),
                                min_conf=0), [])
        self.assertIn("conf", model.kwargs)
        self.assertEqual(model.kwargs["conf"], 0.0)


class ShippedDefaultsTest(unittest.TestCase):
    """The two policy numbers the pipeline now ships, and what they mean.

    Both were measured before being changed: the reconfirm frame cost 1,173
    covered own-hand frames over 145 recordings to keep 103 of a colleague's,
    and the grace keeps 144 own hand-frames for 17 of a colleague's. They are
    asserted here because the values live in `demo_video` and a silent revert
    would look exactly like nothing."""

    def test_a_reacquired_self_hand_is_judged_on_the_frame_it_returns(self):
        from src.rig.demo_video import SELF_RECONFIRM_FRAMES
        hand = _det(400, owner=True, owner_p=0.95)
        hold = OwnHold(rule_w=0.0, self_reconfirm_frames=SELF_RECONFIRM_FRAMES)
        hold.update([hand], [(True, 0.95)], ids=[7])
        hold.update([], [], ids=[])                       # the detector drops it
        back = hold.update([hand], [(True, 0.95)], ids=[7], reacquired={7})[0][0]
        self.assertTrue(back)
        # ...and the evidence still has to say so: a hand that comes back
        # looking foreign is covered, which is what the reset is for.
        hold2 = OwnHold(rule_w=0.0, self_reconfirm_frames=SELF_RECONFIRM_FRAMES)
        hold2.update([hand], [(True, 0.95)], ids=[7])
        hold2.update([], [], ids=[])
        self.assertFalse(hold2.update([_det(405, owner=False, owner_p=0.40)],
                                      [(False, 0.40)], ids=[7],
                                      reacquired={7})[0][0])

    def test_a_new_box_counts_as_beside_a_hand_only_when_it_touches_one(self):
        from src.rig.demo_video import NEAR_SELF, NEW_HAND_GRACE, _touches
        self.assertGreaterEqual(NEW_HAND_GRACE, 1)
        palm, fingers = [100, 200, 200, 300], [100, 120, 200, 200]
        self.assertTrue(_touches(fingers, palm, NEAR_SELF * 100))
        across_the_bench = [800, 120, 900, 200]
        self.assertFalse(_touches(across_the_bench, palm, NEAR_SELF * 100))


if __name__ == "__main__":
    unittest.main()
