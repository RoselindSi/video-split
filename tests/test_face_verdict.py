"""The size cap as a question: what it must keep, drop, and refuse to guess."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig import face_mask  # noqa: E402

SHAPE = (512, 1000, 3)          # W = 1000, so the cap at 0.18 is 180 px
SMALL = (10, 10, 90, 90, 0.9)   # 80 px, under the cap
BIG = (100, 10, 500, 300, 0.9)  # 400 px, over it


class FaceVerdictTest(unittest.TestCase):

    def test_without_verdicts_the_cap_still_refuses(self):
        """The shipped behaviour, unchanged when nobody was asked. It uncovers
        a real face about once per eighty seconds of this material and that
        was accepted knowingly; this test is what stops it changing by
        accident."""
        h = face_mask.Hold(frames=1, max_frac=0.18)
        self.assertEqual(h.update([BIG], shape=SHAPE), [])
        self.assertEqual(len(h.refused), 1)

    def test_a_verdict_of_face_keeps_a_box_the_cap_would_refuse(self):
        """The point of the whole change. On cam3 this is one box in thirteen
        and it is a colleague leaning into the camera -- the most identifiable
        person in the recording, and the one the size rule threw away."""
        h = face_mask.Hold(frames=1, max_frac=0.18,
                           verdicts={BIG[:4]: True})
        self.assertEqual(h.update([BIG], shape=SHAPE), [BIG[:4]])
        self.assertEqual(h.refused, [])

    def test_a_verdict_of_not_a_face_drops_it(self):
        """The other twelve: walls, tables, a wok, a parts bin."""
        h = face_mask.Hold(frames=1, max_frac=0.18,
                           verdicts={BIG[:4]: False})
        self.assertEqual(h.update([BIG], shape=SHAPE), [])
        self.assertEqual(len(h.refused), 1)

    def test_an_unjudged_box_is_covered_not_dropped(self):
        """Verdicts were supplied and this box is not among them, so the run
        that produced them is not this run. Covering is the safe side of a
        question about a face -- a wrongly kept mosaic costs a patch of bench,
        a wrongly dropped one puts a person in the video -- and the count is
        what makes the divergence findable."""
        h = face_mask.Hold(frames=1, max_frac=0.18,
                           verdicts={(1, 2, 3, 4): True})
        self.assertEqual(h.update([BIG], shape=SHAPE), [BIG[:4]])
        self.assertEqual(h.unjudged, 1)

    def test_small_boxes_are_never_asked(self):
        """Below the cap the probe's not-a-face verdict is right 74 times in
        100, against 13 of 13 above it. A gate built on the first number would
        trade a measured leak for an unmeasured one, so nothing under the cap
        is filtered at all -- with or without verdicts."""
        for v in (None, {SMALL[:4]: False}):
            h = face_mask.Hold(frames=1, max_frac=0.18, verdicts=v)
            self.assertEqual(h.update([SMALL], shape=SHAPE), [SMALL[:4]],
                             "a box under the cap was filtered")

    def test_a_kept_box_still_gets_its_hold(self):
        """Admission and persistence are separate; a box kept by a verdict
        keeps the cover for HOLD_FRAMES like any other."""
        h = face_mask.Hold(frames=3, max_frac=0.18, verdicts={BIG[:4]: True})
        h.update([BIG], shape=SHAPE)
        self.assertEqual(h.update([], shape=SHAPE), [BIG[:4]])


if __name__ == "__main__":
    unittest.main()
