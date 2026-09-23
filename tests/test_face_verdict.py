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

    def test_a_small_box_with_no_verdict_is_kept(self):
        """The default below the cap, and it covers most boxes there. A row is
        written only where the model was confident, so no row means nobody
        answered and the mosaic stays."""
        for v in (None, {}, {(1, 2, 3, 4): False}):
            h = face_mask.Hold(frames=1, max_frac=0.18, verdicts=v)
            self.assertEqual(h.update([SMALL], shape=SHAPE), [SMALL[:4]],
                             "a box under the cap was filtered without a verdict")

    def test_a_small_box_can_be_refused_by_a_verdict_but_none_is_written(self):
        """The mechanism works and the shipped default does not use it.

        It was turned on at p<=0.05 on the strength of 78 correct rejections
        out of 78, and it uncovered two real faces -- p=1.7e-05 and p=0.0, the
        second on a box the detector scored 0.81 -- both found by watching the
        video. 78 of 78 bounds precision at 95.3%, which over 136 episodes
        permits about six wrong drops, so the sample was not unlucky and the
        reading of it was wrong. Precision is the wrong quantity to gate a
        privacy decision on.

        `face_verdicts` therefore writes no rows for small boxes by default,
        and this test drives `Hold` directly to keep the mechanism honest.

        The original note, kept because it is the evidence that was over-read:

        Aggregated over every score it gave, face-ness on small boxes was
        right 74 times in 100 -- unusable, and this test said nothing under
        the cap may be filtered. Split by score, all of its errors sit at
        p >= 0.060 and below 0.05 it is 78 for 78 over two sheets, lower bound
        95.3%. The nearest error at 0.060 makes 0.05 a boundary rather than a
        round number.

        So a verdict may refuse a small box, and `face_verdicts` writes a row
        for one only when it is under that threshold. What this buys is half
        the small-box mosaic -- 2,382 frame instances of 4,406 -- off tables,
        food and bench."""
        h = face_mask.Hold(frames=1, max_frac=0.18,
                           verdicts={SMALL[:4]: False})
        self.assertEqual(h.update([SMALL], shape=SHAPE), [])
        self.assertEqual(len(h.refused), 1)
        self.assertEqual(h.unjudged, 0, "a small box must not count as unjudged")

    def test_a_kept_box_still_gets_its_hold(self):
        """Admission and persistence are separate; a box kept by a verdict
        keeps the cover for HOLD_FRAMES like any other."""
        h = face_mask.Hold(frames=3, max_frac=0.18, verdicts={BIG[:4]: True})
        h.update([BIG], shape=SHAPE)
        self.assertEqual(h.update([], shape=SHAPE), [BIG[:4]])


if __name__ == "__main__":
    unittest.main()
