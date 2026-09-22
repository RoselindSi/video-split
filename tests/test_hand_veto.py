"""The third outcome, and the one thing it is never allowed to do."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig import hand_veto  # noqa: E402
from src.rig.owner_vote import apply_vote, vote_tracks  # noqa: E402


def _row(own, x0=0, w=300, frame=0, rec="R"):
    return {"rec": rec, "frame": frame, "own": str(own),
            "x0": str(x0), "y0": "0", "x1": str(x0 + w), "y1": "200"}


class HandVetoTest(unittest.TestCase):

    def test_a_non_hand_stops_being_an_own_hand(self):
        rows = [_row(1, x0=0)]
        rows, vetoed, _skipped = hand_veto.apply(
            rows, {("R", 0, 0): 0.01})
        self.assertEqual(vetoed, 1)
        self.assertEqual(rows[0]["is_hand"], "0")
        self.assertEqual(rows[0]["own_kept"], "0-nothand")

    def test_a_covered_box_is_never_uncovered(self):
        """The invariant. The probe rejects 15% of other people's hands and 29
        of 30 of those rejections were real hands, so a veto wired to the mask
        would leak one foreign hand in seven. `own=0` goes in and comes out
        unchanged whatever the score says."""
        rows = [_row(0, x0=0)]
        rows, _v, _s = hand_veto.apply(rows, {("R", 0, 0): 0.0001})
        self.assertEqual(rows[0]["own_kept"], "0")
        self.assertTrue(hand_veto.check_never_uncovers(rows))

    def test_the_guard_raises_rather_than_returning(self):
        """A rule that silently uncovers is worth more as a crash."""
        rows = [_row(0)]
        rows[0]["own_kept"] = "1"
        with self.assertRaises(hand_veto.WouldUncover):
            hand_veto.check_never_uncovers(rows)

    def test_small_boxes_are_not_asked(self):
        """Below 150 px the probe's non-hand verdict was right 3 times in 55.
        Boxes there are left alone rather than judged by an instrument known
        to break on them."""
        rows = [_row(1, x0=0, w=80)]
        rows, vetoed, _s = hand_veto.apply(rows, {("R", 0, 0): 0.001})
        self.assertEqual(vetoed, 0)
        self.assertEqual(rows[0]["is_hand"], "")
        self.assertEqual(rows[0]["own_kept"], "1")

    def test_an_unscored_box_keeps_its_verdict(self):
        rows = [_row(1, x0=0)]
        rows, vetoed, _s = hand_veto.apply(rows, {})
        self.assertEqual((vetoed, rows[0]["own_kept"]), (0, "1"))


class OwnerVoteTest(unittest.TestCase):

    def test_a_majority_of_own_carries_the_track(self):
        rows = [dict(tid="1", own="1") for _ in range(8)]
        rows += [dict(tid="1", own="0") for _ in range(2)]
        rows, up, down, tracks = apply_vote(rows)
        self.assertEqual({r["own_voted"] for r in rows}, {"1"})
        self.assertEqual((up, down, tracks), (2, 0, 1))

    def test_a_tie_covers(self):
        """Not a strict majority for the wearer, so it is covered. The cost is
        some of the wearer's own pixels; the alternative cost is a person."""
        rows = [dict(tid="1", own="1"), dict(tid="1", own="0")]
        rows, _up, down, _t = apply_vote(rows)
        self.assertEqual({r["own_voted"] for r in rows}, {"0"})
        self.assertEqual(down, 1)

    def test_a_mostly_foreign_track_becomes_wholly_foreign(self):
        rows = [dict(tid="1", own="0") for _ in range(7)]
        rows += [dict(tid="1", own="1") for _ in range(3)]
        rows, _up, down, _t = apply_vote(rows)
        self.assertEqual({r["own_voted"] for r in rows}, {"0"})
        self.assertEqual(down, 3)

    def test_tracks_are_independent(self):
        rows = [dict(tid="1", own="1"), dict(tid="1", own="1"),
                dict(tid="2", own="0"), dict(tid="2", own="0")]
        self.assertEqual(vote_tracks(rows), {"1": 1, "2": 0})


if __name__ == "__main__":
    unittest.main()
