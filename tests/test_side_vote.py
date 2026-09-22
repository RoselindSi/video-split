"""The handedness vote: what it fixes, and the two things it must not do."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rig.side_vote import apply_vote, vote_tracks  # noqa: E402


def _row(tid, side, own="1", frame=0):
    return {"tid": str(tid), "side": side, "own": own, "frame": frame}


class SideVoteTest(unittest.TestCase):

    def test_a_minority_of_flipped_frames_is_overruled(self):
        """The case the vote exists for. Per-frame labels are 8.2% wrong, so a
        long track carries a handful of frames of the other side; those are
        errors by construction because a hand does not change which hand it
        is."""
        rows = [_row(1, "left", frame=i) for i in range(10)]
        rows += [_row(1, "right", frame=i) for i in range(10, 12)]
        rows, changed, tracks, tied = apply_vote(rows)
        self.assertEqual({r["side_voted"] for r in rows}, {"left"})
        self.assertEqual(changed, 2)
        self.assertEqual((tracks, tied), (1, 0))

    def test_tracks_do_not_pool(self):
        """Two hands in one frame are two tracks and must be counted apart.
        Pooling them is the identity-key bug this project has made four times:
        it does not raise, it just moves the answer."""
        rows = [_row(1, "left") for _ in range(5)] + \
               [_row(2, "right") for _ in range(3)]
        rows, _changed, tracks, _tied = apply_vote(rows)
        self.assertEqual(tracks, 2)
        self.assertEqual({r["tid"]: r["side_voted"] for r in rows},
                         {"1": "left", "2": "right"})

    def test_a_tie_says_nothing_rather_than_guessing(self):
        """A track with no majority has no answer. Returning one taken from
        the iteration order would be a coin toss wearing a measurement's
        clothes, and the 99.2% quoted for this vote was not measured on coin
        tosses."""
        rows = [_row(1, "left"), _row(1, "right")]
        rows, _changed, _tracks, tied = apply_vote(rows)
        self.assertEqual(tied, 1)
        self.assertEqual({r["side_voted"] for r in rows}, {""})

    def test_foreign_hands_do_not_vote(self):
        """A box the pipeline called somebody else's has no side anyone uses,
        and letting it vote would let an ownership error move a handedness
        answer -- two failures with separate evidence, joined for no reason."""
        rows = [_row(1, "left"), _row(1, "right", own="0"),
                _row(1, "right", own="0")]
        self.assertEqual(vote_tracks(rows), {"1": "left"})

    def test_unlabelled_frames_neither_vote_nor_block(self):
        """The detector returns no side on some frames. Those are absent
        evidence, not evidence of a tie."""
        rows = [_row(1, "left"), _row(1, ""), _row(1, None), _row(1, "left")]
        rows, changed, _tracks, tied = apply_vote(rows)
        self.assertEqual((changed, tied), (0, 0))
        self.assertEqual({r["side_voted"] for r in rows}, {"left"})


if __name__ == "__main__":
    unittest.main()
