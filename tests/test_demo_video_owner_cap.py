"""Regression tests for the final owner cap in rendered videos."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig.demo_video import _enforce_owner_cap  # noqa: E402


class FinalOwnerCapTests(unittest.TestCase):
    def test_replayed_third_owner_is_demoted_by_score(self):
        flags = [(True, 0.45), (True, 0.76), (True, 1.0)]

        capped, demoted = _enforce_owner_cap(flags, 2)

        self.assertEqual(capped, [(False, 0.45), (True, 0.76), (True, 1.0)])
        self.assertEqual(demoted, [(0, 0.45)])

    def test_not_hand_does_not_consume_an_owner_slot(self):
        flags = [(True, 0.99), (True, 0.80), (True, 0.70)]

        capped, demoted = _enforce_owner_cap(flags, 2, excluded={0})

        self.assertEqual(capped, flags)
        self.assertEqual(demoted, [])

    def test_disabled_cap_preserves_all_flags(self):
        flags = [(True, 0.20), (True, 0.30), (True, 0.40)]

        capped, demoted = _enforce_owner_cap(flags, None)

        self.assertEqual(capped, flags)
        self.assertEqual(demoted, [])


if __name__ == "__main__":
    unittest.main()
