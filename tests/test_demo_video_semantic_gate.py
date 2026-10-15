"""Regression tests for semantic admission of a second owner hand."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig.demo_video import (_apply_semantic_owner_gate,
                                _apply_owner_constraints,
                                _display_frames)  # noqa: E402


class SemanticOwnerGateTests(unittest.TestCase):
    def test_rejects_other_second_owner(self):
        flags = [(True, 1.0), (True, 0.76)]

        got, demoted = _apply_semantic_owner_gate(
            flags, [0, 3], {"3": "other"})

        self.assertEqual(got, [(True, 1.0), (False, 0.76)])
        self.assertEqual(demoted, [(3, 0.76, "other")])

    def test_admits_semantically_confirmed_second_owner(self):
        flags = [(True, 0.95), (True, 0.91)]

        got, demoted = _apply_semantic_owner_gate(
            flags, [4, 8], {"8": "owner"})

        self.assertEqual(got, flags)
        self.assertEqual(demoted, [])

    def test_missing_verdict_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "track 8"):
            _apply_semantic_owner_gate(
                [(True, 0.95), (True, 0.91)], [4, 8], {})

    def test_foreign_track_stays_rejected_when_it_becomes_primary(self):
        got, demoted = _apply_semantic_owner_gate(
            [(True, 0.82)], [11], {"11": "other"})

        self.assertEqual(got, [(False, 0.82)])
        self.assertEqual(demoted, [(11, 0.82, "other")])

    def test_raw_review_uses_clean_pixels_in_both_panels(self):
        clean, face_covered, suppressed = object(), object(), object()

        annotated, delivered = _display_frames(
            clean, face_covered, suppressed, no_mosaic=True)

        self.assertIs(annotated, clean)
        self.assertIs(delivered, clean)

    def test_normal_render_keeps_privacy_outputs(self):
        clean, face_covered, suppressed = object(), object(), object()

        annotated, delivered = _display_frames(
            clean, face_covered, suppressed, no_mosaic=False)

        self.assertIs(annotated, face_covered)
        self.assertIs(delivered, suppressed)

    def test_semantic_rejection_frees_slot_for_valid_owner(self):
        flags = [(True, 1.0), (True, 0.952), (True, 0.819)]

        got, semantic_demoted, cap_demoted = _apply_owner_constraints(
            flags, [17, 19, 21], {"19": "unsure", "21": "owner"}, 2)

        self.assertEqual(got, [(True, 1.0), (False, 0.952), (True, 0.819)])
        self.assertEqual(semantic_demoted, [(19, 0.952, "unsure")])
        self.assertEqual(cap_demoted, [])

    def test_ungated_third_candidate_can_be_removed_by_cap(self):
        flags = [(True, 1.0), (True, 0.952), (True, 0.900), (True, 0.800)]

        got, semantic_demoted, cap_demoted = _apply_owner_constraints(
            flags, [17, 19, 21, 1], {"19": "unsure", "21": "owner"}, 2)

        self.assertEqual(
            got,
            [(True, 1.0), (False, 0.952), (True, 0.900), (False, 0.800)])
        self.assertEqual(
            semantic_demoted,
            [(19, 0.952, "unsure"), (1, 0.800, "missing")])
        self.assertEqual(cap_demoted, [])


if __name__ == "__main__":
    unittest.main()
