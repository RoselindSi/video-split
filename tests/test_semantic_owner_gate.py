"""Unit tests for semantic second-owner candidate selection and aggregation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semhand.owner_gate import (aggregate, contested_tracks,
                                    context_frame_numbers,
                                    representative_rows)  # noqa: E402


class CandidateSelectionTests(unittest.TestCase):
    def test_only_lower_ranked_owner_is_sent_to_semantics(self):
        rows = [
            {"frame": "10", "tid": "0", "own": "1", "p": "0.99"},
            {"frame": "10", "tid": "3", "own": "1", "p": "0.76"},
            {"frame": "10", "tid": "1", "own": "0", "p": "0.45"},
        ]

        got = contested_tracks(rows)

        self.assertEqual(set(got), {"3"})
        self.assertEqual(got["3"][0]["owner_rank"], 2)

    def test_representative_rows_include_both_ends(self):
        rows = [{"frame": str(frame)} for frame in range(10, 20)]

        got = representative_rows(rows, 5)

        self.assertEqual(got[0]["frame"], "10")
        self.assertEqual(got[-1]["frame"], "19")
        self.assertEqual(len(got), 5)

    def test_context_is_denser_near_the_current_frame(self):
        self.assertEqual(
            context_frame_numbers(100, source_fps=10, seconds=2),
            [80, 90, 95, 100])

    def test_context_never_requests_negative_frames(self):
        self.assertEqual(
            context_frame_numbers(3, source_fps=10, seconds=2),
            [0, 0, 0, 3])


class AggregateTests(unittest.TestCase):
    def test_confident_track_is_admitted(self):
        rows = [{"rec": "r", "tid": "3", "p": p}
                for p in (0.91, 0.88, 0.72, 0.84, 0.40)]

        self.assertEqual(aggregate(rows)[0]["verdict"], "owner")

    def test_confident_foreign_track_is_rejected(self):
        rows = [{"rec": "r", "tid": "3", "p": p}
                for p in (0.01, 0.04, 0.08, 0.12, 0.02)]

        self.assertEqual(aggregate(rows)[0]["verdict"], "other")

    def test_borderline_or_incomplete_track_is_unsure(self):
        borderline = [{"rec": "r", "tid": "3", "p": p}
                      for p in (0.45, 0.50, 0.55, 0.49, 0.51)]
        incomplete = [{"rec": "r", "tid": "4", "p": p}
                      for p in (0.99, 0.98)]

        got = {row["tid"]: row["verdict"]
               for row in aggregate(borderline + incomplete)}

        self.assertEqual(got, {"3": "unsure", "4": "unsure"})

    def test_too_few_context_views_can_use_stable_full_track_evidence(self):
        rows = [{"rec": "r", "tid": "8", "p": 0.999,
                 "base_p": 0.998, "base_track_p_median": 0.9985,
                 "base_track_strong_frac": 1.0}]

        got = aggregate(rows)[0]

        self.assertEqual(got["verdict"], "owner")
        self.assertEqual(got["reason"], "too_few_base_track_consensus")

    def test_too_few_context_views_ignore_unstable_base_evidence(self):
        rows = [{"rec": "r", "tid": "6", "p": 0.999,
                 "base_p": 0.989, "base_track_p_median": 0.989,
                 "base_track_strong_frac": 0.667}]

        got = aggregate(rows)[0]

        self.assertEqual(got["verdict"], "unsure")
        self.assertEqual(got["reason"], "too_few_context_views")

    def test_second_owner_requires_four_of_five_context_views(self):
        rows = [{"rec": "r", "tid": "19", "p": p}
                for p in (0.932, 0.047, 0.500, 0.990, 0.998)]

        got = aggregate(rows, min_positive_frac=0.80)[0]

        self.assertEqual(got["verdict"], "unsure")
        self.assertEqual(got["positive_frac"], 0.6)

    def test_stable_base_track_rescues_uncertain_context(self):
        rows = [{"rec": "r", "tid": "21", "p": p,
                 "base_p": b, "base_track_p_median": 0.997,
                 "base_track_strong_frac": 0.855}
                for p, b in zip((0.998, 0.407, 0.165, 0.679, 0.562),
                                (0.878, 0.992, 0.974, 0.999, 0.554))]

        got = aggregate(rows, min_positive_frac=0.80)[0]

        self.assertEqual(got["verdict"], "owner")
        self.assertEqual(got["reason"], "base_track_consensus")

    def test_weak_base_track_does_not_rescue_conflicting_context(self):
        rows = [{"rec": "r", "tid": "19", "p": p,
                 "base_p": b, "base_track_p_median": 0.689,
                 "base_track_strong_frac": 0.277}
                for p, b in zip((0.932, 0.047, 0.500, 0.990, 0.998),
                                (0.440, 0.792, 0.491, 0.992, 0.733))]

        got = aggregate(rows, min_positive_frac=0.80)[0]

        self.assertEqual(got["verdict"], "unsure")
        self.assertEqual(got["reason"], "conflicting_evidence")

    def test_stable_base_track_rescues_context_reject(self):
        rows = [{"rec": "r", "tid": "27", "p": p,
                 "base_p": b, "base_track_p_median": 0.982,
                 "base_track_strong_frac": 0.882}
                for p, b in zip((0.0008, 0.0003, 0.0003, 0.0124, 0.0534),
                                (0.946, 0.980, 0.984, 0.992, 0.981))]

        got = aggregate(rows, min_positive_frac=0.80)[0]

        self.assertEqual(got["verdict"], "owner")
        self.assertEqual(got["context_state"], "other")
        self.assertEqual(got["base_state"], "owner")
        self.assertEqual(got["reason"],
                         "base_track_overrides_context_reject")

    def test_fragment_views_are_aggregated_by_canonical_track(self):
        rows = [
            {"rec": "r", "tid": "1", "p": 0.99},
            {"rec": "r", "tid": "7", "p": 0.98},
            {"rec": "r", "tid": "7", "p": 0.97},
        ]

        got = aggregate(
            rows, track_maps={"r": {"1": "1", "7": "1"}})[0]

        self.assertEqual(got["tid"], "1")
        self.assertEqual(got["n"], 3)
        self.assertEqual(got["verdict"], "owner")


if __name__ == "__main__":
    unittest.main()
