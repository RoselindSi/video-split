from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig.post_pass import decide, materialize_rows  # noqa: E402


class CanonicalVoteTests(unittest.TestCase):
    def test_votes_are_aggregated_by_canonical_id_but_written_to_raw_ids(self):
        rows = [
            {"frame": "1", "tid": "1", "own": "1", "side": "right",
             "x0": "0", "y0": "0", "x1": "20", "y1": "20"},
            {"frame": "2", "tid": "7", "own": "0", "side": "right",
             "x0": "1", "y0": "0", "x1": "21", "y1": "20"},
            {"frame": "3", "tid": "7", "own": "0", "side": "right",
             "x0": "2", "y0": "0", "x1": "22", "y1": "20"},
        ]

        decisions, stats = decide(
            rows, "r", {}, track_map={"1": "1", "7": "1"})

        self.assertEqual(decisions[(1, "1")][0], 0)
        self.assertEqual(decisions[(2, "7")][0], 0)
        self.assertEqual(stats["tracks"], 2)
        self.assertEqual(stats["canonical_tracks"], 1)

    def test_materialized_rows_apply_decisions_then_owner_cap(self):
        rows = [
            {"frame": "1", "tid": str(tid), "own": "0", "side": "left",
             "p": str(score), "not_hand": "0", "covered": "0.8"}
            for tid, score in ((1, 0.7), (2, 0.9), (3, 0.8), (4, 0.99))
        ]
        decisions = {
            (1, "1"): (1, "right", 1),
            (1, "2"): (1, "left", 1),
            (1, "3"): (1, "left", 1),
            (1, "4"): (0, "", 0),
        }

        got = materialize_rows(rows, decisions, max_owner=2)

        self.assertEqual([row["own"] for row in got], ["0", "1", "1", "0"])
        self.assertEqual(got[0]["side"], "right")
        self.assertEqual(got[3]["not_hand"], "1")
        self.assertTrue(all(row["covered"] == "0" for row in got))


if __name__ == "__main__":
    unittest.main()
