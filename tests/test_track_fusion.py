from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig.track_fusion import fuse_tracks  # noqa: E402


def row(frame, tid, box, side="right", own="1"):
    return {"frame": str(frame), "tid": str(tid), "own": own,
            "side": side, **dict(zip(("x0", "y0", "x1", "y1"), box))}


class TrackFusionTests(unittest.TestCase):
    def test_joins_clear_short_handoff(self):
        rows = [row(1, 1, (0, 0, 10, 10)),
                row(2, 1, (1, 0, 11, 10)),
                row(3, 7, (2, 0, 12, 10))]

        mapping, links, _ = fuse_tracks(rows)

        self.assertEqual(mapping["1"], mapping["7"])
        self.assertEqual([(r["source_tid"], r["target_tid"])
                          for r in links], [("1", "7")])

    def test_never_joins_temporally_overlapping_tracks(self):
        rows = [row(1, 1, (0, 0, 10, 10)),
                row(2, 1, (1, 0, 11, 10)),
                row(2, 7, (1, 0, 11, 10))]

        mapping, links, _ = fuse_tracks(rows)

        self.assertNotEqual(mapping["1"], mapping["7"])
        self.assertEqual(links, [])

    def test_strong_side_conflict_stays_separate(self):
        rows = [row(1, 1, (0, 0, 10, 10), "left"),
                row(2, 7, (0, 0, 10, 10), "right")]

        mapping, links, audit = fuse_tracks(rows)

        self.assertNotEqual(mapping["1"], mapping["7"])
        self.assertEqual(links, [])
        self.assertEqual(audit[0]["status"], "strong_side_conflict")

    def test_near_tied_successors_are_left_ambiguous(self):
        rows = [row(1, 1, (0, 0, 10, 10)),
                row(2, 7, (0, 0, 10, 10)),
                row(2, 8, (0, 0, 10, 10))]

        mapping, links, audit = fuse_tracks(rows)

        self.assertEqual(len(set(mapping.values())), 3)
        self.assertEqual(links, [])
        self.assertEqual({r["status"] for r in audit}, {"ambiguous"})


if __name__ == "__main__":
    unittest.main()
