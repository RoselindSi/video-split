from pathlib import Path
import csv
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semhand.three_probe import items_for  # noqa: E402


class CanonicalSamplingTests(unittest.TestCase):
    def test_samples_canonical_track_instead_of_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for frame in range(10, 20):
                rows.append({
                    "frame": frame,
                    "own": 1,
                    "x0": 0,
                    "y0": 0,
                    "x1": 200,
                    "y1": 200,
                    "conf": 0.9,
                    "tid": 1 if frame < 15 else 7,
                })
            with (root / "rec.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            got = items_for(
                str(root), {"rec": ("unused", 10, 10)}, 0.10, 1,
                {"rec": {"1": "1", "7": "1"}},
                frames_per_track=5, min_px=150)

            self.assertEqual(len(got), 5)
            self.assertEqual({row["canonical_tid"] for row in got}, {"1"})
            self.assertEqual(got[0]["frame"], 10)
            self.assertEqual(got[-1]["frame"], 19)

    def test_filters_boxes_that_post_pass_would_not_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"frame": 1, "own": 1, "x0": 0, "y0": 0,
                     "x1": 149, "y1": 200, "conf": 0.9, "tid": 1}]
            with (root / "rec.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            got = items_for(
                str(root), {"rec": ("unused", 1, 1)}, 0.10, 1,
                frames_per_track=5, min_px=150)

            self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
