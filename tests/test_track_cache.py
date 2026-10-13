"""Tests for the versioned detector/tracker replay cache."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rig import track_cache  # noqa: E402


class TrackCacheTests(unittest.TestCase):
    def test_round_trip_numpy_payload_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tracks.jsonl"
            manifest = {"start": 10, "stride": 2,
                        "config": {"continue_conf": 0.1}}
            writer = track_cache.Writer(path, manifest)
            self.assertFalse(path.exists())
            writer.write_frame({
                "frame": 10,
                "dets": [{"box": np.asarray([1, 2, 3, 4]),
                           "score": np.float32(0.75)}],
            })
            writer.close()
            self.assertTrue(path.exists())

            reader = track_cache.Reader(
                path, {"start": 10,
                       "config": {"continue_conf": 0.1}})
            row = reader.read_frame(10)
            self.assertEqual(row["dets"][0]["box"], [1, 2, 3, 4])
            self.assertAlmostEqual(row["dets"][0]["score"], 0.75)
            reader.finish()

    def test_manifest_mismatch_rejects_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tracks.jsonl"
            writer = track_cache.Writer(path, {
                "start": 10, "config": {"continue_conf": 0.1}})
            writer.close()

            with self.assertRaisesRegex(ValueError, "continue_conf"):
                track_cache.Reader(
                    path, {"config": {"continue_conf": 0.25}})

    def test_frame_number_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tracks.jsonl"
            writer = track_cache.Writer(path, {"start": 10})
            writer.write_frame({"frame": 10})
            writer.close()

            reader = track_cache.Reader(path)
            with self.assertRaisesRegex(ValueError, "expected frame 11"):
                reader.read_frame(11)
            reader.close()

    def test_manifest_fingerprint_detects_manual_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tracks.jsonl"
            writer = track_cache.Writer(path, {"start": 10})
            writer.close()
            text = path.read_text()
            path.write_text(text.replace('"start":10', '"start":11'))

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                track_cache.Reader(path)

    def test_partial_cache_is_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tracks.jsonl"
            writer = track_cache.Writer(path, {"start": 10})
            writer.write_frame({"frame": 10})
            writer.abort()

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
