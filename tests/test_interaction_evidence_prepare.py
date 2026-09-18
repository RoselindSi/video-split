import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_evidence_prepare import clip_bounds, selected_events, video_index


class PrepareTest(unittest.TestCase):
    def test_clip_bounds_preserve_candidate_and_clip_at_endpoints(self):
        self.assertEqual(clip_bounds(50.0, 100.0), (35.0, 65.0, 15.0))
        self.assertEqual(clip_bounds(2.0, 100.0), (0.0, 17.0, 2.0))
        self.assertEqual(clip_bounds(98.0, 100.0), (83.0, 100.0, 15.0))
        for args in ((-1.0, 100.0), (101.0, 100.0), (2.0, 100.0, 0.0)):
            with self.assertRaises(ValueError):
                clip_bounds(*args)

    def test_video_index_rejects_conflicting_recording_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            a, b = d / "a.json", d / "b.json"
            a.write_text(json.dumps([{"recording_id": "recording_000001", "video": "/one.mp4"}]))
            b.write_text(json.dumps([{"recording_id": "recording_000001", "video": "/two.mp4"}]))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                video_index([a, b])

    def test_current_packet_has_150_unique_review_aliases(self):
        packet = Path(__file__).resolve().parents[1] / "results/auditor/interaction_evidence_v2_20260914_review150_verified"
        rows = selected_events(packet)
        self.assertEqual(len(rows), 150)
        self.assertEqual(len({r["review_id"] for r in rows}), 150)
        self.assertEqual(len({r["event_id"] for r in rows}), 150)


if __name__ == "__main__":
    unittest.main()
