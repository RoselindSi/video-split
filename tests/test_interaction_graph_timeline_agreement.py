import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.build_graph_timeline_offline_packet import build_packet
from src.auditor.boundary.interaction_graph_timeline_agreement import (
    adjusted_rand,
    match_boundaries,
    restricted_growth,
    score,
)


def annotation(video_id, event_ids, boundaries, repeats=None):
    starts = [0.0, *boundaries]
    ends = [*boundaries, 6.0]
    repeats = repeats or [False] * len(event_ids)
    events = sorted(set(event_ids))
    return {
        "schema_version": "interaction_graph_timeline_v1",
        "video_id": video_id,
        "events": [{"event_id": value, "description": value} for value in events],
        "segments": [
            {"start_s": start, "end_s": end, "event_id": event_id,
             "repeats_within_segment": repeat, "note": ""}
            for start, end, event_id, repeat in zip(
                starts, ends, event_ids, repeats
            )
        ],
        "confirmed_full_video": True,
        "status": "final",
    }


class InteractionGraphTimelineAgreementTest(unittest.TestCase):
    def test_matching_is_one_to_one_and_order_preserving(self):
        left = [{"time_s": 1.0}, {"time_s": 2.0}]
        right = [{"time_s": 1.1}, {"time_s": 1.9}, {"time_s": 4.0}]
        self.assertEqual(match_boundaries(left, right, 0.25), [(0, 0), (1, 1)])

    def test_event_letters_are_compared_as_partitions_not_names(self):
        self.assertEqual(restricted_growth(["A", "B", "A"]), [0, 1, 0])
        self.assertEqual(restricted_growth(["X", "Q", "X"]), [0, 1, 0])
        self.assertAlmostEqual(adjusted_rand(["A", "A", "B"],
                                             ["X", "X", "Q"]), 1.0)
        self.assertAlmostEqual(adjusted_rand(["A", "A", "A"],
                                             ["X", "X", "X"]), 1.0)

    def test_score_recognizes_equivalent_local_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"video")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "interaction_graph_timeline_manifest_v1",
                "n_videos": 1,
                "videos": [{"video_id": "sample", "video": str(video),
                            "duration_s": 6.0}],
            }))
            first = root / "first"
            first.mkdir()
            (first / "sample.final.1.json").write_text(json.dumps(
                annotation("sample", ["A", "A", "B"], [2.0, 4.0],
                           [True, False, False])
            ))
            second = root / "second.json"
            second.write_text(json.dumps({
                "schema_version": "interaction_graph_timeline_offline_export_v1",
                "videos": [annotation("sample", ["X", "X", "Q"],
                                       [2.1, 4.1], [True, False, False])],
            }))
            result = score(manifest, first, second)
            self.assertEqual(result["n_compared_videos"], 1)
            self.assertEqual(
                result["metrics"]["boundary_micro_by_tolerance_s"]["0.25"]["f1"],
                1.0,
            )
            self.assertEqual(
                result["metrics"]["matched_relation_at_1s"]["agreement"], 1.0
            )
            self.assertEqual(
                result["metrics"]["median_temporal_type_partition_ari"], 1.0
            )

    def test_offline_packet_contains_only_blind_public_case_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "secret.mp4"
            video.write_bytes(b"video")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "videos": [{"video_id": "blind_0001", "video": str(video),
                            "duration_s": 6.0, "source_ref": "secret-label"}]
            }))
            result = build_packet(
                manifest, root / "packet", "reviewer_2", "batch", "盲标",
                expected_count=1,
            )
            packet = json.loads(
                (root / "packet" / "packet_manifest.json").read_text()
            )
            self.assertNotIn("secret-label", json.dumps(packet))
            self.assertEqual(packet["cases"][0]["video_id"], "blind_0001")
            self.assertTrue(Path(result["archive"]).is_file())


if __name__ == "__main__":
    unittest.main()
