import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_graph_release import (
    CONFIG_SCHEMA,
    build_release,
)


class InteractionGraphReleaseTest(unittest.TestCase):
    def test_release_preserves_special_gap_and_classifies_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "videos": [{"video_id": "v1", "duration_s": 5,
                            "video": str(video)}],
            }))
            annotations = root / "annotations"
            annotations.mkdir()
            annotation = {
                "video_id": "v1",
                "status": "final",
                "saved_at": "2026-09-20T00:00:00Z",
                "confirmed_full_video": True,
                "events": [
                    {"event_id": "A", "description": "动作甲"},
                    {"event_id": "B", "description": "动作乙"},
                    {"event_id": "S", "description": "特殊事件"},
                ],
                "segments": [
                    {"start_s": 0, "end_s": 1, "event_id": "A",
                     "repeats_within_segment": True, "note": ""},
                    {"start_s": 1, "end_s": 2, "event_id": "B",
                     "repeats_within_segment": False, "note": ""},
                    {"start_s": 2, "end_s": 3, "event_id": "S",
                     "repeats_within_segment": False, "note": ""},
                    {"start_s": 3, "end_s": 4, "event_id": "A",
                     "repeats_within_segment": False, "note": ""},
                    {"start_s": 4, "end_s": 5, "event_id": "A",
                     "repeats_within_segment": False, "note": ""},
                ],
            }
            (annotations / "v1.final.1.json").write_text(json.dumps(annotation))
            config = root / "config.json"
            config.write_text(json.dumps({
                "schema_version": CONFIG_SCHEMA,
                "expected_videos": 1,
                "expected_action_nodes": 2,
                "special_labels": ["特殊事件"],
                "grouping_contract": "test",
                "batches": [{"batch_id": "batch", "manifest": str(manifest),
                             "annotation_dir": str(annotations)}],
                "procedure_groups": {
                    "g": {"name": "同一流程", "definition": "测试"},
                },
                "assignments": {
                    "batch/v1/A": {"original_description": "动作甲",
                                    "canonical_description": "规范甲",
                                    "procedure_group_id": "g"},
                    "batch/v1/B": {"original_description": "动作乙",
                                    "canonical_description": "规范乙",
                                    "procedure_group_id": "g"},
                },
            }))
            result = build_release(config, root / "out")
            types = CounterForTest(row["edge_type"]
                                   for row in result["fine_edge_occurrences"])
            self.assertEqual(types["exact_state_repeat"], 2)
            self.assertEqual(types["intra_group_transition"], 1)
            self.assertEqual(result["summary"]["edges_omitted_for_special_endpoint"], 2)
            self.assertEqual(result["summary"]["videos_with_inter_group_transition"], 0)
            self.assertEqual(result["summary"]["videos_with_procedure_return"], 0)
            self.assertEqual(
                result["video_paths"][0]["compressed_group_sequence"],
                ["g", "__SPECIAL_GAP__", "g"],
            )
            self.assertTrue((root / "out" / "graph_explorer.html").is_file())


class CounterForTest(dict):
    def __init__(self, values):
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


if __name__ == "__main__":
    unittest.main()
