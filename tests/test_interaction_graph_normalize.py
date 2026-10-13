import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_graph_normalize import (
    normalize_annotation,
    normalize_batch,
)


class InteractionGraphNormalizeTest(unittest.TestCase):
    def annotation(self):
        return {
            "video_id": "sample",
            "status": "final",
            "saved_at": "2026-09-19T00:00:00Z",
            "events": [
                {"event_id": "A", "description": "擦桌子"},
                {"event_id": "B", "description": "idle"},
                {"event_id": "C", "description": "拿杯子"},
            ],
            "segments": [
                {"start_s": 0, "end_s": 1, "event_id": "A"},
                {"start_s": 1, "end_s": 2, "event_id": "B"},
                {"start_s": 2, "end_s": 3, "event_id": "A"},
                {"start_s": 3, "end_s": 4, "event_id": "C"},
                {"start_s": 4, "end_s": 5, "event_id": "A"},
            ],
            "graph": {
                "nodes": [
                    {"event_id": "A"}, {"event_id": "B"}, {"event_id": "C"},
                ],
                "edges": [],
                "edge_occurrences": [
                    {"source": "A", "target": "A", "kind": "internal_repeat"},
                    {"source": "A", "target": "B", "kind": "transition"},
                    {"source": "B", "target": "A", "kind": "transition"},
                    {"source": "A", "target": "C", "kind": "transition"},
                    {"source": "C", "target": "A", "kind": "transition"},
                ],
                "event_sequence": ["A", "B", "A", "C", "A"],
            },
        }

    def test_special_states_are_separate_without_synthetic_edges(self):
        result = normalize_annotation(self.annotation())
        self.assertEqual(result["special_event_ids"], ["B"])
        self.assertEqual(
            result["action_graph"]["compressed_event_sequence"], ["A", "C", "A"])
        self.assertEqual(
            [row for row in result["action_graph"]["edge_occurrences"]
             if row["source"] == row["target"]],
            [{"source": "A", "target": "A", "kind": "internal_repeat",
              "source_role": "action", "target_role": "action"}],
        )
        self.assertEqual(result["action_graph"]["edge_kind_counts"],
                         {"internal_repeat": 1, "transition": 2})
        self.assertTrue(result["action_graph"]["self_loop_present"])
        self.assertTrue(result["action_graph"]["return_cycle_present"])

    def test_batch_uses_latest_final_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"videos": [{"video_id": "sample"}]}))
            annotations = root / "annotations"
            annotations.mkdir()
            annotation = self.annotation()
            source = annotations / "sample.final.20260919T000000Z.json"
            original = json.dumps(annotation)
            source.write_text(original)
            result = normalize_batch(manifest, annotations, root / "normalized.json")
            self.assertEqual(result["n_videos"], 1)
            self.assertEqual(source.read_text(), original)
            self.assertTrue(result["normalization_policy"]["source_annotations_modified"]
                            is False)


if __name__ == "__main__":
    unittest.main()
