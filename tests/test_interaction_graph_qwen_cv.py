import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_graph_qwen_cv import (
    add_failure_aware_metrics,
    annotation_tags,
    assign_folds,
    build_prompt,
    normalize_prediction,
    prepare,
)


def annotation(video_id, event_ids, boundaries, repeats=None):
    repeats = repeats or [False] * len(event_ids)
    events = sorted(set(event_ids))
    return {
        "schema_version": "interaction_graph_timeline_v1",
        "video_id": video_id,
        "events": [{"event_id": event_id, "description": f"action {event_id}"}
                   for event_id in events],
        "segments": [
            {"start_s": start, "end_s": end, "event_id": event_id,
             "repeats_within_segment": repeat, "note": ""}
            for start, end, event_id, repeat in zip(
                [0.0, *boundaries], [*boundaries, 6.0], event_ids, repeats
            )
        ],
        "confirmed_full_video": True,
        "status": "final",
    }


class QwenTimelineFiveFoldTest(unittest.TestCase):
    def manifest(self):
        strata = ["low", "medium", "high"]
        return {
            "videos": [
                {"video_id": f"v{i:02d}", "duration_s": 6,
                 "video": f"/tmp/v{i:02d}.mp4",
                 "provenance": {"keyframe_stratum": strata[i % 3]}}
                for i in range(30)
            ]
        }

    def test_fold_assignment_is_complete_disjoint_and_balanced(self):
        value = assign_folds(self.manifest(), seed="fixed")
        seen = []
        for fold in value["folds"]:
            self.assertEqual(len(fold["test_video_ids"]), 6)
            self.assertEqual(fold["test_strata"],
                             {"high": 2, "low": 2, "medium": 2})
            self.assertFalse(set(fold["test_video_ids"]) &
                             set(fold["train_video_ids"]))
            seen.extend(fold["test_video_ids"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {f"v{i:02d}" for i in range(30)})

    def test_prepare_never_uses_held_out_video_as_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            videos = [annotation(row["video_id"], ["A"], [])
                      for row in manifest["videos"]]
            gold_path = root / "gold.json"
            gold_path.write_text(json.dumps({"videos": videos}))
            value = prepare(manifest_path, gold_path, root / "out", "fixed", 8)
            for fold in value["folds"]:
                self.assertFalse(set(fold["test_video_ids"]) &
                                 set(fold["demo_video_ids"]))

    def test_structural_tags_capture_the_research_distinctions(self):
        value = annotation("x", ["A", "A", "B", "A"], [1, 2, 3],
                           [True, False, False, False])
        tags = annotation_tags(value)
        self.assertIn("internal_repeat", tags)
        self.assertIn("same_type_new_instance", tags)
        self.assertIn("return_to_prior_type", tags)
        self.assertIn("type_switch", tags)

    def test_prompt_contains_other_examples_and_no_target_answer(self):
        demo = annotation("train", ["A", "B"], [3.0])
        prompt = build_prompt(6.0, [demo])
        self.assertIn("同一种动作已经结束", prompt)
        self.assertIn("action A", prompt)
        self.assertNotIn("target gold", prompt)

    def test_invalid_first_pass_gets_a_bounded_recovery_prompt(self):
        prompt = build_prompt(
            6.0, [], '{"events":[', "JSONDecodeError: truncated"
        )
        self.assertIn("第一遍输出没有通过格式校验", prompt)
        self.assertIn("events和segments各不得超过60项", prompt)
        self.assertIn('{"events":[', prompt)

    def test_model_json_is_normalized_by_production_validator(self):
        raw = "```json\n" + json.dumps({
            "events": [{"event_id": "A", "description": "擦桌子"}],
            "segments": [{"start_s": 0, "end_s": 6, "event_id": "A",
                          "repeats_within_segment": True, "note": ""}],
        }, ensure_ascii=False) + "\n```"
        value = normalize_prediction(raw, "sample", 6.0)
        self.assertEqual(value["status"], "final")
        self.assertTrue(value["segments"][0]["repeats_within_segment"])

    def test_schema_only_canonicalization_is_logged(self):
        raw = json.dumps({
            "events": [
                {"event_id": "A", "description": "擦桌子"},
                {"event_id": "B", "description": "未使用"},
            ],
            "segments": [
                {"start_s": 0.0, "end_s": 3.0, "event_id": "A",
                 "repeats_within_segment": False, "note": ""},
                {"start_s": 3.1, "end_s": 6.0, "event_id": "A",
                 "repeats_within_segment": False, "note": ""},
            ],
        }, ensure_ascii=False)
        fixes = []
        value = normalize_prediction(raw, "sample", 6.0, fixes)
        self.assertEqual(value["events"], [
            {"event_id": "A", "description": "擦桌子"}
        ])
        self.assertEqual(value["segments"][0]["end_s"], 3.05)
        self.assertEqual(value["segments"][1]["start_s"], 3.05)
        self.assertEqual(
            [item["kind"] for item in fixes],
            ["drop_unused_events", "snap_adjacent_boundary"],
        )

    def test_failure_aware_boundary_metric_counts_missing_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            gold_path = Path(directory) / "gold.json"
            gold_path.write_text(json.dumps({
                "videos": [annotation("missing", ["A", "B"], [3.0])]
            }))
            result = {
                "n_compared_videos": 1,
                "n_expected_videos": 2,
                "missing_reviewer_2": ["missing"],
                "metrics": {"boundary_micro_by_tolerance_s": {
                    "1.0": {"matches": 2, "n_reviewer_1": 3,
                            "n_reviewer_2": 2}
                }},
            }
            add_failure_aware_metrics(result, gold_path)
            metric = result["metrics"][
                "boundary_micro_failure_aware_by_tolerance_s"
            ]["1.0"]
            self.assertEqual(metric["n_gold"], 4)
            self.assertEqual(metric["recall"], 0.5)
            self.assertEqual(result["metrics"]["valid_output_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
