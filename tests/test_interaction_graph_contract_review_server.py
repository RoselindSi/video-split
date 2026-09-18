import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_graph_contract_review_server import PAGE, Store, validate


def payload(**overrides):
    value = {
        "review_id": "graph_review_0001", "revision": 0, "relation": "NEW_ACTION",
        "type_equivalence": "different", "prior_episode_status": "completed",
        "next_episode_status": "new", "return_to_prior_type": "no",
        "instance_link": "new_instance", "observed_type_path": "A>B",
        "boundary_start_s": 4.0, "boundary_end_s": 4.5,
        "transition_shape": "gradual", "continuous_contact": "yes",
        "release_observed": "no", "overlap_present": "no", "visibility": "visible",
        "observable_evidence": "The immediate outcome visibly changes.",
        "human_confirmed": True,
    }
    value.update(overrides)
    return value


class GraphContractReviewServerTest(unittest.TestCase):
    def test_page_groups_rules_and_required_judgments(self):
        self.assertIn('判断时记住三条硬规则', PAGE)
        self.assertIn('2. 必须确认', PAGE)
        self.assertIn('可选线索：手有没有一直碰着', PAGE)
        for field in ("evidence_before", "evidence_after", "evidence_cue", "evidence_judgment"):
            self.assertIn(f'id="{field}"', PAGE)
        self.assertIn('之前在${values[0]}；之后在${values[1]}', PAGE)
        self.assertIn("if(relation==='CONTINUE')Object.assign", PAGE)
        self.assertIn('function validateBeforeFinal()', PAGE)
        self.assertNotIn('id="type_equivalence"', PAGE)

    def test_final_requires_boundary_time_and_evidence(self):
        self.assertEqual(validate(payload(), "graph_review_0001", 12, True)["relation"], "NEW_ACTION")
        with self.assertRaisesRegex(ValueError, "boundary time"):
            validate(payload(boundary_start_s="", boundary_end_s=""), "graph_review_0001", 12, True)
        with self.assertRaisesRegex(ValueError, "observable evidence"):
            validate(payload(observable_evidence=""), "graph_review_0001", 12, True)
        with self.assertRaisesRegex(ValueError, "type equivalence"):
            validate(payload(type_equivalence="same"), "graph_review_0001", 12, True)

    def test_draft_may_be_incomplete(self):
        result = validate({"review_id": "graph_review_0001", "revision": 0},
                          "graph_review_0001", 12, False)
        self.assertEqual(result["relation"], "")

    def test_store_requires_30_anonymous_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for index in range(1, 31):
                video = root / f"{index}.mp4"
                video.write_bytes(b"test")
                cases.append({"review_id": f"graph_review_{index:04d}", "video": str(video),
                              "duration_s": 12, "candidate_offset_s": 6})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": "interaction_graph_contract_review_v1",
                                            "n_cases": 30, "cases": cases}))
            store = Store(manifest, root / "out", "reviewer one")
            self.assertEqual(len(store.ids), 30)
            self.assertEqual(store.reviewer, "reviewer_one")


if __name__ == "__main__":
    unittest.main()
