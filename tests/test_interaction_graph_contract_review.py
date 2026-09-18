import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.auditor.boundary.final_timeline_audit import file_sha256
from src.auditor.boundary.interaction_graph_contract_review import (
    FIXED_CASES, REVIEW_FIELDS, contextual_bounds, main, normalized_label, score,
    structural_candidates,
)


class InteractionGraphContractReviewTest(unittest.TestCase):
    def test_fixed_cases_have_required_coverage(self):
        self.assertEqual(len(FIXED_CASES), 26)
        values = list(FIXED_CASES.values())
        for stratum in ("continuous_contact_boundary", "release_without_boundary",
                        "self_loop", "cross_node"):
            self.assertGreaterEqual(values.count(stratum), 4)

    def test_fixed_cases_are_unique(self):
        self.assertEqual(len(FIXED_CASES), len(set(FIXED_CASES)))

    def test_normalized_label_removes_instance_ordinals(self):
        self.assertEqual(normalized_label("Fold second tissue (again)"), "fold tissue")
        self.assertNotEqual(normalized_label("wind cable"), normalized_label("unwind cable"))

    def test_context_bounds_include_nearby_segments_and_cap_length(self):
        segments = [{"start_s": 10, "end_s": 20}, {"start_s": 40, "end_s": 80}]
        self.assertEqual(contextual_bounds(45, segments, 100), (7.0, 83.0))
        long = [{"start_s": 0, "end_s": 300}]
        self.assertEqual(contextual_bounds(150, long, 300), (75.0, 225.0))

    def test_structural_candidates_are_distinct_recordings_and_not_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                rid = f"recording_{index + 1:06d}"
                source = root / rid
                source.mkdir()
                (source / "mid.mp4").write_bytes(b"present")
                if index < 2:
                    triples = [(0, 8, "Wind cable"), (8, 15, "Unwind cable"),
                               (15, 24, "Wind cable")]
                else:
                    triples = [(0, 20, "Fold cloth"), (10, 25, "Wash cup")]
                segments = [{"start_s": a, "end_s": b, "label_en": label}
                            for a, b, label in triples]
                (source / "segments.json").write_text(json.dumps({"segments": segments}))
            ids = [f"recording_{index + 1:06d}" for index in range(4)]
            cases = structural_candidates(ids, [root], set())
            self.assertEqual([c["kind"] for c in cases],
                             ["aba_candidate", "aba_candidate",
                              "overlap_candidate", "overlap_candidate"])
            self.assertEqual(len({c["recording_id"] for c in cases}), 4)

    def test_score_requires_independent_complete_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f"graph_review_{i:04d}" for i in range(1, 31)]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"n_cases": 30,
                                            "cases": [{"review_id": rid} for rid in ids]}))
            manifest_hash = file_sha256(manifest)
            paths = [root / "one.csv", root / "two.csv"]
            for path in paths:
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS + [
                        "status", "manifest_sha256"])
                    writer.writeheader()
                    for rid in ids:
                        row = {field: "" for field in REVIEW_FIELDS + [
                            "status", "manifest_sha256"]}
                        row.update(review_id=rid, relation="CONTINUE", type_equivalence="same",
                                   prior_episode_status="ongoing", next_episode_status="ongoing",
                                   return_to_prior_type="no", instance_link="same_instance",
                                   transition_shape="no_transition", continuous_contact="yes",
                                   release_observed="no", overlap_present="no", visibility="visible",
                                   observable_evidence="same episode continues", reviewer_id=path.stem,
                                   human_confirmed="true", status="final",
                                   manifest_sha256=manifest_hash)
                        writer.writerow(row)
            out = root / "agreement.json"
            score(manifest, paths[0], paths[1], out)
            result = json.loads(out.read_text())
            self.assertTrue(result["passes_primary_gate"])
            self.assertFalse(result["ratified"])
            cli_out = root / "agreement-cli.json"
            with patch("sys.argv", ["interaction_graph_contract_review", "score",
                                    "--manifest", str(manifest),
                                    "--reviewer-1", str(paths[0]),
                                    "--reviewer-2", str(paths[1]),
                                    "--out", str(cli_out)]):
                main()
            self.assertEqual(json.loads(cli_out.read_text())["primary_exact"], 30)
            with self.assertRaisesRegex(ValueError, "distinct reviewer IDs"):
                score(manifest, paths[0], paths[0], root / "invalid.json")


if __name__ == "__main__":
    unittest.main()
