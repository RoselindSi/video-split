import tempfile
import unittest
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.boundary.final_test_split import (
    build_manifest,
    collect_exclusions,
    inventory,
    selection_rank,
)
from src.auditor.boundary.ontology_constitution import (
    Constitution,
    ConstitutionViolation,
)


def _recording(root: Path, number: int) -> Path:
    directory = root / f"recording_{number:06d}"
    directory.mkdir()
    (directory / "mid.mp4").write_bytes(b"video")
    return directory


class FinalTestSplitTest(unittest.TestCase):
    def test_exclusion_scan_normalizes_numeric_recording_id(self):
        with tempfile.TemporaryDirectory() as name:
            gold = Path(name) / "gold"
            gold.mkdir()
            (gold / "events.csv").write_text(
                "recording_id,label\n4,answer\nrecording_000012,answer\n",
                encoding="utf-8")
            excluded, provenance = collect_exclusions([str(gold)], [])
        self.assertEqual(excluded, {"recording_000004", "recording_000012"})
        self.assertEqual(provenance["n_excluded_recordings"], 2)
        self.assertEqual(len(provenance["aggregate_sha256"]), 64)

    def test_selection_is_deterministic_and_excludes_used_recordings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first = root / "part1"
            second = root / "part2"
            first.mkdir()
            second.mkdir()
            for number in (1, 3, 5):
                _recording(first, number)
            for number in (2, 4, 6):
                _recording(second, number)

            corpus_a = inventory([str(first), str(second)], use_ffprobe=False)
            corpus_b = inventory([str(second), str(first)], use_ffprobe=False)
            excluded = {"recording_000003"}
            provenance = {"n_excluded_recordings": 1}
            one = build_manifest(corpus_a, excluded, provenance, 3, "fixed", 30.0)
            two = build_manifest(corpus_b, excluded, provenance, 3, "fixed", 30.0)

        ids_one = [row["recording_id"] for row in one["selected"]]
        ids_two = [row["recording_id"] for row in two["selected"]]
        self.assertEqual(ids_one, ids_two)
        self.assertNotIn("recording_000003", ids_one)
        self.assertEqual(
            [row["selection_rank"] for row in one["selected"]],
            sorted(row["selection_rank"] for row in one["selected"]))

    def test_sampler_does_not_depend_on_segments_json(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "recordings"
            root.mkdir()
            for number in range(1, 4):
                directory = _recording(root, number)
                (directory / "segments.json").write_bytes(b"not valid json")

            corpus = inventory([str(root)], use_ffprobe=False)
            manifest = build_manifest(corpus, set(), {}, 2, "blind", 30.0)
        self.assertEqual(len(manifest["selected"]), 2)
        self.assertIn("did not open segments.json",
                      manifest["label_blind_guarantee"])

    def test_hash_rank_changes_with_salt(self):
        rid = "recording_000001"
        self.assertNotEqual(selection_rank("a", rid),
                            selection_rank("b", rid))

    def test_ffprobe_failure_is_not_eligible(self):
        corpus = [
            {"recording_id": "recording_000001", "duration_s": None},
            {"recording_id": "recording_000002", "duration_s": 60.0},
        ]
        manifest = build_manifest(corpus, set(), {}, 1, "fixed", 30.0)
        self.assertEqual(manifest["selected"][0]["recording_id"],
                         "recording_000002")
        self.assertEqual(manifest["technical_rejects"], [{
            "recording_id": "recording_000001",
            "reason": "ffprobe_unreadable",
        }])

    @unittest.skipUnless(importlib.util.find_spec("yaml"),
                         "PyYAML is not installed in this environment")
    def test_constitution_protects_final_test_from_design_use(self):
        constitution = Constitution()
        dataset = constitution.check_dataset_use(
            "final_test_v1", "annotate_exhaustive_timeline")
        self.assertEqual(dataset["role"], "final_test")
        for forbidden in ("train", "choose_design", "select_thresholds",
                          "exploratory_ablation", "rerun_after_result"):
            with self.assertRaises(ConstitutionViolation):
                constitution.check_dataset_use("final_test_v1", forbidden)


if __name__ == "__main__":
    unittest.main()
