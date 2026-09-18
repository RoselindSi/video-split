import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_review_server import ReviewStore, validate


def payload(revision=0, **overrides):
    result = {
        "review_id": "review_0001", "revision": revision, "relation": "CONTINUE",
        "boundary_scope": "internal", "visibility": "visible", "context_sufficient": "yes",
        "transition_shape": "no_transition", "continuous_contact": "yes",
        "release_observed": "no", "why_one_line": "The same operation continues.",
        "context_half_s": 6, "human_confirmed": True,
    }
    result.update(overrides)
    return result


class ReviewServerTest(unittest.TestCase):
    def make_store(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        clip = root / "review_0001.mp4"
        clip.write_bytes(b"not decoded by this unit test")
        packet = root / "packet.json"
        contract = root / "contract.md"
        packet.write_text("packet")
        contract.write_text("contract")
        manifest = root / "clips_manifest.json"
        manifest.write_text(json.dumps({"schema_version": "interaction_review_clips_v2", "n_events": 1,
                                        "packet_sha256": "packet-hash", "contract_sha256": "contract-hash",
                                        "clips": [{"review_id": "review_0001", "video": str(clip),
                                                   "candidate_offset_s": 6.0, "duration_s": 12.0}]}))
        return temp, ReviewStore(manifest, root / "out", "test annotator")

    def test_draft_can_be_incomplete_but_final_cannot(self):
        draft = validate({"review_id": "review_0001", "revision": 0}, "review_0001", False)
        self.assertEqual(draft["relation"], "")
        with self.assertRaisesRegex(ValueError, "Choose"):
            validate({"review_id": "review_0001", "revision": 0}, "review_0001", True)
        with self.assertRaisesRegex(ValueError, "One-sided"):
            validate(payload(boundary_scope="initial"), "review_0001", True)
        self.assertEqual(validate(payload(), "review_0001", True)["relation"], "CONTINUE")

    def test_revision_and_final_audit_trail(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        draft = store.save("review_0001", payload(revision=0, human_confirmed=False), False)
        self.assertEqual((draft["status"], draft["revision"]), ("draft", 1))
        with self.assertRaisesRegex(RuntimeError, "another tab"):
            store.save("review_0001", payload(revision=0), True)
        final = store.save("review_0001", payload(revision=1), True)
        self.assertEqual(store.progress()["final"], 1)
        self.assertTrue((store.out / "review_0001.final.00002.json").is_file())
        self.assertIn(b"CONTINUE", store.export())

    def test_manifest_requires_real_alias_and_timing(self):
        temp, store = self.make_store()
        self.addCleanup(temp.cleanup)
        bad = json.loads((Path(temp.name) / "clips_manifest.json").read_text())
        bad["clips"][0]["candidate_offset_s"] = 13.0
        (Path(temp.name) / "bad.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Invalid clip timing"):
            ReviewStore(Path(temp.name) / "bad.json", Path(temp.name) / "out2", "a")


if __name__ == "__main__":
    unittest.main()
