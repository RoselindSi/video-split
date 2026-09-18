import json
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.boundary.final_timeline_audit import (
    Handler,
    MANIFEST_SCHEMA,
    atomic_json,
    file_sha256,
    load_manifest,
    page,
    validate_submission,
)


class FinalTimelineAuditTest(unittest.TestCase):
    def test_manifest_load_is_label_blind(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            recording = root / "recording_000007"
            recording.mkdir()
            video = recording / "mid.mp4"
            video.write_bytes(b"test-video")
            (recording / "segments.json").write_bytes(b"not valid json")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": MANIFEST_SCHEMA,
                "frozen": True,
                "n_selected_recordings": 1,
                "selected": [{
                    "recording_id": "recording_000007",
                    "video": str(video),
                    "duration_s": 12.5,
                }],
            }), encoding="utf-8")
            expected_digest = file_sha256(manifest_path)

            manifest, recordings, digest = load_manifest(manifest_path)

        self.assertTrue(manifest["frozen"])
        self.assertEqual(recordings["recording_000007"]["duration_s"], 12.5)
        self.assertEqual(digest, expected_digest)

    def test_submission_is_normalized_and_sorted(self):
        value = validate_submission({
            "recording_id": "recording_000007",
            "no_internal_boundary": False,
            "events": [
                {"start_s": 8.1239, "end_s": 8.4,
                 "instance_relation": "same_action_new_instance",
                 "note": "  restart  "},
                {"start_s": 0, "instance_relation": "initial_action_start"},
            ],
        }, "recording_000007", 10.0)

        self.assertEqual([event["start_s"] for event in value["events"]],
                         [0.0, 8.124])
        self.assertEqual(value["events"][1]["note"], "restart")
        self.assertEqual(value["schema_version"],
                         "episode_boundary_final_timeline_v1")

    def test_submission_rejects_invalid_or_conflicting_content(self):
        valid = {
            "recording_id": "recording_000007",
            "no_internal_boundary": False,
            "events": [{
                "start_s": 2.0,
                "end_s": 2.0,
                "instance_relation": "new_action",
            }],
        }
        cases = [
            dict(valid, recording_id="recording_000008"),
            dict(valid, events=[]),
            dict(valid, events=[dict(valid["events"][0], start_s=-0.1)]),
            dict(valid, events=[dict(valid["events"][0], end_s=11.0)]),
            dict(valid, events=[dict(valid["events"][0],
                                     instance_relation="same_instance")]),
            dict(valid, no_internal_boundary=True),
            dict(valid, no_internal_boundary=True, events=[{
                "start_s": 2.0,
                "instance_relation": "cannot_determine",
            }]),
            dict(valid, events=valid["events"] * 2),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_submission(case, "recording_000007", 10.0)

    def test_no_boundary_allows_only_endpoint_events(self):
        value = validate_submission({
            "recording_id": "recording_000007",
            "no_internal_boundary": True,
            "events": [
                {"start_s": 0, "instance_relation": "initial_action_start"},
                {"start_s": 10, "instance_relation": "terminal_action_end"},
            ],
        }, "recording_000007", 10.0)
        self.assertTrue(value["no_internal_boundary"])

    def test_atomic_json_replaces_target_without_temp_files(self):
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "nested" / "answer.json"
            atomic_json(target, {"version": 1})
            atomic_json(target, {"version": 2})
            leftovers = list(target.parent.glob(".*.tmp"))
            value = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(value, {"version": 2})
        self.assertEqual(leftovers, [])

    def test_page_exposes_only_raw_video_annotation_controls(self):
        source = page({
            "recording_id": "recording_000007",
            "duration_s": 10.0,
        }, None).decode("utf-8")
        self.assertIn('src="/video/recording_000007"', source)
        self.assertIn('data-add="same_action_new_instance"', source)
        for forbidden in ("segments.json", "model score", "candidate pool"):
            self.assertNotIn(forbidden, source.lower())

    def test_video_supports_byte_ranges(self):
        with tempfile.TemporaryDirectory() as name:
            video = Path(name) / "mid.mp4"
            video.write_bytes(b"0123456789")

            class Response:
                headers = {"Range": "bytes=2-5"}

                def __init__(self):
                    self.status = None
                    self.response_headers = {}
                    self.wfile = io.BytesIO()

                def send_response(self, status):
                    self.status = status

                def send_header(self, key, value):
                    self.response_headers[key] = value

                def end_headers(self):
                    pass

            response = Response()
            Handler._video(response, video)

        self.assertEqual(response.status, 206)
        self.assertEqual(response.wfile.getvalue(), b"2345")
        self.assertEqual(response.response_headers["Content-Range"],
                         "bytes 2-5/10")


if __name__ == "__main__":
    unittest.main()
