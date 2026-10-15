import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.auditor.boundary.interaction_graph_timeline import (
    MANIFEST_SCHEMA,
    Store,
    blank_state,
    build_graph,
    load_manifest,
    page,
    validate_submission,
)
from src.auditor.boundary.interaction_graph_timeline_prepare import (
    CATALOG_SCHEMA,
    build_databag_catalog,
    manifest_from_databag_catalog,
    prepare_video_manifest,
)


class InteractionGraphTimelineTest(unittest.TestCase):
    def annotation(self):
        return {
            "schema_version": "interaction_graph_timeline_v1",
            "video_id": "sample",
            "events": [
                {"event_id": "A", "description": "擦桌子"},
                {"event_id": "B", "description": "冲洗塞子"},
            ],
            "segments": [
                {"start_s": 0, "end_s": 2, "event_id": "A",
                 "repeats_within_segment": True, "note": ""},
                {"start_s": 2, "end_s": 4, "event_id": "B",
                 "repeats_within_segment": False, "note": ""},
                {"start_s": 4, "end_s": 6, "event_id": "A",
                 "repeats_within_segment": False, "note": ""},
            ],
            "confirmed_full_video": True,
        }

    def test_graph_records_internal_self_loop_and_aba_path(self):
        graph = build_graph(self.annotation())
        self.assertEqual(graph["event_sequence"], ["A", "B", "A"])
        edges = {(row["source"], row["target"]): row for row in graph["edges"]}
        self.assertEqual(edges[("A", "A")]["kinds"], {"internal_repeat": 1})
        self.assertEqual(edges[("A", "B")]["count"], 1)
        self.assertEqual(edges[("B", "A")]["count"], 1)
        nodes = {row["event_id"]: row for row in graph["nodes"]}
        self.assertEqual(nodes["A"]["segment_count"], 2)
        self.assertEqual(nodes["A"]["total_duration_s"], 4.0)

    def test_final_validation_requires_complete_partition_and_descriptions(self):
        valid = validate_submission(self.annotation(), "sample", 6, final=True)
        self.assertEqual(valid["segments"][-1]["end_s"], 6)

        missing = self.annotation()
        missing["segments"][1]["event_id"] = ""
        with self.assertRaisesRegex(ValueError, "还没有选择事件"):
            validate_submission(missing, "sample", 6, final=True)

        gap = self.annotation()
        gap["segments"][1]["start_s"] = 2.2
        with self.assertRaisesRegex(ValueError, "空白或重叠"):
            validate_submission(gap, "sample", 6)

    def test_page_exposes_cut_reuse_and_explicit_repeat_controls(self):
        video = {"video_id": "sample", "duration_s": 6,
                 "source_pool": "test", "video": "/tmp/sample.mp4"}
        html = page(video, blank_state("sample", 6), "token").decode()
        self.assertIn("在当前时间切开", html)
        self.assertIn("这段是什么事件", html)
        self.assertIn("这段里同一动作反复发生", html)
        self.assertIn("把这段开头改到当前时间", html)
        self.assertIn("把这段结尾改到当前时间", html)
        self.assertIn('aria-label="选择上一段"', html)
        self.assertIn('aria-label="选择下一段"', html)
        self.assertIn("$('#prevSegment').onclick", html)
        self.assertIn("function moveBoundary", html)
        self.assertIn('id="playhead"', html)
        self.assertIn("function nextId()", html)

    def test_store_saves_final_and_exports_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"video placeholder")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": MANIFEST_SCHEMA,
                "n_videos": 1,
                "videos": [{"video_id": "sample", "video": str(video),
                            "duration_s": 6, "source_pool": "test"}],
            }))
            loaded, videos, digest = load_manifest(manifest)
            self.assertEqual(loaded["n_videos"], 1)
            self.assertEqual(videos["sample"]["duration_s"], 6)
            self.assertEqual(len(digest), 64)

            store = Store(manifest, root / "out", "reviewer 1")
            saved = store.save("sample", self.annotation(), final=True)
            self.assertEqual(saved["status"], "final")
            exported = json.loads(store.export_json())
            self.assertEqual(exported["videos"][0]["graph"]["event_sequence"],
                             ["A", "B", "A"])
            self.assertTrue(store.export_csv().startswith(b"\xef\xbb\xbf"))


class InteractionGraphTimelinePrepareTest(unittest.TestCase):
    def test_prepare_video_manifest_probes_all_supported_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "one.mp4").write_bytes(b"one")
            (root / "nested" / "two.mov").write_bytes(b"two")
            (root / "ignore.txt").write_text("ignore")
            out = root / "out" / "manifest.json"
            with patch(
                "src.auditor.boundary.interaction_graph_timeline_prepare.probe_duration",
                side_effect=[4.5, 7.0],
            ):
                value = prepare_video_manifest(root, out, "repeated_actions")
            self.assertEqual(value["n_videos"], 2)
            self.assertEqual({row["source_pool"] for row in value["videos"]},
                             {"repeated_actions"})
            self.assertEqual(json.loads(out.read_text())["schema_version"],
                             MANIFEST_SCHEMA)

    def test_databag_catalog_prefers_latest_final_and_keeps_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "databag-26_0904_112658"
            bag.mkdir()
            base = {
                "recording": "R0904_112658",
                "source": "/shared/raw/databag-26_0904_112658/cam34.mp4",
                "clip_start_frame": 10770,
                "fps": 30,
            }
            (bag / "R__draft.json").write_text(json.dumps(base))
            older = bag / "R__final_2026-09-16T120000.json"
            older.write_text(json.dumps({**base, "clip_start_frame": 1}))
            latest = bag / "R__final_2026-09-17T120000.json"
            latest.write_text(json.dumps(base))
            out = root / "catalog.json"
            value = build_databag_catalog(root, out)
            self.assertEqual(value["schema_version"], CATALOG_SCHEMA)
            self.assertEqual(value["n_databags"], 1)
            row = value["databags"][0]
            self.assertEqual(row["clip_start_frame"], 10770)
            self.assertEqual(row["mosaic_annotation"], str(latest.resolve()))
            self.assertFalse(row["source_exists_here"])

    def test_catalog_manifest_can_rewrite_the_data_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounted = root / "mounted" / "bag" / "cam34.mp4"
            mounted.parent.mkdir(parents=True)
            mounted.write_bytes(b"video")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema_version": CATALOG_SCHEMA,
                "databags": [{
                    "recording_id": "R0904_112658",
                    "databag_id": "databag-26_0904_112658",
                    "source_video": "/shared/original/bag/cam34.mp4",
                    "mosaic_annotation": "/annotations/final.json",
                }],
            }))
            out = root / "manifest.json"
            with patch(
                "src.auditor.boundary.interaction_graph_timeline_prepare.probe_duration",
                return_value=42.0,
            ):
                value = manifest_from_databag_catalog(
                    catalog, out, "diverse_databags",
                    path_maps=[("/shared/original", str(root / "mounted"))],
                )
            self.assertEqual(value["n_videos"], 1)
            self.assertEqual(value["videos"][0]["video"], str(mounted.resolve()))
            self.assertEqual(value["videos"][0]["duration_s"], 42.0)


if __name__ == "__main__":
    unittest.main()
