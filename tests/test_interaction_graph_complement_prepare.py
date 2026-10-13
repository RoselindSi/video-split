import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_graph_complement_prepare import (
    SELECTION_SCHEMA,
    clip_bounds,
    select_complement,
)


class InteractionGraphComplementPrepareTest(unittest.TestCase):
    def test_clip_bounds_centers_and_clamps(self):
        self.assertEqual(clip_bounds(100, 300, 120), (40.0, 120.0))
        self.assertEqual(clip_bounds(10, 300, 120), (0.0, 120.0))
        self.assertEqual(clip_bounds(290, 300, 120), (180.0, 120.0))
        self.assertEqual(clip_bounds(10, 40, 120), (0.0, 40.0))

    def test_selection_is_balanced_deterministic_and_traceable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            counts = [2, 4, 10] * 4
            for index, count in enumerate(counts):
                bag = f"databag-26_09{index % 6 + 1:02d}_{100000 + index}"
                annotation = root / f"annotation_{index}.json"
                annotation.write_text(json.dumps({
                    "eyes": {
                        "cam3": {"keyframes": [{} for _ in range(count // 2)]},
                        "cam4": {"keyframes": [{} for _ in range(count - count // 2)]},
                    },
                }))
                rows.append({
                    "recording_id": f"R{index:04d}",
                    "databag_id": bag,
                    "source_video": f"/shared/raw/group_{index}/{bag}/cam34.mp4",
                    "source_exists_here": False,
                    "clip_start_frame": index * 30,
                    "fps": 30,
                    "mosaic_annotation": str(annotation),
                    "mosaic_annotation_status": "final",
                })
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"databags": rows}))
            first = select_complement(catalog, root / "first.json", sample_size=9)
            second = select_complement(catalog, root / "second.json", sample_size=9)

            self.assertEqual(first["schema_version"], SELECTION_SCHEMA)
            self.assertEqual(first["clips"], second["clips"])
            self.assertEqual(
                first["realized"]["by_keyframe_stratum"],
                {"high": 3, "low": 3, "medium": 3},
            )
            self.assertEqual(first["realized"]["unique_source_groups"], 9)
            self.assertEqual(first["clips"][0]["video_id"], "mosaic_graph_0001")
            self.assertTrue(all("mosaic_annotation_sha256" in row
                                for row in first["clips"]))

            excluded = root / "excluded.json"
            excluded.write_text(json.dumps({"clips": [first["clips"][0]]}))
            third = select_complement(
                catalog, root / "third.json", sample_size=9,
                exclude_selections=[excluded], video_prefix="blind_graph",
            )
            self.assertNotIn(
                first["clips"][0]["databag_id"],
                {row["databag_id"] for row in third["clips"]},
            )
            self.assertEqual(third["excluded_databag_count"], 1)
            self.assertEqual(third["clips"][0]["video_id"], "blind_graph_0001")


if __name__ == "__main__":
    unittest.main()
