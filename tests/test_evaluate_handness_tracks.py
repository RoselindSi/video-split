import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_handness_tracks import aggregate_tracks, align_rows


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_alignment_uses_frame_iou_and_canonical_map(tmp_path):
    package = tmp_path / "pkg"
    write(package / "hands.csv",
          ["stem", "w_frac", "h_frac", "cx_frac", "cy_frac"],
          [{"stem": "sample", "w_frac": .2, "h_frac": .2,
            "cx_frac": .5, "cy_frac": .5}])
    manifest = tmp_path / "manifest.csv"
    write(manifest, ["item_id", "rec", "frame", "split", "human_label",
                     "sources", "stem", "full_path"],
          [{"item_id": "a", "rec": "R1", "frame": 10, "split": "val",
            "human_label": "hand", "sources": "pkg", "stem": "sample",
            "full_path": package / "context" / "sample.jpg"}])
    predictions = tmp_path / "predictions.csv"
    write(predictions, ["item_id", "p_hand"], [{"item_id": "a", "p_hand": .8}])
    tracks = tmp_path / "tracks"
    write(tracks / "R1.csv", ["frame", "tid", "x0", "y0", "x1", "y1"],
          [{"frame": 10, "tid": 7, "x0": 40, "y0": 40, "x1": 60, "y1": 60}])
    maps = tmp_path / "maps"
    write(maps / "R1.track_map.csv", ["raw_tid", "canonical_tid"],
          [{"raw_tid": 7, "canonical_tid": 3}])
    rows = align_rows(manifest, [predictions], tracks, maps, width=100, height=100)
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["raw_tid"] == "7"
    assert rows[0]["canonical_tid"] == "3"


def test_fail_open_aggregation_requires_three_unanimous_low_views():
    base = {"rec": "R1", "canonical_tid": "3", "match_status": "matched",
            "human_label": "hand", "item_id": "a"}
    two = [{**base, "item_id": str(i), "p_hand": .1} for i in range(2)]
    assert aggregate_tracks(two, .3)[0]["verdict"] == "unsure"
    three = two + [{**base, "item_id": "2", "p_hand": .1}]
    assert aggregate_tracks(three, .3)[0]["verdict"] == "nothand"
    rescued = three + [{**base, "item_id": "3", "p_hand": .8}]
    assert aggregate_tracks(rescued, .3)[0]["verdict"] == "hand"


def test_identical_duplicate_package_rows_are_safe(tmp_path):
    package = tmp_path / "pkg"
    duplicate = {"stem": "sample", "w_frac": .2, "h_frac": .2,
                 "cx_frac": .5, "cy_frac": .5}
    write(package / "hands.csv",
          ["stem", "w_frac", "h_frac", "cx_frac", "cy_frac"],
          [duplicate, duplicate])
    manifest = tmp_path / "manifest.csv"
    write(manifest, ["item_id", "rec", "frame", "split", "human_label",
                     "sources", "stem", "full_path"],
          [{"item_id": "a", "rec": "R1", "frame": 10, "split": "val",
            "human_label": "hand", "sources": "pkg", "stem": "sample",
            "full_path": package / "context" / "sample.jpg"}])
    predictions = tmp_path / "predictions.csv"
    write(predictions, ["item_id", "p_hand"], [{"item_id": "a", "p_hand": .8}])
    tracks = tmp_path / "tracks"
    write(tracks / "R1.csv", ["frame", "tid", "x0", "y0", "x1", "y1"],
          [{"frame": 10, "tid": 7, "x0": 40, "y0": 40, "x1": 60, "y1": 60}])
    maps = tmp_path / "maps"
    write(maps / "R1.track_map.csv", ["raw_tid", "canonical_tid"],
          [{"raw_tid": 7, "canonical_tid": 3}])
    assert align_rows(manifest, [predictions], tracks, maps,
                      width=100, height=100)[0]["match_status"] == "matched"


def test_conflicting_track_truth_fails_open():
    rows = [
        {"rec": "R1", "canonical_tid": "3", "match_status": "matched",
         "human_label": label, "item_id": str(index), "p_hand": .1}
        for index, label in enumerate(("hand", "hand", "nothand"))
    ]
    result = aggregate_tracks(rows, .3)[0]
    assert result["verdict"] == "unsure"
    assert result["reason"] == "label_conflict"
