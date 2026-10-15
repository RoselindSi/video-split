import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_handness_track_feature_manifest import build


def test_feature_manifest_keeps_canonical_identity(tmp_path):
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "rec", "frame", "tid",
                                                       "canonical_tid"])
        writer.writeheader()
        writer.writerow({"stem": "x", "rec": "R1", "frame": 9,
                         "tid": 7, "canonical_tid": 3})
    output = tmp_path / "manifest.csv"
    assert build(index, tmp_path / "views", output) == 1
    row = next(csv.DictReader(open(output)))
    assert row["item_id"] == "x"
    assert row["raw_tid"] == "7"
    assert row["canonical_tid"] == "3"
    assert row["split"] == "test"
    assert row["human_label"] == ""
