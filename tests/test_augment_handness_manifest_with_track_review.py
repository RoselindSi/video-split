import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.augment_handness_manifest_with_track_review import augment
from scripts.build_handness_distill_manifest import FIELDS


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_augments_views_with_track_truth_and_resplits_by_recording(tmp_path):
    base = tmp_path / "base.csv"
    write(base, FIELDS, [{"item_id": "old", "rec": "R0", "frame": 1,
                          "human_label": "hand", "split": "train"}])
    full, crop = tmp_path / "full.jpg", tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    features = tmp_path / "features.csv"
    write(features, ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
                     "full_path", "crop_path"],
          [{"item_id": "new", "rec": "R1", "frame": 2, "raw_tid": 7,
            "canonical_tid": 3, "full_path": full, "crop_path": crop}])
    review = tmp_path / "review.csv"
    write(review, ["rec", "canonical_tid", "label"],
          [{"rec": "R1", "canonical_tid": 3, "label": "nothand"}])
    output = tmp_path / "out.csv"
    result = augment(base, features, review, output, "salt", {"R1": "val"})
    assert result["added_labels"] == {"nothand": 1}
    rows = list(csv.DictReader(open(output)))
    added = next(row for row in rows if row["item_id"] == "new")
    assert added["human_label"] == "nothand"
    assert added["review_status"] == "reviewed_track_v1"
    assert added["full_sha256"]
    assert {row["split"] for row in rows if row["rec"] == "R1"} == {"val"}
