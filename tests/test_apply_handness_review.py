import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_handness_review import apply_review


FIELDS = ["item_id", "split", "human_label", "review_status"]


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_apply_review_preserves_split_and_updates_truth(tmp_path):
    manifest = tmp_path / "manifest.csv"
    review = tmp_path / "review.csv"
    output = tmp_path / "reviewed.csv"
    write_csv(manifest, [
        {"item_id": "a", "split": "train", "human_label": "",
         "review_status": "required"},
        {"item_id": "b", "split": "test", "human_label": "hand",
         "review_status": "accepted_positive_identity"},
    ])
    write_csv(review, [
        {"item_id": "a", "label": "nothand"},
    ], fields=["item_id", "label"])

    report = apply_review(manifest, review, output)
    rows = list(csv.DictReader(output.open()))
    assert rows[0] == {"item_id": "a", "split": "train",
                       "human_label": "nothand", "review_status": "reviewed_v2"}
    assert rows[1]["split"] == "test"
    assert report["applied_rows"] == 1


def test_apply_review_rejects_unknown_id(tmp_path):
    manifest = tmp_path / "manifest.csv"
    review = tmp_path / "review.csv"
    write_csv(manifest, [{"item_id": "a", "split": "train",
                          "human_label": "", "review_status": "required"}])
    write_csv(review, [{"item_id": "missing", "label": "hand"}],
              fields=["item_id", "label"])
    with pytest.raises(ValueError, match="absent from manifest"):
        apply_review(manifest, review, tmp_path / "out.csv")


def test_apply_review_can_resplit_after_labels_are_known(tmp_path):
    manifest = tmp_path / "manifest.csv"
    review = tmp_path / "review.csv"
    manifest_rows = []
    review_rows = []
    for index in range(12):
        for offset in range(4):
            item_id = f"{index}-{offset}"
            manifest_rows.append({"item_id": item_id, "split": "train",
                                  "human_label": "hand",
                                  "review_status": "accepted",
                                  "rec": f"R{index}"})
        if index < 9:
            item_id = f"{index}-negative"
            manifest_rows.append({"item_id": item_id, "split": "train",
                                  "human_label": "",
                                  "review_status": "required",
                                  "rec": f"R{index}"})
            review_rows.append({"item_id": item_id, "label": "nothand"})
    fields = FIELDS + ["rec"]
    write_csv(manifest, manifest_rows, fields=fields)
    write_csv(review, review_rows, fields=["item_id", "label"])
    output = tmp_path / "out.csv"
    apply_review(manifest, review, output, resplit_salt="reviewed")
    rows = list(csv.DictReader(output.open()))
    assert {row["split"] for row in rows if row["human_label"] == "nothand"} == {
        "train", "val", "test"
    }


def test_apply_review_can_override_whole_recording_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    review = tmp_path / "review.csv"
    output = tmp_path / "out.csv"
    write_csv(manifest, [
        {"item_id": "a", "split": "train", "human_label": "",
         "review_status": "required", "rec": "R1"},
        {"item_id": "b", "split": "train", "human_label": "hand",
         "review_status": "accepted", "rec": "R1"},
    ], fields=FIELDS + ["rec"])
    write_csv(review, [{"item_id": "a", "label": "nothand"}],
              fields=["item_id", "label"])
    report = apply_review(manifest, review, output,
                          split_overrides={"R1": "val"})
    assert {row["split"] for row in csv.DictReader(open(output))} == {"val"}
    assert report["split_overrides"] == {"R1": "val"}


def test_apply_review_preserves_explicit_review_status(tmp_path):
    manifest = tmp_path / "manifest.csv"
    review = tmp_path / "review.csv"
    output = tmp_path / "out.csv"
    write_csv(manifest, [{
        "item_id": "a", "split": "train", "human_label": "",
        "review_status": "required",
    }])
    write_csv(review, [{
        "item_id": "a", "label": "nothand",
        "review_status": "reviewed_track_hardneg_v1",
    }], fields=["item_id", "label", "review_status"])
    apply_review(manifest, review, output)
    assert next(csv.DictReader(open(output)))["review_status"] == (
        "reviewed_track_hardneg_v1")
