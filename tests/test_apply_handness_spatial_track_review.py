import csv

from scripts.apply_handness_spatial_track_review import apply


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_applies_track_truth_and_requires_teacher_agreement(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["rec", "canonical_tid", "human_label", "review_status",
              "spatial_teacher_label", "spatial_status",
              "spatial_train_label"]
    write(manifest, fields, [
        {"rec": "R1", "canonical_tid": "1", "human_label": "",
         "spatial_teacher_label": "hand"},
        {"rec": "R1", "canonical_tid": "1", "human_label": "",
         "spatial_teacher_label": "nothand"},
        {"rec": "R2", "canonical_tid": "2", "human_label": "",
         "spatial_teacher_label": "nothand"},
    ])
    review = tmp_path / "review.csv"
    write(review, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": "1", "label": "hand"},
        {"rec": "R2", "canonical_tid": "2", "label": "nothand"},
    ])
    output = tmp_path / "out.csv"
    report = apply(manifest, review, output)
    rows = list(csv.DictReader(output.open()))
    assert rows[0]["spatial_train_label"] == "hand"
    assert rows[1]["spatial_train_label"] == "unsure"
    assert rows[2]["spatial_train_label"] == "nothand"
    assert all(row["review_status"] == "reviewed_spatial_track_v1"
               for row in rows)
    assert report["review_tracks"] == 2
    assert report["conflict_human_hand"] == 1
