import csv

from src.semhand.pixel_student import load_samples, sample_weights


def test_load_samples_excludes_complete_recording(tmp_path):
    path = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "canonical_tid", "split", "human_label",
              "teacher_p", "review_status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"item_id": "a", "rec": "held", "canonical_tid": 1,
             "split": "val", "human_label": "nothand", "teacher_p": .9},
            {"item_id": "b", "rec": "cal", "canonical_tid": 2,
             "split": "val", "human_label": "hand", "teacher_p": .1},
        ])
    rows = load_samples(path, "val", exclude_recs={"held"})
    assert [row["item_id"] for row in rows] == ["b"]
    assert rows[0]["target"] == 1.0


def test_hard_review_weight_is_stronger_within_class():
    rows = [
        {"group": "regular", "target": 0.0, "supervision": "human",
         "review_status": "reviewed_v2"},
        {"group": "hard", "target": 0.0, "supervision": "human",
         "review_status": "reviewed_track_hardneg_v1"},
        {"group": "positive", "target": 1.0, "supervision": "human",
         "review_status": "reviewed_v2"},
    ]
    weights = sample_weights(rows, hard_example_weight=8.0)
    assert weights[1] == 8.0 * weights[0]
