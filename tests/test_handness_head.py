import csv
import math
from pathlib import Path

from src.semhand.handness_head import (
    binary_auc, load_samples, metrics, sample_weights, target_of,
)


def test_human_label_overrides_teacher():
    assert target_of({"human_label": "nothand", "teacher_p": "0.99"}) == (0.0, "human")
    assert target_of({"human_label": "hand", "teacher_p": "0.01"}) == (1.0, "human")
    assert target_of({"human_label": "unsure", "teacher_p": "0.99"}) == (None, None)
    assert target_of({"human_label": "", "teacher_p": "0.25"}) == (0.25, "teacher")


def test_auc_and_metrics():
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    report = metrics([
        {"p": 0.9, "target": 1.0, "supervision": "human", "human_label": "hand"},
        {"p": 0.1, "target": 0.0, "supervision": "human", "human_label": "nothand"},
    ])
    assert report["hard_agreement"] == 1.0
    assert report["human_recall"] == 1.0
    assert report["human_specificity"] == 1.0


def test_track_and_class_weights_are_finite():
    rows = [
        {"group": "long", "target": 1.0, "supervision": "teacher"},
        {"group": "long", "target": 1.0, "supervision": "teacher"},
        {"group": "negative", "target": 0.0, "supervision": "human"},
    ]
    weights = sample_weights(rows)
    assert len(weights) == 3
    assert all(math.isfinite(value) and value > 0 for value in weights)
    assert weights[2] > weights[0]


def test_human_emphasis_does_not_break_class_balance():
    rows = [
        {"group": "human-pos", "target": 1.0, "supervision": "human"},
        {"group": "teacher-pos", "target": 1.0, "supervision": "teacher"},
        {"group": "teacher-neg-a", "target": 0.0, "supervision": "teacher"},
        {"group": "teacher-neg-b", "target": 0.0, "supervision": "teacher"},
    ]
    weights = sample_weights(rows, human_weight=4.0)
    positive = sum(weight for row, weight in zip(rows, weights)
                   if row["target"] >= 0.5)
    negative = sum(weight for row, weight in zip(rows, weights)
                   if row["target"] < 0.5)
    assert positive == negative
    assert weights[0] > weights[1]


def test_load_samples_keeps_recording_split(tmp_path):
    path = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
              "split", "human_label", "teacher_p"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"item_id": "a", "rec": "R1", "frame": 1,
                         "raw_tid": 1, "canonical_tid": 1, "split": "train",
                         "human_label": "", "teacher_p": 0.8})
        writer.writerow({"item_id": "b", "rec": "R2", "frame": 2,
                         "raw_tid": 2, "canonical_tid": 2, "split": "test",
                         "human_label": "hand", "teacher_p": ""})
    rows = load_samples(str(path), "train")
    assert [row["item_id"] for row in rows] == ["a"]
    assert rows[0]["group"] == "R1|1"
