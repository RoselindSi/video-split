import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_handness_hardneg_cv import evaluate


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_applies_fail_open_and_any_view_rescue(tmp_path):
    candidates = tmp_path / "candidates.csv"
    review = tmp_path / "review.csv"
    predictions = tmp_path / "predictions.csv"
    write(candidates, ["item_id", "rec", "canonical_tid"], [
        {"item_id": "a", "rec": "R1", "canonical_tid": 1},
        {"item_id": "b", "rec": "R1", "canonical_tid": 1},
        {"item_id": "c", "rec": "R1", "canonical_tid": 1},
        {"item_id": "d", "rec": "R1", "canonical_tid": 2},
    ])
    write(review, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 1, "label": "nothand"},
        {"rec": "R1", "canonical_tid": 2, "label": "nothand"},
    ])
    write(predictions, ["item_id", "p_hand"], [
        {"item_id": "a", "p_hand": .1}, {"item_id": "b", "p_hand": .2},
        {"item_id": "c", "p_hand": .3}, {"item_id": "d", "p_hand": .1},
    ])
    tracks, report = evaluate(
        predictions, candidates, review, "R1", threshold=.4)
    assert [row["verdict"] for row in tracks] == ["nothand", "unsure"]
    assert report["eligible_negative_tracks"] == 1
    assert report["rejected_eligible_negative_tracks"] == 1
    assert report["negative_track_rejection_rate"] == 1.0
    assert report["eligible_positive_tracks"] == 0
    assert report["positive_track_recall"] is None
    assert report["fail_open_tracks"] == 1


def test_evaluate_reports_positive_track_recall(tmp_path):
    candidates = tmp_path / "candidates.csv"
    review = tmp_path / "review.csv"
    predictions = tmp_path / "predictions.csv"
    write(candidates, ["item_id", "rec", "canonical_tid"], [
        {"item_id": "a", "rec": "R1", "canonical_tid": 1},
        {"item_id": "b", "rec": "R1", "canonical_tid": 1},
        {"item_id": "c", "rec": "R1", "canonical_tid": 1},
        {"item_id": "d", "rec": "R1", "canonical_tid": 2},
        {"item_id": "e", "rec": "R1", "canonical_tid": 2},
        {"item_id": "f", "rec": "R1", "canonical_tid": 2},
    ])
    write(review, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 1, "label": "hand"},
        {"rec": "R1", "canonical_tid": 2, "label": "hand"},
    ])
    write(predictions, ["item_id", "p_hand"], [
        {"item_id": "a", "p_hand": .1}, {"item_id": "b", "p_hand": .2},
        {"item_id": "c", "p_hand": .8}, {"item_id": "d", "p_hand": .1},
        {"item_id": "e", "p_hand": .2}, {"item_id": "f", "p_hand": .3},
    ])
    tracks, report = evaluate(
        predictions, candidates, review, "R1", threshold=.4)
    assert [row["verdict"] for row in tracks] == ["hand", "nothand"]
    assert report["eligible_positive_tracks"] == 2
    assert report["retained_eligible_positive_tracks"] == 1
    assert report["positive_track_recall"] == .5


def test_evaluate_combines_review_rounds_and_rejects_duplicate_items(tmp_path):
    candidates_1 = tmp_path / "candidates_1.csv"
    candidates_2 = tmp_path / "candidates_2.csv"
    review_1 = tmp_path / "review_1.csv"
    review_2 = tmp_path / "review_2.csv"
    predictions = tmp_path / "predictions.csv"
    fields = ["item_id", "rec", "canonical_tid"]
    write(candidates_1, fields, [
        {"item_id": "a", "rec": "R1", "canonical_tid": 1},
    ])
    write(candidates_2, fields, [
        {"item_id": "b", "rec": "R1", "canonical_tid": 2},
    ])
    write(review_1, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 1, "label": "hand"},
    ])
    write(review_2, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 2, "label": "nothand"},
    ])
    write(predictions, ["item_id", "p_hand"], [
        {"item_id": "a", "p_hand": .8},
        {"item_id": "b", "p_hand": .1},
    ])
    tracks, report = evaluate(
        predictions, [candidates_1, candidates_2], [review_1, review_2],
        "R1", threshold=.4, min_views=1)
    assert [row["truth"] for row in tracks] == ["hand", "nothand"]
    assert report["eligible_tracks"] == 2

    write(candidates_2, fields, [
        {"item_id": "a", "rec": "R1", "canonical_tid": 2},
    ])
    with pytest.raises(ValueError, match="duplicate candidate item_id"):
        evaluate(predictions, [candidates_1, candidates_2],
                 [review_1, review_2], "R1", threshold=.4, min_views=1)
