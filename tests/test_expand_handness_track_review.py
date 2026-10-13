import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.expand_handness_track_review import expand


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_expand_labels_only_candidate_views(tmp_path):
    candidates = tmp_path / "candidates.csv"
    review = tmp_path / "review.csv"
    output = tmp_path / "items.csv"
    write(candidates, ["item_id", "rec", "canonical_tid"], [
        {"item_id": "a", "rec": "R1", "canonical_tid": 3},
        {"item_id": "b", "rec": "R1", "canonical_tid": 3},
        {"item_id": "c", "rec": "R2", "canonical_tid": 8},
    ])
    write(review, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 3, "label": "hand"},
        {"rec": "R2", "canonical_tid": 8, "label": "nothand"},
    ])
    report = expand(candidates, review, output)
    assert list(csv.DictReader(open(output))) == [
        {"item_id": "a", "label": "hand",
         "review_status": "reviewed_track_hardneg_v1"},
        {"item_id": "b", "label": "hand",
         "review_status": "reviewed_track_hardneg_v1"},
        {"item_id": "c", "label": "nothand",
         "review_status": "reviewed_track_hardneg_v1"},
    ]
    assert report["track_labels"] == {"hand": 1, "nothand": 1}
    assert report["view_labels"] == {"hand": 2, "nothand": 1}


def test_expand_requires_every_reviewed_track_in_candidates(tmp_path):
    candidates = tmp_path / "candidates.csv"
    review = tmp_path / "review.csv"
    write(candidates, ["item_id", "rec", "canonical_tid"], [
        {"item_id": "a", "rec": "R1", "canonical_tid": 3},
    ])
    write(review, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": 3, "label": "hand"},
        {"rec": "R2", "canonical_tid": 8, "label": "nothand"},
    ])
    with pytest.raises(ValueError, match="no candidate views"):
        expand(candidates, review, tmp_path / "out.csv")
