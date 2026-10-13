import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_handness_track_packet import evaluate, report


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_track_packet_evaluation_is_fail_open_and_treats_mixed_as_keep(tmp_path):
    index = tmp_path / "index.csv"
    rows = []
    predictions = []
    for track, probabilities in (("1", (.1, .2)), ("2", (.1, .2, .8)),
                                 ("3", (.1, .2, .3))):
        for view, probability in enumerate(probabilities):
            stem = f"t{track}v{view}"
            rows.append({"stem": stem, "rec": "R1", "tid": track,
                         "canonical_tid": track, "frame": view})
            predictions.append({"item_id": stem, "p_hand": probability})
    write(index, rows[0].keys(), rows)
    prediction_path = tmp_path / "predictions.csv"
    write(prediction_path, predictions[0].keys(), predictions)
    review_path = tmp_path / "review.csv"
    write(review_path, ["rec", "canonical_tid", "label"], [
        {"rec": "R1", "canonical_tid": "1", "label": "nothand"},
        {"rec": "R1", "canonical_tid": "2", "label": "mixed"},
        {"rec": "R1", "canonical_tid": "3", "label": "nothand"},
    ])
    tracks = evaluate(index, prediction_path, review_path, threshold=.5, min_views=3)
    assert [row["verdict"] for row in tracks] == ["unsure", "hand", "nothand"]
    result = report(tracks)
    assert result["false_deletions"] == 0
    assert result["safety_recall"] == 1
    assert result["nothand_rejection_rate"] == .5
