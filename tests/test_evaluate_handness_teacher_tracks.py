import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_handness_teacher_tracks import evaluate


def test_direct_teacher_uses_same_fail_open_track_rule(tmp_path):
    review = tmp_path / "review.csv"
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rec", "canonical_tid", "label"])
        writer.writeheader()
        writer.writerows([
            {"rec": "R1", "canonical_tid": 1, "label": "hand"},
            {"rec": "R1", "canonical_tid": 2, "label": "nothand"},
        ])
    scores = tmp_path / "scores.jsonl"
    with scores.open("w") as handle:
        for tid, values in ((1, (.1, .8)), (2, (.1, .2, .3))):
            for value in values:
                handle.write(json.dumps({"rec": "R1", "canonical_tid": tid,
                                         "p": value}) + "\n")
    tracks, report = evaluate(scores, review, threshold=.6)
    assert [row["verdict"] for row in tracks] == ["unsure", "nothand"]
    assert report["hand_retention"] == 1
    assert report["nothand_rejection"] == 1
