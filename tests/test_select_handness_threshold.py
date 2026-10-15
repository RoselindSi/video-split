import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_handness_threshold import build_report, select_threshold


def test_selects_highest_positive_score_with_allowed_false_negatives():
    rows = [(value, 1) for value in (.1, .2, .3, .8)] + [(value, 0) for value in (.05, .25)]
    threshold, allowed = select_threshold(rows, minimum_recall=.75)
    assert allowed == 1
    assert threshold == .2


def test_freezes_validation_threshold_for_test(tmp_path):
    fields = ["p_hand", "target", "supervision"]
    paths = []
    for name, rows in (("val", [(.1, 1), (.8, 1), (.2, 0)]),
                       ("test", [(.15, 1), (.05, 0)])):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({"p_hand": p, "target": y, "supervision": "human"}
                             for p, y in rows)
        paths.append(path)
    result = build_report(paths[0], paths[1], minimum_recall=1.0)
    assert result["threshold"] == .1
    assert result["splits"]["test"]["fn"] == 0


def test_heldout_recording_does_not_select_threshold(tmp_path):
    path = tmp_path / "val.csv"
    fields = ["rec", "p_hand", "target", "supervision"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"rec": "cal", "p_hand": .4, "target": 1, "supervision": "human"},
            {"rec": "cal", "p_hand": .1, "target": 0, "supervision": "human"},
            {"rec": "held", "p_hand": .01, "target": 0, "supervision": "human"},
        ])
    result = build_report(path, None, minimum_recall=1.0,
                          heldout_recs={"held"})
    assert result["threshold"] == .4
    assert result["splits"]["heldout"]["tn"] == 1
    assert result["splits"]["val"]["n"] == 2
