import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_handness_recording_tracks import summarize


def test_summarizes_fail_open_tracks(tmp_path):
    path = tmp_path / "predictions.csv"
    fields = ["item_id", "rec", "canonical_tid", "p_hand", "target"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tid, values, target in (("1", (.1, .2), 0),
                                    ("2", (.1, .2, .3), 0),
                                    ("3", (.1, .8, .2), 1)):
            for index, value in enumerate(values):
                writer.writerow({"item_id": f"{tid}-{index}", "rec": "R1",
                                 "canonical_tid": tid, "p_hand": value,
                                 "target": target})
    rows = summarize(path, "R1", threshold=.5)
    assert [row["verdict"] for row in rows] == ["unsure", "nothand", "hand"]
