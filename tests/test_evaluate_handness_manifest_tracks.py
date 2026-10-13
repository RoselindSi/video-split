import csv

from scripts.evaluate_handness_manifest_tracks import evaluate


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_fail_open_and_any_view_rescue(tmp_path):
    manifest = tmp_path / "manifest.csv"
    predictions = tmp_path / "predictions.csv"
    mrows = []
    prows = []
    for tid, label, values in (("1", "hand", [.1, .2, .8]),
                               ("2", "nothand", [.1, .2, .3]),
                               ("3", "nothand", [.1, .2])):
        for index, value in enumerate(values):
            item = f"{tid}-{index}"
            mrows.append({"item_id": item, "human_label": label})
            prows.append({"item_id": item, "rec": "R", "canonical_tid": tid,
                          "p_hand": value})
    write(manifest, ["item_id", "human_label"], mrows)
    write(predictions, ["item_id", "rec", "canonical_tid", "p_hand"], prows)
    tracks, report = evaluate(manifest, predictions, .5, 3)
    assert report["hand_track_recall"] == 1.0
    assert report["nothand_track_rejection"] == 1.0
    assert report["fail_open_tracks"] == 1
    assert next(row for row in tracks if row["canonical_tid"] == "3")[
        "reason"] == "too_few_views"
