import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mine_handness_hard_negative_tracks import mine


def test_mine_selects_unlabeled_disagreements_by_track(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
              "full_path", "crop_path", "human_label", "teacher_p"]
    rows = []
    for index in range(7):
        rows.append({"item_id": f"x{index}", "rec": "R1", "frame": index,
                     "raw_tid": 8, "canonical_tid": 3,
                     "full_path": f"f{index}", "crop_path": f"c{index}",
                     "human_label": "", "teacher_p": .1})
    rows.append({"item_id": "reviewed", "rec": "R1", "frame": 8,
                 "raw_tid": 8, "canonical_tid": 3, "full_path": "f",
                 "crop_path": "c", "human_label": "nothand", "teacher_p": .1})
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "p_hand"])
        writer.writeheader()
        writer.writerows({"item_id": row["item_id"], "p_hand": .9} for row in rows)
    output = tmp_path / "selected.csv"
    report = mine(manifest, [predictions], output, max_views=5)
    selected = list(csv.DictReader(open(output)))
    assert report == {"tracks": 1, "candidate_views": 7, "selected_views": 5}
    assert {row["canonical_tid"] for row in selected} == {"3"}
    assert all(row["item_id"] != "reviewed" for row in selected)


def test_mine_excludes_reviewed_tracks_and_short_candidates(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
              "full_path", "crop_path", "human_label", "teacher_p"]
    rows = [
        {"item_id": "a", "rec": "R1", "frame": 1, "raw_tid": 1,
         "canonical_tid": 1, "full_path": "f", "crop_path": "c",
         "human_label": "", "teacher_p": .1},
        {"item_id": "b", "rec": "R2", "frame": 1, "raw_tid": 2,
         "canonical_tid": 2, "full_path": "f", "crop_path": "c",
         "human_label": "", "teacher_p": .1},
        {"item_id": "c", "rec": "R2", "frame": 2, "raw_tid": 2,
         "canonical_tid": 2, "full_path": "f", "crop_path": "c",
         "human_label": "", "teacher_p": .1},
    ]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "p_hand"])
        writer.writeheader()
        writer.writerows({"item_id": row["item_id"], "p_hand": .9} for row in rows)
    output = tmp_path / "selected.csv"
    report = mine(manifest, [predictions], output, min_candidate_views=2,
                  excluded_tracks={("R1", "1")})
    assert report == {"tracks": 1, "candidate_views": 2, "selected_views": 2}
    assert {row["rec"] for row in csv.DictReader(open(output))} == {"R2"}
