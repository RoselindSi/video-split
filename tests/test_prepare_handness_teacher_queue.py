import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_handness_teacher_queue import prepare


def test_prepare_links_only_unscored_required_rows(tmp_path):
    full = tmp_path / "full.jpg"
    crop = tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
              "full_path", "crop_path", "review_status", "teacher_p"]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"item_id": "keep", "rec": "R1", "frame": 3,
                         "raw_tid": 4, "canonical_tid": 5,
                         "full_path": full, "crop_path": crop,
                         "review_status": "required", "teacher_p": ""})
        writer.writerow({"item_id": "scored", "rec": "R1", "frame": 4,
                         "raw_tid": 6, "canonical_tid": 6,
                         "full_path": full, "crop_path": crop,
                         "review_status": "required", "teacher_p": "0.2"})
        writer.writerow({"item_id": "positive", "rec": "R2", "frame": 5,
                         "raw_tid": 7, "canonical_tid": 7,
                         "full_path": full, "crop_path": crop,
                         "review_status": "teacher_only", "teacher_p": ""})
    views = tmp_path / "views"
    assert prepare(str(manifest), str(views)) == 1
    assert (views / "keep_full.jpg").is_symlink()
    rows = list(csv.DictReader(open(views / "index.csv")))
    assert [row["stem"] for row in rows] == ["keep"]
    assert rows[0]["canonical_tid"] == "5"
