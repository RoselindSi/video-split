import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_handness_review_packet import build


def test_review_packet_is_blind_and_self_contained(tmp_path):
    full = tmp_path / "full.jpg"
    crop = tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "frame", "full_path", "crop_path",
              "review_status", "teacher_p", "legacy_labels"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"item_id": "abc", "rec": "R1", "frame": 4,
                         "full_path": full, "crop_path": crop,
                         "review_status": "required", "teacher_p": "0.99",
                         "legacy_labels": "nothand"})
    out = tmp_path / "packet"
    assert build(str(manifest), str(out)) == 1
    page = (out / "打开标注页面.html").read_text()
    assert "0.99" not in page
    assert "legacy_labels" not in page
    assert 'lines.join("\\n")' in page
    assert 'lines.join("\n")' not in page
    assert (out / "images/abc_full.jpg").read_bytes() == b"full"
    rows = list(csv.DictReader(open(out / "review.csv")))
    assert rows[0]["label"] == ""
