import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_handness_track_review_packet import build


def test_packet_groups_views_by_canonical_track_and_is_blind(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    rows = []
    for index in range(3):
        stem = f"sample_{index}"
        (views / f"{stem}_full.jpg").write_bytes(f"full{index}".encode())
        (views / f"{stem}_crop.jpg").write_bytes(f"crop{index}".encode())
        rows.append({"stem": stem, "rec": "R1", "frame": 10 + index,
                     "tid": "7", "canonical_tid": "3", "conf": .9,
                     "n_claimed": 1})
    index_path = views / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "packet"
    assert build(str(index_path), str(views), str(out)) == 1
    page = (out / "打开标注页面.html").read_text()
    assert "canonical track 3" not in page
    assert "conf" not in page
    assert "mixed" in page
    assert 'lines.join("\\n")' in page
    review = list(csv.DictReader(open(out / "review.csv")))
    assert review == [{"rec": "R1", "canonical_tid": "3", "label": "",
                       "note": "", "n_views": "3"}]
    assert len(list((out / "images").glob("*.jpg"))) == 6


def test_packet_accepts_explicit_image_paths(tmp_path):
    full, crop = tmp_path / "a.jpg", tmp_path / "b.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "stem", "rec", "frame", "tid", "canonical_tid",
            "full_path", "crop_path"])
        writer.writeheader()
        writer.writerow({"stem": "x", "rec": "R1", "frame": 1, "tid": 2,
                         "canonical_tid": 2, "full_path": full,
                         "crop_path": crop})
    out = tmp_path / "packet"
    assert build(index, tmp_path / "unused", out) == 1
    assert sorted(path.read_bytes() for path in (out / "images").glob("*.jpg")) == [
        b"crop", b"full"]
