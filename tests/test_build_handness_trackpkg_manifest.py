import csv

from scripts.build_handness_trackpkg_manifest import build


def test_build_maps_track_labels_and_keeps_unlabeled(tmp_path):
    package = tmp_path / "trackpkg_R1"
    (package / "crops").mkdir(parents=True)
    (package / "context").mkdir()
    with (package / "tracks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tid", "label"])
        writer.writeheader()
        writer.writerows([{"tid": 1, "label": "owner"},
                          {"tid": 2, "label": "nothand"},
                          {"tid": 3, "label": ""}])
    fields = ["stem", "tid", "frame", "wrist_x", "wrist_y", "box_cx",
              "box_cy", "box_w", "box_h"]
    with (package / "hands.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tid in (1, 2, 3):
            stem = f"R1_f00000{tid}_h{tid}"
            writer.writerow({"stem": stem, "tid": tid, "frame": tid})
            (package / "crops" / f"{stem}.jpg").write_bytes(f"c{tid}".encode())
            (package / "context" / f"{stem}.jpg").write_bytes(f"f{tid}".encode())
    output = tmp_path / "manifest.csv"
    report = build([package], output)
    rows = list(csv.DictReader(output.open()))
    assert [row["human_label"] for row in rows] == ["hand", "nothand", ""]
    assert rows[2]["review_status"] == "spatial_teacher_pending"
    assert report["tracks"] == 3
