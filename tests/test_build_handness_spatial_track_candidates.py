import csv

from scripts.build_handness_spatial_track_candidates import build


def test_build_tracks_suggests_mixed_without_leaking_suggestion(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "canonical_tid", "frame", "human_label",
              "spatial_parse_ok", "spatial_teacher_label", "full_path",
              "crop_path"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, label in enumerate(("hand", "hand", "nothand", "nothand")):
            writer.writerow({"item_id": str(frame), "rec": "R1",
                             "canonical_tid": 1, "frame": frame,
                             "human_label": "", "spatial_parse_ok": 1,
                             "spatial_teacher_label": label,
                             "full_path": "f", "crop_path": "c"})
    candidates, summary = tmp_path / "candidates.csv", tmp_path / "summary.csv"
    result = build(manifest, candidates, summary)
    candidate_rows = list(csv.DictReader(candidates.open()))
    summary_rows = list(csv.DictReader(summary.open()))
    assert result["suggestions"] == {"mixed": 1}
    assert summary_rows[0]["suggested_label"] == "mixed"
    assert "suggested_label" not in candidate_rows[0]
