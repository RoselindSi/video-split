import csv

from scripts.select_handness_spatial_probe import select


def test_selects_all_negative_tracks_and_limits_other_groups(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "rec", "canonical_tid", "frame", "human_label"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec, tid, label in (("R1", 1, "nothand"), ("R2", 2, "hand"),
                                ("R3", 3, "")):
            for frame in range(5):
                writer.writerow({"item_id": f"{rec}-{frame}", "rec": rec,
                                 "canonical_tid": tid, "frame": frame,
                                 "human_label": label})
    output = tmp_path / "probe.csv"
    result = select(manifest, output, hand_tracks=1, unlabeled_tracks=1,
                    views_per_track=3)
    assert result["tracks"] == 3
    assert result["views"] == 9
    assert result["labels"] == {"missing": 3, "hand": 3, "nothand": 3}
