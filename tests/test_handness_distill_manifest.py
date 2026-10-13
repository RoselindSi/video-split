import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_handness_distill_manifest import (
    apply_review, assign_grouped_splits, merge_rows, split_of,
)


def _image(path, payload):
    path.write_bytes(payload)
    return str(path)


def _row(tmp_path, source, stem, full, crop, legacy="", human="", p="",
         rec="R0821_000001", tid="1", canonical=""):
    return {
        "rec": rec, "frame": "10", "raw_tid": tid,
        "canonical_tid": canonical, "source": source, "stem": stem,
        "full_path": _image(tmp_path / full, b"full"),
        "crop_path": _image(tmp_path / crop, b"crop"),
        "legacy_label": legacy, "human_label": human,
        "teacher_p": p, "teacher_source": source if p != "" else "",
        "review_status": "required" if not human else "accepted_positive_identity",
    }


def test_split_is_grouped_by_recording():
    assert split_of("R0821_000001") == split_of("R0821_000001")
    assert split_of("R0821_000001", "x") == split_of("R0821_000001", "x")


def test_grouped_split_balances_rows_without_recording_leakage():
    rows = []
    for index, size in enumerate((40, 25, 15, 10, 5, 5)):
        rows.extend({"rec": f"R{index}", "split": ""} for _ in range(size))
    counts = assign_grouped_splits(rows, "salt")
    by_rec = {}
    for row in rows:
        by_rec.setdefault(row["rec"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_rec.values())
    assert sum(counts.values()) == 100
    assert counts["train"] >= counts["val"]
    assert counts["train"] >= counts["test"]


def test_grouped_split_distributes_reviewed_negative_recordings():
    rows = []
    for index in range(12):
        rows.extend({"rec": f"R{index}", "split": "", "human_label": "hand"}
                    for _ in range(10))
        if index < 9:
            rows.extend({"rec": f"R{index}", "split": "",
                         "human_label": "nothand"} for _ in range(3))
    assign_grouped_splits(rows, "reviewed")
    negative_splits = {
        row["split"] for row in rows if row["human_label"] == "nothand"
    }
    assert negative_splits == {"train", "val", "test"}


def test_same_stem_different_images_stays_distinct(tmp_path):
    a = _row(tmp_path, "detector_a", "same", "a_full", "a_crop")
    b = _row(tmp_path, "detector_b", "same", "b_full", "b_crop")
    Path(b["full_path"]).write_bytes(b"other-full")
    Path(b["crop_path"]).write_bytes(b"other-crop")
    rows = merge_rows([a, b], {}, "salt")
    assert len(rows) == 2


def test_exact_views_merge_teacher_and_identity_label(tmp_path):
    a = _row(tmp_path, "legacy", "a", "full", "crop", legacy="owner",
             human="hand")
    b = _row(tmp_path, "qwen", "b", "full2", "crop2", p="0.8",
             canonical="9")
    Path(b["full_path"]).write_bytes(b"full")
    Path(b["crop_path"]).write_bytes(b"crop")
    rows = merge_rows([a, b], {}, "salt")
    assert len(rows) == 1
    assert rows[0]["human_label"] == "hand"
    assert rows[0]["canonical_tid"] == "9"
    assert rows[0]["teacher_p"] == "0.800000000"


def test_review_overrides_legacy_negative(tmp_path):
    row = _row(tmp_path, "legacy", "a", "full", "crop", legacy="nothand")
    rows = merge_rows([row], {}, "salt")
    review = tmp_path / "review.csv"
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "label"])
        writer.writeheader()
        writer.writerow({"item_id": rows[0]["item_id"], "label": "hand"})
    apply_review(rows, str(review))
    assert rows[0]["human_label"] == "hand"
    assert rows[0]["review_status"] == "reviewed_v2"
