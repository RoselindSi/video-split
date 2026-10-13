import csv
import json

from scripts.apply_handness_spatial_teacher import apply


def test_apply_uses_only_human_teacher_agreement_for_training(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["item_id", "human_label"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{"item_id": "a", "human_label": "hand"},
                          {"item_id": "b", "human_label": "nothand"},
                          {"item_id": "c", "human_label": ""}])
    scores = tmp_path / "scores.jsonl"
    values = [
        {"item_id": "a", "parse_ok": True, "teacher_hand": True,
         "teacher_bbox": [1, 2, 10, 20]},
        {"item_id": "b", "parse_ok": True, "teacher_hand": True,
         "teacher_bbox": [1, 2, 10, 20]},
        {"item_id": "c", "parse_ok": True, "teacher_hand": False,
         "teacher_bbox": None},
    ]
    scores.write_text("\n".join(json.dumps(value) for value in values))
    output = tmp_path / "out.csv"
    report = apply(manifest, scores, output)
    rows = list(csv.DictReader(output.open()))
    assert rows[0]["spatial_train_label"] == "hand"
    assert rows[1]["spatial_train_label"] == "unsure"
    assert rows[2]["spatial_train_label"] == ""
    assert report["agreement_hand"] == 1
    assert report["conflict_human_nothand"] == 1
