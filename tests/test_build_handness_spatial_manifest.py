import csv

from scripts.build_handness_spatial_manifest import build


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_merge_reconciles_split_and_excludes_conflicts(tmp_path):
    base = tmp_path / "base.csv"
    spatial = tmp_path / "spatial.csv"
    output = tmp_path / "out.csv"
    fields = ["item_id", "rec", "canonical_tid", "human_label",
              "teacher_p", "split"]
    write_csv(base, fields, [{"item_id": "a", "rec": "R1",
                              "canonical_tid": "1", "human_label": "hand",
                              "teacher_p": "", "split": "train"}])
    spatial_fields = fields + ["spatial_status", "spatial_train_label"]
    write_csv(spatial, spatial_fields, [
        {"item_id": "b", "rec": "R1", "canonical_tid": "2",
         "human_label": "hand", "teacher_p": "", "split": "test",
         "spatial_status": "conflict_human_hand",
         "spatial_train_label": "unsure"},
        {"item_id": "c", "rec": "R2", "canonical_tid": "3",
         "human_label": "hand", "teacher_p": "", "split": "val",
         "spatial_status": "agreement_hand",
         "spatial_train_label": "hand"},
    ])
    report = build(base, spatial, output)
    rows = list(csv.DictReader(output.open()))
    assert rows[1]["split"] == "train"
    assert rows[1]["human_label"] == "unsure"
    assert rows[2]["human_label"] == "hand"
    assert report["split_reconciled"] == 1
    assert report["conflict_excluded"] == 1
