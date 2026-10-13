from scripts.extract_handness_qwen_features import feature_kind, token_counts
from src.semhand.qwen_feature_head import group_weights, load_rows


def test_unlabeled_rows_can_be_loaded_for_frozen_scoring(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "item_id,rec,canonical_tid,split,human_label,teacher_p\n"
        "a,R1,3,test,,\n")
    features = {"a": object()}
    assert load_rows(manifest, "test", features) == []
    rows = load_rows(manifest, "test", features, allow_unlabeled=True)
    assert rows[0]["target"] is None
    assert rows[0]["supervision"] == "unlabeled"


def test_token_counts_follow_spatial_merge():
    assert token_counts([[1, 16, 16], [1, 8, 12]], 2) == [64, 24]


def test_feature_kind_records_the_visual_input():
    assert feature_kind("crop_path").endswith(":crop_path")
    assert feature_kind("full_path").endswith(":full_path")


def test_group_weights_do_not_rebalance_classes():
    rows = [
        {"group": "track-a", "supervision": "teacher", "target": 1.0},
        {"group": "track-a", "supervision": "teacher", "target": 1.0},
        {"group": "box-b", "supervision": "teacher", "target": 0.0},
    ]
    weights = group_weights(rows)
    assert weights[0] == weights[1]
    assert weights[2] == 2 * weights[0]


def test_group_weights_can_emphasize_reviewed_hard_examples():
    rows = [
        {"group": "ordinary", "supervision": "human", "target": 1.0,
         "review_status": "reviewed_v2"},
        {"group": "hard", "supervision": "human", "target": 0.0,
         "review_status": "reviewed_track_hardneg_v1"},
    ]
    weights = group_weights(rows, hard_example_weight=4)
    assert weights[1] == 4 * weights[0]


def test_load_rows_can_exclude_validation_recording(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "item_id,rec,canonical_tid,split,human_label,teacher_p\n"
        "a,R1,3,val,hand,\n"
        "b,R2,4,val,hand,\n")
    rows = load_rows(manifest, "val", {"a": object(), "b": object()},
                     exclude_recs={"R2"})
    assert [row["rec"] for row in rows] == ["R1"]
