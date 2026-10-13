from src.semhand.spatial_student import spatial_target


def test_spatial_target_marks_bbox_cells():
    target, mask = spatial_target({
        "spatial_train_label": "hand",
        "spatial_bbox_x1": "400", "spatial_bbox_y1": "400",
        "spatial_bbox_x2": "600", "spatial_bbox_y2": "600",
    })
    assert mask == 1.0
    assert sum(target) >= 1
    assert target[3 * 7 + 3] == 1.0


def test_spatial_target_handles_negative_and_missing():
    target, mask = spatial_target({"spatial_train_label": "nothand"})
    assert mask == 1.0
    assert sum(target) == 0
    target, mask = spatial_target({"spatial_train_label": ""})
    assert mask == 0.0
    assert sum(target) == 0


def test_dataset_batches_heatmaps_as_tensor(tmp_path):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    from src.semhand.spatial_student import make_dataset

    image = tmp_path / "crop.jpg"
    Image.new("RGB", (20, 20), "white").save(image)
    sample = {
        "crop_path": str(image), "full_path": str(image), "target": 1.0,
        "heatmap": [0.0] * 49, "spatial_mask": 1.0,
    }
    batch = next(iter(DataLoader(make_dataset([sample]), batch_size=1)))
    assert isinstance(batch[4], torch.Tensor)
    assert tuple(batch[4].shape) == (1, 7, 7)


def test_dual_model_requires_and_uses_context():
    import pytest
    import torch

    from src.semhand.spatial_student import build_model

    model = build_model("dual")
    crop = torch.randn(2, 3, 224, 224)
    full = torch.randn(2, 3, 224, 224)
    logits, heatmap = model(crop, full)
    assert tuple(logits.shape) == (2,)
    assert tuple(heatmap.shape) == (2, 7, 7)
    with pytest.raises(ValueError):
        model(crop)
