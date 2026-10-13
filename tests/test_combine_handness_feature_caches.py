import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from scripts.combine_handness_feature_caches import combine


def cache(path, ids, values, kind):
    torch.save({"model": "qwen", "feature_kind": kind,
                "item_ids": ids, "features": torch.tensor(values)}, path)


def test_combines_shards_by_item_id(tmp_path):
    left_a, left_b = tmp_path / "la.pt", tmp_path / "lb.pt"
    right = tmp_path / "r.pt"
    cache(left_a, ["a"], [[1, 2]], "crop")
    cache(left_b, ["b"], [[3, 4]], "crop")
    cache(right, ["b", "a"], [[7], [6]], "context")
    output = tmp_path / "out.pt"
    assert combine([left_a, left_b], [right], output) == (2, 3)
    result = torch.load(output, weights_only=False)
    assert result["item_ids"] == ["a", "b"]
    assert result["features"].float().tolist() == [[1, 2, 6], [3, 4, 7]]


def test_rejects_mismatched_item_sets(tmp_path):
    left, right = tmp_path / "l.pt", tmp_path / "r.pt"
    cache(left, ["a"], [[1]], "crop")
    cache(right, ["b"], [[2]], "context")
    with pytest.raises(ValueError, match="item mismatch"):
        combine([left], [right], tmp_path / "out.pt")
