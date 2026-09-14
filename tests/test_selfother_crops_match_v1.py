"""Inputs A and B are V1's own crops, pixel for pixel.

This is the one place that imports V1 on purpose: to prove the
re-implementation in `src/selfother/crops.py` feeds the V1-initialised trunks
exactly the distribution they were fitted to, without the package itself
depending on V1's code. Boxes include the frame corners and a degenerate
three-pixel box, where clipping and the too-small rule decide the result.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from src.rig import own_cnn, own_ctx          # noqa: E402
from src.selfother import crops, model         # noqa: E402

IMG = np.random.default_rng(7).integers(0, 256, size=(904, 2048, 3),
                                        dtype=np.uint8)
BOXES = [(100, 600, 260, 880), (0, 0, 40, 30), (1900, 700, 2047, 903),
         (1000, 400, 1003, 402), (500, 100, 900, 500)]


@pytest.mark.parametrize("box", BOXES)
def test_input_b_is_v1_hand_crop(box):
    ours = crops.hand_crop(IMG, box)
    theirs = own_cnn.crop_of(IMG, {"box": box})
    if theirs is None:
        assert ours is None
    else:
        assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("box", BOXES)
def test_input_a_is_v1_context_crop(box):
    H, W = IMG.shape[:2]
    small = crops.shrink(IMG)
    v1_small = cv2.resize(IMG, (own_ctx.CONTEXT_STORE_W,
                                int(own_ctx.CONTEXT_STORE_W * H / W)),
                          interpolation=cv2.INTER_AREA)
    assert np.array_equal(small, v1_small)
    x0, y0, x1, y1 = box
    norm = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
    raw = own_ctx.context_crop(v1_small, norm, own_ctx.CTX_SCALE)
    ours = crops.context_crop(small, box, W, H)
    if raw is None:
        assert ours is None
    else:
        theirs = own_ctx.Pairs([])._fit(raw, own_ctx.CTX_PX)[0]
        assert np.array_equal(ours, theirs)


def test_tensor_is_v1_prep():
    img = np.ascontiguousarray(IMG[:128, :128])
    assert torch.allclose(model.to_tensor(img),
                          own_ctx.Pairs([])._prep(img, own_ctx.HAND_PX))
