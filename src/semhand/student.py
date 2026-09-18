"""The shipped student at inference: same view it was trained on, in the live pipeline.

`own_ctx.predict` hands the classifier two crops cut from the frame; the
student reads the WHOLE frame with the hand's box drawn on it, plus a zoom on
that box. The transforms here are the training ones, step for step
(`qwen.views` then `crossv1.Views`), because a resize in a different order is
a different input distribution:

    frame  PIL RGB -> 1280x704, green box (width 4) -> 448x246
    crop   1.6x the box's longer side, clipped -> 256 -> 224
    both   /255, ImageNet mean/std

WHAT SHIPS WITH IT. The prior and the cap are V1's, fitted for V1's
classifier, and the ablation found both hurt the student: `geom_w 0`, no
`max_owner`. OwnHold's smoothing stays. `predict` returns what
`own_ctx.predict` returns, so the pipeline's call site is one line.

SEEDS ARE AVERAGED, not voted: the student is three checkpoints trained with
different seeds and the reported numbers are the mean probability.
"""
from __future__ import annotations

import glob
import os

import numpy as np

FRAME_PX = (448, 246)
DRAW_PX = (1280, 704)
CROP_PX = 224
CROP_MID = 256
BOX_RGB = (0, 230, 0)


def load(pattern, device=None):
    """-> (models, device). `pattern` is a glob over the seeds' checkpoints."""
    import torch
    from src.rig import own_ctx
    paths = sorted(glob.glob(pattern)) if not os.path.isfile(pattern) else [pattern]
    if not paths:
        return [], None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = own_ctx.build(ck.get("arm", "both")).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        out.append(m)
    return out, device


def views(rgb, box):
    """-> (crop 224 RGB, frame 448x246 RGB) as uint8 arrays, as trained."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb[:, :, ::-1])          # the pipeline works in BGR
    W, H = img.size
    x0, y0, x1, y1 = (float(v) for v in box)
    sx, sy = DRAW_PX[0] / W, DRAW_PX[1] / H
    frame = img.resize(DRAW_PX, Image.BICUBIC)
    ImageDraw.Draw(frame).rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                                    outline=BOX_RGB, width=4)
    side = max(x1 - x0, y1 - y0) * 1.6
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    crop = img.crop((int(cx - side / 2), int(cy - side / 2),
                     int(cx + side / 2), int(cy + side / 2))).resize((CROP_MID, CROP_MID), Image.BICUBIC)
    return (np.asarray(crop.resize((CROP_PX, CROP_PX), Image.BICUBIC)),
            np.asarray(frame.resize(FRAME_PX, Image.BICUBIC)))


def predict(models, device, rgb, dets, batch=32):
    """-> [(is_owner, p_owner)] per detection, like `own_ctx.predict`."""
    import torch
    from src.semhand.crossv1 import norm_rgb
    if not dets or not models:
        return [(True, 1.0)] * len(dets)
    hs, cs = [], []
    for d in dets:
        h, c = views(rgb, d["box"])
        hs.append(norm_rgb(h))
        cs.append(norm_rgb(c))
    out = []
    zero = torch.zeros(len(hs), 14)
    with torch.no_grad():
        for i in range(0, len(hs), batch):
            h = torch.stack(hs[i:i + batch]).to(device)
            c = torch.stack(cs[i:i + batch]).to(device)
            g = zero[i:i + batch].to(device)
            p = np.mean([torch.softmax(m(h, c, g), 1)[:, 1].cpu().numpy() for m in models], 0)
            out.extend(float(x) for x in p)
    return [(x >= 0.5, x) for x in out]
