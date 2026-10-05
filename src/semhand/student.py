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

# 训练池的帧口径。S_wide_g6 的正例来自 1600x900 的全景渲染，而 FRAME_PX
# 和 DRAW_PX 都是照着它定的。部署喂的是 cam3 原始帧 1920x1520。
TRAINED_FRAME = (1600, 900)
ASPECT_TOLERANCE = 0.08
_WARNED = set()


def check_frame(shape, source="", trained=TRAINED_FRAME):
    """输入帧的宽高比和训练池对不上就说出来，每种尺寸只说一次。

    这一类错误不会抛异常、不会让数字看起来可疑，只会让每个预测静默偏掉。
    这个项目里它已经发生过三次：hand-ness 学生训在全景跑在原始帧（对照
    94.5% 掉到 12%）、三分类的负例是原始帧而正例是全景（模型学到的是渲染
    不是内容）、以及这里 —— 整帧分支被纵向压扁三成，实测 2.9% 的框判定翻转。
    三次里有两次只要有这一行检查就不会发生。

    只警告不阻断：口径不符时正确的做法不是停机，是让读数字的人知道数字
    是在什么条件下产生的。
    """
    height, width = shape[0], shape[1]
    if not height or not width:
        return
    actual, expected = width / height, trained[0] / trained[1]
    if abs(actual - expected) <= ASPECT_TOLERANCE:
        return
    key = (width, height)
    if key in _WARNED:
        return
    _WARNED.add(key)
    print("！输入帧 %dx%d 宽高比 %.2f，训练池 %dx%d 是 %.2f%s\n"
          "  整帧分支会被缩到 %s，形变约 %.0f%%。裁剪分支是正方形不受影响。\n"
          "  `views(..., letterbox=True)` 保形，但会改变已上线模型的每一个预测。"
          % (width, height, actual, trained[0], trained[1], expected,
             "（%s）" % source if source else "",
             "x".join(str(v) for v in DRAW_PX),
             100 * abs(actual / expected - 1)))


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


def views(rgb, box, letterbox=False):
    """-> (crop 224 RGB, frame 448x246 RGB) as uint8 arrays, as trained.

    THE FRAME BRANCH IS RESIZED TO A FIXED SHAPE, and that shape came from the
    training pool: 1600x900 at 1.78, which lands on 1280x704 at 1.82 almost
    unchanged. The pipeline feeds it a cam3 frame at 1920x1520, which is 1.26 --
    the same hand arrives about a third shorter than anything the weights were
    fitted on. The crop branch is square and unaffected; only this one is.

    `letterbox` scales by the smaller factor and pads instead, so the shape is
    preserved and the model sees the proportions it was trained on with blank
    margins. It is off by default because turning it on changes every
    prediction the deployed model has ever made, and that is a decision with
    numbers attached rather than a bug fix to slip in.
    """
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb[:, :, ::-1])          # the pipeline works in BGR
    W, H = img.size
    x0, y0, x1, y1 = (float(v) for v in box)
    if letterbox:
        scale = min(DRAW_PX[0] / W, DRAW_PX[1] / H)
        new = (max(1, int(W * scale)), max(1, int(H * scale)))
        pad_x, pad_y = (DRAW_PX[0] - new[0]) // 2, (DRAW_PX[1] - new[1]) // 2
        frame = Image.new("RGB", DRAW_PX, (0, 0, 0))
        frame.paste(img.resize(new, Image.BICUBIC), (pad_x, pad_y))
        sx = sy = scale
        ImageDraw.Draw(frame).rectangle(
            [x0 * sx + pad_x, y0 * sy + pad_y, x1 * sx + pad_x, y1 * sy + pad_y],
            outline=BOX_RGB, width=4)
    else:
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


def predict(models, device, rgb, dets, batch=32, letterbox=False):
    """-> [(is_owner, p_owner)] per detection, like `own_ctx.predict`."""
    import torch
    from src.semhand.crossv1 import norm_rgb
    if not dets or not models:
        return [(True, 1.0)] * len(dets)
    check_frame(rgb.shape, source="semhand.student.predict")
    hs, cs = [], []
    for d in dets:
        h, c = views(rgb, d["box"], letterbox=letterbox)
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
