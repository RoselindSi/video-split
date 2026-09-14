"""The network every arm shares, and the only door V1's weights come through.

ONE ARCHITECTURE, TWO STARTING POINTS. Each arm is a ResNet18 trunk with the
same small head. What differs is only where the trunk starts: torchvision's
ImageNet release, or the matching trunk lifted out of the V1 checkpoint --
`ctx` for input A, `hand` for input B, because those are the inputs each was
fitted to. Labels, crops, sampling, epochs and the head are the same, so a gap
between an ImageNet row and its V1 row is the initialisation.

V1 WEIGHTS COME IN BY PATH, NEVER BY IMPORT, AND ONLY INTO THE TRUNK. The
state dict is filtered to one branch and loaded with `strict=True`, so a
renamed or missing layer fails loudly instead of leaving half a trunk at
random. V1's head, geometry branch and post-processing never enter. The head
here is new in every arm, including the V1 ones: V1's head reads a 1088-wide
vector of hand, context and geometry features that no single-input arm has.

THE INPUT IS 128 x 256, cam3 beside cam4. ResNet18 ends in adaptive pooling,
so the trunk takes the wider image unchanged and no layer is resized -- which
is what lets V1's trunk weights load as they are.

CLASS 1 IS SELF, as in V1, so `softmax(...)[:, 1]` reads the same way in both.
"""
from __future__ import annotations

import hashlib
import os

V1_BRANCH = {"A": "ctx", "B": "hand"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def to_tensor(img_bgr):
    """BGR uint8 -> normalised RGB tensor, as V1 prepares its inputs."""
    import torch
    x = torch.from_numpy(img_bgr[:, :, ::-1].copy()).float().permute(2, 0, 1)
    x = x / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(input_kind, init, v1_ckpt=None):
    """-> nn.Module. init: "imagenet" | "v1" | "none" (architecture only)."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    if input_kind not in V1_BRANCH:
        raise ValueError(f"input must be A or B, got {input_kind!r}")
    if init == "imagenet":
        if v1_ckpt:
            raise ValueError("ImageNet 初始化的 arm 不能带 V1 checkpoint")
        trunk = resnet18(weights="IMAGENET1K_V1")
    elif init == "v1":
        if not v1_ckpt or not os.path.exists(v1_ckpt):
            raise ValueError(f"V1 初始化需要存在的 checkpoint，得到 {v1_ckpt!r}")
        trunk = resnet18(weights=None)
    elif init == "none":
        trunk = resnet18(weights=None)
    else:
        raise ValueError(f"unknown init {init!r}")
    trunk.fc = nn.Identity()

    if init == "v1":
        state = torch.load(v1_ckpt, map_location="cpu",
                           weights_only=False)["state"]
        prefix = V1_BRANCH[input_kind] + "."
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        trunk.load_state_dict(sub, strict=True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(),
                nn.Linear(128, 2))

        def forward(self, x):
            return self.head(self.trunk(x))

    return Net()


def load_trained(path, device):
    """-> (model, metadata) for a checkpoint written by train.py."""
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build(ck["input"], "none").to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, {k: v for k, v in ck.items() if k != "state"}
