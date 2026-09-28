"""Train a Qwen-distilled Handness student with auxiliary spatial supervision.

The ResNet-18 is initialized from scratch. Qwen supplies soft hand-presence
targets and, for reviewed views, a hand box that becomes a 7x7 heatmap target.
No Qwen or ImageNet model is needed at inference time.

THE FULL FRAME NOW CARRIES THE BOX, as a fourth channel. Until it did, the
dual view encoded the crop and the whole frame with one shared stem and
concatenated them, with nothing saying which part of that frame the crop came
from. The only thing the full branch could learn from that is "is there a hand
somewhere in this picture", which is nearly always yes -- and that is why
turning the full view on pushed head false positives from 0.772 to 0.897
instead of down. The teacher never worked blind: Qwen was shown the frame and
told which crop to judge.

A MASK RATHER THAN A DRAWN RECTANGLE, because the colour jitter in the train
transform would repaint a drawn box while leaving the mask alone, and the
horizontal flip has to move the frame and the box together or the supervision
is simply wrong. So `full` is built by hand here instead of going through the
shared `transform()`: resize, flip both or neither, jitter the colours only,
then append the mask.

THE BOXES WERE RECOVERED FIRST (`box_backfill.py`). They existed on 2,417 of
14,592 rows and that subset was almost all positive -- 12 negatives in train --
so box and negatives were mutually exclusive until the source packages were
joined back in. Rows that still lack a box get an all-zero mask rather than
being dropped, which keeps the negative supply intact and tells the model
"no box known" in a way it can learn to distrust.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random

from src.semhand.pixel_student import metrics, sample_weights, transform
from src.semhand.handness_head import target_of


def spatial_target(row, size=7):
    """Return a flattened heatmap and whether this row has spatial supervision."""
    label = (row.get("spatial_train_label") or "").strip()
    if label not in {"hand", "nothand"}:
        return [0.0] * (size * size), 0.0
    target = [0.0] * (size * size)
    if label == "nothand":
        return target, 1.0
    values = [row.get(f"spatial_bbox_{name}", "")
              for name in ("x1", "y1", "x2", "y2")]
    if not all(values):
        return target, 0.0
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(value) / 1000.0))
                      for value in values]
    if x2 <= x1 or y2 <= y1:
        return target, 0.0
    hit = False
    for y in range(size):
        for x in range(size):
            cx, cy = (x + 0.5) / size, (y + 0.5) / size
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                target[y * size + x] = 1.0
                hit = True
    if not hit:
        x = min(size - 1, max(0, int(((x1 + x2) / 2) * size)))
        y = min(size - 1, max(0, int(((y1 + y2) / 2) * size)))
        target[y * size + x] = 1.0
    return target, 1.0


def load_samples(manifest, split, exclude_recs=None):
    exclude_recs = set(exclude_recs or ())
    samples = []
    for row in csv.DictReader(open(manifest, encoding="utf-8-sig")):
        if row["split"] != split or row.get("rec") in exclude_recs:
            continue
        target, supervision = target_of(row)
        if target is None:
            continue
        group = (f"{row['rec']}|{row['canonical_tid']}"
                 if row.get("canonical_tid") else row["item_id"])
        heatmap, spatial_mask = spatial_target(row)
        samples.append({**row, "target": target, "supervision": supervision,
                        "group": group, "heatmap": heatmap,
                        "spatial_mask": spatial_mask})
    return samples


def build_model(view="dual", box_channel=True):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    def trunk(in_ch=3):
        b = resnet18(weights=None)
        if in_ch != 3:
            b.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        return nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool,
                             b.layer1, b.layer2, b.layer3, b.layer4)

    class SpatialStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = trunk(3)
            self.view = view
            self.box_channel = box_channel and view == "dual"
            # 全图这一路不再和裁剪共用 stem：它多一个框通道，而且看的是完全不同的
            # 分布（整个场景 vs 一只手），共享权重只会让两边互相拖累。
            self.full_stem = trunk(4 if self.box_channel else 3) \
                if view == "dual" else None
            self.classifier = nn.Linear(1024 if view == "dual" else 512, 1)
            self.spatial_head = nn.Conv2d(512, 1, kernel_size=1)

        def forward(self, crop, full=None):
            crop_features = self.stem(crop)
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                crop_features, 1).flatten(1)
            if self.view == "dual":
                if full is None:
                    raise ValueError("dual view requires full context images")
                full_features = self.full_stem(full)
                full_pooled = torch.nn.functional.adaptive_avg_pool2d(
                    full_features, 1).flatten(1)
                pooled = torch.cat((pooled, full_pooled), dim=1)
            logits = self.classifier(pooled).squeeze(1)
            heatmap_logits = self.spatial_head(crop_features).squeeze(1)
            return logits, heatmap_logits

    return SpatialStudent()


def box_mask(row, size=224):
    """-> 224x224 的框掩码；没有框就全 0（而不是丢掉这一行）。"""
    import torch
    m = torch.zeros(1, size, size)
    try:
        cx, cy = float(row["box_cx"]), float(row["box_cy"])
        w, h = float(row["box_w"]), float(row["box_h"])
    except (KeyError, TypeError, ValueError):
        return m
    x0 = int(max(0, min(1, cx - w / 2)) * size)
    x1 = int(max(0, min(1, cx + w / 2)) * size)
    y0 = int(max(0, min(1, cy - h / 2)) * size)
    y1 = int(max(0, min(1, cy + h / 2)) * size)
    if x1 > x0 and y1 > y0:
        m[:, y0:y1, x0:x1] = 1.0
    return m


def make_dataset(samples, train=False, weights=None, box_channel=True):
    import random as _random

    from PIL import Image
    from torch.utils.data import Dataset

    crop_transform = transform(train)
    full_transform = transform(train)

    class SpatialDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            import torch
            from torchvision import transforms as T

            row = samples[index]
            crop = crop_transform(Image.open(row["crop_path"]).convert("RGB"))
            if not box_channel:
                full = full_transform(Image.open(row["full_path"]).convert("RGB"))
            else:
                # 手写这一路：翻转必须让图和掩码一起翻，抖色只能动 RGB
                im = Image.open(row["full_path"]).convert("RGB").resize((224, 224))
                mask = box_mask(row)
                flip = train and _random.random() < 0.5
                if flip:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = torch.flip(mask, dims=[2])
                if train:
                    im = T.ColorJitter(brightness=.15, contrast=.15,
                                       saturation=.10, hue=.02)(im)
                rgb = T.Normalize((.5, .5, .5), (.25, .25, .25))(T.ToTensor()(im))
                full = torch.cat([rgb, mask], dim=0)
            weight = weights[index] if weights is not None else 1.0
            return (crop, full, float(row["target"]), float(weight),
                    torch.tensor(row["heatmap"], dtype=torch.float32).view(7, 7),
                    float(row["spatial_mask"]), index)

    return SpatialDataset()


def spatial_loss(logits, targets, mask):
    import torch
    import torch.nn.functional as functional

    active = mask > 0
    if not torch.any(active):
        return logits.sum() * 0.0
    logits, targets = logits[active], targets[active]
    pixel_weights = 1.0 + 4.0 * targets
    bce = functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    bce = (bce * pixel_weights).mean()
    positive = targets.flatten(1).sum(1) > 0
    if not torch.any(positive):
        return bce
    probs = torch.sigmoid(logits[positive]).flatten(1)
    truth = targets[positive].flatten(1)
    dice = 1.0 - ((2.0 * (probs * truth).sum(1) + 1.0) /
                  (probs.sum(1) + truth.sum(1) + 1.0))
    return bce + dice.mean()


def records_for(model, loader, samples, device, view):
    import torch

    model.eval()
    records = []
    spatial_losses = []
    with torch.no_grad():
        for crop, full, targets, _weights, heatmaps, masks, indices in loader:
            crop = crop.to(device)
            full = full.to(device) if view == "dual" else None
            logits, heatmap_logits = model(crop, full)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            loss = spatial_loss(heatmap_logits, heatmaps.float().to(device),
                                masks.float().to(device))
            if masks.sum().item():
                spatial_losses.append(float(loss.cpu()))
            for probability, target, index in zip(
                    probabilities, targets.tolist(), indices.tolist()):
                records.append({**samples[index], "p": float(probability),
                                "target": float(target)})
    return records, (sum(spatial_losses) / len(spatial_losses)
                     if spatial_losses else math.nan)


def train(args):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_samples(args.manifest, "train")
    val_rows = load_samples(args.manifest, "val",
                            exclude_recs=args.exclude_val_rec)
    if not train_rows or not val_rows:
        raise SystemExit("manifest needs non-empty train and validation samples")
    weights = sample_weights(train_rows, args.human_weight,
                             args.hard_example_weight)
    train_loader = DataLoader(
        make_dataset(train_rows, True, weights, box_channel=not args.no_box),
        batch_size=args.batch,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        make_dataset(val_rows, box_channel=not args.no_box),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.view, box_channel=not args.no_box).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best = math.inf
    stale = 0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for crop, full, targets, batch_weights, heatmaps, masks, _ in train_loader:
            crop = crop.to(device)
            full = full.to(device) if args.view == "dual" else None
            targets = targets.float().to(device)
            batch_weights = batch_weights.float().to(device)
            heatmaps = heatmaps.float().to(device)
            masks = masks.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, heatmap_logits = model(crop, full)
            classification = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none")
            classification = (classification * batch_weights).mean()
            localization = spatial_loss(heatmap_logits, heatmaps, masks)
            loss = classification + args.spatial_weight * localization
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            count += len(targets)
        records, val_spatial = records_for(
            model, val_loader, val_rows, device, args.view)
        report = metrics(records)
        report.update({"epoch": epoch, "train_loss": total / count,
                       "val_spatial_loss": val_spatial,
                       "train_spatial_n": sum(
                           row["spatial_mask"] > 0 for row in train_rows)})
        history.append(report)
        print(json.dumps(report, allow_nan=True), flush=True)
        objective = report["brier"]
        if objective < best:
            best = objective
            stale = 0
            torch.save({
                "state": {key: value.cpu()
                          for key, value in model.state_dict().items()},
                "architecture": "resnet18_qwen_spatial_distill",
                "view": args.view,
                "normalization": {"mean": [.5] * 3, "std": [.25] * 3},
                "heatmap_size": 7, "seed": args.seed, "epoch": epoch,
                "spatial_weight": args.spatial_weight, "val": report,
            }, args.out)
        else:
            stale += 1
            if stale >= args.patience:
                break
    with open(f"{args.out}.history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=True)


def score(args):
    import torch
    from torch.utils.data import DataLoader

    rows = load_samples(args.manifest, args.split)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    models = []
    checkpoint_view = None
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "resnet18_qwen_spatial_distill":
            raise ValueError("unsupported spatial student checkpoint")
        view = checkpoint.get("view", "crop")
        if checkpoint_view is not None and view != checkpoint_view:
            raise ValueError("checkpoint view mismatch")
        checkpoint_view = view
        model = build_model(view, box_channel=not getattr(args, "no_box", False)).to(device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        models.append(model)
    loader = DataLoader(make_dataset(rows, box_channel=not getattr(args, "no_box", False)),
                        batch_size=args.batch,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True,
                        persistent_workers=args.workers > 0)
    records = []
    with torch.no_grad():
        for crop, full, targets, _weights, _heatmaps, _masks, indices in loader:
            crop = crop.to(device)
            full = full.to(device) if checkpoint_view == "dual" else None
            probabilities = torch.stack([
                torch.sigmoid(model(crop, full)[0]) for model in models
            ]).mean(0).cpu().tolist()
            for probability, target, index in zip(
                    probabilities, targets.tolist(), indices.tolist()):
                records.append({**rows[index], "p": float(probability),
                                "target": float(target)})
    print(json.dumps(metrics(records), indent=2, allow_nan=True))
    if args.predictions:
        with open(args.predictions, "w", newline="", encoding="utf-8") as handle:
            fields = ["item_id", "rec", "frame", "raw_tid",
                      "canonical_tid", "p_hand", "target", "supervision"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({
                    "item_id": row["item_id"], "rec": row["rec"],
                    "frame": row["frame"], "raw_tid": row["raw_tid"],
                    "canonical_tid": row["canonical_tid"],
                    "p_hand": row["p"], "target": row["target"],
                    "supervision": row["supervision"],
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=6)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--workers", type=int, default=6)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--human-weight", type=float, default=4.0)
    fit.add_argument("--hard-example-weight", type=float, default=16.0)
    fit.add_argument("--spatial-weight", type=float, default=.35)
    fit.add_argument("--view", choices=("crop", "dual"), default="dual")
    # 关掉框通道 = 复现旧版本的对照臂，不是默认路径
    fit.add_argument("--no-box", action="store_true",
                     help="全图不带框掩码（旧口径，用作对照）")
    fit.add_argument("--exclude-val-rec", action="append", default=[])
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device")
    run = commands.add_parser("score")
    run.add_argument("--manifest", required=True)
    run.add_argument("--checkpoint", action="append", required=True)
    run.add_argument("--split", choices=("train", "val", "test"),
                     default="test")
    run.add_argument("--batch", type=int, default=256)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--device")
    run.add_argument("--predictions")
    run.add_argument("--no-box", action="store_true")
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
