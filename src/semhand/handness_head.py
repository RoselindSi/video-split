"""Rejected ImageNet control for Handness V2.

The retained path is ``semhand.qwen_feature_head``. This module remains only
to reproduce the 2026-09-26 backbone comparison and must not be deployed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random


def target_of(row):
    label = (row.get("human_label") or "").strip()
    if label == "hand":
        return 1.0, "human"
    if label == "nothand":
        return 0.0, "human"
    if label == "unsure":
        return None, None
    value = (row.get("teacher_p") or "").strip()
    if value:
        return float(value), "teacher"
    return None, None


def load_samples(manifest, split):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    out = []
    for row in rows:
        if row["split"] != split:
            continue
        target, supervision = target_of(row)
        if target is None:
            continue
        group = (f"{row['rec']}|{row['canonical_tid']}"
                 if row.get("canonical_tid") else row["item_id"])
        out.append({**row, "target": target, "supervision": supervision,
                    "group": group})
    return out


def sample_weights(samples, human_weight=4.0):
    groups = collections.Counter(row["group"] for row in samples)
    class_totals = collections.Counter()
    base_values = []
    for row in samples:
        cls = int(row["target"] >= 0.5)
        value = 1.0 / groups[row["group"]]
        if row["supervision"] == "human":
            value *= human_weight
        base_values.append((cls, value))
        class_totals[cls] += value
    values = [value / class_totals[cls] for cls, value in base_values]
    scale = len(values) / max(1e-12, sum(values))
    return [value * scale for value in values]


def binary_auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(records, threshold=0.5):
    if not records:
        return {"n": 0}
    probs = [row["p"] for row in records]
    targets = [row["target"] for row in records]
    hard = [int(value >= 0.5) for value in targets]
    pred = [int(value >= threshold) for value in probs]
    result = {
        "n": len(records),
        "auc_target_hard": binary_auc(hard, probs),
        "teacher_mae": sum(abs(p - y) for p, y in zip(probs, targets)) / len(records),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, targets)) / len(records),
        "hard_agreement": sum(a == b for a, b in zip(pred, hard)) / len(records),
    }
    human = [row for row in records if row["supervision"] == "human"
             and row.get("human_label") in {"hand", "nothand"}]
    if human:
        hp = [int(row["p"] >= threshold) for row in human]
        hy = [int(row["human_label"] == "hand") for row in human]
        pos = sum(hy)
        neg = len(hy) - pos
        result.update({
            "human_n": len(human),
            "human_auc": binary_auc(hy, [row["p"] for row in human]),
            "human_recall": (sum(a and b for a, b in zip(hp, hy)) / pos
                             if pos else float("nan")),
            "human_specificity": (sum(not a and not b for a, b in zip(hp, hy)) / neg
                                  if neg else float("nan")),
        })
    return result


def build_model(view="crop", pretrained=True):
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet18(weights=weights)
    width = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class HandnessHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            inputs = width if view == "crop" else width * 2
            self.head = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, crop, full=None):
            crop_features = self.backbone(crop)
            if view == "crop":
                features = crop_features
            else:
                if full is None:
                    raise ValueError("dual view needs full images")
                features = torch.cat((crop_features, self.backbone(full)), dim=1)
            return self.head(features).squeeze(1)

    return HandnessHead()


def transforms(train=False):
    from torchvision import transforms as T

    augment = [T.RandomHorizontalFlip(),
               T.ColorJitter(brightness=0.15, contrast=0.15,
                             saturation=0.10, hue=0.02)] if train else []
    normalize = [T.ToTensor(),
                 T.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))]
    crop = T.Compose([T.Resize((224, 224)), *augment, *normalize])
    full = T.Compose([T.Resize((246, 448)), *augment, *normalize])
    return crop, full


def make_dataset(samples, train=False, weights=None):
    from PIL import Image
    from torch.utils.data import Dataset

    crop_transform, full_transform = transforms(train)

    class ManifestDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            row = samples[index]
            crop = crop_transform(Image.open(row["crop_path"]).convert("RGB"))
            full = full_transform(Image.open(row["full_path"]).convert("RGB"))
            return (crop, full, float(row["target"]),
                    float(weights[index] if weights is not None else 1.0), index)

    return ManifestDataset()


def evaluate(model, loader, samples, device, view):
    import torch

    model.eval()
    records = []
    with torch.no_grad():
        for crop, full, target, _weight, indices in loader:
            crop, full = crop.to(device), full.to(device)
            logits = model(crop, full if view == "dual" else None)
            probs = torch.sigmoid(logits).cpu().tolist()
            for p, y, index in zip(probs, target.tolist(), indices.tolist()):
                records.append({**samples[index], "p": float(p), "target": float(y)})
    return records


def train(args):
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_samples(args.manifest, "train")
    val_rows = load_samples(args.manifest, "val")
    if not train_rows or not val_rows:
        raise SystemExit("manifest needs non-empty train and val samples")
    weights = sample_weights(train_rows, args.human_weight)
    train_data = make_dataset(train_rows, train=True, weights=weights)
    val_data = make_dataset(val_rows, train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.view, not args.no_pretrained).to(device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
                                  weight_decay=args.weight_decay)
    best = math.inf
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        total = count = 0.0
        for crop, full, target, weight, _indices in train_loader:
            crop, full = crop.to(device), full.to(device)
            target, weight = target.float().to(device), weight.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(crop, full if args.view == "dual" else None)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            loss = (loss * weight).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(target)
            count += len(target)
        records = evaluate(model, val_loader, val_rows, device, args.view)
        report = metrics(records)
        report.update({"epoch": epoch, "train_loss": total / max(1, count)})
        history.append(report)
        print(json.dumps(report, allow_nan=True), flush=True)
        if report["brier"] < best:
            best = report["brier"]
            torch.save({"state": model.state_dict(), "view": args.view,
                        "seed": args.seed, "epoch": epoch,
                        "freeze_backbone": args.freeze_backbone,
                        "val": report}, args.out)
    with open(args.out + ".history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=True)


def score(args):
    import torch
    from torch.utils.data import DataLoader

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    view = checkpoint["view"]
    rows = load_samples(args.manifest, args.split)
    dataset = make_dataset(rows, train=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(view, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state"])
    records = evaluate(model, loader, rows, device, view)
    print(json.dumps(metrics(records), indent=2, allow_nan=True))
    if args.predictions:
        with open(args.predictions, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["item_id", "rec", "frame",
                                                        "raw_tid", "canonical_tid",
                                                        "p_hand", "target",
                                                        "supervision"])
            writer.writeheader()
            for row in records:
                writer.writerow({"item_id": row["item_id"], "rec": row["rec"],
                                 "frame": row["frame"], "raw_tid": row["raw_tid"],
                                 "canonical_tid": row["canonical_tid"],
                                 "p_hand": row["p"], "target": row["target"],
                                 "supervision": row["supervision"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--view", choices=("crop", "dual"), default="crop")
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--batch", type=int, default=32)
    fit.add_argument("--workers", type=int, default=4)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--weight_decay", type=float, default=1e-4)
    fit.add_argument("--human_weight", type=float, default=4.0)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device")
    fit.add_argument("--no_pretrained", action="store_true")
    fit.add_argument("--freeze_backbone", action="store_true")
    run = sub.add_parser("score")
    run.add_argument("--manifest", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--split", choices=("train", "val", "test"), default="test")
    run.add_argument("--batch", type=int, default=64)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--device")
    run.add_argument("--predictions")
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
