"""Distill Qwen vision features into a Qwen-free Handness pixel student."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random

from src.semhand.pixel_student import (
    load_samples,
    metrics,
    sample_weights,
    transform,
)
from src.semhand.qwen_feature_head import load_feature_map


def build_model(teacher_dim):
    import torch.nn as nn
    from torchvision.models import resnet18

    class FeatureStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = resnet18(weights=None)
            width = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(width, 1)
            self.projector = nn.Linear(width, teacher_dim)

        def forward(self, images):
            features = self.backbone(images)
            return self.classifier(features).squeeze(1), self.projector(features)

    return FeatureStudent()


def make_dataset(samples, teacher_features, train=False, weights=None):
    from PIL import Image
    from torch.utils.data import Dataset

    image_transform = transform(train)

    class FeatureDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            row = samples[index]
            image = image_transform(Image.open(row["crop_path"]).convert("RGB"))
            weight = weights[index] if weights is not None else 1.0
            return (image, teacher_features[row["item_id"]],
                    float(row["target"]), float(weight), index)

    return FeatureDataset()


def records_for(models, loader, samples, device):
    import torch

    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        model.eval()
    records = []
    with torch.no_grad():
        for images, _teacher, targets, _weights, indices in loader:
            images = images.to(device)
            probabilities = torch.stack([
                torch.sigmoid(model(images)[0]) for model in models
            ]).mean(0).cpu().tolist()
            for probability, target, index in zip(
                    probabilities, targets.tolist(), indices.tolist()):
                records.append({**samples[index], "p": float(probability),
                                "target": float(target)})
    return records


def train(args):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    teacher_features, metadata = load_feature_map(args.features)
    train_rows = load_samples(args.manifest, "train")
    val_rows = load_samples(
        args.manifest, "val", exclude_recs=args.exclude_val_rec)
    train_rows = [row for row in train_rows if row["item_id"] in teacher_features]
    val_rows = [row for row in val_rows if row["item_id"] in teacher_features]
    if not train_rows or not val_rows:
        raise SystemExit("teacher features must cover train and validation rows")
    teacher_dim = len(next(iter(teacher_features.values())))
    weights = sample_weights(
        train_rows, args.human_weight, args.hard_example_weight)
    train_loader = DataLoader(
        make_dataset(train_rows, teacher_features, True, weights),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        make_dataset(val_rows, teacher_features), batch_size=args.batch,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(teacher_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    stale = 0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for images, teacher, targets, batch_weights, _indices in train_loader:
            images = images.to(device)
            teacher = teacher.float().to(device)
            targets = targets.float().to(device)
            batch_weights = batch_weights.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, projected = model(images)
            label_loss = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none")
            label_loss = (label_loss * batch_weights).mean()
            feature_loss = (1 - functional.cosine_similarity(
                projected, teacher, dim=1)).mean()
            loss = label_loss + args.feature_weight * feature_loss
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(targets)
            count += len(targets)
        report = metrics(records_for(model, val_loader, val_rows, device))
        report.update({"epoch": epoch, "train_loss": total / count})
        history.append(report)
        print(json.dumps(report, allow_nan=True), flush=True)
        if report["brier"] < best:
            best = report["brier"]
            stale = 0
            torch.save({
                "state": {key: value.cpu()
                          for key, value in model.state_dict().items()},
                "architecture": "resnet18_qwen_feature_distill",
                "teacher_dim": teacher_dim,
                "feature_metadata": metadata,
                "normalization": {"mean": [.5] * 3, "std": [.25] * 3},
                "seed": args.seed, "epoch": epoch, "val": report,
            }, args.out)
        else:
            stale += 1
            if stale >= args.patience:
                break
    with open(args.out + ".history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=True)


def score(args):
    import torch
    from torch.utils.data import DataLoader

    rows = load_samples(args.manifest, args.split)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False)
                   for path in args.checkpoint]
    teacher_dim = checkpoints[0]["teacher_dim"]
    models = []
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for checkpoint in checkpoints:
        if (checkpoint.get("architecture") !=
                "resnet18_qwen_feature_distill"):
            raise ValueError("unsupported feature student checkpoint")
        if checkpoint["teacher_dim"] != teacher_dim:
            raise ValueError("teacher dimension mismatch")
        model = build_model(teacher_dim).to(device)
        model.load_state_dict(checkpoint["state"])
        models.append(model)
    dummy = {row["item_id"]: torch.empty(teacher_dim) for row in rows}
    loader = DataLoader(
        make_dataset(rows, dummy), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    records = records_for(models, loader, rows, device)
    print(json.dumps(metrics(records), indent=2, allow_nan=True))
    if args.predictions:
        with open(args.predictions, "w", newline="", encoding="utf-8") as handle:
            fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
                      "p_hand", "target", "supervision"]
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("train")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--features", action="append", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=6)
    fit.add_argument("--batch", type=int, default=96)
    fit.add_argument("--workers", type=int, default=6)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--human-weight", type=float, default=4.0)
    fit.add_argument("--hard-example-weight", type=float, default=4.0)
    fit.add_argument("--feature-weight", type=float, default=1.0)
    fit.add_argument("--exclude-val-rec", action="append", default=[])
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device")
    run = subparsers.add_parser("score")
    run.add_argument("--manifest", required=True)
    run.add_argument("--checkpoint", action="append", required=True)
    run.add_argument("--split", choices=("train", "val", "test"),
                     default="test")
    run.add_argument("--batch", type=int, default=256)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--device")
    run.add_argument("--predictions")
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
