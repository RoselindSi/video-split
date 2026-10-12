"""Train a fast crop-only Handness student from Qwen and human labels.

The student uses a randomly initialized ResNet-18. It never loads ImageNet
weights and does not require Qwen at inference time.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random

from src.semhand.handness_head import binary_auc, target_of


def load_samples(manifest, split, exclude_recs=None):
    exclude_recs = set(exclude_recs or ())
    samples = []
    for row in csv.DictReader(open(manifest, encoding="utf-8")):
        if row["split"] != split or row.get("rec") in exclude_recs:
            continue
        target, supervision = target_of(row)
        if target is None:
            continue
        group = (f"{row['rec']}|{row['canonical_tid']}"
                 if row.get("canonical_tid") else row["item_id"])
        samples.append({**row, "target": target, "supervision": supervision,
                        "group": group})
    return samples


def sample_weights(samples, human_weight=4.0, hard_example_weight=16.0):
    groups = collections.Counter(row["group"] for row in samples)
    class_totals = collections.Counter()
    weighted = []
    for row in samples:
        cls = int(row["target"] >= 0.5)
        value = 1.0 / groups[row["group"]]
        if row["supervision"] == "human":
            value *= human_weight
        if row.get("review_status") == "reviewed_track_hardneg_v1":
            value *= hard_example_weight
        weighted.append((cls, value))
        class_totals[cls] += value
    values = [value / class_totals[cls] for cls, value in weighted]
    scale = len(values) / max(1e-12, sum(values))
    return [value * scale for value in values]


def build_model():
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def transform(train=False):
    from torchvision import transforms as T

    augment = [
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=.15, contrast=.15, saturation=.10, hue=.02),
    ] if train else []
    return T.Compose([
        T.Resize((224, 224)),
        *augment,
        T.ToTensor(),
        T.Normalize((.5, .5, .5), (.25, .25, .25)),
    ])


def make_dataset(samples, train=False, weights=None):
    from PIL import Image
    from torch.utils.data import Dataset

    image_transform = transform(train)

    class ManifestDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            row = samples[index]
            image = image_transform(Image.open(row["crop_path"]).convert("RGB"))
            weight = weights[index] if weights is not None else 1.0
            return image, float(row["target"]), float(weight), index

    return ManifestDataset()


def records_for(models, loader, samples, device):
    import torch

    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        model.eval()
    records = []
    with torch.no_grad():
        for images, targets, _weights, indices in loader:
            images = images.to(device)
            probabilities = torch.stack([
                torch.sigmoid(model(images).squeeze(1)) for model in models
            ]).mean(0).cpu().tolist()
            for probability, target, index in zip(
                    probabilities, targets.tolist(), indices.tolist()):
                records.append({**samples[index], "p": float(probability),
                                "target": float(target)})
    return records


def metrics(records, threshold=.5):
    if not records:
        return {"n": 0}
    probabilities = [row["p"] for row in records]
    targets = [row["target"] for row in records]
    hard = [int(target >= .5) for target in targets]
    predicted = [int(probability >= threshold)
                 for probability in probabilities]
    human = [row for row in records if row["supervision"] == "human"]
    result = {
        "n": len(records),
        "auc_target_hard": binary_auc(hard, probabilities),
        "teacher_mae": sum(abs(a - b) for a, b in zip(probabilities, targets))
        / len(records),
        "brier": sum((a - b) ** 2 for a, b in zip(probabilities, targets))
        / len(records),
        "hard_agreement": sum(a == b for a, b in zip(predicted, hard))
        / len(records),
        "human_n": len(human),
    }
    if human:
        positive = [row for row in human if row["human_label"] == "hand"]
        negative = [row for row in human if row["human_label"] == "nothand"]
        result["human_recall"] = (
            sum(row["p"] >= threshold for row in positive) / len(positive)
            if positive else math.nan)
        result["human_specificity"] = (
            sum(row["p"] < threshold for row in negative) / len(negative)
            if negative else math.nan)
    return result


def train(args):
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_samples(args.manifest, "train")
    val_rows = load_samples(
        args.manifest, "val", exclude_recs=args.exclude_val_rec)
    if not train_rows or not val_rows:
        raise SystemExit("manifest needs non-empty train and validation samples")
    weights = sample_weights(
        train_rows, args.human_weight, args.hard_example_weight)
    train_loader = DataLoader(
        make_dataset(train_rows, train=True, weights=weights),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        make_dataset(val_rows), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    stale = 0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for images, targets, batch_weights, _indices in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)
            batch_weights = batch_weights.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none")
            loss = (loss * batch_weights).mean()
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
                "architecture": "resnet18_random_init",
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
    loader = DataLoader(
        make_dataset(rows), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "resnet18_random_init":
            raise ValueError(f"unsupported pixel student checkpoint: {path}")
        model = build_model().to(device)
        model.load_state_dict(checkpoint["state"])
        models.append(model)
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
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=6)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--workers", type=int, default=6)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--human-weight", type=float, default=4.0)
    fit.add_argument("--hard-example-weight", type=float, default=16.0)
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
