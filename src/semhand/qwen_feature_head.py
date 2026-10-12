"""Train a small Handness V2 head directly on cached Qwen vision features."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random

from src.semhand.handness_head import binary_auc, target_of


def load_feature_map(paths):
    import torch

    result = {}
    metadata = []
    for path in paths:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        metadata.append({key: cache[key] for key in ("model", "feature_kind")})
        for item_id, feature in zip(cache["item_ids"], cache["features"]):
            if item_id in result:
                raise ValueError(f"duplicate feature for {item_id}")
            result[item_id] = feature.float()
    return result, metadata


def load_rows(manifest, split, features, allow_unlabeled=False,
              exclude_recs=None):
    exclude_recs = set(exclude_recs or ())
    rows = []
    for row in csv.DictReader(open(manifest, encoding="utf-8")):
        if (row["split"] != split or row["item_id"] not in features or
                row.get("rec") in exclude_recs):
            continue
        target, supervision = target_of(row)
        if target is None and not allow_unlabeled:
            continue
        if target is None:
            supervision = "unlabeled"
        group = (f"{row['rec']}|{row['canonical_tid']}"
                 if row.get("canonical_tid") else row["item_id"])
        rows.append({**row, "target": target, "supervision": supervision,
                     "group": group})
    return rows


def group_weights(rows, human_weight=1.0, hard_example_weight=1.0):
    groups = collections.Counter(row["group"] for row in rows)
    values = []
    for row in rows:
        value = 1.0 / groups[row["group"]]
        if row["supervision"] == "human":
            value *= human_weight
        if row.get("review_status") == "reviewed_track_hardneg_v1":
            value *= hard_example_weight
        values.append(value)
    scale = len(values) / max(1e-12, sum(values))
    return [value * scale for value in values]


def build_head(feature_dim, hidden=256):
    import torch.nn as nn

    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, 1),
    )


def records_for(models, rows, features, device, batch):
    import torch

    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            x = torch.stack([features[row["item_id"]] for row in part]).to(device)
            probs = torch.stack([
                torch.sigmoid(model(x).squeeze(1)) for model in models
            ]).mean(0).cpu().tolist()
            records.extend({**row, "p": float(probability)}
                           for row, probability in zip(part, probs))
    return records


def metrics(records, threshold=0.5):
    if not records:
        return {"n": 0}
    probs = [row["p"] for row in records]
    targets = [float(row["target"]) for row in records]
    hard = [int(target >= 0.5) for target in targets]
    predicted = [int(probability >= threshold) for probability in probs]
    human = [row for row in records if row["supervision"] == "human"]
    result = {
        "n": len(records),
        "auc_target_hard": binary_auc(hard, probs),
        "teacher_mae": sum(abs(a - b) for a, b in zip(probs, targets)) / len(records),
        "brier": sum((a - b) ** 2 for a, b in zip(probs, targets)) / len(records),
        "hard_agreement": sum(a == b for a, b in zip(predicted, hard)) / len(records),
        "human_n": len(human),
    }
    if human:
        positives = [row for row in human if row["human_label"] == "hand"]
        negatives = [row for row in human if row["human_label"] == "nothand"]
        result["human_recall"] = (sum(row["p"] >= threshold for row in positives)
                                  / len(positives) if positives else math.nan)
        result["human_specificity"] = (sum(row["p"] < threshold for row in negatives)
                                       / len(negatives) if negatives else math.nan)
    return result


def train(args):
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    features, metadata = load_feature_map(args.features)
    train_rows = load_rows(args.manifest, "train", features)
    val_rows = load_rows(
        args.manifest, "val", features, exclude_recs=args.exclude_val_rec)
    if not train_rows or not val_rows:
        raise SystemExit("features must cover non-empty train and val splits")
    feature_dim = len(next(iter(features.values())))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_head(feature_dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    weights = group_weights(
        train_rows, args.human_weight, args.hard_example_weight)
    best = math.inf
    history = []
    stale = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        total = count = 0
        for start in range(0, len(order), args.batch):
            indices = order[start:start + args.batch]
            rows = [train_rows[index] for index in indices]
            x = torch.stack([features[row["item_id"]] for row in rows]).to(device)
            target = torch.tensor([row["target"] for row in rows], device=device)
            weight = torch.tensor([weights[index] for index in indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            loss = (loss * weight).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(rows)
            count += len(rows)
        records = records_for(model, val_rows, features, device, args.batch)
        report = metrics(records)
        report.update({"epoch": epoch, "train_loss": total / count})
        history.append(report)
        print(json.dumps(report, allow_nan=True), flush=True)
        if report["brier"] < best:
            best = report["brier"]
            stale = 0
            torch.save({"state": {key: value.cpu() for key, value in model.state_dict().items()},
                        "feature_dim": feature_dim, "hidden": args.hidden,
                        "seed": args.seed, "epoch": epoch, "val": report,
                        "feature_metadata": metadata}, args.out)
        else:
            stale += 1
            if stale >= args.patience:
                break
    with open(args.out + ".history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=True)


def score(args):
    import torch

    features, _ = load_feature_map(args.features)
    rows = load_rows(args.manifest, args.split, features, args.allow_unlabeled)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = []
    signature = None
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        current = (checkpoint["feature_dim"], checkpoint["hidden"])
        if signature is not None and current != signature:
            raise ValueError(f"checkpoint architecture mismatch: {current} != {signature}")
        signature = current
        model = build_head(*current).to(device)
        model.load_state_dict(checkpoint["state"])
        models.append(model)
    records = records_for(models, rows, features, device, args.batch)
    labeled = [row for row in records if row["target"] is not None]
    report = metrics(labeled) if labeled else {"n": len(records), "labeled_n": 0}
    print(json.dumps(report, indent=2, allow_nan=True))
    if args.predictions:
        with open(args.predictions, "w", newline="", encoding="utf-8") as handle:
            fields = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
                      "p_hand", "target", "supervision"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({"item_id": row["item_id"], "rec": row["rec"],
                                 "frame": row["frame"], "raw_tid": row["raw_tid"],
                                 "canonical_tid": row["canonical_tid"],
                                 "p_hand": row["p"], "target": row["target"],
                                 "supervision": row["supervision"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("train")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--features", action="append", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--patience", type=int, default=8)
    fit.add_argument("--batch", type=int, default=512)
    fit.add_argument("--hidden", type=int, default=256)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--weight_decay", type=float, default=1e-4)
    fit.add_argument("--human_weight", type=float, default=1.0)
    fit.add_argument("--hard-example-weight", type=float, default=1.0)
    fit.add_argument("--exclude-val-rec", action="append", default=[])
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device")
    run = subparsers.add_parser("score")
    run.add_argument("--manifest", required=True)
    run.add_argument("--features", action="append", required=True)
    run.add_argument("--checkpoint", action="append", required=True)
    run.add_argument("--split", choices=("train", "val", "test"), default="test")
    run.add_argument("--batch", type=int, default=512)
    run.add_argument("--device")
    run.add_argument("--predictions")
    run.add_argument("--allow-unlabeled", action="store_true",
                     help="score manifest rows without requiring labels")
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
