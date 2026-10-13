#!/usr/bin/env python3
"""Join Qwen spatial answers onto a Handness manifest with agreement gating."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


EXTRA_FIELDS = [
    "spatial_parse_ok", "spatial_teacher_label", "spatial_bbox_x1",
    "spatial_bbox_y1", "spatial_bbox_x2", "spatial_bbox_y2",
    "spatial_bbox_area", "spatial_status", "spatial_train_label",
]


def read_scores(path):
    scores = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row["item_id"]
        if item_id in scores:
            raise ValueError(f"duplicate spatial score: {item_id}")
        scores[item_id] = row
    return scores


def apply(manifest, scores_path, output):
    scores = read_scores(scores_path)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    fields = list(rows[0]) if rows else []
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    counts = collections.Counter()
    for row in rows:
        score = scores.get(row["item_id"])
        for field in EXTRA_FIELDS:
            row[field] = ""
        if score is None:
            row["spatial_status"] = "missing_score"
            counts["missing_score"] += 1
            continue
        if not score.get("parse_ok"):
            row["spatial_parse_ok"] = "0"
            row["spatial_status"] = "parse_error"
            counts["parse_error"] += 1
            continue
        row["spatial_parse_ok"] = "1"
        teacher_label = "hand" if score["teacher_hand"] else "nothand"
        row["spatial_teacher_label"] = teacher_label
        bbox = score.get("teacher_bbox")
        if bbox:
            for field, value in zip(EXTRA_FIELDS[2:6], bbox):
                row[field] = str(value)
            row["spatial_bbox_area"] = str(
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        human = row.get("human_label", "")
        if not human:
            status = f"teacher_only_{teacher_label}"
        elif human == teacher_label:
            status = f"agreement_{human}"
            row["spatial_train_label"] = human
        elif human in {"hand", "nothand"}:
            status = f"conflict_human_{human}"
            row["spatial_train_label"] = "unsure"
        else:
            status = "human_unsure"
        row["spatial_status"] = status
        counts[status] += 1

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"rows": len(rows), "scores": len(scores), **dict(counts)}
    audit_path = f"{output}.audit.json"
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.manifest, args.scores, args.out),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
