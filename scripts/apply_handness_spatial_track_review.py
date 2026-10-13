#!/usr/bin/env python3
"""Apply blind track labels and recompute trusted spatial supervision."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


VALID = {"hand", "nothand", "unsure"}


def read_reviews(path):
    reviews = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        key = ((row.get("rec") or "").strip(),
               (row.get("canonical_tid") or "").strip())
        label = (row.get("label") or "").strip()
        if not all(key):
            raise ValueError(f"review row has an incomplete track key: {key}")
        if key in reviews:
            raise ValueError(f"duplicate reviewed track: {key}")
        if label not in VALID:
            raise ValueError(f"invalid label for {key}: {label!r}")
        reviews[key] = label
    return reviews


def apply(manifest, review_path, output):
    reviews = read_reviews(review_path)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    fields = list(rows[0]) if rows else []
    seen = set()
    counts = collections.Counter()
    for row in rows:
        key = (row.get("rec", ""), row.get("canonical_tid", ""))
        label = reviews.get(key)
        if label is None:
            continue
        seen.add(key)
        row["human_label"] = label
        row["review_status"] = "reviewed_spatial_track_v1"
        teacher = row.get("spatial_teacher_label", "")
        if label == "unsure" or teacher not in {"hand", "nothand"}:
            row["spatial_status"] = "human_unsure"
            row["spatial_train_label"] = ""
            counts["human_unsure"] += 1
        elif label == teacher:
            row["spatial_status"] = f"agreement_{label}"
            row["spatial_train_label"] = label
            counts[f"agreement_{label}"] += 1
        else:
            row["spatial_status"] = f"conflict_human_{label}"
            row["spatial_train_label"] = "unsure"
            counts[f"conflict_human_{label}"] += 1
        counts[f"review_view_{label}"] += 1

    missing = sorted(set(reviews) - seen)
    if missing:
        raise ValueError(f"{len(missing)} reviewed tracks absent from manifest")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "rows": len(rows), "review_tracks": len(reviews),
        "review_labels": dict(collections.Counter(reviews.values())),
        **dict(counts),
    }
    with open(f"{output}.audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.manifest, args.review, args.out), indent=2,
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
