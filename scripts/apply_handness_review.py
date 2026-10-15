#!/usr/bin/env python3
"""Apply a complete Handness V2 review export to an existing manifest."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


VALID_LABELS = {"hand", "nothand", "unsure"}


def read_reviews(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    reviews = {}
    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        label = (row.get("label") or "").strip()
        if not item_id:
            raise ValueError("review row has no item_id")
        if item_id in reviews:
            raise ValueError(f"duplicate review item_id: {item_id}")
        if label not in VALID_LABELS:
            raise ValueError(f"invalid review label for {item_id}: {label!r}")
        reviews[item_id] = {
            "label": label,
            "review_status": (row.get("review_status") or "reviewed_v2").strip(),
        }
    return reviews


def apply_review(manifest, review, output, resplit_salt=None,
                 split_overrides=None):
    reviews = read_reviews(review)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    fields = list(rows[0]) if rows else []
    manifest_ids = {row["item_id"] for row in rows}
    unknown = sorted(set(reviews) - manifest_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} review IDs are absent from manifest")

    applied = 0
    for row in rows:
        review_row = reviews.get(row["item_id"])
        if review_row is None:
            continue
        row["human_label"] = review_row["label"]
        row["review_status"] = review_row["review_status"]
        applied += 1
    if applied != len(reviews):
        raise ValueError(f"applied {applied}/{len(reviews)} review rows")

    if resplit_salt:
        from scripts.build_handness_distill_manifest import assign_grouped_splits
        assign_grouped_splits(rows, resplit_salt)
    split_overrides = split_overrides or {}
    invalid = set(split_overrides.values()) - {"train", "val", "test"}
    if invalid:
        raise ValueError(f"invalid split overrides: {sorted(invalid)}")
    for row in rows:
        if row.get("rec") in split_overrides:
            row["split"] = split_overrides[row["rec"]]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_split = collections.Counter(
        (row["split"], row["human_label"] or "missing") for row in rows)
    return {
        "manifest_rows": len(rows),
        "review_rows": len(reviews),
        "applied_rows": applied,
        "resplit_salt": resplit_salt or "",
        "split_overrides": split_overrides,
        "review_labels": dict(collections.Counter(
            row["label"] for row in reviews.values())),
        "human_labels_by_split": {
            split: {
                label: by_split[(split, label)]
                for label in ("hand", "nothand", "unsure", "missing")
            }
            for split in ("train", "val", "test", "review")
            if any(by_split[(split, label)]
                   for label in ("hand", "nothand", "unsure", "missing"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resplit-salt")
    parser.add_argument("--split-override", action="append", default=[],
                        metavar="REC:SPLIT")
    args = parser.parse_args()
    overrides = dict(value.split(":", 1) for value in args.split_override)
    print(json.dumps(apply_review(args.manifest, args.review, args.out,
                                  args.resplit_salt, overrides),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
