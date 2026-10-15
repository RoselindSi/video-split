#!/usr/bin/env python3
"""Expand canonical-track review labels onto only the reviewed candidate views."""
from __future__ import annotations

import argparse
import collections
import csv
import os


VALID_LABELS = {"hand", "nothand", "mixed", "unsure"}


def expand(candidates_path, review_path, output):
    reviews = {}
    with open(review_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["rec"].strip(), row["canonical_tid"].strip())
            label = row.get("label", "").strip()
            if key in reviews:
                raise ValueError(f"duplicate reviewed track: {key}")
            if label not in VALID_LABELS:
                raise ValueError(f"invalid or missing label for {key}: {label!r}")
            reviews[key] = "unsure" if label in {"mixed", "unsure"} else label

    output_rows = []
    seen_tracks = set()
    seen_items = set()
    with open(candidates_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["rec"].strip(), row["canonical_tid"].strip())
            if key not in reviews:
                raise ValueError(f"candidate track has no review: {key}")
            item_id = (row.get("item_id") or row.get("stem") or "").strip()
            if not item_id or item_id in seen_items:
                raise ValueError(f"missing or duplicate candidate item_id: {item_id!r}")
            seen_items.add(item_id)
            seen_tracks.add(key)
            output_rows.append({
                "item_id": item_id, "label": reviews[key],
                "review_status": "reviewed_track_hardneg_v1",
            })

    missing = sorted(set(reviews) - seen_tracks)
    if missing:
        raise ValueError(f"{len(missing)} reviewed tracks have no candidate views")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "label", "review_status"])
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "tracks": len(reviews), "views": len(output_rows),
        "track_labels": dict(collections.Counter(reviews.values())),
        "view_labels": dict(collections.Counter(row["label"] for row in output_rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = expand(args.candidates, args.review, args.out)
    print(" ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
