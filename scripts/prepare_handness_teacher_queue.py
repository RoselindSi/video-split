#!/usr/bin/env python3
"""Expose manifest views under stable item ids for one-time Qwen scoring."""
from __future__ import annotations

import argparse
import csv
import os


FIELDS = ["stem", "rec", "frame", "conf", "tid", "canonical_tid",
          "n_claimed"]


def selected(rows, statuses, missing_only=True):
    for row in rows:
        if statuses and row["review_status"] not in statuses:
            continue
        if missing_only and row.get("teacher_p"):
            continue
        yield row


def link_view(source, destination):
    if os.path.lexists(destination):
        if os.path.realpath(destination) == os.path.realpath(source):
            return
        raise FileExistsError(destination)
    os.symlink(source, destination)


def prepare(manifest, views, statuses=("required",), missing_only=True):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    queue = list(selected(rows, set(statuses), missing_only))
    os.makedirs(views, exist_ok=True)
    index_rows = []
    for row in queue:
        stem = row["item_id"]
        link_view(row["full_path"], os.path.join(views, stem + "_full.jpg"))
        link_view(row["crop_path"], os.path.join(views, stem + "_crop.jpg"))
        index_rows.append({
            "stem": stem,
            "rec": row["rec"],
            "frame": row["frame"] or 0,
            "conf": 0,
            "tid": row["raw_tid"] or "",
            "canonical_tid": row["canonical_tid"] or row["raw_tid"] or "",
            "n_claimed": 1,
        })
    with open(os.path.join(views, "index.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    return len(index_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--status", action="append", default=["required"])
    parser.add_argument("--include_scored", action="store_true")
    args = parser.parse_args()
    count = prepare(args.manifest, args.views, tuple(args.status),
                    missing_only=not args.include_scored)
    print(f"{count} items -> {args.views}")


if __name__ == "__main__":
    main()
