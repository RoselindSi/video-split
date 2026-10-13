#!/usr/bin/env python3
"""Convert canonical-track view indexes into an unlabeled feature manifest."""
from __future__ import annotations

import argparse
import csv
import os


FIELDS = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
          "full_path", "crop_path", "split", "human_label", "teacher_p"]


def build(index_path, views_dir, output):
    rows = []
    with open(index_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "item_id": row["stem"], "rec": row["rec"],
                "frame": row["frame"], "raw_tid": row.get("tid", ""),
                "canonical_tid": row.get("canonical_tid") or row.get("tid", ""),
                "full_path": os.path.join(views_dir, row["stem"] + "_full.jpg"),
                "crop_path": os.path.join(views_dir, row["stem"] + "_crop.jpg"),
                "split": "test", "human_label": "", "teacher_p": "",
            })
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(f"{build(args.index, args.views, args.out)} views -> {args.out}")


if __name__ == "__main__":
    main()
