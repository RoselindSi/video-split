#!/usr/bin/env python3
"""Merge the reviewed Handness manifest with spatially supervised views."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


def read_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def build(base_path, spatial_path, output):
    base = read_rows(base_path)
    spatial = read_rows(spatial_path)
    base_ids = {row["item_id"] for row in base}
    split_by_rec = {row["rec"]: row["split"] for row in base}
    fields = list(base[0]) if base else []
    for row in spatial:
        for field in row:
            if field not in fields:
                fields.append(field)

    rows = [{field: row.get(field, "") for field in fields} for row in base]
    stats = collections.Counter(base_rows=len(base))
    for source in spatial:
        if source["item_id"] in base_ids:
            stats["duplicate_item"] += 1
            continue
        row = {field: source.get(field, "") for field in fields}
        if row["rec"] in split_by_rec:
            if row["split"] != split_by_rec[row["rec"]]:
                stats["split_reconciled"] += 1
            row["split"] = split_by_rec[row["rec"]]

        status = row.get("spatial_status", "")
        if status.startswith("conflict_"):
            # A track-level label is not reliable for a frame where the crop
            # itself contains no hand. Keep the row for audit, not training.
            row["human_label"] = "unsure"
            row["teacher_p"] = ""
            stats["conflict_excluded"] += 1
        elif status.startswith("teacher_only_"):
            # Blind review is required before teacher-only tracks become
            # classification targets.
            row["human_label"] = ""
            row["teacher_p"] = ""
            stats["teacher_only_excluded"] += 1
        elif status == "agreement_hand":
            stats["spatial_hand"] += 1
        elif status == "agreement_nothand":
            stats["spatial_nothand"] += 1
        rows.append(row)
        stats["spatial_rows"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "rows": len(rows),
        "recordings": len({row["rec"] for row in rows}),
        "tracks": len({(row["rec"], row.get("canonical_tid", ""))
                       for row in rows}),
        "splits": dict(collections.Counter(row["split"] for row in rows)),
        **dict(stats),
    }
    with open(f"{output}.audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.spatial, args.out), indent=2,
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
