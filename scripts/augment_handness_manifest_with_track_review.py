#!/usr/bin/env python3
"""Append reviewed canonical-track views to a Handness training manifest."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from scripts.build_handness_distill_manifest import (
    FIELDS, assign_grouped_splits, sha256_file,
)


def augment(base_manifest, feature_manifest, track_review, output,
            split_salt="handness-v2-context-hardneg-v1", split_overrides=None):
    base = list(csv.DictReader(open(base_manifest, encoding="utf-8-sig")))
    reviews = {}
    with open(track_review, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["rec"], row["canonical_tid"])
            if key in reviews:
                raise ValueError(f"duplicate reviewed track: {key}")
            label = row.get("label", "").strip()
            if label not in {"hand", "nothand", "mixed", "unsure"}:
                raise ValueError(f"invalid track label for {key}: {label!r}")
            reviews[key] = "unsure" if label in {"mixed", "unsure"} else label

    existing = {row["item_id"] for row in base}
    additions = []
    with open(feature_manifest, encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            key = (source["rec"], source["canonical_tid"])
            if key not in reviews:
                raise ValueError(f"unreviewed track in feature manifest: {key}")
            item_id = source["item_id"]
            if item_id in existing:
                raise ValueError(f"duplicate item_id: {item_id}")
            existing.add(item_id)
            row = {
                "item_id": item_id, "rec": source["rec"],
                "frame": source["frame"], "raw_tid": source.get("raw_tid", ""),
                "canonical_tid": source["canonical_tid"],
                "sources": "canonical_track_review_v1", "stem": item_id,
                "full_path": source["full_path"], "crop_path": source["crop_path"],
                "full_sha256": sha256_file(source["full_path"]),
                "crop_sha256": sha256_file(source["crop_path"]),
                "legacy_labels": "", "human_label": reviews[key],
                "teacher_p": "", "teacher_sources": "",
                "review_status": "reviewed_track_v1", "split": "",
            }
            additions.append({field: row.get(field, "") for field in FIELDS})

    rows = base + additions
    assign_grouped_splits(rows, split_salt)
    split_overrides = split_overrides or {}
    invalid = set(split_overrides.values()) - {"train", "val", "test"}
    if invalid:
        raise ValueError(f"invalid split overrides: {sorted(invalid)}")
    for row in rows:
        if row["rec"] in split_overrides:
            row["split"] = split_overrides[row["rec"]]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_split = collections.Counter((row["split"], row["human_label"] or "missing")
                                   for row in rows)
    return {
        "base_rows": len(base), "added_rows": len(additions),
        "rows": len(rows), "recordings": len({row["rec"] for row in rows}),
        "added_labels": dict(collections.Counter(row["human_label"]
                                                  for row in additions)),
        "human_labels_by_split": {
            split: {label: by_split[(split, label)]
                    for label in ("hand", "nothand", "unsure", "missing")}
            for split in ("train", "val", "test")
        },
        "split_salt": split_salt,
        "split_overrides": split_overrides,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-json")
    parser.add_argument("--split-salt", default="handness-v2-context-hardneg-v1")
    parser.add_argument("--split-override", action="append", default=[],
                        metavar="REC:SPLIT")
    args = parser.parse_args()
    overrides = dict(value.split(":", 1) for value in args.split_override)
    result = augment(args.base, args.features, args.review, args.out,
                     args.split_salt, overrides)
    audit = args.audit_json or args.out + ".audit.json"
    with open(audit, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
