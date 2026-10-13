#!/usr/bin/env python3
"""Build a Handness manifest from existing per-recording track packages."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os

from scripts.build_handness_distill_manifest import FIELDS, assign_grouped_splits


LABEL_MAP = {"owner": "hand", "other": "hand", "nothand": "nothand"}


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build(package_dirs, output, split_salt="handness-trackpkg-v1"):
    output = os.fspath(output)
    rows = []
    seen_hashes = set()
    stats = collections.Counter()
    for package_dir in sorted(set(package_dirs)):
        hands_path = os.path.join(package_dir, "hands.csv")
        tracks_path = os.path.join(package_dir, "tracks.csv")
        if not os.path.isfile(hands_path) or not os.path.isfile(tracks_path):
            continue
        package = os.path.basename(package_dir)
        track_labels = {
            str(row["tid"]): (row.get("label") or "").strip()
            for row in csv.DictReader(open(tracks_path, encoding="utf-8-sig"))
        }
        for source in csv.DictReader(open(hands_path, encoding="utf-8-sig")):
            stem = source["stem"]
            crop = os.path.join(package_dir, "crops", stem + ".jpg")
            full = os.path.join(package_dir, "context", stem + ".jpg")
            if not os.path.isfile(crop) or not os.path.isfile(full):
                stats["missing_views"] += 1
                continue
            full_hash, crop_hash = sha256_file(full), sha256_file(crop)
            pair = (full_hash, crop_hash)
            if pair in seen_hashes:
                stats["duplicate_views"] += 1
                continue
            seen_hashes.add(pair)
            tid = str(source["tid"])
            track_label = track_labels.get(tid, "")
            human_label = LABEL_MAP.get(track_label, "")
            item_id = hashlib.sha256(
                f"{full_hash}|{crop_hash}".encode()).hexdigest()[:24]
            rec = (source.get("recording") or stem.split("_f", 1)[0]).strip()
            rows.append({
                "item_id": item_id,
                "rec": rec,
                "frame": source.get("frame", ""),
                "raw_tid": tid,
                "canonical_tid": tid,
                "sources": package,
                "stem": stem,
                "full_path": full,
                "crop_path": crop,
                "full_sha256": full_hash,
                "crop_sha256": crop_hash,
                "legacy_labels": track_label,
                "human_label": human_label,
                "teacher_p": "",
                "teacher_sources": "",
                "review_status": ("accepted_track_label"
                                  if human_label else "spatial_teacher_pending"),
                "split": "",
                "wrist_x": source.get("wrist_x", ""),
                "wrist_y": source.get("wrist_y", ""),
                "box_cx": source.get("box_cx", ""),
                "box_cy": source.get("box_cy", ""),
                "box_w": source.get("box_w", ""),
                "box_h": source.get("box_h", ""),
            })
            stats[f"track_label:{track_label or 'missing'}"] += 1

    assign_grouped_splits(rows, split_salt)
    extra = ["wrist_x", "wrist_y", "box_cx", "box_cy", "box_w", "box_h"]
    fields = FIELDS + extra
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields}
                         for row in rows)
    report = {
        "rows": len(rows),
        "recordings": len({row["rec"] for row in rows}),
        "tracks": len({(row["rec"], row["canonical_tid"]) for row in rows}),
        "human_labels": dict(collections.Counter(
            row["human_label"] or "missing" for row in rows)),
        "track_labels": dict(collections.Counter(
            row["legacy_labels"] or "missing" for row in rows)),
        "splits": dict(collections.Counter(row["split"] for row in rows)),
        **dict(stats),
    }
    with open(output + ".audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--package-glob", action="append", default=[
        "trackpkg_*", "validpkg_*", "valid2pkg_*",
    ])
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-salt", default="handness-trackpkg-v1")
    args = parser.parse_args()
    package_dirs = sum((glob.glob(os.path.join(args.workspace, pattern))
                        for pattern in args.package_glob), [])
    print(json.dumps(build(package_dirs, args.out, args.split_salt),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
