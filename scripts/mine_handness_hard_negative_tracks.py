#!/usr/bin/env python3
"""Mine teacher-negative/student-positive canonical tracks for blind review."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os


FIELDS = [
    "stem", "item_id", "rec", "frame", "tid", "canonical_tid",
    "full_path", "crop_path", "student_p", "teacher_p", "candidate_views",
]


def _evenly_spaced(rows, limit):
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (limit - 1))
               for index in range(limit)]
    return [rows[index] for index in dict.fromkeys(indices)]


def mine(manifest_path, prediction_paths, output, teacher_max=.20,
         student_min=.60, max_views=5, unlabeled_only=True,
         excluded_tracks=None, min_candidate_views=1):
    excluded_tracks = set(excluded_tracks or ())
    manifest = {
        row["item_id"]: row
        for row in csv.DictReader(open(manifest_path, encoding="utf-8-sig"))
    }
    grouped = collections.defaultdict(list)
    for path in prediction_paths:
        with open(path, encoding="utf-8-sig") as handle:
            for prediction in csv.DictReader(handle):
                row = manifest.get(prediction["item_id"])
                if row is None:
                    continue
                teacher = row.get("teacher_p", "").strip()
                human = row.get("human_label", "").strip()
                canonical_tid = row.get("canonical_tid", "").strip()
                if (not teacher or not canonical_tid or
                        (row["rec"], canonical_tid) in excluded_tracks or
                        float(teacher) > teacher_max or
                        float(prediction["p_hand"]) < student_min or
                        (unlabeled_only and human)):
                    continue
                grouped[(row["rec"], canonical_tid)].append({
                    **row, "student_p": float(prediction["p_hand"]),
                    "teacher_p": float(teacher),
                })

    grouped = collections.defaultdict(list, {
        key: rows for key, rows in grouped.items()
        if len(rows) >= min_candidate_views
    })
    selected = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["frame"] or 0), row["item_id"]))
        for row in _evenly_spaced(rows, max_views):
            selected.append({
                "stem": row["item_id"], "item_id": row["item_id"],
                "rec": row["rec"], "frame": row["frame"],
                "tid": row.get("raw_tid", ""),
                "canonical_tid": row["canonical_tid"],
                "full_path": row["full_path"], "crop_path": row["crop_path"],
                "student_p": f"{row['student_p']:.9f}",
                "teacher_p": f"{row['teacher_p']:.9f}",
                "candidate_views": len(rows),
            })
    selected.sort(key=lambda row: (
        row["rec"], row["canonical_tid"], int(row["frame"] or 0)))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    return {
        "tracks": len(grouped), "candidate_views": sum(map(len, grouped.values())),
        "selected_views": len(selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-max", type=float, default=.20)
    parser.add_argument("--student-min", type=float, default=.60)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--min-candidate-views", type=int, default=1)
    parser.add_argument("--exclude-track-review", action="append", default=[])
    parser.add_argument("--include-reviewed", action="store_true")
    args = parser.parse_args()
    paths = sum((glob.glob(pattern) for pattern in args.predictions), [])
    excluded = set()
    for path in args.exclude_track_review:
        with open(path, encoding="utf-8-sig") as handle:
            excluded.update(
                (row["rec"].strip(), row["canonical_tid"].strip())
                for row in csv.DictReader(handle))
    result = mine(args.manifest, paths, args.out, args.teacher_max,
                  args.student_min, args.max_views, not args.include_reviewed,
                  excluded, args.min_candidate_views)
    print(" ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
