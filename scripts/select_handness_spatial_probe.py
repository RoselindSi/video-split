#!/usr/bin/env python3
"""Select a recording-diverse spatial-teacher probe from a Handness manifest."""
from __future__ import annotations

import argparse
import collections
import csv
import os


def evenly_spaced(rows, limit):
    rows = sorted(rows, key=lambda row: (int(row.get("frame") or 0),
                                         row["item_id"]))
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (limit - 1))
               for index in range(limit)]
    return [rows[index] for index in dict.fromkeys(indices)]


def select(manifest, output, hand_tracks=5, unlabeled_tracks=5,
           views_per_track=3):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["rec"], row["canonical_tid"])].append(row)

    by_label = collections.defaultdict(list)
    for key, track_rows in grouped.items():
        labels = {row.get("human_label", "") for row in track_rows}
        label = next(iter(labels)) if len(labels) == 1 else "conflict"
        by_label[label].append((key, track_rows))
    for label in by_label:
        by_label[label].sort(key=lambda item: item[0])

    chosen = list(by_label.get("nothand", []))
    chosen.extend(by_label.get("hand", [])[:hand_tracks])
    chosen.extend(by_label.get("", [])[:unlabeled_tracks])
    selected = []
    for _key, track_rows in chosen:
        selected.extend(evenly_spaced(track_rows, views_per_track))
    selected.sort(key=lambda row: (
        row.get("human_label", ""), row["rec"], row["canonical_tid"],
        int(row.get("frame") or 0)))

    fields = list(rows[0]) if rows else []
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return {
        "tracks": len(chosen),
        "views": len(selected),
        "labels": dict(collections.Counter(
            row.get("human_label", "") or "missing" for row in selected)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hand-tracks", type=int, default=5)
    parser.add_argument("--unlabeled-tracks", type=int, default=5)
    parser.add_argument("--views-per-track", type=int, default=3)
    args = parser.parse_args()
    result = select(args.manifest, args.out, args.hand_tracks,
                    args.unlabeled_tracks, args.views_per_track)
    print(" ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
