#!/usr/bin/env python3
"""Build blind review candidates for tracks lacking human Handness labels."""
from __future__ import annotations

import argparse
import collections
import csv
import os

from scripts.select_handness_spatial_probe import evenly_spaced


CANDIDATE_FIELDS = [
    "item_id", "rec", "frame", "canonical_tid", "full_path", "crop_path",
]
SUMMARY_FIELDS = [
    "rec", "canonical_tid", "n_views", "n_hand", "n_nothand",
    "n_parse_error", "suggested_label",
]


def build(manifest, candidates_out, summary_out, max_views=5, min_views=3):
    grouped = collections.defaultdict(list)
    for row in csv.DictReader(open(manifest, encoding="utf-8-sig")):
        if row.get("human_label"):
            continue
        grouped[(row["rec"], row["canonical_tid"])].append(row)

    candidate_rows, summaries = [], []
    for (rec, canonical_tid), rows in sorted(grouped.items()):
        parsed = [row for row in rows if row.get("spatial_parse_ok") == "1"]
        hand = [row for row in parsed
                if row.get("spatial_teacher_label") == "hand"]
        nothand = [row for row in parsed
                   if row.get("spatial_teacher_label") == "nothand"]
        if len(parsed) < min_views:
            suggestion = "unsure"
        elif hand and not nothand:
            suggestion = "hand"
        elif nothand and not hand:
            suggestion = "nothand"
        else:
            suggestion = "mixed"
        summaries.append({
            "rec": rec, "canonical_tid": canonical_tid,
            "n_views": len(rows), "n_hand": len(hand),
            "n_nothand": len(nothand),
            "n_parse_error": len(rows) - len(parsed),
            "suggested_label": suggestion,
        })

        if hand and nothand:
            selected = (evenly_spaced(hand, max(1, max_views - 2)) +
                        evenly_spaced(nothand, min(2, max_views)))
            selected = sorted(selected, key=lambda row: int(row.get("frame") or 0))
        else:
            selected = evenly_spaced(parsed or rows, max_views)
        for row in selected[:max_views]:
            candidate_rows.append({field: row.get(field, "")
                                   for field in CANDIDATE_FIELDS})

    for path, fields, data in (
            (candidates_out, CANDIDATE_FIELDS, candidate_rows),
            (summary_out, SUMMARY_FIELDS, summaries)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return {
        "tracks": len(summaries), "selected_views": len(candidate_rows),
        "suggestions": dict(collections.Counter(
            row["suggested_label"] for row in summaries)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--min-views", type=int, default=3)
    args = parser.parse_args()
    result = build(args.manifest, args.candidates_out, args.summary_out,
                   args.max_views, args.min_views)
    print(" ".join(f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
