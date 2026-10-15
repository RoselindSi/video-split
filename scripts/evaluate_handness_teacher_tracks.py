#!/usr/bin/env python3
"""Evaluate direct Qwen handness scores against blind canonical-track labels."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics


def evaluate(scores_path, review_path, threshold=.60, min_views=3):
    reviews = {}
    with open(review_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            reviews[(row["rec"], row["canonical_tid"])] = row["label"]
    grouped = collections.defaultdict(list)
    with open(scores_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("canonical_tid", row.get("tid", "")))
            grouped[(row["rec"], tid)].append(float(row["p"]))

    tracks = []
    for (rec, tid), probabilities in sorted(grouped.items()):
        truth = reviews.get((rec, tid), "unreviewed")
        expected = "hand" if truth in {"hand", "mixed"} else truth
        if len(probabilities) < min_views:
            verdict, reason = "unsure", "too_few_views"
        elif max(probabilities) >= threshold:
            verdict, reason = "hand", "any_view_rescue"
        else:
            verdict, reason = "nothand", "all_views_reject"
        tracks.append({
            "rec": rec, "canonical_tid": tid, "truth": truth,
            "expected_policy": expected, "verdict": verdict, "reason": reason,
            "n_views": len(probabilities), "p_min": min(probabilities),
            "p_median": statistics.median(probabilities),
            "p_max": max(probabilities),
        })
    reviewed = [row for row in tracks if row["expected_policy"] in {"hand", "nothand"}]
    positives = [row for row in reviewed if row["expected_policy"] == "hand"]
    negatives = [row for row in reviewed if row["expected_policy"] == "nothand"]
    report = {
        "threshold": threshold, "tracks": len(tracks),
        "reviewed_tracks": len(reviewed),
        "false_deletions": sum(row["verdict"] == "nothand" for row in positives),
        "hand_retention": (sum(row["verdict"] != "nothand" for row in positives)
                           / len(positives) if positives else None),
        "nothand_rejection": (sum(row["verdict"] == "nothand" for row in negatives)
                              / len(negatives) if negatives else None),
        "fail_open_tracks": sum(row["verdict"] == "unsure" for row in reviewed),
    }
    return tracks, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--threshold", type=float, default=.60)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tracks, report = evaluate(args.scores, args.review, args.threshold,
                              args.min_views)
    os.makedirs(args.out, exist_ok=True)
    fields = list(tracks[0]) if tracks else []
    with open(os.path.join(args.out, "track_results.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tracks)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
