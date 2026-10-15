#!/usr/bin/env python3
"""Evaluate frozen per-view scores with fail-open canonical-track aggregation."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


def read_keyed(path, keys):
    result = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[name] for name in keys)
            if key in result:
                raise ValueError(f"duplicate key {key} in {path}")
            result[key] = row
    return result


def evaluate(index_path, predictions_path, review_path, threshold, min_views=3):
    predictions = read_keyed(predictions_path, ("item_id",))
    reviews = read_keyed(review_path, ("rec", "canonical_tid"))
    grouped = collections.defaultdict(list)
    with open(index_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            canonical_tid = row.get("canonical_tid") or row.get("tid")
            prediction = predictions.get((row["stem"],))
            grouped[(row["rec"], canonical_tid)].append({
                **row,
                "p_hand": "" if prediction is None else prediction["p_hand"],
            })

    tracks = []
    for (rec, canonical_tid), views in sorted(grouped.items()):
        review = reviews.get((rec, canonical_tid), {})
        truth = review.get("label", "") or "unreviewed"
        probabilities = [float(row["p_hand"]) for row in views if row["p_hand"] != ""]
        if len(probabilities) < min_views:
            verdict, reason = "unsure", "too_few_scored_views"
        elif max(probabilities) >= threshold:
            verdict, reason = "hand", "any_view_rescue"
        else:
            verdict, reason = "nothand", "all_views_reject"
        expected = ("hand" if truth in {"hand", "mixed"}
                    else "nothand" if truth == "nothand" else "")
        tracks.append({
            "rec": rec, "canonical_tid": canonical_tid, "truth": truth,
            "expected_policy": expected, "verdict": verdict, "reason": reason,
            "n_views": len(views), "n_scored_views": len(probabilities),
            "p_min": min(probabilities) if probabilities else "",
            "p_median": (sorted(probabilities)[len(probabilities) // 2]
                         if probabilities else ""),
            "p_max": max(probabilities) if probabilities else "",
        })
    return tracks


def report(tracks):
    reviewed = [row for row in tracks if row["expected_policy"]]
    resolved = [row for row in reviewed if row["verdict"] in {"hand", "nothand"}]
    positives = [row for row in reviewed if row["expected_policy"] == "hand"]
    negatives = [row for row in reviewed if row["expected_policy"] == "nothand"]
    matrix = collections.Counter((row["expected_policy"], row["verdict"])
                                 for row in reviewed)
    return {
        "tracks": len(tracks), "reviewed_tracks": len(reviewed),
        "resolved_tracks": len(resolved),
        "mixed_tracks": sum(row["truth"] == "mixed" for row in reviewed),
        "fail_open_tracks": sum(row["verdict"] == "unsure" for row in reviewed),
        "false_deletions": sum(row["expected_policy"] == "hand"
                               and row["verdict"] == "nothand" for row in reviewed),
        "safety_recall": (sum(row["verdict"] != "nothand" for row in positives)
                          / len(positives) if positives else None),
        "nothand_rejection_rate": (sum(row["verdict"] == "nothand" for row in negatives)
                                   / len(negatives) if negatives else None),
        "confusion": {f"{truth}_as_{verdict}": count
                      for (truth, verdict), count in sorted(matrix.items())},
    }


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tracks = evaluate(args.index, args.predictions, args.review,
                      args.threshold, args.min_views)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "track_results.csv"), tracks)
    result = report(tracks)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
