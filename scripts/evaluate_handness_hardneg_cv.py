#!/usr/bin/env python3
"""Evaluate held-out reviewed hard-negative tracks with a frozen threshold."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics


def _paths(value):
    return value if isinstance(value, (list, tuple)) else [value]


def evaluate(predictions_path, candidates_path, review_path, heldout_rec,
             threshold, min_views=3):
    predictions = {
        row["item_id"]: float(row["p_hand"])
        for row in csv.DictReader(open(predictions_path, encoding="utf-8-sig"))
    }
    reviews = {}
    for path in _paths(review_path):
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            key = (row["rec"], row["canonical_tid"])
            label = row["label"]
            if key in reviews and reviews[key] != label:
                raise ValueError(f"conflicting labels for reviewed track {key}")
            reviews[key] = label
    grouped = collections.defaultdict(list)
    seen_items = set()
    for path in _paths(candidates_path):
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                item_id = row["item_id"]
                if item_id in seen_items:
                    raise ValueError(f"duplicate candidate item_id: {item_id}")
                seen_items.add(item_id)
                if row["rec"] != heldout_rec or item_id not in predictions:
                    continue
                grouped[(row["rec"], row["canonical_tid"])].append(
                    predictions[item_id])

    tracks = []
    for key, probabilities in sorted(grouped.items()):
        truth = reviews[key]
        if len(probabilities) < min_views:
            verdict, reason = "unsure", "too_few_views"
        elif max(probabilities) >= threshold:
            verdict, reason = "hand", "any_view_rescue"
        else:
            verdict, reason = "nothand", "all_views_reject"
        tracks.append({
            "rec": key[0], "canonical_tid": key[1], "truth": truth,
            "verdict": verdict, "reason": reason, "n_views": len(probabilities),
            "p_min": min(probabilities),
            "p_median": statistics.median(probabilities),
            "p_max": max(probabilities),
        })
    eligible = [
        row for row in tracks
        if row["n_views"] >= min_views and row["truth"] in {"hand", "nothand"}
    ]
    eligible_negative = [
        row for row in tracks
        if row["truth"] == "nothand" and row["n_views"] >= min_views
    ]
    eligible_positive = [
        row for row in tracks
        if row["truth"] == "hand" and row["n_views"] >= min_views
    ]
    rejected_negative = sum(
        row["verdict"] == "nothand" for row in eligible_negative)
    retained_positive = sum(
        row["verdict"] == "hand" for row in eligible_positive)
    return tracks, {
        "heldout_rec": heldout_rec, "threshold": threshold,
        "tracks": len(tracks),
        "eligible_tracks": len(eligible),
        "eligible_negative_tracks": len(eligible_negative),
        "rejected_eligible_negative_tracks": rejected_negative,
        "negative_track_rejection_rate": (
            rejected_negative / len(eligible_negative)
            if eligible_negative else None),
        "eligible_positive_tracks": len(eligible_positive),
        "retained_eligible_positive_tracks": retained_positive,
        "positive_track_recall": (
            retained_positive / len(eligible_positive)
            if eligible_positive else None),
        "fail_open_tracks": sum(row["verdict"] == "unsure" for row in tracks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--review", action="append", required=True)
    parser.add_argument("--heldout-rec", required=True)
    parser.add_argument("--threshold-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-views", type=int, default=3)
    args = parser.parse_args()
    threshold = json.load(open(args.threshold_report))["threshold"]
    tracks, report = evaluate(
        args.predictions, args.candidates, args.review, args.heldout_rec,
        threshold, args.min_views)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"report": report, "tracks": tracks}, handle,
                  indent=2, sort_keys=True)
    print(json.dumps({"report": report, "tracks": tracks},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
