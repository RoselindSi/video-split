#!/usr/bin/env python3
"""Evaluate fail-open Handness aggregation directly from manifest predictions."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics


def evaluate(manifest_path, predictions_path, threshold=.5, min_views=3):
    truth = {}
    for row in csv.DictReader(open(manifest_path, encoding="utf-8-sig")):
        label = (row.get("human_label") or "").strip()
        if label in {"hand", "nothand"}:
            truth[row["item_id"]] = label
    grouped = collections.defaultdict(list)
    for row in csv.DictReader(open(predictions_path, encoding="utf-8-sig")):
        label = truth.get(row["item_id"])
        if label is None:
            continue
        key = (row["rec"], row["canonical_tid"] or row["item_id"])
        grouped[key].append((label, float(row["p_hand"]), row["item_id"]))

    tracks = []
    for (rec, tid), views in sorted(grouped.items()):
        labels = {label for label, _probability, _item in views}
        probabilities = [probability for _label, probability, _item in views]
        if len(labels) != 1:
            truth_label, verdict, reason = "unsure", "unsure", "label_conflict"
        elif len(views) < min_views:
            truth_label = next(iter(labels))
            verdict, reason = "unsure", "too_few_views"
        else:
            truth_label = next(iter(labels))
            verdict = "hand" if max(probabilities) >= threshold else "nothand"
            reason = "any_view_rescue" if verdict == "hand" else "all_views_reject"
        tracks.append({
            "rec": rec, "canonical_tid": tid, "truth": truth_label,
            "verdict": verdict, "reason": reason, "n_views": len(views),
            "p_min": min(probabilities),
            "p_median": statistics.median(probabilities),
            "p_max": max(probabilities),
        })
    eligible = [row for row in tracks if row["verdict"] != "unsure"
                and row["truth"] in {"hand", "nothand"}]
    positive = [row for row in eligible if row["truth"] == "hand"]
    negative = [row for row in eligible if row["truth"] == "nothand"]
    report = {
        "threshold": threshold, "min_views": min_views,
        "tracks": len(tracks), "eligible_tracks": len(eligible),
        "eligible_hand_tracks": len(positive),
        "retained_hand_tracks": sum(row["verdict"] == "hand"
                                    for row in positive),
        "hand_track_recall": (sum(row["verdict"] == "hand"
                                  for row in positive) / len(positive)
                              if positive else None),
        "eligible_nothand_tracks": len(negative),
        "rejected_nothand_tracks": sum(row["verdict"] == "nothand"
                                       for row in negative),
        "nothand_track_rejection": (sum(row["verdict"] == "nothand"
                                        for row in negative) / len(negative)
                                    if negative else None),
        "fail_open_tracks": sum(row["verdict"] == "unsure" for row in tracks),
    }
    return tracks, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tracks, report = evaluate(args.manifest, args.predictions, args.threshold,
                              args.min_views)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"report": report, "tracks": tracks}, handle,
                  indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
