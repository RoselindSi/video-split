#!/usr/bin/env python3
"""Summarize frozen Handness predictions for canonical tracks in one recording."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics


def summarize(predictions, rec, threshold, min_views=3):
    grouped = collections.defaultdict(list)
    with open(predictions, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["rec"] != rec:
                continue
            tid = row.get("canonical_tid") or row.get("raw_tid") or row["item_id"]
            grouped[tid].append(row)
    tracks = []
    for tid, rows in sorted(grouped.items()):
        probabilities = [float(row["p_hand"]) for row in rows]
        targets = {int(float(row["target"]) >= .5) for row in rows
                   if row.get("target", "") != ""}
        truth = ("mixed" if len(targets) > 1 else
                 "hand" if targets == {1} else
                 "nothand" if targets == {0} else "unknown")
        if len(probabilities) < min_views:
            verdict, reason = "unsure", "too_few_views"
        elif max(probabilities) >= threshold:
            verdict, reason = "hand", "any_view_rescue"
        else:
            verdict, reason = "nothand", "all_views_reject"
        tracks.append({
            "canonical_tid": tid, "truth": truth, "verdict": verdict,
            "reason": reason, "n_views": len(rows),
            "p_min": min(probabilities),
            "p_median": statistics.median(probabilities),
            "p_max": max(probabilities),
        })
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    tracks = summarize(args.predictions, args.recording, args.threshold,
                       args.min_views)
    result = {"recording": args.recording, "threshold": args.threshold,
              "tracks": tracks}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
