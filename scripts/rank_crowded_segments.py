#!/usr/bin/env python3
"""Rank previously scanned segments by sustained multi-hand activity."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import statistics


def read_jobs(root):
    jobs = collections.defaultdict(list)
    paths = (glob.glob(os.path.join(root, "**", "*jobs*.txt"), recursive=True)
             + glob.glob(os.path.join(root, "**", "jobs.txt"), recursive=True))
    for path in paths:
        try:
            lines = open(path, encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) >= 3 and parts[1].startswith("/shared/"):
                n = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 400
                jobs[parts[0]].append((parts[1], int(parts[2]), n, path))
    return jobs


def excluded_recordings(paths):
    out = set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                out.add(line.split("|", 1)[0])
    return out


def score_csv(path):
    by_frame = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        # Ranking only needs per-frame hand counts.  Older screening runs did
        # not expose tracker ids, but are still valid evidence for crowding.
        if not {"frame", "own"}.issubset(reader.fieldnames or []):
            return None
        for row in reader:
            try:
                by_frame[int(row["frame"])].append(row)
            except (TypeError, ValueError):
                pass
    if len(by_frame) < 100:
        return None

    detections, other, owner = [], [], []
    for rows in by_frame.values():
        n_det = max([int(float(row.get("n_det") or 0)) for row in rows] or [0])
        detections.append(n_det or len(rows))
        other.append(sum(row.get("own") == "0"
                         and row.get("not_hand", "0") != "1" for row in rows))
        owner.append(sum(row.get("own") == "1"
                         and row.get("not_hand", "0") != "1" for row in rows))

    frac4 = sum(value >= 4 for value in detections) / len(detections)
    frac5 = sum(value >= 5 for value in detections) / len(detections)
    multi = sum(o >= 2 and s >= 1 for o, s in zip(other, owner)) / len(detections)
    mean_hands = statistics.mean(detections)
    mean_other = statistics.mean(other)
    score = 2.5 * mean_hands + 4 * frac4 + 5 * frac5 + 5 * multi + mean_other
    return {
        "frames": len(by_frame), "mean_hands": mean_hands,
        "frac4": frac4, "frac5": frac5, "multi_other": multi,
        "max_hands": max(detections), "mean_other": mean_other,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--exclude_jobs", action="append", default=[])
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    jobs = read_jobs(args.root)
    exclude = excluded_recordings(args.exclude_jobs)
    best = {}
    for path in glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True):
        if path.endswith(".faces.csv") or "/context/" in path or "/pkg/" in path:
            continue
        rec = os.path.basename(path)[:-4]
        if rec in exclude or rec not in jobs:
            continue
        try:
            metrics = score_csv(path)
        except (OSError, csv.Error, ValueError):
            continue
        if metrics is None:
            continue
        row = {"rec": rec, "csv": path, **metrics}
        if rec not in best or row["score"] > best[rec]["score"]:
            best[rec] = row

    fields = ("rec", "score", "frames", "mean_hands", "frac4", "frac5",
              "multi_other", "max_hands", "mean_other", "start", "n",
              "bag", "csv")
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    for row in sorted(best.values(), key=lambda item: -item["score"])[:args.limit]:
        bag, start, n, _source = jobs[row["rec"]][0]
        writer.writerow({**row, "bag": bag, "start": start, "n": n})


if __name__ == "__main__":
    main()
