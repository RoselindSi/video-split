#!/usr/bin/env python3
"""Compare representative-track handness sampling to an existing full run."""
from __future__ import annotations

import argparse
import csv
import json
import os

from src.rig.post_pass import decide
from src.rig.track_fusion import load_track_map
from src.semhand.three_probe import items_for


def read_jobs(path):
    jobs = {}
    for line in open(path, encoding="utf-8"):
        fields = line.rstrip("\n").split("|")
        if len(fields) >= 4 and not line.startswith("#"):
            jobs[fields[0]] = (fields[1], int(fields[2]), int(fields[3]))
    return jobs


def read_scores(path):
    scores = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            scores[row["stem"]] = float(row["p"])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--track_maps", required=True)
    parser.add_argument("--samples", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--min_px", type=int, default=150)
    args = parser.parse_args()

    jobs = read_jobs(args.jobs)
    scores = read_scores(args.scores)
    maps = {
        rec: load_track_map(os.path.join(
            args.track_maps, rec + ".track_map.csv"))
        for rec in jobs
    }
    rows = {
        rec: list(csv.DictReader(open(
            os.path.join(args.arm, rec + ".csv"), encoding="utf-8")))
        for rec in jobs
    }
    baseline = {
        rec: decide(rec_rows, rec, scores, min_px=args.min_px,
                    track_map=maps[rec])[0]
        for rec, rec_rows in rows.items()
    }

    print(f"full semantic queries: {len(scores)}")
    for count in args.samples:
        selected = items_for(
            args.arm, jobs, 0.10, 1, maps,
            frames_per_track=count, min_px=args.min_px)
        stems = {item["stem"] for item in selected}
        sampled_scores = {stem: scores[stem] for stem in stems if stem in scores}
        changed_boxes = changed_tracks = 0
        changed_recs = []
        for rec, rec_rows in rows.items():
            sampled = decide(
                rec_rows, rec, sampled_scores, min_px=args.min_px,
                track_map=maps[rec])[0]
            changed = [key for key in baseline[rec]
                       if baseline[rec][key] != sampled[key]]
            if changed:
                changed_boxes += len(changed)
                changed_tracks += len({key[1] for key in changed})
                changed_recs.append(f"{rec}:{len(changed)}")
        missing = len(stems - scores.keys())
        print(
            f"samples={count}: queries={len(stems)} "
            f"({len(stems) / max(1, len(scores)):.1%} of full), "
            f"missing_scores={missing}, changed_boxes={changed_boxes}, "
            f"changed_raw_tracks={changed_tracks}, "
            f"changed_recs={','.join(changed_recs) or '-'}")


if __name__ == "__main__":
    main()
