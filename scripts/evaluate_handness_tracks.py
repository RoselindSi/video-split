#!/usr/bin/env python3
"""Align reviewed Handness views to canonical tracks and evaluate fail-open aggregation."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from src.rig.track_fusion import box_iou, load_track_map


def _sample_box(row):
    width = float(row["w_frac"])
    height = float(row["h_frac"])
    cx = float(row["cx_frac"])
    cy = float(row["cy_frac"])
    return (cx - width / 2, cy - height / 2,
            cx + width / 2, cy + height / 2)


def _track_box(row, width, height):
    return (float(row["x0"]) / width, float(row["y0"]) / height,
            float(row["x1"]) / width, float(row["y1"]) / height)


def _read_keyed(paths, key):
    result = {}
    for path in paths:
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            value = row[key]
            if value in result:
                raise ValueError(f"duplicate {key}: {value}")
            result[value] = row
    return result


def package_row(manifest_row, cache):
    package = os.path.dirname(os.path.dirname(manifest_row["full_path"]))
    path = os.path.join(package, "hands.csv")
    if path not in cache:
        grouped = collections.defaultdict(list)
        if os.path.isfile(path):
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                grouped[row["stem"]].append(row)
        cache[path] = grouped
    rows = cache[path].get(manifest_row["stem"], [])
    if not rows:
        return None, "missing_package_box"
    boxes = {tuple(row.get(key, "") for key in
                   ("w_frac", "h_frac", "cx_frac", "cy_frac"))
             for row in rows}
    if len(boxes) != 1:
        return None, "ambiguous_package_box"
    return rows[0], ""


def align_rows(manifest, predictions, track_dir, map_dir, width=1920,
               height=1520, iou_min=0.30, ambiguity_margin=0.10):
    prediction_map = _read_keyed(predictions, "item_id")
    track_cache = {}
    map_cache = {}
    package_cache = {}
    output = []
    for row in csv.DictReader(open(manifest, encoding="utf-8-sig")):
        prediction = prediction_map.get(row["item_id"])
        if prediction is None or row.get("human_label") not in {"hand", "nothand"}:
            continue
        rec = row["rec"]
        if rec not in track_cache:
            path = os.path.join(track_dir, rec + ".csv")
            frames = collections.defaultdict(list)
            if os.path.isfile(path):
                for track_row in csv.DictReader(open(path, encoding="utf-8")):
                    frames[int(track_row["frame"])].append(track_row)
            track_cache[rec] = frames
            map_cache[rec] = load_track_map(
                os.path.join(map_dir, rec + ".track_map.csv"))
        sample, package_status = package_row(row, package_cache)
        candidates = track_cache[rec].get(int(row["frame"] or -1), [])
        status = "matched"
        raw_tid = canonical_tid = ""
        best_iou = second_iou = 0.0
        if sample is None:
            status = package_status
        elif not candidates:
            status = "missing_track_frame"
        else:
            scored = sorted(
                ((box_iou(_sample_box(sample),
                          _track_box(candidate, width, height)), candidate)
                 for candidate in candidates),
                key=lambda pair: (-pair[0], str(pair[1].get("tid", ""))))
            best_iou, best = scored[0]
            second_iou = scored[1][0] if len(scored) > 1 else 0.0
            if best_iou < iou_min:
                status = "low_iou"
            elif second_iou and best_iou - second_iou < ambiguity_margin:
                status = "ambiguous"
            else:
                raw_tid = str(best["tid"])
                canonical_tid = map_cache[rec].get(raw_tid, raw_tid)
        output.append({
            "item_id": row["item_id"], "rec": rec, "frame": row["frame"],
            "split": row["split"], "human_label": row["human_label"],
            "p_hand": prediction["p_hand"], "sources": row["sources"],
            "stem": row["stem"], "raw_tid": raw_tid,
            "canonical_tid": canonical_tid, "match_iou": best_iou,
            "second_iou": second_iou, "match_status": status,
        })
    return output


def aggregate_tracks(rows, threshold, min_views=3):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["match_status"] == "matched" and row["canonical_tid"]:
            grouped[(row["rec"], row["canonical_tid"])].append(row)
    output = []
    for (rec, canonical_tid), views in sorted(grouped.items()):
        labels = {row["human_label"] for row in views}
        probabilities = [float(row["p_hand"]) for row in views]
        if len(labels) != 1:
            truth = "unsure"
            verdict = "unsure"
            reason = "label_conflict"
        elif len(views) < min_views:
            truth = next(iter(labels))
            verdict = "unsure"
            reason = "too_few_views"
        else:
            truth = next(iter(labels))
            verdict = "hand" if max(probabilities) >= threshold else "nothand"
            reason = "any_view_rescue" if verdict == "hand" else "all_views_reject"
        output.append({
            "rec": rec, "canonical_tid": canonical_tid, "truth": truth,
            "verdict": verdict, "reason": reason, "n_views": len(views),
            "p_min": min(probabilities), "p_median": sorted(probabilities)[len(probabilities) // 2],
            "p_max": max(probabilities),
            "item_ids": ";".join(row["item_id"] for row in views),
        })
    return output


def report(aligned, tracks):
    statuses = collections.Counter(row["match_status"] for row in aligned)
    resolved = [row for row in tracks if row["verdict"] in {"hand", "nothand"}
                and row["truth"] in {"hand", "nothand"}]
    matrix = collections.Counter((row["truth"], row["verdict"]) for row in resolved)
    return {
        "views": len(aligned), "alignment_status": dict(statuses),
        "tracks": len(tracks), "resolved_tracks": len(resolved),
        "confusion": {f"{truth}_as_{verdict}": count
                      for (truth, verdict), count in sorted(matrix.items())},
        "unsure_tracks": sum(row["verdict"] == "unsure" for row in tracks),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--track-dir", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--width", type=float, default=1920)
    parser.add_argument("--height", type=float, default=1520)
    parser.add_argument("--iou-min", type=float, default=0.30)
    parser.add_argument("--ambiguity-margin", type=float, default=0.10)
    args = parser.parse_args()
    aligned = align_rows(
        args.manifest, args.predictions, args.track_dir, args.map_dir,
        args.width, args.height, args.iou_min, args.ambiguity_margin)
    tracks = aggregate_tracks(aligned, args.threshold, args.min_views)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "aligned_views.csv"), aligned)
    write_csv(os.path.join(args.out_dir, "track_results.csv"), tracks)
    result = report(aligned, tracks)
    with open(os.path.join(args.out_dir, "report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
