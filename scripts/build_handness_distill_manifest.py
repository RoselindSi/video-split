#!/usr/bin/env python3
"""Build a leakage-safe manifest for the Handness V2 student."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import statistics


HAND_LABELS = {"owner", "other", "hand"}
REC_RE = re.compile(r"(R(?:26_)?\d{4}_\d{6})")
FIELDS = [
    "item_id", "rec", "frame", "raw_tid", "canonical_tid", "sources",
    "stem", "full_path", "crop_path", "full_sha256", "crop_sha256",
    "legacy_labels", "human_label", "teacher_p", "teacher_sources",
    "review_status", "split",
]


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def recording_of(row):
    rec = (row.get("rec") or "").strip()
    if rec:
        return rec
    match = REC_RE.search(row.get("stem", ""))
    return match.group(1) if match else ""


def split_of(rec, salt="handness-v2"):
    value = int(hashlib.sha256(f"{salt}|{rec}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def assign_grouped_splits(rows, salt="handness-v2",
                          fractions=(0.70, 0.15, 0.15)):
    """Assign whole recordings while balancing rows and reviewed labels."""
    names = ("train", "val", "test")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["rec"]].append(row)

    def measures(group):
        labels = collections.Counter(row.get("human_label", "") for row in group)
        return {
            "rows": len(group),
            "hand": labels["hand"],
            "nothand": labels["nothand"],
            "hand_recordings": int(labels["hand"] > 0),
            "nothand_recordings": int(labels["nothand"] > 0),
        }

    features = {rec: measures(group) for rec, group in grouped.items() if rec}
    totals = {
        metric: sum(values[metric] for values in features.values())
        for metric in next(iter(features.values()), {"rows": 0})
    }
    targets = {
        name: {metric: total * fraction for metric, total in totals.items()}
        for name, fraction in zip(names, fractions)
    }
    counts = {name: dict.fromkeys(totals, 0) for name in names}
    weights = {"rows": 4.0, "hand": 2.0, "nothand": 2.0,
               "hand_recordings": 0.5, "nothand_recordings": 2.0}
    ordered = sorted(
        ((rec, group) for rec, group in grouped.items() if rec),
        key=lambda item: (
            -max((features[item[0]][metric] / max(1, totals[metric]))
                 for metric in totals),
            -len(item[1]),
                          hashlib.sha256(f"{salt}|{item[0]}".encode()).hexdigest()))
    for rec, group in ordered:
        values = features[rec]
        choice = min(
            names,
            key=lambda name: (
                sum(
                    weights[metric] *
                    ((counts[other][metric] +
                      (values[metric] if other == name else 0) -
                      targets[other][metric]) /
                     max(1.0, targets[other][metric])) ** 2
                    for other in names for metric in totals
                    if totals[metric] > 0),
                names.index(name)))
        for row in group:
            row["split"] = choice
        for metric, value in values.items():
            counts[choice][metric] += value
    for row in grouped.get("", []):
        row["split"] = "review"
    return {name: counts[name]["rows"] for name in names}


def canonical_maps(root):
    maps = {}
    for path in glob.glob(os.path.join(root, "*_track_maps", "*.track_map.csv")):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            maps[(row["rec"], str(row["raw_tid"]))] = str(row["canonical_tid"])
    return maps


def package_candidates(root):
    seen = set()
    patterns = (os.path.join(root, "*pkg", "hands.csv"),
                os.path.join(root, "*pkg_*", "hands.csv"))
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def add_package_rows(root, output, stats):
    for csv_path in package_candidates(root):
        package = os.path.basename(os.path.dirname(csv_path))
        crop_dir = os.path.join(os.path.dirname(csv_path), "crops")
        full_dir = os.path.join(os.path.dirname(csv_path), "context")
        for row in csv.DictReader(open(csv_path, encoding="utf-8")):
            legacy = (row.get("label") or "").strip()
            if legacy not in HAND_LABELS | {"nothand", "unsure"}:
                continue
            stem = row.get("stem", "")
            crop = os.path.join(crop_dir, stem + ".jpg")
            full = os.path.join(full_dir, stem + ".jpg")
            if not (os.path.isfile(crop) and os.path.isfile(full)):
                stats["missing_package_views"] += 1
                continue
            output.append({
                "rec": recording_of(row),
                "frame": row.get("frame", ""),
                "raw_tid": row.get("tid", ""),
                "canonical_tid": row.get("canonical_tid", ""),
                "source": package,
                "stem": stem,
                "full_path": full,
                "crop_path": crop,
                "legacy_label": legacy,
                "human_label": "hand" if legacy in HAND_LABELS else "",
                "teacher_p": "",
                "teacher_source": "",
                "review_status": ("accepted_positive_identity"
                                  if legacy in HAND_LABELS else "required"),
            })


def read_scores(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def add_teacher_run(output, stats, name, views, scores_path, track_maps):
    for row in read_scores(scores_path):
        stem = row["stem"]
        crop = os.path.join(views, stem + "_crop.jpg")
        full = os.path.join(views, stem + "_full.jpg")
        if not (os.path.isfile(crop) and os.path.isfile(full)):
            stats["missing_teacher_views"] += 1
            continue
        rec = recording_of(row)
        raw_tid = str(row.get("tid", ""))
        canonical_tid = str(row.get("canonical_tid", ""))
        if not canonical_tid and raw_tid:
            canonical_tid = track_maps.get((rec, raw_tid), raw_tid)
        legacy = str(row.get("gold", ""))
        output.append({
            "rec": rec,
            "frame": row.get("frame", ""),
            "raw_tid": raw_tid,
            "canonical_tid": canonical_tid,
            "source": name,
            "stem": stem,
            "full_path": full,
            "crop_path": crop,
            "legacy_label": legacy,
            "human_label": "hand" if legacy in HAND_LABELS else "",
            "teacher_p": float(row["p"]),
            "teacher_source": name,
            "review_status": ("accepted_positive_identity"
                              if legacy in HAND_LABELS else
                              "required" if legacy in {"nothand", "unsure"}
                              else "teacher_only"),
        })


def merge_rows(rows, stats, salt):
    merged = {}
    for row in rows:
        full_hash = sha256_file(row["full_path"])
        crop_hash = sha256_file(row["crop_path"])
        key = (full_hash, crop_hash)
        if key not in merged:
            merged[key] = {**row, "sources_set": {row["source"]},
                           "legacy_set": {row["legacy_label"]} if row["legacy_label"] else set(),
                           "teacher_values": ([float(row["teacher_p"])]
                                              if row["teacher_p"] != "" else []),
                           "teacher_set": ({row["teacher_source"]}
                                           if row["teacher_source"] else set()),
                           "full_sha256": full_hash, "crop_sha256": crop_hash}
            continue
        dst = merged[key]
        dst["sources_set"].add(row["source"])
        if row["legacy_label"]:
            dst["legacy_set"].add(row["legacy_label"])
        if row["teacher_p"] != "":
            dst["teacher_values"].append(float(row["teacher_p"]))
        if row["teacher_source"]:
            dst["teacher_set"].add(row["teacher_source"])
        if not dst["canonical_tid"] and row["canonical_tid"]:
            dst["canonical_tid"] = row["canonical_tid"]
            dst["raw_tid"] = row["raw_tid"]
        if row["human_label"] == "hand":
            dst["human_label"] = "hand"
            dst["review_status"] = "accepted_positive_identity"

    out = []
    for (full_hash, crop_hash), row in merged.items():
        item_id = hashlib.sha256(f"{full_hash}|{crop_hash}".encode()).hexdigest()[:24]
        teacher_values = row.pop("teacher_values")
        row["item_id"] = item_id
        row["sources"] = ";".join(sorted(row.pop("sources_set")))
        row["legacy_labels"] = ";".join(sorted(row.pop("legacy_set")))
        row["teacher_p"] = (f"{statistics.median(teacher_values):.9f}"
                            if teacher_values else "")
        row["teacher_sources"] = ";".join(sorted(row.pop("teacher_set")))
        row["split"] = ""
        row.pop("source", None)
        row.pop("legacy_label", None)
        row.pop("teacher_source", None)
        out.append({field: row.get(field, "") for field in FIELDS})
    assign_grouped_splits(out, salt)
    stats["pre_dedup"] = len(rows)
    stats["deduplicated"] = len(out)
    return sorted(out, key=lambda r: (r["split"], r["rec"], int(r["frame"] or -1), r["item_id"]))


def apply_review(rows, path):
    if not path:
        return
    reviews = {r["item_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    for row in rows:
        review = reviews.get(row["item_id"])
        if not review:
            continue
        label = (review.get("label") or "").strip()
        if label not in {"hand", "nothand", "unsure"}:
            continue
        row["human_label"] = label
        row["review_status"] = "reviewed_v2"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, stats):
    stats["splits"] = dict(collections.Counter(r["split"] for r in rows))
    stats["human_labels"] = dict(collections.Counter(r["human_label"] or "missing" for r in rows))
    stats["review_status"] = dict(collections.Counter(r["review_status"] for r in rows))
    stats["teacher_rows"] = sum(bool(r["teacher_p"]) for r in rows)
    stats["recordings"] = len({r["rec"] for r in rows if r["rec"]})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit_json")
    parser.add_argument("--review_csv")
    parser.add_argument("--split_salt", default="handness-v2")
    parser.add_argument(
        "--teacher", action="append", default=[], metavar="NAME:VIEWS:SCORES",
        help="Qwen teacher run; may be supplied more than once")
    args = parser.parse_args()

    candidates = []
    stats = collections.Counter()
    add_package_rows(args.workspace, candidates, stats)
    maps = canonical_maps(args.workspace)
    for spec in args.teacher:
        name, views, scores = spec.split(":", 2)
        add_teacher_run(candidates, stats, name, views, scores, maps)
    rows = merge_rows(candidates, stats, args.split_salt)
    apply_review(rows, args.review_csv)
    if args.review_csv:
        assign_grouped_splits(rows, args.split_salt)
    write_csv(args.out, rows)
    report = summary(rows, dict(stats))
    audit_path = args.audit_json or args.out + ".audit.json"
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
