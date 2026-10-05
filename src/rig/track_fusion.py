"""Conservatively join tracker fragments without rewriting raw track ids.

The online tracker remains the source of every detection and ``tid``.  This
offline pass adds a canonical id only when one track ends and another starts
within a few frames at the same image location.  Ambiguous continuations and
strong handedness conflicts stay separate.  The mapping is therefore
auditable, reversible, and reusable by every later semantic pass.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os


BOX_FIELDS = ("x0", "y0", "x1", "y1")
SIDES = ("left", "right")


def _box(row):
    return tuple(float(row[key]) for key in BOX_FIELDS)


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = ((a[2] - a[0]) * (a[3] - a[1])
            + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _predict(rows, target_frame, from_end):
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    anchor = ordered[-1] if from_end else ordered[0]
    if len(ordered) < 2:
        return _box(anchor)
    neighbour = ordered[-2] if from_end else ordered[1]
    af = int(anchor["frame"])
    nf = int(neighbour["frame"])
    if af == nf:
        return _box(anchor)
    anchor_box, neighbour_box = _box(anchor), _box(neighbour)
    velocity = tuple((anchor_box[i] - neighbour_box[i]) / (af - nf)
                     for i in range(4))
    delta = int(target_frame) - af
    return tuple(anchor_box[i] + velocity[i] * delta for i in range(4))


def _stable_side(rows, min_frac=0.80):
    counts = collections.Counter(
        str(row.get("side") or "").strip().lower() for row in rows)
    counts = {key: value for key, value in counts.items() if key in SIDES}
    if not counts:
        return ""
    side, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return side if count / sum(counts.values()) >= min_frac else ""


def summarize_tracks(rows, side_min_frac=0.80):
    grouped = collections.defaultdict(list)
    for row in rows:
        tid = str(row.get("tid") or "").strip()
        if tid:
            grouped[tid].append(row)
    out = {}
    for tid, track_rows in grouped.items():
        track_rows.sort(key=lambda row: int(row["frame"]))
        out[tid] = {
            "tid": tid,
            "first": int(track_rows[0]["frame"]),
            "last": int(track_rows[-1]["frame"]),
            "first_box": _box(track_rows[0]),
            "last_box": _box(track_rows[-1]),
            "side": _stable_side(track_rows, side_min_frac),
            "n": len(track_rows),
            "rows": track_rows,
        }
    return out


def _natural_tid(tid):
    try:
        return (0, int(tid))
    except ValueError:
        return (1, tid)


def fuse_tracks(rows, max_gap=3, iou_min=0.20, max_center_norm=1.0,
                ambiguity_margin=0.10, side_min_frac=0.80):
    """Return ``(raw_to_canonical, accepted_links, audit_candidates)``."""
    tracks = summarize_tracks(rows, side_min_frac)
    tids = sorted(tracks, key=_natural_tid)
    candidates = []
    for source_tid in tids:
        source = tracks[source_tid]
        for target_tid in tids:
            if source_tid == target_tid:
                continue
            target = tracks[target_tid]
            delta = target["first"] - source["last"]
            if not 1 <= delta <= max_gap:
                continue
            endpoint = box_iou(source["last_box"], target["first_box"])
            forward = box_iou(
                _predict(source["rows"], target["first"], True),
                target["first_box"])
            backward = box_iou(
                source["last_box"],
                _predict(target["rows"], source["last"], False))
            score = max(endpoint, forward, backward)
            ac, bc = _centre(source["last_box"]), _centre(target["first_box"])
            scale = max(_diag(source["last_box"]),
                        _diag(target["first_box"]), 1.0)
            center_norm = math.hypot(ac[0] - bc[0], ac[1] - bc[1]) / scale
            side_conflict = (source["side"] and target["side"]
                             and source["side"] != target["side"])
            eligible = (score >= iou_min and center_norm <= max_center_norm
                        and not side_conflict)
            reason = "eligible"
            if score < iou_min or center_norm > max_center_norm:
                reason = "geometry_low"
            elif side_conflict:
                reason = "strong_side_conflict"
            candidates.append({
                "source_tid": source_tid,
                "target_tid": target_tid,
                "frame_delta": delta,
                "endpoint_iou": endpoint,
                "forward_iou": forward,
                "backward_iou": backward,
                "score": score,
                "center_norm": center_norm,
                "source_side": source["side"],
                "target_side": target["side"],
                "eligible": eligible,
                "status": reason,
            })

    eligible = [row for row in candidates if row["eligible"]]
    by_source = collections.defaultdict(list)
    by_target = collections.defaultdict(list)
    for row in eligible:
        by_source[row["source_tid"]].append(row)
        by_target[row["target_tid"]].append(row)

    def unambiguous(row, group):
        ranked = sorted(group, key=lambda item: (
            -item["score"], _natural_tid(item["source_tid"]),
            _natural_tid(item["target_tid"])))
        if ranked[0] is not row:
            return False
        return len(ranked) == 1 or ranked[0]["score"] - ranked[1]["score"] \
            >= ambiguity_margin

    accepted = []
    for row in sorted(eligible, key=lambda item: (
            -item["score"], _natural_tid(item["source_tid"]),
            _natural_tid(item["target_tid"]))):
        if (unambiguous(row, by_source[row["source_tid"]])
                and unambiguous(row, by_target[row["target_tid"]])):
            row["status"] = "accepted"
            accepted.append(row)
        else:
            row["status"] = "ambiguous"

    parent = {tid: tid for tid in tids}

    def find(tid):
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in accepted:
        union(row["source_tid"], row["target_tid"])
    components = collections.defaultdict(list)
    for tid in tids:
        components[find(tid)].append(tid)
    mapping = {}
    for members in components.values():
        canonical = min(members, key=lambda tid: (
            tracks[tid]["first"], _natural_tid(tid)))
        for tid in members:
            mapping[tid] = canonical
    return mapping, accepted, candidates


def load_track_map(path):
    if not path or not os.path.exists(path):
        return {}
    return {str(row["raw_tid"]): str(row["canonical_tid"])
            for row in csv.DictReader(open(path, encoding="utf-8"))}


def write_fusion(rec, rows, out_dir, **kwargs):
    mapping, links, candidates = fuse_tracks(rows, **kwargs)
    tracks = summarize_tracks(rows, kwargs.get("side_min_frac", 0.80))
    groups = collections.Counter(mapping.values())
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, f"{rec}.track_map.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as fh:
        fields = ["rec", "raw_tid", "canonical_tid", "group_size",
                  "first_frame", "last_frame", "n"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for tid in sorted(mapping, key=_natural_tid):
            writer.writerow({
                "rec": rec, "raw_tid": tid,
                "canonical_tid": mapping[tid],
                "group_size": groups[mapping[tid]],
                "first_frame": tracks[tid]["first"],
                "last_frame": tracks[tid]["last"], "n": tracks[tid]["n"],
            })
    audit_path = os.path.join(out_dir, f"{rec}.track_links.csv")
    fields = ["rec", "source_tid", "target_tid", "frame_delta",
              "endpoint_iou", "forward_iou", "backward_iou", "score",
              "center_norm", "source_side", "target_side", "status"]
    with open(audit_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            out = {key: row[key] for key in fields if key != "rec"}
            out["rec"] = rec
            for key in ("endpoint_iou", "forward_iou", "backward_iou",
                        "score", "center_norm"):
                out[key] = round(float(out[key]), 6)
            writer.writerow(out)
    return mapping, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        help="directory containing <rec>.csv tracker output")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_gap", type=int, default=3)
    parser.add_argument("--iou_min", type=float, default=0.20)
    parser.add_argument("--max_center_norm", type=float, default=1.0)
    parser.add_argument("--ambiguity_margin", type=float, default=0.10)
    parser.add_argument("--side_min_frac", type=float, default=0.80)
    args = parser.parse_args()
    recs = []
    for line in open(args.jobs, encoding="utf-8"):
        fields = line.strip().split("|")
        if len(fields) >= 4 and not line.startswith("#"):
            recs.append(fields[0])
    total_raw = total_canonical = total_links = 0
    for rec in recs:
        path = os.path.join(args.arm, rec + ".csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        mapping, links = write_fusion(
            rec, rows, args.out, max_gap=args.max_gap,
            iou_min=args.iou_min, max_center_norm=args.max_center_norm,
            ambiguity_margin=args.ambiguity_margin,
            side_min_frac=args.side_min_frac)
        total_raw += len(mapping)
        total_canonical += len(set(mapping.values()))
        total_links += len(links)
        print(f"  {rec}: {len(mapping)} raw -> "
              f"{len(set(mapping.values()))} canonical, {len(links)} links")
    print(f"合计: {total_raw} raw -> {total_canonical} canonical; "
          f"accepted links {total_links}")


if __name__ == "__main__":
    main()
