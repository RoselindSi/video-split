#!/usr/bin/env python3
"""Propagate reviewed singleton nothand boxes to nearby packaged frames."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os

from PIL import Image


FIELDS = ["item_id", "rec", "frame", "raw_tid", "canonical_tid",
          "full_path", "crop_path", "source_frame", "propagated"]


def nearest_frames(frames, center, limit=5, radius=12):
    eligible = sorted({frame for frame in frames if abs(frame - center) <= radius},
                      key=lambda frame: (abs(frame - center), frame))
    chosen = sorted(eligible[:limit])
    return chosen


def frame_of(path):
    name = os.path.basename(path)
    return int(name.split("_f", 1)[1].split("_", 1)[0])


def crop_box(image, source, padding=.25):
    cx, cy = float(source["box_cx"]), float(source["box_cy"])
    width, height = float(source["box_w"]), float(source["box_h"])
    width *= 1.0 + 2.0 * padding
    height *= 1.0 + 2.0 * padding
    x1 = max(0.0, cx - width / 2)
    y1 = max(0.0, cy - height / 2)
    x2 = min(1.0, cx + width / 2)
    y2 = min(1.0, cy + height / 2)
    return image.crop((round(x1 * image.width), round(y1 * image.height),
                       round(x2 * image.width), round(y2 * image.height)))


def expand(manifest, output, max_views=5, radius=12, padding=.25):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    targets = [row for row in rows
               if row.get("review_status") == "reviewed_spatial_track_v1"
               and row.get("human_label") == "nothand"]
    os.makedirs(os.path.join(output, "full"), exist_ok=True)
    os.makedirs(os.path.join(output, "crops"), exist_ok=True)
    expanded = []
    for row in targets:
        package = os.path.dirname(os.path.dirname(row["full_path"]))
        hand_rows = {source["stem"]: source for source in csv.DictReader(
            open(os.path.join(package, "hands.csv"), encoding="utf-8-sig"))}
        source = hand_rows[row["stem"]]
        paths_by_frame = {}
        for path in sorted(glob.glob(os.path.join(package, "context", "*.jpg"))):
            paths_by_frame.setdefault(frame_of(path), path)
        center = int(row["frame"])
        for frame in nearest_frames(paths_by_frame, center, max_views, radius):
            source_path = paths_by_frame[frame]
            digest = hashlib.sha256(
                f"{row['rec']}|{row['canonical_tid']}|{frame}|{center}".encode()
            ).hexdigest()[:24]
            base = f"{row['rec']}__t{row['canonical_tid']}__f{frame:06d}"
            full_path = os.path.join(output, "full", base + ".jpg")
            crop_path = os.path.join(output, "crops", base + ".jpg")
            image = Image.open(source_path).convert("RGB")
            image.save(full_path, quality=92)
            crop_box(image, source, padding).save(crop_path, quality=95)
            expanded.append({
                "item_id": digest, "rec": row["rec"], "frame": frame,
                "raw_tid": row["raw_tid"],
                "canonical_tid": row["canonical_tid"],
                "full_path": full_path, "crop_path": crop_path,
                "source_frame": center,
                "propagated": "0" if frame == center else "1",
            })
    index = os.path.join(output, "candidates.csv")
    with open(index, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(expanded)
    return {"tracks": len(targets), "views": len(expanded), "index": index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--padding", type=float, default=.25)
    args = parser.parse_args()
    report = expand(args.manifest, args.out, args.max_views, args.radius,
                    args.padding)
    print(" ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()
