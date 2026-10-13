#!/usr/bin/env python3
"""Render accepted canonical-track links as endpoint frame pairs."""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np


def read_jobs(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        fields = line.rstrip("\n").split("|")
        if len(fields) >= 4 and not line.startswith("#"):
            out[fields[0]] = int(fields[2])
    return out


def draw_endpoint(frame, row, label, colour):
    x0, y0, x1, y1 = [int(float(row[key]))
                       for key in ("x0", "y0", "x1", "y1")]
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 8)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (20, 20, 20), -1)
    cv2.putText(frame, label, (24, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def video_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {path}")
    return frame


def resize_width(image, width):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, round(image.shape[0] * scale)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel_width", type=int, default=650)
    parser.add_argument("--contact_columns", type=int, default=4)
    args = parser.parse_args()
    starts = read_jobs(args.jobs)
    os.makedirs(args.out, exist_ok=True)
    pairs = []
    for link_path in sorted(glob.glob(os.path.join(
            args.maps, "*.track_links.csv"))):
        rec = os.path.basename(link_path).split(".track_links.csv")[0]
        rows = list(csv.DictReader(open(
            os.path.join(args.base, rec + ".csv"), encoding="utf-8")))
        by_tid = {}
        for row in rows:
            by_tid.setdefault(str(row["tid"]), []).append(row)
        for values in by_tid.values():
            values.sort(key=lambda row: int(row["frame"]))
        for link in csv.DictReader(open(link_path, encoding="utf-8")):
            if link["status"] != "accepted":
                continue
            source_tid, target_tid = link["source_tid"], link["target_tid"]
            source_row = by_tid[source_tid][-1]
            target_row = by_tid[target_tid][0]
            source_frame, target_frame = (int(source_row["frame"]),
                                          int(target_row["frame"]))
            video = os.path.join(args.base, rec + ".mp4")
            left = video_frame(video, source_frame - starts[rec])
            right = video_frame(video, target_frame - starts[rec])
            left = draw_endpoint(
                left, source_row,
                f"{rec} raw tid {source_tid} LAST frame {source_frame}",
                (0, 255, 255))
            right = draw_endpoint(
                right, target_row,
                f"raw tid {target_tid} FIRST frame {target_frame}  "
                f"score {float(link['score']):.2f}", (255, 255, 0))
            pair = np.hstack((resize_width(left, args.panel_width),
                              resize_width(right, args.panel_width)))
            name = f"{rec}__{source_tid}_to_{target_tid}.jpg"
            cv2.imwrite(os.path.join(args.out, name), pair,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            pairs.append(resize_width(pair, 1000))
    if pairs:
        width = max(image.shape[1] for image in pairs)
        padded = []
        for image in pairs:
            if image.shape[1] < width:
                image = cv2.copyMakeBorder(
                    image, 0, 0, 0, width - image.shape[1],
                    cv2.BORDER_CONSTANT, value=(15, 15, 15))
            padded.append(image)
        blank = np.zeros_like(padded[0])
        contact_rows = []
        for start in range(0, len(padded), args.contact_columns):
            row = padded[start:start + args.contact_columns]
            row.extend([blank] * (args.contact_columns - len(row)))
            contact_rows.append(np.hstack(row))
        cv2.imwrite(os.path.join(args.out, "all_accepted_links.jpg"),
                    np.vstack(contact_rows),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"rendered {len(pairs)} accepted links -> {args.out}")


if __name__ == "__main__":
    main()
