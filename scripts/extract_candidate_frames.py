#!/usr/bin/env python3
"""Extract raw cam3 midpoint frames from a ranked segment CSV."""
from __future__ import annotations

import argparse
import csv
import os

import cv2

from src.rig.seam_fix import RawCameraReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_csv")
    parser.add_argument("out_dir")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.rank_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:args.limit]

    for row in rows:
        bag = row["bag"]
        start = int(row["start"])
        length = int(row["n"])
        frame = start + max(0, length // 2)
        videos = {name: os.path.join(bag, f"{name}.mp4")
                  for name in ("cam12", "cam34", "cam56")}
        try:
            reader = RawCameraReader(videos, "cam3", frame)
            image = reader.next()
            reader.close()
        except SystemExit as error:
            print(f"MISS {row['rec']} frame={frame}: {error}")
            continue
        if image is None:
            print(f"MISS {row['rec']} frame={frame}")
            continue
        path = os.path.join(args.out_dir, f"{row['rec']}_f{frame:06d}.jpg")
        cv2.imwrite(path, image)
        print(path)


if __name__ == "__main__":
    main()
