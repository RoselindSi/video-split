#!/usr/bin/env python3
"""Render crop-local Qwen hand boxes as review contact sheets."""
from __future__ import annotations

import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


def render(input_path, output_dir, columns=4, rows_per_page=3, tile=256):
    records = [json.loads(line) for line in open(input_path, encoding="utf-8")
               if line.strip()]
    page_size = columns * rows_per_page
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()
    outputs = []
    for page_index in range(math.ceil(len(records) / page_size)):
        page_rows = records[page_index * page_size:(page_index + 1) * page_size]
        canvas = Image.new("RGB", (columns * tile, rows_per_page * (tile + 38)),
                           "#151918")
        for index, record in enumerate(page_rows):
            x = (index % columns) * tile
            y = (index // columns) * (tile + 38)
            image = Image.open(record["crop_path"]).convert("RGB").resize(
                (tile, tile))
            draw = ImageDraw.Draw(image)
            bbox = record.get("teacher_bbox")
            if record.get("teacher_hand") and bbox:
                scaled = [round(value * tile / 1000) for value in bbox]
                draw.rectangle(scaled, outline="#35e08b", width=4)
            canvas.paste(image, (x, y))
            label = record.get("human_label") or "unlabeled"
            prediction = "hand" if record.get("teacher_hand") else "nothand"
            text = (f"{record['rec']} t{record['canonical_tid']} "
                    f"f{record['frame']}\ntruth={label} pred={prediction}")
            ImageDraw.Draw(canvas).multiline_text(
                (x + 4, y + tile + 3), text, fill="white", font=font, spacing=2)
        path = os.path.join(output_dir, f"page_{page_index + 1:02d}.jpg")
        canvas.save(path, quality=92)
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for path in render(args.input, args.out):
        print(path)


if __name__ == "__main__":
    main()
