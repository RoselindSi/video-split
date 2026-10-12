"""Put the採集 negatives' boxes back into the manifest, so the box channel has something to say.

WHY THE BOX CHANNEL COULD NOT BE TESTED. `spatial_student` documents that the
full-frame branch is useless without a box -- with nothing saying which part of
the frame the crop came from, all it can learn is "is there a hand somewhere",
which is nearly always yes, and turning the full view on pushed head false
positives from 0.772 to 0.897 instead of down. The fourth channel exists to fix
exactly that. But on the current manifest only 24.6% of train rows carry a box,
and only 17% of the negatives do, because the rows that grew the manifest from
14k to 41k came from harvesting and arrived without one. Training with the
channel on that manifest would test "a box on a quarter of the data", and would
additionally teach the model that an all-zero mask is itself a feature.

THE BOXES WERE NEVER LOST, just not joined. Every harvest run wrote an
`index.csv` next to its crops with `stem,x0,y0,x1,y1`. Joining on `stem` takes
train coverage from 24.6% to 99.5% and the negatives from 17% to 100%.

THE NORMALISATION IS THE PART THAT CAN SILENTLY BE WRONG. The index stores
pixels in the original 1920x1520 frame; `box_mask` wants fractions of the image
it loads. Those agree only when `full_path` is that frame rescaled -- 1280x1013
has the same 1.263 aspect ratio and is, while the older 900x506 rendering at
1.78 is not. So rows are only filled when their full image matches the source
aspect ratio, and the rest are left empty rather than filled with a plausible
wrong number. The fill was checked by drawing the computed box back onto the
full image and confirming it lands on the hand in the crop.

Rows that already have a box are never overwritten: those came from a different
source with its own convention, and quietly replacing them would mix two
conventions in one column.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

BOX = ("box_cx", "box_cy", "box_w", "box_h")
SOURCE_W, SOURCE_H = 1920, 1520
ASPECT_TOLERANCE = 0.02


def load_index(patterns):
    """-> {stem: (x0, y0, x1, y1)}"""
    out = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                stem = (row.get("stem") or "").strip()
                if not stem or not (row.get("x1") or "").strip():
                    continue
                try:
                    x0, y0, x1, y1 = (float(row["x0"]), float(row["y0"]),
                                      float(row["x1"]), float(row["y1"]))
                except (KeyError, ValueError, TypeError):
                    continue
                if x1 > x0 and y1 > y0:
                    out[stem] = (x0, y0, x1, y1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", nargs="+", required=True,
                    help="采集时的 index.csv，可给 glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=200,
                    help="抽查多少行的整图尺寸来判断宽高比是否匹配")
    a = ap.parse_args()

    from PIL import Image

    index = load_index(a.index)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    print("索引里有框的 stem %d；manifest %d 行" % (len(index), len(rows)))

    source_aspect = SOURCE_W / SOURCE_H
    sizes = {}

    def aspect_ok(path):
        if path not in sizes:
            try:
                with Image.open(path) as im:
                    sizes[path] = im.size
            except Exception:
                sizes[path] = None
        size = sizes[path]
        if not size:
            return False
        return abs(size[0] / size[1] - source_aspect) <= ASPECT_TOLERANCE

    filled = skipped_aspect = already = 0
    for row in rows:
        if (row.get("box_cx") or "").strip():
            already += 1
            continue
        box = index.get((row.get("stem") or "").strip())
        if not box:
            continue
        if not aspect_ok(row.get("full_path", "")):
            skipped_aspect += 1
            continue
        x0, y0, x1, y1 = box
        row["box_cx"] = "%.6f" % ((x0 + x1) / 2 / SOURCE_W)
        row["box_cy"] = "%.6f" % ((y0 + y1) / 2 / SOURCE_H)
        row["box_w"] = "%.6f" % ((x1 - x0) / SOURCE_W)
        row["box_h"] = "%.6f" % ((y1 - y0) / SOURCE_H)
        filled += 1

    fields = list(rows[0].keys())
    for name in BOX:
        if name not in fields:
            fields.append(name)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    have = sum(1 for r in rows if (r.get("box_cx") or "").strip())
    print("本来就有 %d；补上 %d；宽高比不符跳过 %d"
          % (already, filled, skipped_aspect))
    print("补后有框 %d/%d = %.1f%% -> %s"
          % (have, len(rows), 100 * have / len(rows), a.out))
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r.get("split") == split]
        if not sub:
            continue
        n = sum(1 for r in sub if (r.get("box_cx") or "").strip())
        neg = [r for r in sub
               if (r.get("teacher_p") or "") and float(r["teacher_p"]) < 0.5]
        nn = sum(1 for r in neg if (r.get("box_cx") or "").strip())
        print("  %-6s %5d/%5d = %5.1f%%；负例 %4d/%4d = %5.1f%%"
              % (split, n, len(sub), 100 * n / len(sub),
                 nn, len(neg), 100 * nn / max(1, len(neg))))


if __name__ == "__main__":
    main()
