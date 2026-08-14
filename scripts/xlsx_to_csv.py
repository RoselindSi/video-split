"""Read a .xlsx without openpyxl. A workbook is a zip of XML.

The annotator returns sheets as .xlsx and this environment has no spreadsheet
library, so rather than add a dependency for a format that is a zip containing
sharedStrings.xml and one XML file per sheet, it is parsed directly.

WHAT IT HANDLES AND WHAT IT DOES NOT. Shared strings, inline strings, numbers,
booleans, and gaps -- a row that skips column C emits an empty field there
rather than shifting every later value one to the left, which is the failure
mode that makes a naive parse produce a plausible-looking wrong table. Dates
come out as the underlying serial number, formulas as their cached value, and
neither is converted; if a sheet needs either, that is a reason to say so
rather than to guess a format.

Usage:
    python scripts/xlsx_to_csv.py in.xlsx --out out.csv
    python scripts/xlsx_to_csv.py in.xlsx --sheet 2 --head 5
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CELL = re.compile(r"^([A-Z]+)(\d+)$")


def col_index(ref):
    """`A` -> 0, `AA` -> 26. Cells carry their column, and using it is what
    keeps a row with gaps aligned."""
    m = CELL.match(ref or "")
    if not m:
        return None
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def shared_strings(z):
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall(f"{NS}si"):
        out.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    return out


def read_sheet(path, sheet=1):
    z = zipfile.ZipFile(path)
    ss = shared_strings(z)
    name = f"xl/worksheets/sheet{sheet}.xml"
    if name not in z.namelist():
        raise SystemExit(f"{name} not in the workbook; it has "
                         f"{[n for n in z.namelist() if 'worksheets' in n]}")
    root = ET.fromstring(z.read(name))
    rows = []
    for r in root.iter(f"{NS}row"):
        cells = {}
        for c in r.findall(f"{NS}c"):
            i = col_index(c.get("r"))
            if i is None:
                continue
            t = c.get("t")
            if t == "s":
                v = c.find(f"{NS}v")
                cells[i] = ss[int(v.text)] if v is not None else ""
            elif t == "inlineStr":
                cells[i] = "".join(x.text or ""
                                   for x in c.iter(f"{NS}t"))
            else:
                v = c.find(f"{NS}v")
                cells[i] = v.text if v is not None else ""
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])
    w = max((len(r) for r in rows), default=0)
    return [r + [""] * (w - len(r)) for r in rows]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx")
    ap.add_argument("--sheet", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()

    rows = read_sheet(a.xlsx, a.sheet)
    print(f"{len(rows)} rows x {len(rows[0]) if rows else 0} columns")
    if rows:
        print(f"header: {rows[0]}")
    for r in rows[1:1 + a.head]:
        print("  ", r)
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
