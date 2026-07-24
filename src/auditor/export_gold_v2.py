"""Regenerate the committed gold JSONL + context JSONL from the source
spreadsheet, so `data/gold/*.jsonl` is reproducible rather than hand-edited.

The visual auditor is scored against `Gold_72_Normalized` (the orthogonal
temporal/semantic/policy labels) in audit_72_gold_v2_machine_readable.xlsx.
The per-event annotation *context* (the original segment labels the auditor
verifies) comes from the raw audit CSV. This script writes both:

    data/gold/audit_72_gold_v2.jsonl   (frozen gold labels)
    data/gold/audit_72_context.jsonl   (original labels + gt/pred/score)

It uses only the stdlib (unzips the xlsx and reads the XML directly), so it
runs anywhere without openpyxl/pandas.

Usage:
    python -m src.auditor.export_gold_v2 \
        --xlsx ~/Downloads/audit_72_gold_v2_machine_readable.xlsx \
        --audit_csv ~/Documents/audit_sample.csv \
        --out_dir data/gold
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_BOOL = {"no_valid_boundary", "boundary_time_unresolved", "corrected_target_known", "auto_proposal_eligible"}
_FLOAT = {"gt_time", "pred_time", "pred_score", "primary_corrected_boundary_time",
          "boundary_interval_start", "boundary_interval_end"}
_JSONF = {"corrected_boundary_times_json", "corrected_boundary_intervals_json"}


def _colnum(ref):
    letters = "".join(c for c in ref if c.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _shared_strings(z):
    ss = []
    for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
        ss.append("".join(t.text or "" for t in si.iter(NS + "t")))
    return ss


def _sheet_target(z, name):
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    RN = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    rid2t = {r.get("Id"): r.get("Target").lstrip("/") for r in rels.findall(RN + "Relationship")}
    for s in wb.find(NS + "sheets"):
        if s.get("name") == name:
            rid = s.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            t = rid2t[rid]
            return t if t.startswith("xl/") else "xl/" + t
    raise KeyError(name)


def _read_sheet(z, name):
    ss = _shared_strings(z)
    root = ET.fromstring(z.read(_sheet_target(z, name)))
    rows = []
    for row in root.find(NS + "sheetData").findall(NS + "row"):
        cells, maxc = {}, 0
        for c in row.findall(NS + "c"):
            ci = _colnum(c.get("r"))
            maxc = max(maxc, ci)
            t, v, isv = c.get("t"), c.find(NS + "v"), c.find(NS + "is")
            if t == "s" and v is not None:
                val = ss[int(v.text)]
            elif t == "inlineStr" and isv is not None:
                val = "".join(x.text or "" for x in isv.iter(NS + "t"))
            elif v is not None:
                val = v.text
            else:
                val = None
            cells[ci] = val
        rows.append([cells.get(i) for i in range(maxc + 1)])
    return rows


def _conv(k, v):
    if v is None or v == "":
        return None
    if k in _BOOL:
        return bool(int(float(v)))
    if k in _FLOAT:
        try:
            return float(v)
        except ValueError:
            return v
    if k in _JSONF:
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def export_gold(xlsx, out_path):
    with zipfile.ZipFile(xlsx) as z:
        rows = _read_sheet(z, "Gold_72_Normalized")
    hdr = rows[0]
    recs = [{k: _conv(k, r[i] if i < len(r) else None) for i, k in enumerate(hdr)} for r in rows[1:]]
    with open(out_path, "w", encoding="utf-8") as f:
        for d in recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return recs


def export_context(audit_csv, out_path):
    ctx = {}
    with open(audit_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            eid = (r.get("event_id") or "").strip()
            if not eid:
                continue
            def g(k):
                v = (r.get(k) or "").strip()
                return v or None
            def fl(k):
                return float(r[k]) if g(k) else None
            ctx[eid] = {
                "event_id": eid, "recording_id": g("recording_id"),
                "source_category": g("category"),
                "gt_time": fl("gt_time"), "pred_time": fl("Matched pred_time"),
                "pred_score": fl("pred_score"),
                "prev_segment_label": g("prev_segment_label"),
                "next_segment_label": g("next_segment_label"),
                "containing_segment_label": g("containing_segment_label"),
                "nearest_previous_segment_label": g("nearest_previous_segment_label"),
                "nearest_next_segment_label": g("nearest_next_segment_label"),
            }
    with open(out_path, "w", encoding="utf-8") as f:
        for d in ctx.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--out_dir", default="data/gold")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    gold = export_gold(a.xlsx, os.path.join(a.out_dir, "audit_72_gold_v2.jsonl"))
    ctx = export_context(a.audit_csv, os.path.join(a.out_dir, "audit_72_context.jsonl"))
    miss = [g["event_id"] for g in gold if g["event_id"] not in ctx]
    print(f"gold rows={len(gold)}  context rows={len(ctx)}  missing_context={miss}")


if __name__ == "__main__":
    main()
