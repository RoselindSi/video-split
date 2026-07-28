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


def export_merged(xlsx, dev_event_ids, out_gold, out_context):
    """Export the MERGED 188-event workbook, stamping every row with an
    explicit `split` field: 'dev_original72' (the events the HAL scorer and
    the 0.85 threshold were developed on) vs 'test_batch2' (the held-out
    events, from disjoint recordings, that must ONLY be used for final
    evaluation).

    The split is derived by matching event_id against the already-committed
    72-event dev gold, NOT by row order or sheet membership, so it stays
    correct even if the workbook is re-sorted. Carrying it in the JSONL (not
    just as separate sheets) is what stops a later script from silently
    pooling the two and re-tuning on the test half.

    Context (the original segment labels) comes from the per-batch audit
    sheets, which carry the label columns the Gold_*_Normalized sheet drops.
    """
    with zipfile.ZipFile(xlsx) as z:
        rows = _read_sheet(z, "Gold_Combined_Normalized")
        ctx_rows = []
        for sheet in ("Original_Audit_72", "Batch2_Audit"):
            ctx_rows.append(_read_sheet(z, sheet))
    hdr = rows[0]
    recs = []
    for r in rows[1:]:
        d = {k: _conv(k, r[i] if i < len(r) else None) for i, k in enumerate(hdr)}
        d["split"] = "dev_original72" if d.get("event_id") in dev_event_ids else "test_batch2"
        recs.append(d)
    with open(out_gold, "w", encoding="utf-8") as f:
        for d in recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    ctx = {}
    for sheet_rows in ctx_rows:
        chdr = sheet_rows[0]
        for r in sheet_rows[1:]:
            row = {k: (r[i] if i < len(r) else None) for i, k in enumerate(chdr)}
            eid = (row.get("event_id") or "").strip()
            if not eid:
                continue
            def g(k):
                v = (row.get(k) or "").strip() if isinstance(row.get(k), str) else row.get(k)
                return v or None
            def fl(k):
                v = g(k)
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None
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
    with open(out_context, "w", encoding="utf-8") as f:
        for d in ctx.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return recs, ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--audit_csv", help="required for the original 72-event export")
    ap.add_argument("--merged", action="store_true",
                    help="export the 188-event merged workbook (Gold_Combined_Normalized + "
                         "both audit sheets) with a split field, instead of the original 72")
    ap.add_argument("--dev_gold", default="data/gold/audit_72_gold_v2.jsonl",
                    help="--merged only: defines which event_ids count as dev_original72")
    ap.add_argument("--out_dir", default="data/gold")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.merged:
        with open(a.dev_gold, encoding="utf-8") as f:
            dev_ids = {json.loads(l)["event_id"] for l in f if l.strip()}
        gold, ctx = export_merged(
            a.xlsx, dev_ids,
            os.path.join(a.out_dir, "audit_188_gold_v2.jsonl"),
            os.path.join(a.out_dir, "audit_188_context.jsonl"))
        from collections import Counter
        by_split = Counter(g["split"] for g in gold)
        print(f"merged gold rows={len(gold)}  by split={dict(by_split)}")
        for split in ("dev_original72", "test_batch2"):
            sub = [g for g in gold if g["split"] == split]
            tt = Counter(g.get("temporal_truth") for g in sub)
            recs = {g.get("recording_id") for g in sub}
            print(f"  {split}: n={len(sub)} recordings={len(recs)} temporal_truth={dict(tt)}")
        dev_recs = {g["recording_id"] for g in gold if g["split"] == "dev_original72"}
        test_recs = {g["recording_id"] for g in gold if g["split"] == "test_batch2"}
        overlap = dev_recs & test_recs
        print(f"  RECORDING OVERLAP between splits: {sorted(overlap) if overlap else 'NONE (clean held-out)'}")
        miss = [g["event_id"] for g in gold if g["event_id"] not in ctx]
        print(f"  context rows={len(ctx)}  missing_context={len(miss)}")
        return

    if not a.audit_csv:
        ap.error("--audit_csv is required unless --merged is given")
    gold = export_gold(a.xlsx, os.path.join(a.out_dir, "audit_72_gold_v2.jsonl"))
    ctx = export_context(a.audit_csv, os.path.join(a.out_dir, "audit_72_context.jsonl"))
    miss = [g["event_id"] for g in gold if g["event_id"] not in ctx]
    print(f"gold rows={len(gold)}  context rows={len(ctx)}  missing_context={miss}")


if __name__ == "__main__":
    main()
