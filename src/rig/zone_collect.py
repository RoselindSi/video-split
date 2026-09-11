"""Fold a directory of zone annotations into one table, whatever their names.

ONE SHARED CSV IS THE WRONG SHAPE FOR GIT. Four annotators appending lines to
the same tracked file conflict on every push, and the conflict lands in the
middle of a row of coordinates where nobody can resolve it correctly. So each
person's work stays a separate file -- git merges disjoint files without being
asked -- and the single table everyone wants is GENERATED from them, never
edited. Regenerating is free; reconstructing a mangled row is not.

FILENAMES ARE NOT IDENTITY. Some files arrive from the collector already named
by recording and annotator, some arrive out of a browser's download folder as
`wearer_zone_R0821_104146 (2).json`, and some have been renamed by whoever
emailed them. Every one of them carries the recording and the annotator INSIDE,
so that is what is read; the filename is only ever reported, never parsed.

A FINAL BEATS A DRAFT FOR THE SAME PAIR, and a later one beats an earlier one.
The collector writes drafts continuously while someone works, so the same
person on the same recording legitimately appears several times; taking all of
them would multiply-count one afternoon's work.

TWO TABLES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. `index` is one row per
person per recording -- who did what, how much of it they drew rather than
copied -- and is what you read to see whether the batch is done. `keyframes`
is one row per curve and carries the geometry, which is what an analysis
consumes. A single table would make the first question unreadable.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

EYE_ORDER = ("cam3", "cam4")


def load(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return None, f"读不了 ({e})"
    if not isinstance(d, dict) or "eyes" not in d:
        return None, "不是 zone 标注文件（没有 eyes）"
    if not d.get("recording"):
        return None, "没有 recording 字段"
    return d, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="where the json files are (searched recursively)")
    ap.add_argument("--index", default="zone_index.csv")
    ap.add_argument("--keyframes", default="zone_keyframes.csv")
    ap.add_argument("--drafts", action="store_true",
                    help="keep a draft even when that pair also has a final")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "**", "*.json"),
                             recursive=True))
    best, skipped = {}, []
    for p in files:
        d, why = load(p)
        if d is None:
            skipped.append((p, why))
            continue
        rec = str(d.get("recording"))
        who = str(d.get("annotator") or "").strip() or "anon"
        key = (rec, who)
        # final > draft, then later > earlier; `submitted_at` is only present
        # on files that came through the collector, so a hand-delivered
        # download sorts as the empty string and loses to a submitted one.
        rank = (d.get("status") == "final", str(d.get("submitted_at") or ""),
                os.path.getmtime(p))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, d, p)

    idx_rows, kf_rows = [], []
    for (rec, who), (_, d, p) in sorted(best.items()):
        counts, frames = collections.Counter(), []
        for eye in (list(d["eyes"]) if not set(EYE_ORDER) <= set(d["eyes"])
                    else EYE_ORDER):
            for k in d["eyes"][eye].get("keyframes", []):
                counts[(eye, k.get("src", "?"))] += 1
                frames.append(k["frame"])
                sp = k.get("own_side_point") or [None, None]
                kf_rows.append({
                    "recording": rec, "annotator": who, "eye": eye,
                    "frame": k["frame"], "t_s": k.get("t"),
                    "src": k.get("src"),
                    "n_points": len(k.get("curve") or []),
                    "own_side_x": sp[0], "own_side_y": sp[1],
                    "eye_w": d["eyes"][eye].get("size", [None, None])[0],
                    "eye_h": d["eyes"][eye].get("size", [None, None])[1],
                    "curve_json": json.dumps(k.get("curve"),
                                             separators=(",", ":")),
                    "own_polygon_json": json.dumps(k.get("own_polygon"),
                                                   separators=(",", ":")),
                })
        row = {"recording": rec, "annotator": who,
               "status": d.get("status", "downloaded"),
               "submitted_at": d.get("submitted_at", ""),
               "clip_start_frame": d.get("clip_start_frame"),
               "fps": d.get("fps"), "copy_shift_px": d.get("copy_shift_px"),
               "problem": d.get("problem", ""),
               "first_frame": min(frames) if frames else "",
               "last_frame": max(frames) if frames else "",
               "file": os.path.relpath(p, a.dir)}
        for eye in d["eyes"]:
            for src in ("drawn", "held", "copied"):
                row[f"{eye}_{src}"] = counts[(eye, src)]
            row[f"{eye}_total"] = sum(n for (e, _), n in counts.items()
                                      if e == eye)
        idx_rows.append(row)

    def write(path, rows):
        if not rows:
            print(f"  （没有行，没写 {path}）")
            return
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  {len(rows):5d} 行 -> {path}")

    write(a.index, idx_rows)
    write(a.keyframes, kf_rows)

    print(f"\n看了 {len(files)} 个文件，"
          f"{len(best)} 个（录像×标注员）组合")
    for p, why in skipped:
        print(f"  跳过 {os.path.relpath(p, a.dir)}: {why}")
    # WHO OVERLAPS WITH WHOM is the reason to collect centrally at all: two
    # people on one recording is the only thing that can tell you whether
    # `the wearer's side` means the same to both of them.
    bad = [(rec, who, d.get("problem", ""))
           for (rec, who), (_, d, _p) in sorted(best.items())
           if d.get("status") == "unusable"]
    if bad:
        print(f"\n{len(bad)} 段被报告有问题，别拿去分析:")
        for rec, who, why in bad:
            print(f"  {rec}  ({who}) {why}")
    per_rec = collections.defaultdict(set)
    for rec, who in best:
        per_rec[rec].add(who)
    dbl = {r: sorted(w) for r, w in per_rec.items() if len(w) > 1}
    print(f"{len(per_rec)} 段录像，{len({w for _, w in best})} 个标注员；"
          f"有两人以上标过的: {len(dbl)}")
    for r, w in sorted(dbl.items()):
        print(f"  {r}: {', '.join(w)}")


if __name__ == "__main__":
    main()
