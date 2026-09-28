"""Recover the candidate box for every view, from the packages the views came from.

WHY THE STUDENT CANNOT SEE THE BOX TODAY. The dual-view student encodes the
crop and the whole frame with one shared stem and concatenates them. Nothing
tells it which part of that frame the crop came from, so the full-frame branch
can only learn "is there a hand anywhere in this picture", which is nearly
always yes -- and that is why adding the full view pushed head false positives
up rather than down. The teacher never had this problem: Qwen was shown the
frame *and* told which crop to judge.

WHY IT CANNOT SIMPLY BE TURNED ON. `box_cx/cy/w/h` are present on 2,417 of
14,592 rows, and that subset is almost entirely positive -- 12 negatives in
train, 2 in val, 1 in test. A box-aware student trained on only those rows
would have no negatives at all, so the box and the negatives are, as the
manifest stands, mutually exclusive.

THEY DO NOT HAVE TO BE. Every view was cut from a package that recorded where
it cut, in one of three layouts:

    otherpkg / basepkg / trackpkg   `stem` -> box_cx, box_cy, box_w, box_h
    facepkg                         `stem` -> cx_frac, cy_frac, w_frac, h_frac
    crowd10                         (rec, frame, tid) -> x0, y0, x1, y1 in pixels

The first two are already normalized. The third is in pixels and is divided by
the frame size read from the image itself rather than assumed, because the
pipeline has both 1920x1520 raw frames and 1600x900 panorama renders in
circulation and guessing wrong would put every box in the wrong place.

NOTHING IS OVERWRITTEN. A row that already carries a box keeps it, and the
recovered value is written only where the field was empty; the report prints
how far the two agree on the rows that have both, because a join that is
subtly wrong looks exactly like a join that worked.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

BOX = ("box_cx", "box_cy", "box_w", "box_h")


def load_stem_tables(roots):
    """-> {stem: (cx, cy, w, h)}，两种列名都吃。"""
    out = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "*", "hands.csv")):
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                stem = (row.get("stem") or "").strip()
                if not stem:
                    continue
                if (row.get("box_cx") or "").strip():
                    v = [row["box_cx"], row["box_cy"], row["box_w"], row["box_h"]]
                elif (row.get("cx_frac") or "").strip():
                    v = [row["cx_frac"], row["cy_frac"], row["w_frac"], row["h_frac"]]
                else:
                    continue
                try:
                    out[stem] = tuple(float(x) for x in v)
                except ValueError:
                    pass
    return out


def load_crowd_tables(roots, size_of):
    """-> {(rec, frame, tid): (cx, cy, w, h)}，像素框按真实帧尺寸归一。"""
    out = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "*.csv")):
            base = os.path.basename(path)
            if base.endswith(".faces.csv") or base.endswith(".decisions.csv"):
                continue
            rec = base[:-4]
            wh = size_of(rec)
            if not wh:
                continue
            W, H = wh
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                try:
                    x0, y0 = float(row["x0"]), float(row["y0"])
                    x1, y1 = float(row["x1"]), float(row["y1"])
                    key = (rec, str(int(float(row["frame"]))),
                           str(int(float(row["tid"]))))
                except (KeyError, ValueError, TypeError):
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                out.setdefault(key, ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                                     (x1 - x0) / W, (y1 - y0) / H))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg_roots", nargs="+", default=["/workspace"])
    ap.add_argument("--crowd_roots", nargs="+",
                    default=["/workspace/crowd10_base",
                             "/workspace/crowd10_fast_v3_base"])
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    for k in BOX:
        if k not in fields:
            fields.append(k)

    # 帧尺寸从图片本身读，不猜：流水线里同时有 1920x1520 原始帧和 1600x900 全景渲染
    from PIL import Image
    cache = {}

    def size_of(rec):
        if rec in cache:
            return cache[rec]
        wh = None
        for r in rows:
            if r.get("rec") == rec and (r.get("full_path") or "").strip():
                try:
                    with Image.open(r["full_path"]) as im:
                        wh = im.size
                except Exception:
                    wh = None
                break
        cache[rec] = wh
        return wh

    stem_box = load_stem_tables(a.pkg_roots)
    crowd_box = load_crowd_tables(a.crowd_roots, size_of)
    print("按 stem 可查的框 %d 个；按 (rec,frame,tid) 可查的框 %d 个"
          % (len(stem_box), len(crowd_box)))

    filled = collections.Counter()
    agree, checked = 0, 0
    for r in rows:
        had = (r.get("box_cx") or "").strip() != ""
        v = stem_box.get((r.get("stem") or "").strip())
        how = "stem"
        if v is None:
            for tid_col in ("raw_tid", "canonical_tid"):
                t = (r.get(tid_col) or "").strip()
                if not t:
                    continue
                try:
                    key = (r.get("rec", ""), str(int(float(r.get("frame", "")))),
                           str(int(float(t))))
                except ValueError:
                    continue
                v = crowd_box.get(key)
                if v is not None:
                    how = "crowd"
                    break
        if v is None:
            filled["查不到"] += 0 if had else 1
            continue
        if had:
            # 两边都有就比一下：连错的 join 看起来和连对的一模一样
            checked += 1
            try:
                old = tuple(float(r[k]) for k in BOX)
                if max(abs(o - n) for o, n in zip(old, v)) < 0.02:
                    agree += 1
            except ValueError:
                pass
            continue
        for k, x in zip(BOX, v):
            r[k] = "%.6f" % x
        filled[how] += 1

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    now = sum(1 for r in rows if (r.get("box_cx") or "").strip())
    print("补上 %s；查不到 %d" % (dict(filled), filled["查不到"]))
    if checked:
        print("两边都有框的 %d 行里，一致（误差<0.02）的 %d 行 = %.1f%%  ← 低了说明 join 连错了"
              % (checked, agree, 100 * agree / checked))
    print("有框行数：%d -> %d / %d (%.1f%%)"
          % (sum(1 for r in rows if (r.get("box_cx") or "").strip()) - sum(
              v for k, v in filled.items() if k != "查不到"),
             now, len(rows), 100 * now / len(rows)))
    print("\n补全后各 split 的负例：")
    print("%-8s %8s %8s %8s" % ("split", "有框", "其中hand", "其中nothand"))
    for s in ("train", "val", "test"):
        g = [r for r in rows if r["split"] == s and (r.get("box_cx") or "").strip()]
        c = collections.Counter((r.get("human_label") or "(空)") for r in g)
        print("%-8s %8d %8d %8d" % (s, len(g), c["hand"], c["nothand"]))
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
