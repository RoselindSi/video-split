"""Build the sheet that decides whether a forced fragment join is the same hand.

WHAT IS AND IS NOT ESTABLISHED. On 52 clean recordings, 31.8% of OWNER
fragment ends have exactly one continuation inside 30 frames and 3 box widths,
so for those there is no question of *which* fragment follows. What geometry
cannot answer is whether that fragment is the same hand at all -- the other
hand can move into the vacated spot, and so can somebody else's. That is the
only question on this sheet.

LEFT/RIGHT IS NOT ASKED, AND NOT USED TO PRE-FILTER. 21.6% of these joins have
disagreeing side majorities, which first looked like a fifth of them being
wrong. But inside a single fragment, where identity is not in doubt, 4.4% of
frames already disagree with that fragment's own majority and 55.6% of
fragments flip at least once; the implied majority-level reliability is about
0.88, which accounts for the 21.6% on its own. Gating on a label that is 12%
unreliable would discard good joins to avoid a problem it cannot see, so the
sheet carries all 102 and the join recomputes side afterwards from the merged
track -- which is strictly more evidence than either fragment had.

THE STRIP IS THE LAST FRAMES BEFORE AND THE FIRST FRAMES AFTER, in order, with
the gap marked. A reviewer needs to see the hand leave and arrive; a pair of
still crops cannot show that a hand went out of frame right and came back from
the left, which is the case most likely to be a different hand.

ONE RECORDING IS NOT ONE SAMPLE HERE but it is close: the joins sit in 23
recordings with at most 3 each, so a single bad scene cannot carry the result.
The sheet records `rec` so the agreement can be clustered by it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

PAD = 4                  # 接缝两侧各取几帧
THUMB = 220


def read_jobs(path):
    jobs = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 4:
            jobs[parts[0]] = parts[1]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, help="frag_audit 的 json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=int, default=PAD)
    a = ap.parse_args()

    import cv2
    from src.rig.seam_fix import RawCameraReader

    audit = json.load(open(a.audit, encoding="utf-8"))
    forced = [e for e in audit["ends"] if e["n_cands"] == 1]
    jobs = read_jobs(a.jobs)
    os.makedirs(a.out, exist_ok=True)
    print("强制接续 %d 条，分布在 %d 条录像"
          % (len(forced), len({e["rec"] for e in forced})), flush=True)

    by_rec = collections.defaultdict(list)
    for e in forced:
        by_rec[e["rec"]].append(e)

    rows = []
    for rec in sorted(by_rec):
        if rec not in jobs:
            print("  %-22s jobs 里没有，跳过" % rec, flush=True)
            continue
        boxes = collections.defaultdict(dict)
        path = os.path.join(a.arm, "%s.csv" % rec)
        for row in csv.DictReader(open(path, encoding="utf-8")):
            tid = row.get("tid")
            if tid in (None, ""):
                continue
            try:
                boxes[tid][int(row["frame"])] = [
                    int(float(row[c])) for c in ("x0", "y0", "x1", "y1")]
            except (KeyError, TypeError, ValueError):
                continue
        wanted = {}
        for e in by_rec[rec]:
            cand = e["cands"][0]
            a_frames = sorted(boxes[e["tid"]])[-a.pad:]
            b_frames = sorted(boxes[cand["tid"]])[:a.pad]
            for f in a_frames:
                wanted.setdefault(f, []).append((e["tid"], "A", e, f))
            for f in b_frames:
                wanted.setdefault(f, []).append((cand["tid"], "B", e, f))
        if not wanted:
            continue
        bag = jobs[rec]
        videos = {k: os.path.join(bag, "%s.mp4" % k)
                  for k in ("cam12", "cam34", "cam56")}
        order = sorted(wanted)
        reader = RawCameraReader(videos, "cam3", order[0])
        current, image = order[0] - 1, None
        made = collections.defaultdict(list)
        for f in order:
            while current < f:
                image = reader.next()
                current += 1
                if image is None:
                    break
            if image is None:
                break
            for tid, which, e, frame in wanted[f]:
                box = boxes[tid].get(frame)
                if not box:
                    continue
                vis = image.copy()
                x0, y0, x1, y1 = box
                cv2.rectangle(vis, (x0, y0), (x1, y1),
                              (0, 230, 0) if which == "A" else (0, 140, 255), 5)
                side = max(x1 - x0, y1 - y0) * 2.2
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                h, w = image.shape[:2]
                p0, q0 = max(0, cx - int(side / 2)), max(0, cy - int(side / 2))
                p1, q1 = min(w, cx + int(side / 2)), min(h, cy + int(side / 2))
                crop = vis[q0:q1, p0:p1]
                if crop.size == 0:
                    continue
                name = "%s__%s_%s__%s_f%06d.jpg" % (
                    rec, e["tid"], e["cands"][0]["tid"], which, frame)
                cv2.imwrite(os.path.join(a.out, name),
                            cv2.resize(crop, (THUMB, THUMB)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                made[(e["tid"], e["cands"][0]["tid"])].append((which, frame, name))
        reader.close()
        for e in by_rec[rec]:
            cand = e["cands"][0]
            key = (e["tid"], cand["tid"])
            got = sorted(made.get(key, []))
            if not got:
                continue
            rows.append({
                "rec": rec, "tid_a": e["tid"], "tid_b": cand["tid"],
                "frames_a": e["n"], "gap": cand["gap"],
                "dist_rel": cand["dist_rel"], "size_ratio": cand["size_ratio"],
                "same_side": cand["same_side"],
                "n_views": len(got),
                "views": " ".join(n for _w, _f, n in got),
                "判定": "", "备注": "",
            })
        print("  %-22s %d 个接缝" % (rec, len(by_rec[rec])), flush=True)

    sheet = os.path.join(a.out, "join_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                ["rec", "tid_a", "tid_b", "判定"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("\n%d 个接缝 -> %s" % (len(rows), sheet))
    print("判定填：same = 同一只手可以接 / diff = 不是同一只手 / unsure")
    print("绿框 = 接缝前（A），橙框 = 接缝后（B）；按文件名里的帧号排序看")


if __name__ == "__main__":
    main()
