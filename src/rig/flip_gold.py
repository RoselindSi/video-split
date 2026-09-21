"""The moments a chain changed hands, put to a person without the answer shown.

WHAT THIS DECIDES. 22 of 87 judgeable own-hand chains carry a sustained flip
to the other side label. Three things produce that and they are repaired in
three different places:

    the label is wrong           fix handedness, not identity
    the stitching merged them    fix the post-processing
    the tracker swapped them     fix the tracker, and re-examine C1

Nothing in the output distinguishes them and there are only 52 events, so
they go to a person.

THE MODEL'S ANSWER IS NOT ON THE PICTURE. No side letters, no track id, no
confidence: a strip showing `L L L | R R R` under the boxes tells the reader
what to conclude, and what is being tested is whether the hand changed, not
whether the reader agrees with the label. Both are in the CSV and can be
joined after.

WHOLE FRAMES, NOT CROPS. Two hands that cross are the case at issue, and a
crop of one of them removes the only evidence that separates them -- where
the other hand is. Eleven frames either side of the flip, laid out as a grid.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def chains(own, frames, min_iou=0.3):
    """-> [[(frame, box, side, tid)]] built by overlap alone, as measured."""
    live, out = [], []
    for f in frames:
        cur = own.get(f, [])
        used, nxt = set(), []
        for seq in live:
            box = seq[-1][1]
            best, bi = 0.0, None
            for i, (c, sd, tid) in enumerate(cur):
                if i in used:
                    continue
                v = iou(box, c)
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= min_iou:
                used.add(bi)
                nxt.append(seq + [(f, cur[bi][0], cur[bi][1], cur[bi][2])])
            else:
                out.append(seq)
        for i, (c, sd, tid) in enumerate(cur):
            if i not in used:
                nxt.append([(f, c, sd, tid)])
        live = nxt
    return out + live


def flips(seq, min_run):
    """-> [(index of the first frame of the new run, from_side, to_side)]"""
    lab = [(i, s) for i, (_, _, s, _) in enumerate(seq) if s in ("left", "right")]
    if len(lab) < 2 * min_run:
        return []
    runs = []
    for i, s in lab:
        if runs and runs[-1][0] == s:
            runs[-1][2].append(i)
        else:
            runs.append([s, None, [i]])
    long = [r for r in runs if len(r[2]) >= min_run]
    out = []
    for a, b in zip(long, long[1:]):
        if a[0] != b[0]:
            out.append((b[2][0], a[0], b[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/workspace/cam3_side")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--min_run", type=int, default=3)
    ap.add_argument("--span", type=int, default=5, help="frames either side")
    ap.add_argument("--cell_w", type=int, default=620)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--crop_px", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cv2
    import numpy as np
    from src.rig.seam_fix import RawCameraReader

    gone = set()
    for r in csv.DictReader(open("/workspace/visiblepkg/hands.csv")):
        if r["label"] == "other":
            rec, rest = r["stem"].rsplit("_f", 1)
            gone.add((rec, int(rest.split("_h")[0])))
    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    events = []
    for rec in sorted(jobs):
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        s, n = jobs[rec][1], jobs[rec][2]
        own = collections.defaultdict(list)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["own"] != "1":
                continue
            own[int(r["frame"])].append(
                ([float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                 r.get("side") or "", r.get("tid") or ""))
        frames = [f for f in range(s, s + n) if (rec, f) not in gone]
        for seq in chains(own, frames):
            for at, fr, to in flips(seq, a.min_run):
                events.append({"rec": rec, "seq": seq, "at": at,
                               "from": fr, "to": to})
    print("持续翻边事件 %d 个" % len(events))
    if not events:
        return

    for sub in ("crops", "context"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for e in events:
        by_rec[e["rec"]].append(e)
    rows = []
    for rec in sorted(by_rec):
        bag = jobs[rec][0]
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        need = set()
        for e in by_rec[rec]:
            lo = max(0, e["at"] - a.span)
            hi = min(len(e["seq"]) - 1, e["at"] + a.span)
            e["win"] = list(range(lo, hi + 1))
            need |= {e["seq"][i][0] for i in e["win"]}
        need = sorted(need)
        rd = RawCameraReader(vids, "cam3", need[0])
        cur = need[0] - 1
        cache = {}
        for f in need:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            cache[f] = img
        rd.close()
        for j, e in enumerate(by_rec[rec]):
            tiles = []
            for i in e["win"]:
                f, box, side, tid = e["seq"][i]
                img = cache.get(f)
                if img is None:
                    continue
                v = img.copy()
                b = [int(x) for x in box]
                cv2.rectangle(v, (b[0], b[1]), (b[2], b[3]), (0, 230, 0), 5)
                # frame number only. No side letter, no track id, no score:
                # the picture must not carry the answer being asked for.
                cv2.putText(v, "f%d" % f, (24, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, (0, 255, 255), 3, cv2.LINE_AA)
                sc = a.cell_w / float(v.shape[1])
                tiles.append(cv2.resize(v, (a.cell_w, int(v.shape[0] * sc))))
            if not tiles:
                continue
            h, w = tiles[0].shape[:2]
            nr = (len(tiles) + a.cols - 1) // a.cols
            sheet = np.zeros((nr * h, a.cols * w, 3), np.uint8)
            for k, t in enumerate(tiles):
                r_, c_ = divmod(k, a.cols)
                sheet[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t
            f0, box0, s0, t0 = e["seq"][e["at"]]
            stem = "%s_f%06d_h%d" % (rec, f0, 900 + j)
            cv2.imwrite(os.path.join(a.out, "context", stem + ".jpg"), sheet,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            img = cache.get(f0)
            if img is not None:
                b = [int(x) for x in box0]
                pad = max(160, (b[2] - b[0]))
                y0, y1 = max(0, b[1] - pad), min(img.shape[0], b[3] + pad)
                x0, x1 = max(0, b[0] - pad), min(img.shape[1], b[2] + pad)
                crop = img[y0:y1, x0:x1]
                if crop.size:
                    cv2.imwrite(os.path.join(a.out, "crops", stem + ".jpg"),
                                cv2.resize(crop, (a.crop_px, a.crop_px)),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            tids = {t for _, _, _, t in e["seq"] if t}
            rows.append({
                "stem": stem, "frame": f0, "hand": 900 + j, "conf": 0.0,
                "w_frac": round((box0[2] - box0[0]) / 1920.0, 5),
                "h_frac": round((box0[3] - box0[1]) / 1520.0, 5),
                "cx_frac": round((box0[0] + box0[2]) / 2.0 / 1920.0, 5),
                "cy_frac": round((box0[1] + box0[3]) / 2.0 / 1520.0, 5),
                "model": "%s->%s" % (e["from"], e["to"]),
                "chain_len": len(e["seq"]), "tids": "|".join(sorted(tids)),
                "n_tids": len(tids), "label": "", "label_mode": ""})
        print("  %-16s %d 个事件" % (rec, len(by_rec[rec])), flush=True)
    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    same = sum(1 for r in rows if r["n_tids"] == 1)
    print("\n%d 个事件 -> %s" % (len(rows), a.out))
    print("  整条链只有一个 tracker id 的 %d 个；跨多个 id 的 %d 个"
          % (same, len(rows) - same))


if __name__ == "__main__":
    main()
