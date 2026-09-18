"""Four arms on four recordings, scored the way the downstream use cares about.

THE WEARER'S HANDS ARE THE PRODUCT. They are what the robot-hand model is
meant to learn from, so the first number is not how often a label flipped but
how much of those pixels the cover destroyed -- and a frame damaged in the
middle of a demonstration is worse than its share of the total suggests, which
is why the count of damaged frames is reported beside the pixel share.

A COLLEAGUE'S HAND LEFT UNCOVERED IS THE OTHER SIDE, and every arm here is
allowed to trade one for the other, so both are always printed. `blur_check`
wrote one row per detected hand per frame with the fraction of its box the
delivered frame altered; the dump supplies the track ids and the human or zone
label says whose hand each track is.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

from src.semhand.blur_check import _iou, read
from src.semhand.distil_eval import read_zone_gold


def join_gold(rows, dump_path, labels_path, rec):
    gold = read_zone_gold(labels_path)
    dump = collections.defaultdict(list)
    for r in csv.DictReader(open(dump_path, encoding="utf-8")):
        if r["rec"] == rec:
            dump[int(r["frame"])].append(r)
    for r in rows:
        box = [r[c] for c in ("x0", "y0", "x1", "y1")]
        best, bg = 0.0, None
        for q in dump[r["frame"]]:
            v = _iou(box, [float(q[c]) for c in ("x0", "y0", "x1", "y1")])
            if v > best:
                best, bg = v, q
        r["gold"] = gold.get((rec, str(bg["tid"]))) if bg is not None and best >= 0.5 else None
    return rows


def score(rows, thr=0.15):
    own = [r for r in rows if r["gold"] == "owner"]
    oth = [r for r in rows if r["gold"] == "other"]
    px = sum(r["covered"] * (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in own)
    tot = sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in own)
    frames = sorted({r["frame"] for r in own if r["covered"] >= thr})
    runs = sum(1 for i, f in enumerate(frames) if i == 0 or f != frames[i - 1] + 1)
    return {"own_frames": len(own), "own_damaged": sum(1 for r in own if r["covered"] > 0.01),
            "own_covered": len(frames), "own_events": runs,
            "own_pixels": px / max(1.0, tot),
            "other_frames": len(oth),
            # A colleague's hand the pipeline left visible: it was there, and
            # nothing in the delivered frame changed over it.
            "other_exposed": sum(1 for r in oth if r["covered"] <= 0.01)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/workspace/exp4")
    ap.add_argument("--pick", default="/workspace/pick4.json")
    ap.add_argument("--arms", default="E,A,B,D")
    ap.add_argument("--out", default="/workspace/exp4/summary.json")
    a = ap.parse_args()
    picks = {p["rec"]: p for p in json.load(open(a.pick))}
    arms = a.arms.split(",")
    report, pool = {}, collections.defaultdict(collections.Counter)
    for rec, p in picks.items():
        n = p["batch"].replace("batch", "")
        dump, labels = f"/workspace/own_dump_b{n}.csv", f"/workspace/labels_b{n}.csv"
        print(f"\n=== {rec}（{p['batch']}，上线规则在这段误盖自己 {p['own_covered']} 帧）")
        print(f"  {'':<6}{'自己手帧':>9}{'被破坏':>8}{'糊>=15%':>9}{'段数':>6}"
              f"{'像素破坏':>10}   {'别人手帧':>9}{'漏糊':>7}{'翻转':>6}")
        for arm in arms:
            f = os.path.join(a.dir, f"{rec}_{arm}.csv")
            if not os.path.exists(f):
                continue
            s = score(join_gold(read(f), dump, labels, rec))
            m = None
            for log in (os.path.join(a.dir, "logs", f"{rec}_{arm}.log"),
                        os.path.join(a.dir, "logs", f"{arm}.log")):
                if os.path.exists(log):
                    m = re.search(r"TRACK-LEVEL FLIPS: (\d+)", open(log).read())
                    if m:
                        break
            s["flips"] = int(m.group(1)) if m else -1
            report[f"{rec}|{arm}"] = s
            for k, v in s.items():
                if v is not None and k != "own_pixels":
                    pool[arm][k] += v
            pool[arm]["px_num"] += s["own_pixels"] * s["own_frames"]
            print(f"  {arm:<6}{s['own_frames']:>9}{s['own_damaged']:>8}{s['own_covered']:>9}"
                  f"{s['own_events']:>6}{s['own_pixels']:>9.3%}   {s['other_frames']:>9}"
                  f"{s['other_exposed']:>7}{s['flips']:>6}")
    print(f"\n=== 四段合并 ===")
    print(f"  {'':<6}{'自己手帧':>9}{'被破坏':>8}{'糊>=15%':>9}{'段数':>6}{'像素破坏':>10}"
          f"   {'别人手帧':>9}{'漏糊':>7}{'翻转':>6}")
    for arm in arms:
        c = pool[arm]
        if not c:
            continue
        print(f"  {arm:<6}{c['own_frames']:>9}{c['own_damaged']:>8}{c['own_covered']:>9}"
              f"{c['own_events']:>6}{c['px_num'] / max(1, c['own_frames']):>9.3%}"
              f"   {c['other_frames']:>9}{c['other_exposed']:>7}{c['flips']:>6}")
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
