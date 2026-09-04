"""What each detector input delivers at the far end: covered pixels.

ADMISSION IS NOT THE DELIVERABLE. `22/26 admitted` and `0/8 false` say the
larger detector input lets more real hands into the pipeline. They do not say
more foreign hands end up covered, because between admission and a mosaic sit
the tracker, the ownership model, the two-hand cap and the cut, and each can
lose a hand that was admitted. A resolution change also shifts the detection
distribution those stages were tuned on. So the number that decides whether
1024 ships is measured here, at the output.

HOW THE TWO ARMS ARE COMPARED ON THE SAME HANDS. Every track here comes from
the 1024 pass and carries, per frame, whether the 512 pass admitted an
overlapping detection. So both arms are scored on the SAME physical hands and
the same frames; what differs is which of those frames each arm could see.
A hand 512 never admitted contributes all its frames to 512's misses, which
is exactly the failure being measured.

THE APPROXIMATION, STATED. The 512 arm is credited with the ownership verdict
computed on the 1024 crop, on the frames where 512 admitted the hand. Running
ownership on 512's own slightly different box would move some verdicts, and
this cannot say by how much. It is deliberate: the comparison is of admission,
and giving 512 the better arm's ownership makes the comparison conservative
rather than flattering.
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load(pkg):
    """-> rows with the track label, the frame, and 512's verdict on it."""
    lab = {}
    for r in csv.DictReader(open(os.path.join(pkg, "tracks.csv"),
                                 encoding="utf-8-sig")):
        if r.get("label"):
            lab[int(r["tid"])] = r["label"]
    out = []
    for r in csv.DictReader(open(os.path.join(pkg, "hands.csv"),
                                 encoding="utf-8-sig")):
        tid = int(r["tid"])
        if tid not in lab:
            continue
        crop = os.path.join(pkg, "crops", r["stem"] + ".jpg")
        if not os.path.exists(crop):
            continue
        out.append({"tid": tid, "tag": os.path.basename(pkg).strip("/")
                    .replace("rescue_", ""),
                    "label": lab[tid], "frame": int(r["frame"]),
                    "cmp": int(r["cmp_admitted"] or 0),
                    "_crop": crop,
                    "_ctx": os.path.join(pkg, "context", r["stem"] + ".jpg"),
                    "raw": r, "pkg": pkg,
                    "y": 1 if lab[tid] == "owner" else 0})
    return out


def table(rows, p_owner, thresh=0.5):
    """The four privacy numbers, per arm. -> None

    `blur` on a frame means the arm admitted the hand AND ownership called it
    foreign. Those are in series, so a frame can be lost at either."""
    def blurred(r, i, arm):
        if arm == "512" and not r["cmp"]:
            return False
        return p_owner[i] < thresh

    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["tag"], r["tid"]), []).append(i)

    print(f"\n  {'':<34} {'512':>18} {'1024':>18}")
    out = {}
    for arm in ("512", "1024"):
        oth = [(i, r) for i, r in enumerate(rows) if r["label"] == "other"]
        own = [(i, r) for i, r in enumerate(rows) if r["label"] == "owner"]
        non = [(i, r) for i, r in enumerate(rows) if r["label"] == "nothand"]
        o_bl = sum(1 for i, r in oth if blurred(r, i, arm))
        w_bl = sum(1 for i, r in own if blurred(r, i, arm))
        n_bl = sum(1 for i, r in non if blurred(r, i, arm))
        # Track-level: covered at all, and how late.
        lat, cov, zero, tot = [], 0, 0, 0
        for k, idx in groups.items():
            if rows[idx[0]]["label"] != "other":
                continue
            tot += 1
            idx = sorted(idx, key=lambda i: rows[i]["frame"])
            hit = [j for j, i in enumerate(idx) if blurred(rows[i], i, arm)]
            if hit:
                cov += 1
                d = rows[idx[hit[0]]]["frame"] - rows[idx[0]]["frame"]
                lat.append(d)
                zero += (d == 0)
        out[arm] = {"o": (o_bl, len(oth)), "w": (w_bl, len(own)),
                    "n": (n_bl, len(non)),
                    "lat": sorted(lat), "cov": (cov, tot), "zero": zero}

    def line(name, f):
        print(f"  {name:<34} {f('512'):>18} {f('1024'):>18}")

    def frac(a, key):
        k, n = out[a][key]
        return f"{k}/{n} = {k / n:.3f}" if n else "-"
    line("别人的手 帧被糊（召回）", lambda a: frac(a, "o"))
    line("佩戴者的手 帧被误糊", lambda a: frac(a, "w"))
    line("不是手 帧被误糊", lambda a: frac(a, "n"))
    line("别人的手 track 覆盖率",
         lambda a: (f"{out[a]['cov'][0]}/{out[a]['cov'][1]} = "
                    f"{out[a]['cov'][0] / out[a]['cov'][1]:.3f}"
                    if out[a]["cov"][1] else "-"))
    line("首次打码延迟 中位（帧）",
         lambda a: (f"{out[a]['lat'][len(out[a]['lat']) // 2]}"
                    if out[a]["lat"] else "-"))
    line("零延迟 track 数",
         lambda a: f"{out[a]['zero']}/{out[a]['cov'][1]}")

    for a in ("512", "1024"):
        k, n = out[a]["o"]
        lo, hi = wilson(k, n)
        print(f"\n  {a} 别人的手帧召回 {k}/{n} = {k / n:.3f} "
              f"[{lo:.3f}, {hi:.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--clf_ctx", required=True)
    ap.add_argument("--merge", action="append", default=[],
                    help="pkg=csv, write a downloaded sheet back first")
    a = ap.parse_args()

    from src.rig.track_pool import merge_labels, frame_scores
    for spec in a.merge:
        pkg, path = spec.split("=", 1)
        merge_labels(pkg, path)

    rows = []
    for p in a.pkg:
        rows += load(p)
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    n_tracks = len({(r["tag"], r["tid"]) for r in rows})
    seen512 = sum(r["cmp"] for r in rows)
    print(f"  {len(rows)} 帧样本 over {n_tracks} 条已标 track；"
          f"其中 512 也接纳的帧 {seen512} ({seen512 / len(rows):.1%})")

    p = frame_scores(rows, clf_ctx=a.clf_ctx)
    table(rows, p)
    print("\n  两列打分用的是同一批 1024 裁剪图的归属判决；512 那列只在它当帧"
          "也接纳了这只手时\n  才允许打码。所以差距来自准入，不来自归属模型 —— "
          "这让比较偏保守而不是偏向 1024。")
    print("  帧样本是按 crop_stride 存的抽样帧，不是全部帧；比例是无偏的，"
          "绝对帧数不是。")


if __name__ == "__main__":
    main()
