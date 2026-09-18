"""Flips at 30 fps, not every third frame: what the demo video shows and the metric does not.

WHY. Every ownership number in this project samples every STRIDE-th frame, so
a label that flips and flips back inside three frames is invisible to it. The
deployed student decides each frame on its own, and the video of it blinks
more than V1's; this measures that directly, on all 400 consecutive frames of
a clip, for:

    V1 deployed      the dump's own verdict (prior 0.5, cap 2, OwnHold)
    V1 raw           its classifier alone
    student raw      the shipped student's P >= 0.5, no smoothing
    student held     the shipped configuration: prior 0, no cap, OwnHold

and prints the same quantity at stride 1 and stride 3 so the gap between what
the metric saw and what the eye sees is explicit. Frames come from the clean
render already on disk; boxes and track ids come from the dump, so every row
is the same hand for every arm.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

from src.semhand.holdreplay import invert_prior, key, replay


def tracks(rows, lab, gold, stride, want):
    """-> {track: [(frame, label)]} for the tracks `want` selects, subsampled."""
    seq = collections.defaultdict(list)
    for r in rows:
        t = (r["rec"], str(r["tid"]))
        g = gold.get(t)
        if want == "owner" and g != "owner":
            continue
        if want == "other" and g != "other":
            continue
        if int(r["frame"]) % stride == 0 or stride == 1:
            seq[t].append((int(r["frame"]), lab[key(r)]))
    for v in seq.values():
        v.sort()
    return seq


def flips(seq, stride):
    """-> (flips per 100 consecutive pairs, pairs, tracks touched, tracks, blips).

    A blip is one frame whose label differs from both of its neighbours: the
    single-frame blink the eye notices and a track-level number never shows."""
    n = f = touched = blips = 0
    for v in seq.values():
        hit = False
        for i in range(1, len(v)):
            if v[i][0] == v[i - 1][0] + stride:
                n += 1
                if v[i][1] != v[i - 1][1]:
                    f += 1
                    hit = True
        for i in range(1, len(v) - 1):
            if (v[i - 1][0] + stride == v[i][0] == v[i + 1][0] - stride
                    and v[i][1] != v[i - 1][1] == v[i + 1][1]):
                blips += 1
        touched += hit
    return (100.0 * f / n if n else float("nan")), n, touched, len(seq), blips


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def eye_flips(rows, lab, thr=0.3):
    """Flicker as the VIDEO shows it: match boxes between neighbouring frames by
    overlap, ignoring track ids, and count label changes. A hand whose track
    breaks and comes back under a new id keeps its within-track flip count at
    zero while the blur on screen still switches; this catches that."""
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["frame"])].append(r)
    frames = sorted(by)
    n = f = changed_id = 0
    ev = []
    for a, b in zip(frames, frames[1:]):
        if b != a + 1:
            continue
        used = set()
        for r in by[b]:
            box = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
            best, bi = 0.0, None
            for j, q in enumerate(by[a]):
                if j in used:
                    continue
                v = iou(box, [float(q[c]) for c in ("x0", "y0", "x1", "y1")])
                if v > best:
                    best, bi = v, j
            if bi is None or best < thr:
                continue
            used.add(bi)
            q = by[a][bi]
            n += 1
            if str(q["tid"]) != str(r["tid"]):
                changed_id += 1
            if lab[key(q)] != lab[key(r)]:
                f += 1
                ev.append((b, str(q["tid"]), str(r["tid"]),
                           "自己->别人" if lab[key(q)] else "别人->自己"))
    return (100.0 * f / n if n else float("nan")), n, changed_id, ev


def show(rows, labels, gold, want, title):
    print(f"\n{title}")
    print(f"  {'':<26}{'每帧 翻转/100':>13}{'对数':>6}{'翻转轨迹':>9}{'单帧闪烁':>9}   "
          f"{'每3帧 翻转/100':>14}{'对数':>6}")
    out = {}
    for n, lab in labels.items():
        s1 = tracks(rows, lab, gold, 1, want)
        r1, n1, t1, ntr, b1 = flips(s1, 1)
        r3, n3, _, _, _ = flips(tracks(rows, lab, gold, 3, want), 3)
        out[n] = {"per_frame": r1, "pairs_1": n1, "tracks_flipping": t1, "tracks": ntr,
                  "blips": b1, "per_3_frames": r3, "pairs_3": n3}
        print(f"  {n:<26}{r1:>13.2f}{n1:>6}{f'{t1}/{ntr}':>9}{b1:>9}   {r3:>14.2f}{n3:>6}")
    return out


def where(rows, labels, gold, arm, want):
    """The flips themselves: which track, which frame, which way."""
    seq = tracks(rows, labels[arm], gold, 1, want)
    ev = []
    for t, v in seq.items():
        for i in range(1, len(v)):
            if v[i][0] == v[i - 1][0] + 1 and v[i][1] != v[i - 1][1]:
                ev.append((t[1], v[i][0], "自己->别人" if v[i - 1][1] else "别人->自己"))
    if ev:
        print(f"\n{arm} 的翻转（轨迹 / 帧 / 方向）:")
        for tid, f, d in sorted(ev, key=lambda e: e[1])[:20]:
            print(f"  tid {tid:>4}  frame {f}  {d}")
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="/workspace/vlmhand/clip/clean",
                    help="clean renders named <rec>_f<frame>.jpg")
    ap.add_argument("--rec", default="R0824_160752")
    ap.add_argument("--dump", default="/workspace/own_dump_fresh.csv")
    ap.add_argument("--gold", action="append",
                    default=["/workspace/selfother/gold/fresh_gold_p1.csv",
                             "/workspace/selfother/gold/fresh_gold_p2.csv"])
    ap.add_argument("--ckpt", default="/workspace/distil/student/S_wide_g6_seed*.pt")
    ap.add_argument("--out", default="/workspace/distil/student/flicker.json")
    a = ap.parse_args()
    import cv2
    from src.selfother.train import read_gold
    from src.semhand import student
    cv2.setNumThreads(4)

    dump = [r for r in csv.DictReader(open(a.dump, encoding="utf-8")) if r["rec"] == a.rec]
    by_frame = collections.defaultdict(list)
    for r in dump:
        by_frame[int(r["frame"])].append(r)
    gold = read_gold(a.gold)
    models, device = student.load(a.ckpt)
    prior = invert_prior(dump)
    p_student, rows = {}, []
    for f in sorted(by_frame):
        img = os.path.join(a.frames, f"{a.rec}_f{f:06d}.jpg")
        if not os.path.exists(img):
            continue
        rgb = cv2.imread(img)
        rs = by_frame[f]
        dets = [{"box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")]} for r in rs]
        for r, (_, p) in zip(rs, student.predict(models, device, rgb, dets)):
            p_student[key(r)] = p
            rows.append(r)
    print(f"{a.rec}: {len({int(r['frame']) for r in rows})} 帧、{len(rows)} 只手、"
          f"{len({(r['rec'], r['tid']) for r in rows} & set(gold))} 条 gold 轨迹")

    p_v1 = {key(r): float(r["p_owner_raw"]) for r in rows}
    labels = {
        "V1 deployed": {key(r): r["final_owner_post_cap"] == "1" for r in rows},
        "V1 raw": {k: v >= 0.5 for k, v in p_v1.items()},
        "student raw": {k: v >= 0.5 for k, v in p_student.items()},
        "student held (w0, no cap)": replay(rows, p_student, prior, geom_w=0.0, cap=None),
        "student held (w0, cap2)": replay(rows, p_student, prior, geom_w=0.0, cap=2),
        "student held (w0.5, cap2)": replay(rows, p_student, prior, geom_w=0.5, cap=2),
    }
    out = {"gold_owner": show(rows, labels, gold, "owner", "M1 的口径：有 gold 的自己手轨迹"),
           "gold_other": show(rows, labels, gold, "other", "别人的手（有 gold）"),
           "all": show(rows, labels, gold, "all", "视频里看得到的全部轨迹（含没有 gold 的）")}
    for arm in ("V1 deployed", "student held (w0, no cap)"):
        out.setdefault("events", {})[arm] = where(rows, labels, gold, arm, "all")

    seq = tracks(rows, labels["V1 deployed"], gold, 1, "owner")
    lens = sorted(len(v) for v in seq.values())
    print(f"\n轨迹碎片化（自己手 gold 轨迹）：{len(lens)} 条覆盖 {sum(lens)} 帧手，"
          f"中位长度 {lens[len(lens) // 2]} 帧，最长 {lens[-1]}")
    print(f"\n视频看到的闪烁（按框重叠配对，不看 track id）")
    print(f"  {'':<26}{'翻转/100 配对':>14}{'配对':>7}{'其中换了 id':>12}")
    for n, lab in labels.items():
        r, npair, cid, ev = eye_flips(rows, lab)
        out.setdefault("eye", {})[n] = {"per_100": r, "pairs": npair, "id_changed": cid,
                                        "events": ev[:50]}
        print(f"  {n:<26}{r:>14.2f}{npair:>7}{cid:>12}")
    for n in ("V1 deployed", "student held (w0, no cap)"):
        ev = out["eye"][n]["events"]
        if ev:
            print(f"\n{n} 的画面闪烁（帧 / 前 tid -> 后 tid / 方向）:")
            for f, t0, t1, d in ev[:20]:
                print(f"  frame {f}  {t0} -> {t1}  {d}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
