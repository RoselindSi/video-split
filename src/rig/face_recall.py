"""Recall against every face anybody has confirmed, not only the ones the
shipped detector proposed.

WHY THE OLD NUMBER WAS A CEILING ON ITSELF. Every labelled face came from a
box YOLOv8-face or YuNet put there, so "recall 69.7%" was recall over the
faces the shipped detector already found. A face neither of them proposed
could not be in the denominator, and a candidate that finds one got no credit
for it -- its proposal went into the unknown pile instead. Two of those piles
have now been labelled by hand, SCRFD's and CrowdHuman-head's, and the faces
confirmed in them belong in the denominator of every model.

THREE ANSWERS, NOT TWO. The head package asks a different question from the
face packages -- a head with no face visible is a real head and not a face --
so its `other` is not the face packages' `other`. Folding them together would
count a correctly found head as a false positive. Here:

    face        somebody confirmed a visible face there
    head        a real head, face not visible (head package only)
    non         not a face, and in the head package not a head either

MATCHING IS BY CONTAINMENT, BOTH WAYS. A head box holds the face box inside
it at roughly twice the area, so their IoU sits near 0.3 even when they name
the same person. Scoring a face detector against a head-derived box by IoU
would report it as blind on faces it found perfectly well. `inter / min(area)`
says "one of these sits inside the other", which is the question actually
being asked, and it is symmetric, so every model is read the same way.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

STEM_RE = re.compile(r"^(.*)_f(\d{6})_h\d+$")


def contained(a, b):
    """-> fraction of the SMALLER box that lies inside the other."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def known(pkgs, head_pkgs=()):
    """-> {context image: [{kind, cx, cy, w, h}]}, one entry per FRAME.

    `head_pkgs` are packages whose `other` means "a head with no visible
    face". Everywhere else `other` means "not a face"."""
    heads = {os.path.abspath(p) for p in head_pkgs}
    out, seen, frame_ctx = collections.defaultdict(list), set(), {}
    for d in pkgs:
        is_head_pkg = os.path.abspath(d) in heads
        for r in csv.DictReader(open(os.path.join(d, "hands.csv"))):
            if r["label"] not in ("owner", "other", "nothand"):
                continue
            if r["label"] == "nothand" and not is_head_pkg:
                continue            # only the head sheet offers a third answer
            ctx = os.path.join(d, "context", r["stem"] + ".jpg")
            if not os.path.exists(ctx):
                continue
            m = STEM_RE.match(r["stem"])
            frame = (m.group(1), m.group(2)) if m else (r["stem"], "")
            ctx = frame_ctx.setdefault(frame, ctx)
            key = (r["stem"], round(float(r["w_frac"]), 4),
                   round(float(r["conf"]), 3))
            if key in seen:
                continue
            seen.add(key)
            if r["label"] == "owner":
                kind = "face"
            elif r["label"] == "nothand":
                kind = "non"
            else:
                kind = "head" if is_head_pkg else "non"
            out[ctx].append({"kind": kind, "src": os.path.basename(d),
                             "cx": float(r["cx_frac"]),
                             "cy": float(r["cy_frac"]), "w": float(r["w_frac"]),
                             "h": float(r["h_frac"])})
    return out


def box_of(it, W, H):
    cx, cy, w, h = it["cx"] * W, it["cy"] * H, it["w"] * W, it["h"] * H
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", default=None)
    ap.add_argument("--head_pkg", action="append", default=[],
                    help="a package whose `other` means head-without-face")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--match", type=float, default=0.5,
                    help="fraction of the smaller box that must be inside")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/workspace/face_recall.json")
    a = ap.parse_args()
    import cv2
    from src.rig import face_mask

    pkgs = a.pkg or sorted(glob.glob("/workspace/facepkg_*"))
    frames = known(pkgs, a.head_pkg)
    keys = sorted(frames)
    if a.limit:
        keys = keys[:a.limit]
    tally = collections.Counter(it["kind"] for k in keys for it in frames[k])
    print(f"{len(keys)} 张帧；已确认 真脸 {tally['face']}、"
          f"是头没露脸 {tally['head']}、非脸/非头 {tally['non']}")

    report = {}
    for path in a.model:
        det = face_mask.load_detector(path, a.floor)
        if det is None:
            print(f"  !! 打不开 {path}")
            continue
        name = os.path.basename(path)
        scores = {"face": [], "head": [], "non": []}
        # EVERY DENOMINATOR IS PARTLY THE CANDIDATE'S OWN WORK. A face only
        # entered the population because some detector proposed it and a
        # person confirmed it, so a model is guaranteed to cover the faces
        # that came from its own unknown pile, and its recall over the whole
        # population is flattered by exactly that share. Splitting by where
        # each face came from is the only way to read the comparison: the
        # faces a model did NOT contribute are the ones it can be judged on.
        by_src = collections.defaultdict(list)
        extra = []
        for k, ctx in enumerate(keys):
            img = cv2.imread(ctx)
            if img is None:
                continue
            H, W = img.shape[:2]
            props = face_mask.detect_faces(det, img)
            used = set()
            for it in frames[ctx]:
                gb = box_of(it, W, H)
                best, bi = 0.0, None
                for i, p in enumerate(props):
                    if i in used:
                        continue
                    v = contained(list(p[:4]), gb)
                    if v > best:
                        best, bi = v, i
                hit = bi is not None and best >= a.match
                if hit:
                    used.add(bi)
                s = float(props[bi][4]) if hit else 0.0
                scores[it["kind"]].append(s)
                if it["kind"] == "face":
                    by_src[it["src"]].append(s)
            extra += [float(p[4]) for i, p in enumerate(props) if i not in used]
            if (k + 1) % 300 == 0:
                print(f"    [{k + 1}/{len(keys)}] {name}", flush=True)
        rows = []
        print(f"\n{name}")
        print(f"  {'阈值':>6}{'覆盖真脸':>12}{'召回':>9}{'覆盖无脸头':>12}"
              f"{'碰到非脸':>10}{'仍未判':>9}")
        for t in (0.20, 0.30, 0.35, 0.40, 0.50, 0.60):
            f_ = sum(1 for s in scores["face"] if s >= t)
            h_ = sum(1 for s in scores["head"] if s >= t)
            n_ = sum(1 for s in scores["non"] if s >= t)
            e_ = sum(1 for s in extra if s >= t)
            rows.append({"thresh": t, "face": f_, "head": h_, "non": n_,
                         "extra": e_})
            print(f"  {t:>6.2f}{f_:>8}/{len(scores['face']):<4}"
                  f"{f_ / max(1, len(scores['face'])):>8.1%}"
                  f"{h_:>8}/{len(scores['head']):<4}{n_:>10}{e_:>9}")
        print("  真脸按来源（谁提出了这个框、后来被人确认）：")
        for src in sorted(by_src):
            v = by_src[src]
            print(f"    {src:<28} n={len(v):<5} "
                  f"@0.35 {sum(1 for s in v if s >= 0.35) / len(v):.1%}  "
                  f"@0.50 {sum(1 for s in v if s >= 0.50) / len(v):.1%}")
        report[name] = {"rows": rows, "n": {k: len(v) for k, v in scores.items()},
                        "by_src": {s: {"n": len(v),
                                       "at35": sum(1 for x in v if x >= 0.35),
                                       "at50": sum(1 for x in v if x >= 0.50)}
                                   for s, v in by_src.items()}}
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
