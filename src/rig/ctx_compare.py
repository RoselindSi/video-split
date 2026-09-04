"""Did the indicator fix the confusions without breaking what worked.

THREE QUESTIONS, NOT ONE SCORE. An aggregate f1 can rise while the tracks the
change was made for stay wrong and three others quietly break. So this reports
movement between the four quadrants track by track:

    did the `both_wrong` tracks flip to correct
    did the `both_right` tracks stay right
    did the `context_rescues` tracks stay rescued

The evaluation set is the labelled tracks, which are ground truth a person
gave once per hand. `testpkg_2` is not read here: it has already been scored
twice and a third architecture on it makes it a selection set.

THE BASELINE IS RUN THROUGH THE SAME CODE PATH. Both checkpoints are scored
in one pass over the same rows with the same loader, so a difference cannot
come from a preprocessing change that only one arm received -- which is the
mistake that would be easiest to make here, because the arms differ precisely
in preprocessing.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def quadrant(y, h_ok, c_ok):
    return ("both_right" if h_ok and c_ok else
            "context_hurts" if h_ok and not c_ok else
            "context_rescues" if c_ok else "both_wrong")


def by_track(rows, scores):
    """-> {(tag, tid): (y, median P(owner))}"""
    g = {}
    for r, p in zip(rows, scores):
        g.setdefault((r["tag"], r["tid"]), {"y": r["y"], "p": []})["p"].append(
            float(p))
    return {k: (v["y"], float(np.median(v["p"]))) for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--clf", required=True, help="the hand-only baseline")
    ap.add_argument("--a", required=True, help="checkpoint A")
    ap.add_argument("--b", required=True, help="checkpoint B")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    args = ap.parse_args()

    from src.rig.track_pool import load_pkg, frame_scores

    rows = []
    for p in args.pkg:
        rows += load_pkg(p)[1]
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    print(f"  {len(rows)} 帧样本 over "
          f"{len({(r['tag'], r['tid']) for r in rows})} 条已标 track")

    hand = by_track(rows, frame_scores(rows, clf=args.clf))
    A = by_track(rows, frame_scores(rows, clf_ctx=args.a))
    B = by_track(rows, frame_scores(rows, clf_ctx=args.b))

    qa, qb = {}, {}
    for k in hand:
        y, hp = hand[k]
        qa[k] = quadrant(y, (hp >= 0.5) == y, (A[k][1] >= 0.5) == y)
        qb[k] = quadrant(y, (hp >= 0.5) == y, (B[k][1] >= 0.5) == y)

    names = ("both_right", "context_rescues", "context_hurts", "both_wrong")
    print(f"\n  {'象限':<18} {args.name_a:>8} {args.name_b:>8}")
    for n in names:
        print(f"  {n:<18} {sum(1 for v in qa.values() if v == n):>8} "
              f"{sum(1 for v in qb.values() if v == n):>8}")

    print(f"\n=== 三个问题 ===")
    for n, q in (("both_wrong 修了几条", "both_wrong"),
                 ("both_right 保住了几条", "both_right"),
                 ("context_rescues 保住了几条", "context_rescues")):
        was = [k for k in qa if qa[k] == q]
        if q == "both_wrong":
            fixed = [k for k in was if qb[k] in ("both_right",
                                                 "context_rescues")]
            print(f"  {n:<26} {len(fixed)}/{len(was)}"
                  + (("   " + ", ".join(f"{t}#{i}" for t, i in fixed))
                     if fixed else ""))
        else:
            kept = [k for k in was if qb[k] == q]
            lost = [k for k in was if qb[k] != q]
            print(f"  {n:<26} {len(kept)}/{len(was)}"
                  + (("   丢了 " + ", ".join(f"{t}#{i}->{qb[k]}"
                                            for k in lost
                                            for t, i in [k]))
                     if lost else ""))

    moved = [k for k in qa if qa[k] != qb[k]]
    if moved:
        print(f"\n=== 换了象限的 {len(moved)} 条 ===")
        print(f"  {'track':<24} {'真值':>5} {'hand':>6} "
              f"{args.name_a:>7} {args.name_b:>7}   变化")
        for k in sorted(moved):
            y = "owner" if hand[k][0] == 1 else "other"
            print(f"  {k[0] + ' #' + str(k[1]):<24} {y:>5} "
                  f"{hand[k][1]:>6.2f} {A[k][1]:>7.2f} {B[k][1]:>7.2f}   "
                  f"{qa[k]} -> {qb[k]}")
    else:
        print("\n  没有任何 track 换象限。")


if __name__ == "__main__":
    main()
