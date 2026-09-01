"""Which geometric cues separate a colleague's hand from the wearer's.

WHY MEASURE INSTEAD OF PICKING. The pipeline has one geometric rule -- the
forearm's exit height -- and it is excellent upright and worthless upside
down. The temptation after that is to reason about which OTHER cue should
help. Every hand already carries a dozen of them in `hands.csv`, and there are
now 392 labelled `other` hands over 21 recordings, which is enough to stop
reasoning and start ranking.

WITHIN RECORDING, NOT POOLED. A cue can look strong pooled purely because
recordings differ: one workstation has colleagues leaning in and another does
not, so anything correlated with WHICH RECORDING a hand came from scores well
without separating anything inside a frame. This has bitten this project
before, on a different head, where a pooled correlation of .744 became .531
per recording. So every cue here is scored inside each recording and the
recordings are averaged with equal weight.

THE RELATIVE CUES ARE THE INTERESTING ONES. Absolute position asks "is this
hand low in the frame". Relative position asks "is this hand lower than the
others in this same frame", which is the question the 2014 egocentric work
answered with a body-part model: the wearer's hands are the near ones, so they
are the large ones and the low ones, and a third hand in a two-handed person's
frame belongs to somebody else. These cost one pass of grouping and no model.

AUC, NOT ACCURACY. A cue does not come with a threshold, and inventing one to
report accuracy would measure the threshold rather than the cue. AUC is the
probability that a random `other` outranks a random `owner`, which is what
"does this carry signal" means. 0.5 is nothing; below 0.5 means the cue points
the other way and is just as useful.
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(pkgs, verbose=True):
    """-> [row] with the stored geometry, labels only."""
    rows = []
    for p in pkgs:
        q = os.path.join(p, "hands.csv")
        if not os.path.exists(q):
            continue
        for r in csv.DictReader(open(q, encoding="utf-8-sig")):
            if r.get("label") not in ("owner", "other"):
                continue
            m = STEM_RE.match(r.get("stem", ""))
            if not m:
                continue
            rows.append({
                "tag": m.group(1), "frame": int(m.group(2)),
                "y": 0 if r["label"] == "other" else 1,
                "exit_y": _f(r.get("exit_y")), "exit_x": _f(r.get("exit_x")),
                "cx": _f(r.get("box_cx")), "cy": _f(r.get("box_cy")),
                "w": _f(r.get("box_w")), "h": _f(r.get("box_h")),
                "dir_x": _f(r.get("dir_x")), "dir_y": _f(r.get("dir_y")),
                "span": _f(r.get("hand_span")), "conf": _f(r.get("conf")),
                "n_hands": _f(r.get("n_hands")),
                "wrist_y": _f(r.get("wrist_y")),
                "left": _f(r.get("is_left_hand")),
            })
    if verbose:
        n_o = sum(1 for r in rows if r["y"] == 0)
        print(f"  {len(rows)} labelled hands, {n_o} other, "
              f"{len({r['tag'] for r in rows})} recordings")
    return rows


def add_relative(rows):
    """Cues that only exist relative to the other hands in the same frame.

    A hand cannot be 'the third one' on its own. Grouping is by recording AND
    frame because frame numbers repeat across recordings."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["tag"], r["frame"])].append(r)
    for group in by.values():
        area = [(_r["w"] or 0) * (_r["h"] or 0) for _r in group]
        cy = [_r["cy"] for _r in group]
        big = max(area) if area else 0.0
        n = len(group)
        for r, a, c in zip(group, area, cy):
            r["n_in_frame"] = float(n)
            # 1.0 is the largest hand in the frame. The wearer's hands are the
            # near ones and near things are large.
            r["rel_size"] = (a / big) if big > 0 else np.nan
            # How far below the frame's average hand this one sits.
            r["rel_cy"] = (c - float(np.nanmean(cy))) if n > 1 else 0.0
            # 0 is the lowest hand in the frame.
            r["rank_cy"] = float(sorted(cy, reverse=True).index(c))
            # A two-handed person cannot own three hands. This is a fact about
            # anatomy, and it is the only cue here that is not a tendency.
            r["excess"] = max(0.0, n - 2.0)
    return rows


CUES = ["exit_y", "cy", "wrist_y", "rel_cy", "rank_cy", "rel_size",
        "n_in_frame", "excess", "dir_y", "span", "conf", "exit_x", "cx",
        "w", "h", "left"]


def auc(score, y):
    """P(a random `other` scores above a random `owner`). -> float or nan

    `y` is 1 for owner, 0 for other; `other` is the positive class here, so
    the score is negated to keep 'higher means more foreign'."""
    s = np.asarray(score, float)
    y = np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_o, n_w = int((y == 0).sum()), int((y == 1).sum())
    if n_o == 0 or n_w == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks over ties, or a cue with few distinct values (rank_cy,
    # excess) is scored as though its ties were ordered.
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 0].sum() - n_o * (n_o + 1) / 2.0) / (n_o * n_w))


def scan(rows, min_other=3, verbose=True):
    """-> {cue: (macro_auc, pooled_auc, n_recordings)}

    A recording with no `other` cannot rank one against an owner, so it
    contributes nothing and is excluded rather than counted as 0.5."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tag"]].append(r)
    usable = {t: v for t, v in by.items()
              if sum(1 for r in v if r["y"] == 0) >= min_other
              and any(r["y"] == 1 for r in v)}
    out = {}
    for c in CUES:
        per = [auc([r.get(c, np.nan) for r in v], [r["y"] for r in v])
               for v in usable.values()]
        per = [p for p in per if np.isfinite(p)]
        out[c] = (float(np.mean(per)) if per else np.nan,
                  auc([r.get(c, np.nan) for r in rows],
                      [r["y"] for r in rows]),
                  len(per))
    if verbose:
        print(f"\n  {len(usable)} recordings carry at least {min_other} "
              f"`other` and one owner\n")
        print(f"    {'cue':<12}{'macro AUC':>11}{'pooled':>9}{'recs':>6}"
              f"   {'strength':<22}")
        for c, (m, p, n) in sorted(out.items(),
                                   key=lambda kv: -abs((kv[1][0] or .5) - .5)):
            d = abs(m - 0.5) if np.isfinite(m) else 0
            bar = "#" * int(d * 40)
            gap = ("  pooled >> macro" if np.isfinite(m)
                   and abs(p - 0.5) > abs(m - 0.5) + 0.05 else "")
            print(f"    {c:<12}{m:>11.3f}{p:>9.3f}{n:>6}   {bar}{gap}")
        print("\n  AUC is read as distance from 0.500. Below 0.5 means the "
              "cue points the\n  other way, which is signal, not failure. "
              "`pooled >> macro` marks a cue that\n  looks strong only "
              "because recordings differ -- it separates workstations, "
              "not\n  hands, and would not survive a new one.")
    return out


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    chk(abs(auc([1, 2, 3, 4], [1, 1, 0, 0]) - 1.0) < 1e-9,
        "a cue that ranks every `other` above every owner scores 1.0")
    chk(abs(auc([4, 3, 2, 1], [1, 1, 0, 0]) - 0.0) < 1e-9,
        "and one that inverts them scores 0.0 rather than failing")
    chk(abs(auc([1, 1, 1, 1], [1, 1, 0, 0]) - 0.5) < 1e-9,
        "a constant cue scores exactly 0.5 through the tie correction")
    chk(not np.isfinite(auc([1, 2], [1, 1])),
        "a recording with no `other` returns nan, not a number")

    rows = [{"tag": "A_", "frame": 1, "y": 1, "cy": 800, "w": 100, "h": 100},
            {"tag": "A_", "frame": 1, "y": 1, "cy": 780, "w": 90, "h": 90},
            {"tag": "A_", "frame": 1, "y": 0, "cy": 300, "w": 40, "h": 40}]
    add_relative(rows)
    chk(rows[2]["excess"] == 1.0 and rows[0]["excess"] == 1.0,
        "three hands in a frame put an excess of one on all of them")
    chk(rows[2]["rel_size"] < rows[0]["rel_size"],
        "the far hand is the small one relative to the frame")
    chk(rows[2]["rank_cy"] == 2.0,
        "and the highest hand in the frame ranks last from the bottom")
    two = [{"tag": "B_", "frame": 1, "y": 1, "cy": 800, "w": 1, "h": 1},
           {"tag": "B_", "frame": 1, "y": 1, "cy": 700, "w": 1, "h": 1}]
    add_relative(two)
    chk(two[0]["excess"] == 0.0,
        "two hands are what one person has, so no excess")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  Macro over recordings is the number that matters: a cue "
          "separating\n  workstations scores well pooled and explains "
          "nothing inside a frame.")
    return all(ok)


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--min_other", type=int, default=3)
    a = ap.parse_args()
    pkgs = []
    for p in a.pkg:
        pkgs += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    rows = add_relative(load(pkgs))
    if not rows:
        raise SystemExit("no labelled hands with geometry in those packages")
    scan(rows, min_other=a.min_other)


if __name__ == "__main__":
    main()
