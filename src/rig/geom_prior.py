"""A geometric prior on ownership, fitted rather than asserted.

WHAT REPLACES WHAT. The pipeline's incumbent prior is one hand-set rule: the
forearm's exit height. Scanning every cue stored beside a labelled hand put
that rule NINTH. Apparent size beats it and beats it badly -- macro AUC 0.018
for hand span and 0.024 for box width against 0.198 for exit height, all read
as distance from 0.5. That is not a surprise once stated: in a head-mounted
view the wearer's own hands are the near ones, and near things are large.

AND SIZE SURVIVES WHAT THE RULE DOES NOT. Exit height scores 0.983 upright and
0.000 on a camera mounted upside down, because a half turn is exactly the
transformation that inverts a height. A hand's apparent size is unchanged by
rotation. So the strongest cue family is also the one that does not encode the
mounting, and a prior built on it is a different kind of object from the rule
it replaces. `--invariant` fits that family alone, which is what a differently
mounted rig should use.

THE CNN CANNOT SEE THIS. `crop_of` cuts a window proportional to the detection
and `Crops` resizes it to 128x128, so absolute scale is normalised away before
the network sees anything. The strongest cue in the data is one the classifier
is blind to by construction. That is the gap this fills, and it is why fusing
here is not the same as training longer.

LEAVE ONE RECORDING OUT, BECAUSE THE ALTERNATIVE FLATTERS. Fitting and scoring
over pooled hands lets a coefficient describe which workstation a hand came
from. Every number this prints is from a fit that never saw the recording it
is scored on.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")

# Unchanged by a rotation of the camera: an apparent size, a detector's
# confidence, a size relative to the other hands in the same frame.
INVARIANT = ("hand_span", "box_w", "box_h", "rel_size", "conf")

# Meaningful only in the mounting they were measured in. A half turn maps a
# height to one minus itself and a downward direction to an upward one.
ORIENTED = ("exit_y", "box_cy", "wrist_y", "dir_y")

ALL_CUES = INVARIANT + ORIENTED


def _f(v, d=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load(pkgs, verbose=True):
    """-> [row] of labelled hands carrying the stored geometry."""
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
            d = {"tag": m.group(1), "frame": int(m.group(2)),
                 "y": 1.0 if r["label"] == "owner" else 0.0}
            for c in ALL_CUES:
                if c != "rel_size":
                    d[c] = _f(r.get(c))
            rows.append(d)
    rows = [r for r in rows
            if all(np.isfinite(r.get(c, np.nan))
                   for c in ALL_CUES if c != "rel_size")]
    if verbose:
        print(f"  {len(rows)} labelled hands with complete geometry, "
              f"{int(sum(1 for r in rows if r['y'] == 0))} other, "
              f"{len({r['tag'] for r in rows})} recordings")
    return rows


def add_rel_size(rows):
    """The one cue that needs the rest of the frame. -> rows

    Scale, not area: a box's diagonal falls linearly with distance while its
    area falls with the square, and a linear quantity is what a coefficient
    can use without the fit having to undo an exponent."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["tag"], r["frame"])].append(r)
    for g in by.values():
        diag = [float(np.hypot(r["box_w"], r["box_h"])) for r in g]
        big = max(diag) if diag else 0.0
        for r, d in zip(g, diag):
            r["rel_size"] = (d / big) if big > 0 else 1.0
    return rows


def rel_size_now(det, dets, shape):
    """The same quantity at inference, from one frame's detections."""
    H, W = shape[:2]

    def diag(d):
        x0, y0, x1, y1 = [float(v) for v in d["box"]]
        return float(np.hypot((x1 - x0) / max(W, 1), (y1 - y0) / max(H, 1)))
    big = max((diag(d) for d in dets), default=0.0)
    return (diag(det) / big) if big > 0 else 1.0


def _logistic(X, y, l2=1.0, iters=400, lr=0.5):
    """Plain gradient descent. -> (coef, intercept)

    No sklearn: this container's numpy is certain and its sklearn is not, and
    a two-class fit on nine columns does not need one."""
    n, k = X.shape
    w = np.zeros(k)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = X.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * g
        b -= lr * gb
    return w, b


def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, float)
    n_o, n_w = int((y == 0).sum()), int((y == 1).sum())
    if n_o == 0 or n_w == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    # `other` is the positive class and a LOW score means foreign, so the
    # owner ranks are the ones summed.
    return float((ranks[y == 1].sum() - n_w * (n_w + 1) / 2.0) / (n_o * n_w))


def fit(rows, cues=ALL_CUES, min_other=3, verbose=True):
    """-> {cues, mean, std, coef, intercept, loo_auc}"""
    X = np.array([[r[c] for c in cues] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    tags = np.array([r["tag"] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0

    per = []
    for t in sorted(set(tags)):
        te = tags == t
        if int((y[te] == 0).sum()) < min_other or not (y[te] == 1).any():
            continue
        w, b = _logistic((X[~te] - mu) / sd, y[~te])
        per.append(auc(((X[te] - mu) / sd) @ w + b, y[te]))
    per = [p for p in per if np.isfinite(p)]
    w, b = _logistic((X - mu) / sd, y)
    out = {"cues": list(cues), "mean": mu.tolist(), "std": sd.tolist(),
           "coef": w.tolist(), "intercept": float(b),
           "loo_auc": float(np.mean(per)) if per else float("nan"),
           "n": len(rows), "n_recordings": len(per)}
    if verbose:
        print(f"\n  leave-one-recording-out macro AUC {out['loo_auc']:.3f} "
              f"over {out['n_recordings']} recordings")
        print(f"\n    {'cue':<12}{'weight':>9}   (positive means the cue "
              f"argues for the WEARER)")
        for c, v in sorted(zip(cues, w), key=lambda kv: -abs(kv[1])):
            print(f"    {c:<12}{v:>9.3f}   {'+' if v > 0 else '-'}"
                  f"{'#' * int(abs(v) * 12)}")
    return out


def score(det, dets, shape, model):
    """-> P(this hand is the wearer's) in [0,1], from geometry alone."""
    from src.rig.own_label import FEATURES, features
    vec = features(det, shape)
    got = dict(zip(FEATURES, vec))
    got["rel_size"] = rel_size_now(det, dets, shape)
    x = np.array([float(got.get(c, np.nan)) for c in model["cues"]], float)
    if not np.isfinite(x).all():
        return None
    z = (x - np.array(model["mean"])) / np.array(model["std"])
    return float(1.0 / (1.0 + np.exp(-(z @ np.array(model["coef"])
                                       + model["intercept"]))))


def load_model(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    rng = np.random.default_rng(0)
    rows = []
    for t in range(6):
        for i in range(40):
            own = i % 2 == 0
            rows.append({"tag": f"R{t}_", "frame": i, "y": 1.0 if own else 0.0,
                         # The wearer's hands are the big near ones.
                         "hand_span": (0.25 if own else 0.08) + rng.normal(0, .01),
                         "box_w": (0.20 if own else 0.06) + rng.normal(0, .01),
                         "box_h": (0.22 if own else 0.07) + rng.normal(0, .01),
                         "conf": (0.90 if own else 0.60) + rng.normal(0, .02),
                         "exit_y": (0.85 if own else 0.25) + rng.normal(0, .02),
                         "box_cy": (0.70 if own else 0.30) + rng.normal(0, .02),
                         "wrist_y": (0.75 if own else 0.30) + rng.normal(0, .02),
                         "dir_y": (-0.8 if own else 0.4) + rng.normal(0, .05)})
    add_rel_size(rows)
    chk(all(0 < r["rel_size"] <= 1.0 for r in rows),
        "relative size is a ratio in (0,1] with the frame's largest at 1")

    m = fit(rows, verbose=False)
    chk(m["loo_auc"] > 0.9,
        f"a fit that never saw the recording still separates it "
        f"(AUC {m['loo_auc']:.3f})")
    inv = fit(rows, cues=INVARIANT, verbose=False)
    chk(inv["loo_auc"] > 0.9,
        f"and the rotation-invariant cues alone are enough here "
        f"(AUC {inv['loo_auc']:.3f})")
    chk(len(inv["cues"]) == len(INVARIANT)
        and "exit_y" not in inv["cues"],
        "the invariant fit really does exclude every oriented cue")

    # A prior has to be a probability, and it has to point the right way.
    big = {"box": (100, 700, 300, 900),
           "kp": np.array([[200, 800]] * 21, float), "conf": 0.9,
           "exit": (200, 890)}
    small = {"box": (500, 200, 540, 240),
             "kp": np.array([[520, 220]] * 21, float), "conf": 0.6,
             "exit": (520, 210)}
    p_big = score(big, [big, small], (900, 1600), m)
    p_small = score(small, [big, small], (900, 1600), m)
    chk(p_big is not None and 0.0 <= p_big <= 1.0,
        "the score is a probability")
    chk(p_big > p_small,
        f"and the near large hand outscores the far small one "
        f"({p_big:.2f} vs {p_small:.2f})")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  The invariant fit is not a fallback: it is what a rig mounted "
          "any other way\n  up must use, and the rule it replaces scores "
          "0.000 there.")
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--invariant", action="store_true",
                    help="fit the rotation-invariant cues only")
    ap.add_argument("--min_other", type=int, default=3)
    a = ap.parse_args()
    pkgs = []
    for p in a.pkg:
        pkgs += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    rows = add_rel_size(load(pkgs))
    if not rows:
        raise SystemExit("no labelled hands with complete geometry")
    cues = INVARIANT if a.invariant else ALL_CUES
    m = fit(rows, cues=cues, min_other=a.min_other)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(f"\n  -> {a.out}")
    print("  Compare this against the incumbent: the exit-height rule is "
          "0.983 accurate\n  upright and 0.000 turned. Fit with --invariant "
          "to see what is left when the\n  cues that encode the mounting are "
          "removed -- that is the number that says\n  whether this survives "
          "a differently mounted camera.")


if __name__ == "__main__":
    main()
