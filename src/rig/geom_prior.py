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

# Unchanged by a rotation of the camera, AND unchanged by what else the
# detector happened to find in the same frame. `box_w`, `box_h` and
# `hand_span` are already divided by the frame's own dimensions in
# `own_label.features`, so each is a property of this hand alone.
INVARIANT = ("hand_span", "box_w", "box_h", "conf")

# `rel_size` is this hand's diagonal over the LARGEST hand's diagonal in the
# same frame, and that denominator is another detection. It was carrying the
# fourth-largest weight in the fitted prior, and it is a defect rather than a
# feature: when the wearer's own near hand enters the frame, every other
# hand's value drops without any of them having moved, and when the detector
# loses a hand -- which it does on 14% of frames here -- they all rise again.
# A colleague's hand alone in shot is the largest hand in shot, scores 1.0,
# and is called the wearer's; the wearer's hand arrives and the same hand
# flips to foreign. That is exactly the flicker the demos showed. Kept behind
# a flag so the cost of removing it can be measured rather than assumed.
DETECTION_SET_DEPENDENT = ("rel_size",)

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
            crop = os.path.join(p, "crops", r["stem"] + ".jpg")
            d = {"tag": m.group(1), "frame": int(m.group(2)), "_crop": crop,
                 "y": 1.0 if r["label"] == "owner" else 0.0}
            for c in ALL_CUES:
                if c != "rel_size":
                    d[c] = _f(r.get(c))
            rows.append(d)
    rows = [r for r in rows
            if os.path.exists(r["_crop"])
            and all(np.isfinite(r.get(c, np.nan))
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




def wilson(k, n, z=1.96):
    """95% interval for a rate. -> (lo, hi)

    Every rate this project has printed has been a point estimate, which was
    tolerable while the samples were the training pool and the number moved
    with every split anyway. On a held-out package it is not: 6 of 762 and 60
    of 762 are different claims, and so are 6 of 762 and 1 of 30. Wilson
    rather than the normal approximation because the rates of interest here
    are near zero, where the normal interval runs below it."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def deployment_rate(rows, clf_path, blend=0.5, min_other=2, geom_model=None,
                    verbose=True):
    """The rate at which the wearer's own hands get covered. -> dict

    WHY THIS NEEDS ITS OWN FUNCTION. `compare` scores recordings that carry
    enough `other` to compute a recall, and on a deployment-rate sample that
    is almost none of them: 68 recordings, 12 with two or more foreign hands,
    39 hands scored out of 806. The 762 `owner` hands -- the entire reason for
    sampling this stratum -- were discarded, and they are the only thing here
    that can be measured precisely. Forty-two positives cannot carry a recall
    at any confidence worth printing; seven hundred and sixty-two negatives
    can carry a false-positive rate.

    AND IT IS THE ERROR THAT MATTERS MOST. Missing a colleague's hand leaves a
    stranger in one frame. Covering the wearer's hand destroys the pixels the
    downstream model was built to read, in the exact region it is reading. The
    two are not symmetric and only one of them has been measurable until now.

    THE PREVALENCE IS THE POINT, NOT AN INCONVENIENCE. Every precision figure
    this project has quoted came from a sample that is 40% foreign; deployment
    is nearer 5%. Precision falls with prevalence even when the classifier is
    unchanged, because the same false-positive rate is divided by a much
    smaller number of true positives. So the number to carry forward is the
    per-hand FALSE POSITIVE RATE, which does not move with prevalence, and the
    precision implied by it at the real base rate."""
    from src.rig import own_cnn
    model, device = own_cnn.load_model(clf_path)
    if model is None:
        raise SystemExit(f"no checkpoint at {clf_path}")
    torch = own_cnn._torch()
    tags = sorted({r["tag"] for r in rows})
    out = {k: {"fp": 0, "n_own": 0, "tp": 0, "n_oth": 0}
           for k in ("cnn", "geom", "blend")}
    per_rec = {}
    for tag in tags:
        te = [r for r in rows if r["tag"] == tag]
        tr = [r for r in rows if r["tag"] != tag]
        if not te or not tr:
            continue
        if geom_model is None and sum(1 for r in tr
                                      if r["y"] == 0) < min_other:
            continue
        # A SHIPPED PRIOR IS SCORED AS SHIPPED. Refitting per recording is
        # the right correction when the test recordings sit inside the
        # training pool, because otherwise the prior has seen them. On a
        # package of genuinely fresh recordings it is the wrong thing: what
        # goes to a new workstation is one fixed set of coefficients, and
        # refitting reports a prior nobody will deploy.
        m = geom_model or fit(tr, cues=INVARIANT, min_other=min_other,
                              verbose=False)
        X = np.array([[r[c] for c in m["cues"]] for r in te], float)
        z = (X - np.array(m["mean"])) / np.array(m["std"])
        p_geom = 1.0 / (1.0 + np.exp(-(z @ np.array(m["coef"])
                                       + m["intercept"])))
        ds = own_cnn.Crops([dict(r, y=int(r["y"])) for r in te], augment=False)
        ps = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                xs = [ds[j][0] for j in range(i, min(i + 64, len(ds)))]
                ps.append(torch.softmax(model(torch.stack(xs).to(device)),
                                        1)[:, 1].cpu().numpy())
        p_cnn = np.concatenate(ps)
        y = np.array([r["y"] for r in te])
        rec = {}
        for name, p in (("cnn", p_cnn), ("geom", p_geom),
                        ("blend", (1 - blend) * p_cnn + blend * p_geom)):
            pred_other = p < 0.5
            fp = int((pred_other & (y == 1)).sum())
            tp = int((pred_other & (y == 0)).sum())
            out[name]["fp"] += fp
            out[name]["n_own"] += int((y == 1).sum())
            out[name]["tp"] += tp
            out[name]["n_oth"] += int((y == 0).sum())
            rec[name] = (fp, int((y == 1).sum()))
        per_rec[tag] = rec
    if verbose:
        base = (sum(1 for r in rows if r["y"] == 0) / max(len(rows), 1))
        print(f"\n  {len(rows)} hands over {len(per_rec)} recordings, "
              f"{sum(1 for r in rows if r['y'] == 0)} other "
              f"({base:.1%} -- this is the deployment prevalence)")
        print(f"\n    {'scorer':<8}{'own covered':>14}{'FP rate':>8}"
              f"{'  95% CI':<16}{'found':>11}{'recall':>7}"
              f"{'  95% CI':<14}{'prec':>8}")
        for k in ("cnn", "geom", "blend"):
            v = out[k]
            fpr = v["fp"] / max(v["n_own"], 1)
            rc = v["tp"] / max(v["n_oth"], 1)
            pr = v["tp"] / max(v["tp"] + v["fp"], 1)
            flo, fhi = wilson(v["fp"], v["n_own"])
            rlo, rhi = wilson(v["tp"], v["n_oth"])
            print(f"    {k:<8}{v['fp']:>6}/{v['n_own']:<7}{fpr:>8.2%}"
                  f" [{flo:.2%},{fhi:.2%}]"
                  f"{v['tp']:>5}/{v['n_oth']:<5}{rc:>7.3f}"
                  f" [{rlo:.2f},{rhi:.2f}]{pr:>8.3f}")
        worst = sorted(((v["geom"][0] / max(v["geom"][1], 1), t)
                        for t, v in per_rec.items()), reverse=True)[:5]
        print(f"\n  worst recordings for covering the wearer (geom): "
              + ", ".join(f"{t} {r:.1%}" for r, t in worst if r > 0))
        print("\n  `own covered` is the wearer's own hands called foreign and "
              "therefore blurred.\n  Read the FP RATE rather than the "
              "precision: the rate is a property of the\n  classifier, while "
              "the precision also depends on how rare foreign hands are,\n  "
              "and it will fall on any recording quieter than this sample.")
    return out


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
    ap.add_argument("--out", help="where to write the fitted prior. Not "
                                  "needed with --compare, which fits per "
                                  "held-out recording and ships nothing.")
    ap.add_argument("--with_rel_size", action="store_true",
                    help="add the detection-set-dependent cue back, to "
                         "measure what removing it cost")
    ap.add_argument("--invariant", action="store_true",
                    help="fit the rotation-invariant cues only")
    ap.add_argument("--min_other", type=int, default=3)
    ap.add_argument("--compare", metavar="CLF",
                    help="score the CNN, the geometry and their blend "
                         "separately on --holdout recordings")
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--geom_model",
                    help="score with this shipped prior instead of refitting "
                         "one per recording. Correct whenever the package's "
                         "recordings are not in the training pool: refitting "
                         "would report a prior nobody deploys.")
    ap.add_argument("--deployment", metavar="CLF",
                    help="score EVERY labelled hand and report the rate at "
                         "which the wearer's own hands are covered. Use on a "
                         "randomly sampled package, where positives are too "
                         "few for a recall but negatives are plentiful.")
    a = ap.parse_args()
    pkgs = []
    for p in a.pkg:
        pkgs += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    rows = add_rel_size(load(pkgs))
    if not rows:
        raise SystemExit("no labelled hands with complete geometry")
    if a.deployment:
        deployment_rate(rows, a.deployment, blend=a.blend,
                        min_other=a.min_other,
                        geom_model=load_model(a.geom_model))
        raise SystemExit(0)
    if a.compare:
        hold = a.holdout
        if not hold:
            # Every recording that can carry a rate, not a hand-picked few.
            # Three recordings gave geometry a clean sweep of 1.000s; that is
            # either the finding or the sample, and the only way to tell is
            # to stop choosing which recordings to look at.
            import collections
            by = collections.Counter(r["tag"] for r in rows if r["y"] == 0)
            hold = sorted(t for t, n in by.items() if n >= a.min_other)
            print(f"  no --holdout given: comparing on all {len(hold)} "
                  f"recordings with at least {a.min_other} `other`")
        compare(rows, a.compare, hold, blend=a.blend)
        raise SystemExit(0)
    if not a.out:
        raise SystemExit("--out is required when fitting")
    cues = INVARIANT if a.invariant else ALL_CUES
    if a.with_rel_size:
        cues = tuple(cues) + DETECTION_SET_DEPENDENT
    m = fit(rows, cues=cues, min_other=a.min_other)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(f"\n  -> {a.out}")
    print("  Compare this against the incumbent: the exit-height rule is "
          "0.983 accurate\n  upright and 0.000 turned. Fit with --invariant "
          "to see what is left when the\n  cues that encode the mounting are "
          "removed -- that is the number that says\n  whether this survives "
          "a differently mounted camera.")




def compare(rows, clf_path, holdout, blend=0.5, verbose=True):
    """CNN, geometry, and their blend, scored separately on held-out
    recordings. -> {name: {tag: (prec, rec, n, n_other)}}

    THE BLEND HAS NEVER BEEN MEASURED APART FROM ITS PARTS. The demo fuses a
    network and a fitted prior at equal weight and shows one verdict; when
    that verdict is wrong there is no way to tell from the screen whether the
    network dragged the geometry down, the geometry dragged the network down,
    or both were wrong. Three columns cost one pass over crops that are
    already on disk.

    THE GEOMETRY IS REFITTED WITHOUT THE RECORDING IT IS SCORED ON. The
    shipped JSON was fitted on everything, so using it here would give the
    prior an advantage the network does not have and make the blend look
    better than it will be on a new workstation."""
    from src.rig import own_cnn
    model, device = own_cnn.load_model(clf_path)
    if model is None:
        raise SystemExit(f"no checkpoint at {clf_path}")
    torch = own_cnn._torch()
    hold = set(holdout)
    out = {"cnn": {}, "geom": {}, "blend": {}, "cap2": {}}
    for tag in sorted(hold):
        te = [r for r in rows if r["tag"] == tag]
        if not te or not any(r["y"] == 0 for r in te):
            continue
        tr = [r for r in rows if r["tag"] != tag]
        m = fit(tr, cues=INVARIANT, verbose=False)
        X = np.array([[r[c] for c in m["cues"]] for r in te], float)
        z = (X - np.array(m["mean"])) / np.array(m["std"])
        p_geom = 1.0 / (1.0 + np.exp(-(z @ np.array(m["coef"])
                                       + m["intercept"])))
        ds = own_cnn.Crops([dict(r, _crop=r["_crop"], y=int(r["y"]))
                            for r in te], augment=False)
        ps = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                xs = [ds[j][0] for j in range(i, min(i + 64, len(ds)))]
                ps.append(torch.softmax(model(torch.stack(xs).to(device)),
                                        1)[:, 1].cpu().numpy())
        p_cnn = np.concatenate(ps)
        y = np.array([r["y"] for r in te])
        p_bl = (1 - blend) * p_cnn + blend * p_geom
        frames = np.array([r["frame"] for r in te])
        for name, p in (("cnn", p_cnn), ("geom", p_geom), ("blend", p_bl),
                        ("cap2", p_bl)):
            pred_other = p < 0.5
            if name == "cap2":
                # AT MOST TWO OWNERS PER FRAME, because a person has two
                # hands. AUC says the ranking inside a recording is nearly
                # perfect while the threshold is not, and a threshold is
                # exactly what does not transport: a workstation where every
                # hand is a little smaller shifts every score the same way
                # and moves them all to one side of 0.5 with their order
                # intact. A within-frame rank cannot be shifted like that.
                # This only ever turns `owner` into `other`, so it can lose
                # precision and cannot lose recall.
                for fr in np.unique(frames):
                    m_fr = frames == fr
                    idx = np.where(m_fr)[0]
                    if len(idx) <= 2:
                        continue
                    keep = idx[np.argsort(-p[idx])[:2]]
                    demote = np.setdiff1d(idx, keep)
                    pred_other[demote] = True
            tp = int((pred_other & (y == 0)).sum())
            fp = int((pred_other & (y == 1)).sum())
            fn = int((~pred_other & (y == 0)).sum())
            out[name][tag] = (tp / (tp + fp) if tp + fp else float("nan"),
                              tp / (tp + fn) if tp + fn else float("nan"),
                              len(te), int((y == 0).sum()))
    if verbose:
        print(f"\n    {'recording':<16}{'n':>5}{'other':>7}"
              + "".join(f"{k + ' prec':>12}{k + ' rec':>11}"
                        for k in ("cnn", "geom", "blend", "cap2")))
        for tag in sorted(out["cnn"]):
            n, no = out["cnn"][tag][2], out["cnn"][tag][3]
            line = f"    {tag:<16}{n:>5}{no:>7}"
            for k in ("cnn", "geom", "blend", "cap2"):
                pr, rc, _, _ = out[k][tag]
                line += (f"{pr:>12.3f}" if np.isfinite(pr) else f"{'-':>12}")
                line += (f"{rc:>11.3f}" if np.isfinite(rc) else f"{'-':>11}")
            print(line)
        print("\n  `other` is the positive class. The geometry column is a "
              "fit that never saw\n  the recording it scores, so every "
              "column answers the same question about a\n  new workstation. "
              "If the blend sits below its better part, fusing at equal\n  "
              "weight is costing accuracy rather than buying it.")
        print()
        for k in ("cnn", "geom", "blend", "cap2"):
            pr = np.array([v[0] for v in out[k].values()], float)
            rc = np.array([v[1] for v in out[k].values()], float)
            print(f"    {k:<7} median prec {np.nanmedian(pr):.3f}   "
                  f"median rec {np.nanmedian(rc):.3f}   "
                  f"recordings under 0.5 rec: "
                  f"{int(np.nansum(rc < 0.5))}/{len(rc)}")
        print()
        print("  cap2 is the blend with at most two owners per frame. It "
              "trades a threshold\n  for a within-frame rank, which is what "
              "an AUC of 0.988 says this data\n  actually supports -- and a "
              "rank cannot be moved by a workstation where\n  every hand is "
              "uniformly smaller.")
    return out


if __name__ == "__main__":
    main()
