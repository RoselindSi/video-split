"""What is the model's POINT cue actually keyed on? No training, no API.

Two facts sit badly together. NO_TRANSITION has p75 0.590 and p90 0.957 on
P(POINT), so a batch of same-action events are called boundaries with high
confidence. And the 7 events a human called `point_like` on blind review have
a median P(POINT) of 0.004, with none above the POINT median -- the model is
most certain they are NOT boundaries exactly where a person sees a compact
switch. Those are not two errors, they are one disagreement about what a
boundary looks like, and the disagreement has a shape that can be read off
data already on disk.

FOUR SHORTCUTS, each one falsifiable here:

  1 THE FROZEN SCORE. If P(POINT) tracks the old P1 or fused score closely,
    the temporal encoder re-derived the pre/post contrast it was built to
    replace -- the sequence went in and a summary came out. Rank correlation
    overall and WITHIN each morphology class, because a high overall value
    can come from both scores simply separating POINT from NONE.

  2 THE CANDIDATE GENERATOR. The source tag in an event id (gt_boundary,
    raw_change_peak, false_mid_segment, exact, late) correlates with the class
    by construction, and a model that keys on whatever those sources look like
    would score well here and transfer to nothing. Reported with the class
    composition beside it so the confound stays visible rather than being
    read as a finding.

  3 THE RECORDING. If most of the variance in P(POINT) lies BETWEEN
    recordings rather than within them, the model learned which videos look
    like boundary videos, not which moments are boundaries. That is the one
    shortcut grouped folds do not protect against: it survives the split
    because it is a property of the recording, and every fold has recordings.

  4 THE WINDOW. Coverage is not uniform -- the 10 fps global caches hit more
    grid points than the 2 fps local ones -- and if P(POINT) tracks coverage
    the head is reading how much data it got.

NONE OF THESE IS PROVED BY A CORRELATION. Each is a lead with a number
attached, and the extremes are printed in full so a person can look at the
events rather than at the aggregate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

POINT, INTERVAL, NONE, UNOBS = ("POINT_TRANSITION", "INTERVAL_TRANSITION",
                                "NO_TRANSITION", "UNOBSERVABLE")
OLD_ARMS = ["P1 (global) alone", "local alone", "P1 + local, feature-level"]
SOURCE = re.compile(r"_(gt_boundary|raw_change_peak|false_mid_segment|"
                    r"false_gap|false_near_edge|missed_[a-z_]+?|late|early|"
                    r"exact|duplicate)_t")


def source_of(eid):
    m = SOURCE.search(eid)
    return m.group(1) if m else "other"


def spearman(a, b):
    """Rank correlation without scipy. Ties get average ranks."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")

    def rank(x):
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        # average the ranks of tied values, or a constant column reads as a
        # perfect correlation with whatever it is compared against
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    ra, rb = rank(a[ok]), rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--decisions", help="frozen scores, for shortcut 1")
    ap.add_argument("--interval_audit", help="the filled 37-row sheet")
    ap.add_argument("--rechecked", action="append", default=[],
                    help="files listing batch3 events that were re-annotated "
                         "from video (first column; '#' comments skipped). "
                         "Splits batch3 into re-checked and never-checked")
    ap.add_argument("--top_k", type=int, default=12)
    a = ap.parse_args()

    blob = json.load(open(a.predictions, encoding="utf-8"))
    rows = [r for r in blob["events"]
            if r["morphology"] and np.isfinite(r["morphology"][POINT])]
    for r in rows:
        r["_p"] = r["morphology"][POINT]
        r["_src"] = source_of(r["event_id"])
    old = {}
    if a.decisions and os.path.exists(a.decisions):
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = {c: float(r[c]) for c in OLD_ARMS
                     if r.get(c) not in (None, "")}
                if d:
                    old[r["event_id"]] = d
    sub = {}
    if a.interval_audit and os.path.exists(a.interval_audit):
        with open(a.interval_audit, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                call = next((v for k, v in r.items()
                             if k and k.startswith("your_call")), "")
                if (call or "").strip():
                    sub[r["event_id"]] = call.strip()
    checked = set()
    for pth in a.rechecked:
        if not os.path.exists(pth):
            continue
        with open(pth, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e = line.split(",")[0].strip()
                if e and e != "event_id" and not e.startswith("evidence"):
                    checked.add(e)
    print(f"{len(rows)} events with an out-of-fold P(POINT); "
          f"{sum(1 for r in rows if r['event_id'] in old)} carry the frozen "
          f"scores; {len(sub)} carry an interval subtype")

    # ------------------------------------------------------ 1 frozen score
    print(f"\n{'=' * 92}\n1  IS P(POINT) JUST THE OLD SCORE?  Spearman"
          f"\n{'=' * 92}")
    common = [r for r in rows if r["event_id"] in old]
    if common:
        print(f"  {'population':<28} {'n':>4} "
              + "".join(f"{c[:18]:>20}" for c in OLD_ARMS))
        pops = [("all events", common)] + [
            (c, [r for r in common if r.get("morphology_true") == c])
            for c in (POINT, INTERVAL, NONE)]
        for name, g in pops:
            if len(g) < 3:
                continue
            print(f"  {name:<28} {len(g):>4} "
                  + "".join(f"{spearman([r['_p'] for r in g], [old[r['event_id']][c] for r in g]):>20.3f}"
                            for c in OLD_ARMS))
        print("  The WITHIN-class rows are the ones that matter. A high value "
              "on 'all events' only says both scores separate\n  POINT from "
              "NONE, which is what they were both built to do. A high value "
              "INSIDE a class says the new head is\n  ranking that class the "
              "same way the old summary did, with nothing added.")

    # -------------------------------------------------- 2 the generator tag
    print(f"\n{'=' * 92}\n2  DOES P(POINT) TRACK THE CANDIDATE SOURCE?"
          f"\n{'=' * 92}")
    print(f"  {'source':<32} {'n':>4} {'median P(POINT)':>16}   class mix")
    for s, n in Counter(r["_src"] for r in rows).most_common():
        g = [r for r in rows if r["_src"] == s]
        mix = Counter((r.get("morphology_true") or "MASKED").split("_")[0]
                      for r in g)
        print(f"  {s:<32} {n:>4} "
              f"{np.median([r['_p'] for r in g]):>16.3f}   {dict(mix)}")
    print("  The class mix column is not decoration. These sources were built "
          "to produce positives and negatives at different\n  rates, so a "
          "source effect on the score is expected -- what would be a finding "
          "is a source whose score is high or\n  low against its own class "
          "mix.")

    # ------------------------------- 2b source WITHIN a fixed truth class
    print(f"\n{'=' * 92}\n2b  THE CONFIRMATION: does the source still move "
          f"the score INSIDE one truth class?\n{'=' * 92}")
    print("  If P(POINT) varies across sources among events that share a "
          "morphology truth, the head is reading the candidate\n  generator "
          "rather than the video. The class is held fixed, so nothing about "
          "the class mix can explain it.")
    srcs = [s for s, n in Counter(r["_src"] for r in rows).most_common()
            if n >= 8]
    for cls in (POINT, NONE):
        g = [r for r in rows if r.get("morphology_true") == cls]
        if len(g) < 10:
            continue
        print(f"\n  truth = {cls}, {len(g)} events")
        print(f"    {'source':<30} {'n':>4} {'median P(POINT)':>16} "
              f"{'batch3?':>9}")
        for s_ in srcs:
            h = [r for r in g if r["_src"] == s_]
            if len(h) < 3:
                continue
            b3 = sum(1 for r in h if "_batch3_" in r["event_id"])
            print(f"    {s_:<30} {len(h):>4} "
                  f"{np.median([r['_p'] for r in h]):>16.3f} "
                  f"{b3}/{len(h):>7}")
    print("\n  The batch3 column is there because the source tag almost "
          "partitions the two label sources -- gt_boundary and\n  "
          "raw_change_peak occur only in batch3, the rest only in dev. A "
          "source effect that is really a DATASET effect and one\n  that is "
          "really a generator effect look identical until they are separated, "
          "and this table separates them only\n  where a class holds both.")

    # a single number for the whole thing: how much of P(POINT) does source
    # explain once the truth class is already known?
    print(f"\n  variance of P(POINT) explained, nested:")
    allp = np.array([r["_p"] for r in rows])
    tot = allp.var()
    def resid(keyfn):
        out = []
        by = defaultdict(list)
        for r in rows:
            by[keyfn(r)].append(r["_p"])
        for v in by.values():
            out += list(np.array(v) - np.mean(v))
        return np.var(out)
    r_cls = resid(lambda r: r.get("morphology_true") or "MASKED")
    r_both = resid(lambda r: (r.get("morphology_true") or "MASKED", r["_src"]))
    print(f"    truth class alone            {1 - r_cls / tot:>6.1%}")
    print(f"    truth class + source         {1 - r_both / tot:>6.1%}")
    print(f"    what source adds ON TOP      {(r_cls - r_both) / tot:>6.1%}")
    print("    The last line is the part of the score that the candidate "
          "generator explains and the label does not.")

    # ------------------------------------------- 2c dev against batch3
    print(f"\n{'=' * 92}\n2c  THE SPLIT THAT 2b ACTUALLY FOUND: dev against "
          f"batch3\n{'=' * 92}")
    print("  Conditioned on the truth class, the generator ordering in section "
          "2 disappears: every dev source sits at 0.95-1.00\n  for POINT and "
          "0.001-0.065 for NONE. The tag was standing in for the LABEL SOURCE, "
          "because gt_boundary and\n  raw_change_peak occur only in batch3. "
          "So the question is not which generator but which dataset.")
    def au(g):
        y = np.array([r["morphology_true"] == POINT for r in g], float)
        if len(set(y.tolist())) < 2:
            return float("nan"), 0
        p_ = np.array([r["_p"] for r in g])
        o = np.argsort(p_)
        yy = y[o]
        pos, neg = yy.sum(), len(yy) - yy.sum()
        # rank-based AUROC, ties at 0.5
        ranks = np.empty(len(p_), float)
        ranks[o] = np.arange(len(p_)) + 1.0
        _, inv, cnt = np.unique(p_, return_inverse=True, return_counts=True)
        ssum = np.zeros(len(cnt))
        np.add.at(ssum, inv, ranks)
        ranks = (ssum / cnt)[inv]
        return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) /
                     max(pos * neg, 1)), len(g)
    pn = [r for r in rows if r.get("morphology_true") in (POINT, NONE)]
    b3 = [r for r in pn if "_batch3_" in r["event_id"]]
    groups = [("dev (audited-source tags)",
               [r for r in pn if "_batch3_" not in r["event_id"]]),
              ("batch3, all", b3)]
    if checked:
        # the decisive split. 90 batch3 events were re-annotated from video
        # after their subtypes were traced to machine-made calls, and 58 of
        # them changed. The other 150 were never re-checked. If the
        # re-checked ones score much higher, what is left in batch3 is label
        # noise and the fix is annotation, not representation.
        groups += [("  batch3, re-checked from video",
                    [r for r in b3 if r["event_id"] in checked]),
                   ("  batch3, NEVER re-checked",
                    [r for r in b3 if r["event_id"] not in checked])]
    groups += [("both together", pn)]
    print(f"\n  {'population':<30} {'n':>5} {'POINT':>6} {'NONE':>6} "
          f"{'AUROC':>8}")
    for name, g in groups:
        v, n = au(g)
        npos = sum(1 for r in g if r["morphology_true"] == POINT)
        print(f"  {name:<30} {n:>5} {npos:>6} {n - npos:>6} {v:>8.3f}")
    if checked:
        print("\n  The re-checked split is the one that decides the next "
              "move. Those 90 events were re-annotated from video after\n  "
              "their subtypes were traced to machine-made calls, and 58 "
              "changed; the other 150 were never looked at again. If the\n  "
              "re-checked rows score clearly higher, what remains in batch3 "
              "is label noise and the fix is annotation, not a new\n  "
              "representation. If both halves score the same, the labels are "
              "not the explanation and the features are.")
    print("\n  A near-perfect number on one source beside a weak one on the "
          "other is not a model that half works. Either the two\n  label sets "
          "differ in quality -- batch3 came through a relabel of "
          "machine-made calls, and 58 of the 90 re-checked\n  there were "
          "changed -- or the model has something on dev that it should not "
          "have. Both are worth more than another\n  point of aggregate "
          "AUROC, and the aggregate hides both.")

    # --------------------------------------------------- 3 the recording
    print(f"\n{'=' * 92}\n3  IS P(POINT) A PROPERTY OF THE RECORDING?"
          f"\n{'=' * 92}")
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r["_p"])
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    if multi:
        allv = np.array([x for v in multi.values() for x in v])
        means = np.array([np.mean(v) for v in multi.values()])
        w = np.array([len(v) for v in multi.values()])
        between = float(np.average((means - allv.mean()) ** 2, weights=w))
        within = float(np.average([np.var(v) for v in multi.values()],
                                  weights=w))
        frac = between / max(between + within, 1e-12)
        print(f"  {len(multi)} recordings hold 2+ events ({int(w.sum())} "
              f"events)")
        print(f"  between-recording variance {between:.4f}, within "
              f"{within:.4f}  ->  {frac:.1%} of the variance is BETWEEN "
              f"recordings")
        print(f"  Grouped folds do not protect against this one. A cue that is "
              f"a property of the recording survives the split,\n  because "
              f"every fold contains recordings; it only fails when the "
              f"recordings change, which is what batch4 is for.")
        # the same decomposition on the truth, as the reference point
        yb = defaultdict(list)
        for r in rows:
            if r.get("morphology_true"):
                yb[r["recording_id"]].append(
                    float(r["morphology_true"] == POINT))
        ym = {k: v for k, v in yb.items() if len(v) >= 2}
        if ym:
            ally = np.array([x for v in ym.values() for x in v])
            mm = np.array([np.mean(v) for v in ym.values()])
            ww = np.array([len(v) for v in ym.values()])
            b2 = float(np.average((mm - ally.mean()) ** 2, weights=ww))
            w2 = float(np.average([np.var(v) for v in ym.values()],
                                  weights=ww))
            print(f"  for comparison, the TRUTH is {b2 / max(b2 + w2, 1e-12):.1%} "
                  f"between-recording. A score far above that is keying on the "
                  f"recording\n  more than the label does.")

    # ------------------------------------------------------- 4 the window
    print(f"\n{'=' * 92}\n4  DOES P(POINT) TRACK WINDOW COVERAGE?\n{'=' * 92}")
    for k in ("coverage_g", "coverage_l"):
        v = [r.get(k) for r in rows]
        if any(x is not None for x in v):
            print(f"  spearman(P(POINT), {k}) = "
                  f"{spearman([r['_p'] for r in rows], [x if x is not None else np.nan for x in v]):.3f}"
                  f"   (unique values: "
                  f"{len({x for x in v if x is not None})})")

    # --------------------------------------------------------- the extremes
    print(f"\n{'=' * 92}\nTHE DISAGREEMENTS, IN FULL\n{'=' * 92}")
    nones = sorted([r for r in rows if r.get("morphology_true") == NONE],
                   key=lambda r: -r["_p"])[:a.top_k]
    print(f"  HIGHEST P(POINT) among NO_TRANSITION -- the model calls these "
          f"boundaries and a person called them one ongoing action:")
    for r in nones:
        o = old.get(r["event_id"], {})
        print(f"    {r['event_id'][-46:]:<47} {r['_p']:.3f}   src "
              f"{r['_src']:<22}"
              + (f" P1 {o.get(OLD_ARMS[0], float('nan')):.3f}" if o else ""))
    pl = sorted([r for r in rows if sub.get(r["event_id"]) == "point_like"],
                key=lambda r: r["_p"])
    if pl:
        print(f"\n  LOWEST P(POINT) among the human-called point_like -- a "
              f"person saw a compact switch and the model is certain there is "
              f"none:")
        for r in pl:
            o = old.get(r["event_id"], {})
            print(f"    {r['event_id'][-46:]:<47} {r['_p']:.3f}   src "
                  f"{r['_src']:<22}"
                  + (f" P1 {o.get(OLD_ARMS[0], float('nan')):.3f}" if o else ""))
        print(f"\n  If the OLD score is high on these while P(POINT) is low, "
              f"the temporal head lost something the summary had.\n  If both "
              f"are low, neither representation carries what the person used, "
              f"and that is a feature problem.")


if __name__ == "__main__":
    main()
