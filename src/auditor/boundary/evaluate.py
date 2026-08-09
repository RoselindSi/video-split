"""Boundary v1 evaluation, in the three blocks that answer different questions.

THE HYPOTHESES ARE PRE-REGISTERED, and none of them is "is it better".

  H1  LEGACY PRESERVATION. On the clean sharp-versus-same_action subset, is
      P(POINT) not detectably worse than the deployed frozen scorer? This is a
      COMPATIBILITY test. It cannot show the reformulation is better -- both
      numbers come from different models on a target one of them was not built
      for -- and a delta inside the paired grouped-bootstrap interval is
      reported as "no detectable degradation", never as an improvement.

  H2  MORPHOLOGY IDENTIFIABILITY. On held-out recordings, can INTERVAL be
      separated from POINT? This is the new scientific question and the one
      that matters operationally: gradual events were the largest single
      source of the wrong auto-keeps, so INTERVAL-to-POINT contamination is
      reported on its own rather than buried in a macro average.

  H3  PREVIOUSLY EXCLUDED EVENTS. Do gradual, offscreen, camera and
      annotation-convention events now land somewhere coherent? For gradual
      and offscreen that means their own class. For camera and annotation it
      does NOT mean a correct prediction -- they have no morphology target at
      all -- it means the policy abstains without a perception head having
      been asked to invent one. That is architectural correctness and it is
      checked as such.

A CLASS IS SCORED ONLY IF IT SPANS ENOUGH RECORDINGS. Event count is the wrong
test under recording-grouped CV: ten EARLY events drawn from two recordings
cannot be evaluated across five folds however many events they are, because
three folds contain none of them. n_events, n_recordings and the per-fold
support are all printed, and the withholding is on recordings.

Usage:
    python -m src.auditor.boundary.evaluate \
        --predictions .../boundary_v1_oof.json \
        --compare_decisions .../policy_decisions_v4...csv \
        --compare_col 'P1 (global) alone'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

from src.boundary.state_adapter import _auroc
from src.boundary.c3_selective_policy import wilson

MORPHOLOGY = ["POINT_TRANSITION", "INTERVAL_TRANSITION", "NO_TRANSITION",
              "UNOBSERVABLE"]
RELATION = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID"]


def support(rows, key, cls):
    g = [r for r in rows if r[key] == cls]
    return len(g), len({r["recording_id"] for r in g})


def grouped_bootstrap(fn, rows, n_boot, seed):
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        v = fn([x for i in pick for x in by[keys[i]]])
        if v is not None and np.isfinite(v):
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def auroc_point(rows, score_key):
    y = np.array([r["morphology_true"] == "POINT_TRANSITION" for r in rows],
                 float)
    if len(set(y.tolist())) < 2:
        return None
    p = np.array([score_key(r) for r in rows])
    if not np.isfinite(p).all():
        return None
    return _auroc(y, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--compare_decisions",
                    help="the frozen policy decisions csv, for H1")
    ap.add_argument("--compare_col", default="P1 (global) alone",
                    help="the DEPLOYED arm. Not guessed: picking whichever of "
                         "the three wins would put a search inside H1")
    ap.add_argument("--heads", default="configs/auditor/boundary_v1_heads.yaml")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    min_rec, min_ev = a.n_folds, a.n_folds
    if os.path.exists(a.heads):
        import yaml
        rep = (yaml.safe_load(open(a.heads, encoding="utf-8"))
               .get("reporting") or {})
        min_rec = rep.get("min_recordings_per_class", min_rec)
        min_ev = rep.get("min_events_per_class", min_ev)

    blob = json.load(open(a.predictions, encoding="utf-8"))
    rows = [r for r in blob["events"]
            if r["morphology"]
            and np.isfinite(list(r["morphology"].values())[0])]
    sup = [r for r in rows if r["morphology_true"]]
    pred = {r["event_id"]: max(MORPHOLOGY, key=lambda k: r["morphology"][k])
            for r in rows}
    print(f"{os.path.basename(a.predictions)}: {len(blob['events'])} events, "
          f"{len(rows)} with an out-of-fold prediction, {len(sup)} with a "
          f"morphology target")
    print(f"  reporting threshold: a class needs >= {min_ev} events in "
          f">= {min_rec} distinct recordings")

    # ------------------------------------------------------------------ H2
    print(f"\n{'=' * 78}\nH2  MORPHOLOGY IDENTIFIABILITY\n{'=' * 78}")
    print(f"  {'class':<22} {'n_ev':>5} {'n_rec':>6} {'precision':>10} "
          f"{'recall':>8}  status")
    scored = []
    for c in MORPHOLOGY:
        ne, nr = support(sup, "morphology_true", c)
        ok = ne >= min_ev and nr >= min_rec
        g = [r for r in sup if r["morphology_true"] == c]
        p = [r for r in sup if pred[r["event_id"]] == c]
        tp = sum(1 for r in p if r["morphology_true"] == c)
        if ok:
            scored.append(c)
            print(f"  {c:<22} {ne:>5} {nr:>6} "
                  f"{(tp / len(p) if p else float('nan')):>10.3f} "
                  f"{(tp / len(g) if g else float('nan')):>8.3f}  reported")
        else:
            print(f"  {c:<22} {ne:>5} {nr:>6} {'--':>10} {'--':>8}  WITHHELD "
                  f"(insufficient recording-level support)")

    print(f"\n  confusion, rows are truth:")
    print(f"  {'':<22}" + "".join(f"{c[:12]:>14}" for c in MORPHOLOGY))
    for t in MORPHOLOGY:
        g = [r for r in sup if r["morphology_true"] == t]
        cc = Counter(pred[r["event_id"]] for r in g)
        print(f"  {t:<22}" + "".join(f"{cc.get(c, 0):>14}" for c in MORPHOLOGY))

    recalls = []
    for c in scored:
        g = [r for r in sup if r["morphology_true"] == c]
        recalls.append(sum(1 for r in g
                           if pred[r["event_id"]] == c) / max(1, len(g)))
    if recalls:
        print(f"\n  macro balanced accuracy over the {len(scored)} reported "
              f"classes: {np.mean(recalls):.3f}   chance "
              f"{1 / len(scored):.3f}")
    counts = Counter(r["morphology_true"] for r in sup)
    maj = counts.most_common(1)[0]
    acc = sum(1 for r in sup if pred[r["event_id"]] == r["morphology_true"])
    lo, hi = wilson(acc, len(sup))
    print(f"  overall accuracy {acc}/{len(sup)} = {acc / len(sup):.3f}   "
          f"Wilson [{lo:.3f}, {hi:.3f}]   always-{maj[0]} would give "
          f"{maj[1] / len(sup):.3f}")

    # the operationally decisive cell
    pi = [r for r in sup if r["morphology_true"] in
          ("POINT_TRANSITION", "INTERVAL_TRANSITION")]
    ne, nr = support(pi, "morphology_true", "INTERVAL_TRANSITION")
    if ne >= min_ev and nr >= min_rec:
        au = auroc_point(pi, lambda r: r["morphology"]["POINT_TRANSITION"])
        lo2, hi2 = grouped_bootstrap(
            lambda s: auroc_point(s, lambda r: r["morphology"]["POINT_TRANSITION"]),
            pi, a.n_boot, a.seed)
        cont = sum(1 for r in pi if r["morphology_true"] == "INTERVAL_TRANSITION"
                   and pred[r["event_id"]] == "POINT_TRANSITION")
        print(f"\n  POINT vs INTERVAL alone, {len(pi)} events over "
              f"{len({r['recording_id'] for r in pi})} recordings")
        print(f"  AUROC {au:.3f}   grouped bootstrap [{lo2:.3f}, {hi2:.3f}]")
        print(f"  INTERVAL predicted as POINT: {cont}/{ne}. Gradual was the "
              f"largest single source of the wrong auto-keeps, so this\n  cell "
              f"is the one the reformulation has to move.")

    # ------------------------------------------------------------------ H1
    print(f"\n{'=' * 78}\nH1  LEGACY PRESERVATION (compatibility, not "
          f"improvement)\n{'=' * 78}")
    clean = [r for r in sup if r["morphology_true"] in
             ("POINT_TRANSITION", "NO_TRANSITION")]
    au_new = auroc_point(clean, lambda r: r["morphology"]["POINT_TRANSITION"])
    print(f"  the old clean subset: {len(clean)} events over "
          f"{len({r['recording_id'] for r in clean})} recordings")
    if a.compare_decisions and os.path.exists(a.compare_decisions):
        old = {}
        with open(a.compare_decisions, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if a.compare_col not in (rd.fieldnames or []):
                raise SystemExit(f"`{a.compare_col}` is not in "
                                 f"{rd.fieldnames}")
            for r in rd:
                try:
                    old[r["event_id"]] = float(r[a.compare_col])
                except (TypeError, ValueError):
                    pass
        both = [r for r in clean if r["event_id"] in old]
        print(f"  {len(both)} of them also carry the deployed score "
              f"`{a.compare_col}`")
        if both:
            new = auroc_point(both, lambda r: r["morphology"]["POINT_TRANSITION"])
            oldau = auroc_point(both, lambda r: old[r["event_id"]])
            # PAIRED bootstrap on the delta: two separate intervals on
            # correlated estimates cannot be compared by whether they overlap
            lo3, hi3 = grouped_bootstrap(
                lambda s: (auroc_point(s, lambda r: r["morphology"]["POINT_TRANSITION"])
                           - auroc_point(s, lambda r: old[r["event_id"]]))
                if auroc_point(s, lambda r: old[r["event_id"]]) is not None
                else None, both, a.n_boot, a.seed)
            print(f"  deployed scorer   AUROC {oldau:.3f}")
            print(f"  P(POINT)          AUROC {new:.3f}")
            print(f"  paired delta      {new - oldau:+.3f}   grouped bootstrap "
                  f"[{lo3:+.3f}, {hi3:+.3f}]")
            if lo3 <= 0 <= hi3:
                print(f"  H1: NO DETECTABLE DEGRADATION. The interval contains "
                      f"zero, so this is not evidence of an improvement\n"
                      f"      either -- it is evidence that the reformulation "
                      f"did not cost the legacy discrimination.")
            elif hi3 < 0:
                print(f"  H1: FAILS. The reformulation lost legacy "
                      f"discrimination, and the taxonomy gain below has to be "
                      f"weighed against that.")
            else:
                print(f"  H1: the delta is positive and excludes zero. Report "
                      f"it as preserved-and-then-some, but the two models\n"
                      f"      differ in more than the target, so this is not a "
                      f"clean causal claim for the reformulation.")
    else:
        print(f"  no --compare_decisions given, so H1 cannot be evaluated. "
              f"P(POINT) alone reads AUROC {au_new:.3f}, which is\n  a number "
              f"without a baseline and decides nothing.")

    # ------------------------------------------------------------------ H3
    print(f"\n{'=' * 78}\nH3  PREVIOUSLY EXCLUDED EVENTS\n{'=' * 78}")
    for c in ("INTERVAL_TRANSITION", "UNOBSERVABLE"):
        g = [r for r in sup if r["morphology_true"] == c]
        if not g:
            continue
        right = sum(1 for r in g if pred[r["event_id"]] == c)
        print(f"  {c:<22} {right}/{len(g)} placed in their own class; the rest "
              f"go to "
              f"{dict(Counter(pred[r['event_id']] for r in g if pred[r['event_id']] != c))}")
    masked = [r for r in rows if not r["morphology_true"]]
    if masked:
        print(f"\n  {len(masked)} events have NO morphology target "
              f"({dict(Counter(r['subtype'] for r in masked))}).")
        print(f"  Their success is not a prediction. It is that no perception "
              f"head was asked to invent a target for them and the\n  policy "
              f"abstains -- run src.auditor.boundary.policy to confirm they "
              f"all route to REVIEW. A high-confidence\n  morphology output on "
              f"these is not an error in itself; acting on one would be.")
        conf = [max(r["morphology"].values()) for r in masked]
        print(f"  their morphology confidence: median "
              f"{np.median(conf):.2f}, above 0.95 on "
              f"{int(np.sum(np.array(conf) > 0.95))} of {len(masked)}")

    # -------------------------------------------------------- relation etc.
    sr = [r for r in rows if r["relation_true"] in RELATION]
    print(f"\n{'=' * 78}\nCANDIDATE RELATION (interface only, loss weight 0)"
          f"\n{'=' * 78}")
    if sr:
        rpred = {r["event_id"]: max(RELATION, key=lambda k: r["relation"][k])
                 for r in sr}
        print(f"  {'class':<22} {'n_ev':>5} {'n_rec':>6}  status")
        for c in RELATION:
            ne, nr = support(sr, "relation_true", c)
            if ne >= min_ev and nr >= min_rec:
                g = [r for r in sr if r["relation_true"] == c]
                p = [r for r in sr if rpred[r["event_id"]] == c]
                tp = sum(1 for r in p if r["relation_true"] == c)
                print(f"  {c:<22} {ne:>5} {nr:>6}  precision "
                      f"{(tp / len(p) if p else float('nan')):.3f}  recall "
                      f"{tp / len(g):.3f}")
            else:
                print(f"  {c:<22} {ne:>5} {nr:>6}  WITHHELD")
        print("  This head received no gradient. Any structure here comes from "
              "the encoder that morphology trained, which is\n  worth noticing "
              "and is not a result.")


if __name__ == "__main__":
    main()
