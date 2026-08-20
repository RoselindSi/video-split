"""Production auditor v1. Routes candidates to KEEP/REVIEW and ACCEPT/REVIEW.

THERE IS NO REJECT IN THIS FILE, and its absence is enforced rather than
assumed: ACTIONS below is the complete action set, `route()` may only return a
member of it, and --self_test asserts that no input reaches anything else. A
deletion path that exists but is "off by default" is one config edit away from
being on, and this project has turned an off-by-default operating point on
twice.

    boundary   KEEP | REVIEW      a KEEP enters the final recseg unreviewed
    semantic   ACCEPT | REVIEW    an ACCEPT ships the generated label as-is

WHY THIS DEFERS ON DELETION. False rejection is substantially more expensive
than review: a rejected boundary is gone and the segment merges, a reviewed one
costs a person thirty seconds. Automatic rejection is reserved for
independently calibrated evidence that does not exist yet -- `joint_policy`
holds those rules, and holds them disabled.

AND WHY KEEP IS NOT FREE EITHER, which is the part that is easy to skip. KEEP
is also an automatic decision. The one time this project shipped an automatic
keep -- HAL >= 0.85 -- it reached precision 0.767 [0.591, 0.882] on a held-out
batch and 0.467 on the next one, and was withdrawn. So this file DOES NOT SHIP
A THRESHOLD. With no --boundary_thr the router sends everything to REVIEW,
which is safe and worth nothing, and that is the point: the operating point has
to be chosen by someone looking at --calibrate's risk-coverage curve, not
inherited from a default nobody selected.

THE STRUCTURAL VETO OUTRANKS THE SCORE. `boundary/policy.py` blocks automation
for reasons a confidence cannot overcome -- an INTERVAL transition, an
inadequate view, a camera-dominant event. When morphology predictions are
supplied, a veto there is final and no score overrides it. Score alone is used
only when the head's output is absent, and the emitted record says which of the
two decided.

Usage:
    python -m src.auditor.auditor_v1 --self_test

    # what does a threshold actually buy, with an interval on it
    python -m src.auditor.auditor_v1 --calibrate \
        --gold data/gold/pair_schema_v2_migrated.csv \
        --scores results/boundary_v1_oof.json

    # apply chosen operating points to a pipeline output
    python -m src.auditor.auditor_v1 --run \
        --recseg results/pred_recseg.json \
        --boundary_scores results/boundary_v1_oof.json \
        --semantic_scores results/naming_support.jsonl \
        --boundary_thr 0.93 --semantic_thr 0.80 \
        --out results/final_recseg_audited.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

# The complete action set. v1 cannot emit anything else.
ACTIONS = {"boundary": ("KEEP", "REVIEW"), "semantic": ("ACCEPT", "REVIEW")}

TOLERANCE_S = 1.0  # 2026-08-19; see memory/tolerance-is-1s.md


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def route_boundary(score, thr, veto=None):
    """score -> KEEP | REVIEW. A veto outranks any score.

    `thr is None` means no operating point was chosen, and the honest
    consequence is that nothing is automated."""
    if veto:
        return "REVIEW", f"blocked: {veto}"
    if thr is None:
        return "REVIEW", "no operating point chosen (--boundary_thr not given)"
    if score is None:
        return "REVIEW", "no boundary score for this candidate"
    return (("KEEP", f"score {score:.3f} >= {thr}") if score >= thr
            else ("REVIEW", f"score {score:.3f} < {thr}"))


def route_semantic(score, thr):
    """score -> ACCEPT | REVIEW.

    The score is the DIAGONAL term: this segment against ITS OWN label. That is
    what every audit sheet in this project has collected and what the 8B
    verifier was measured on. The CROSS terms -- each segment against the
    other's label -- were only ever needed for the auto-reject conjunction, so
    dropping that conjunction drops the dependency entirely."""
    if thr is None:
        return "REVIEW", "no operating point chosen (--semantic_thr not given)"
    if score is None:
        return "REVIEW", "no semantic score for this segment"
    return (("ACCEPT", f"support {score:.3f} >= {thr}") if score >= thr
            else ("REVIEW", f"support {score:.3f} < {thr}"))


# Morphology classes that may never be automated whatever the score says. This
# is the part of the ontology backed by a head that actually trains.
NEVER_AUTOMATIC = ("INTERVAL_TRANSITION", "UNOBSERVABLE")


def structural_veto(pred, onto, mode="full"):
    """Ask whether automation is permitted at all, independent of the score.

    THE FULL VETO CURRENTLY BLOCKS EVERYTHING, and that is not a bug in this
    file. `boundary_ontology_v1`'s AUTO_KEEP requires relation=EXACT plus two
    observability fields, and those three heads carry no usable gradient -- 6
    EARLY and 4 LATE over 8 recordings, no observability supervision at all. A
    gate whose input is missing blocks, so under `full` the boundary arm
    automates 0% and v1's boundary side is worth nothing.

        full             the ontology as written. Correct, and inert.
        morphology_only  keep the vetoes backed by a head that trains --
                         INTERVAL and UNOBSERVABLE are never automatic -- and
                         drop the relation and observability requirements.
        none             score alone.

    `morphology_only` IS A REDUCTION IN SAFETY AND IT IS NAMED SO IT CAN BE
    PRICED. The requirements it drops are the ones distinguishing "on the
    boundary" from "near a real transition", and near-misses are what an
    automatic keep gets wrong. Calibrate under the same mode you deploy under,
    so the precision interval already contains that cost.

    Returns a reason string when automation is blocked, None when it is not."""
    if not pred or onto is None or mode == "none":
        return None
    m = pred.get("morphology")
    if mode == "morphology_only":
        if not isinstance(m, dict):
            return "no morphology output"
        top = max(m, key=m.get)
        return (f"morphology={top} is never decided automatically"
                if top in NEVER_AUTOMATIC else None)
    from src.auditor.boundary.policy import decide
    try:
        action, reasons, _ = decide(pred, onto)
    except (KeyError, TypeError) as e:
        return f"policy could not evaluate this prediction ({e})"
    return None if action == "AUTO_KEEP" else "; ".join(reasons[-2:])


# --------------------------------------------------------------------------
# calibration -- risk vs coverage, with an interval
# --------------------------------------------------------------------------
def boot_precision(pairs, n=2000, seed=0):
    """Recording-clustered bootstrap on precision among automated items.

    Clustered because candidates inside one recording are not independent, and
    an unclustered interval on this data has been too narrow before."""
    by_rec = defaultdict(list)
    for rec, ok in pairs:
        by_rec[rec].append(ok)
    recs = list(by_rec)
    if not recs:
        return None, None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        hits = [o for _ in recs for o in by_rec[rng.choice(recs)]]
        if hits:
            out.append(sum(hits) / len(hits))
    if not out:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def risk_coverage(items, thresholds=None):
    """items: (recording, score, truth). Prints what each threshold buys.

    COVERAGE IS OVER EVERYTHING, including items with no score. An item the
    scorer never saw still arrives at deployment and still needs a person, so
    excluding it would report a review reduction that does not happen."""
    total = len(items)
    scored = [(r, s, t) for r, s, t in items if s is not None]
    base = sum(t for _, _, t in items) / total if total else 0.0
    print(f"\n  {total} candidates, {len(scored)} scored, "
          f"{total - len(scored)} unscored (each is a REVIEW)")
    print(f"  base rate: {base:.3f} of all candidates are true\n")
    print(f"  {'thr':>6}{'automated':>11}{'coverage':>10}{'precision':>11}"
          f"{'95% CI':>20}{'errors kept':>13}")
    if thresholds is None:
        qs = sorted(s for _, s, _ in scored)
        thresholds = sorted({qs[int(q * (len(qs) - 1))]
                             for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)}) \
            if qs else []
    rows = []
    for thr in thresholds:
        auto = [(r, t) for r, s, t in scored if s >= thr]
        if not auto:
            continue
        prec = sum(t for _, t in auto) / len(auto)
        lo, hi = boot_precision([(r, t) for r, t in auto])
        cov = len(auto) / total
        bad = sum(1 for _, t in auto if not t)
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "—"
        print(f"  {thr:>6.3f}{len(auto):>11}{cov:>9.1%}{prec:>11.3f}"
              f"{ci:>20}{bad:>13}")
        rows.append({"threshold": thr, "n_automated": len(auto),
                     "coverage": cov, "precision": prec,
                     "ci_lo": lo, "ci_hi": hi, "errors_kept": bad})
    print(f"\n  `errors kept` is the count that would enter the output "
          f"unreviewed.\n  Read the CI LOWER bound, not the point estimate: "
          f"the operating point that\n  failed on held-out data had a point "
          f"estimate of 0.789 and a lower bound of 0.591.")
    return rows


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def read_any(path):
    if path.endswith(".csv"):
        return list(csv.DictReader(open(path, newline="",
                                        encoding="utf-8-sig")))
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8-sig")
                if l.strip()]
    b = json.load(open(path, encoding="utf-8-sig"))
    if isinstance(b, list):
        return b
    for k in ("events", "predictions", "items", "segments"):
        if isinstance(b.get(k), list):
            return b[k]
    return [v for v in b.values() if isinstance(v, dict)]


def index_scores(rows, key_fields, score_field):
    out = {}
    for r in rows:
        k = next((str(r[f]) for f in key_fields if r.get(f) is not None), None)
        if k is None:
            continue
        v = r.get(score_field)
        if isinstance(v, dict):  # a probability dict -> P(POINT)
            v = v.get("POINT_TRANSITION", v.get("POINT"))
        if v is not None:
            out[k] = float(v)
    return out


# --------------------------------------------------------------------------
def run(a):
    onto = None
    if a.ontology and os.path.exists(a.ontology):
        from src.auditor.boundary.policy import load_ontology
        onto = load_ontology(a.ontology)

    segs = read_any(a.recseg)
    bs = index_scores(read_any(a.boundary_scores),
                      ("event_id", "candidate_id", "id"), a.boundary_field) \
        if a.boundary_scores else {}
    ss = index_scores(read_any(a.semantic_scores),
                      ("segment_id", "event_id", "id"), a.semantic_field) \
        if a.semantic_scores else {}
    preds = {str(r.get("event_id") or r.get("id")): r
             for r in read_any(a.boundary_scores)} if a.boundary_scores else {}

    out, tab, vetoed = [], Counter(), Counter()
    for s in segs:
        sid = str(s.get("segment_id") or s.get("id") or len(out))
        bid = str(s.get("boundary_id") or s.get("event_id") or sid)
        bscore = bs.get(bid)
        veto = structural_veto(preds.get(bid), onto, a.veto)
        b_act, b_why = route_boundary(bscore, a.boundary_thr, veto)
        sscore = ss.get(sid)
        s_act, s_why = route_semantic(sscore, a.semantic_thr)
        assert b_act in ACTIONS["boundary"] and s_act in ACTIONS["semantic"]
        tab[(b_act, s_act)] += 1
        if veto:
            vetoed[veto.split(";")[0][:56]] += 1
        out.append({
            "segment_id": sid,
            "recording_id": s.get("recording_id") or s.get("recording"),
            "start": s.get("start"), "end": s.get("end"),
            "boundary_time": s.get("boundary_time", s.get("start")),
            "boundary_score": bscore,
            "boundary_audit": b_act,
            "boundary_reason": b_why,
            "boundary_check": ("structural+score" if veto is not None or
                               preds.get(bid) else "score_only"),
            "label": s.get("label") or s.get("action"),
            "semantic_score": sscore,
            "semantic_audit": s_act,
            "semantic_reason": s_why,
            "auditor_version": "v1",
            "tolerance_s": TOLERANCE_S,
        })

    n = len(out)
    print(f"\n{n} segments routed\n")
    print(f"  {'boundary':<10}{'semantic':<10}{'n':>6}{'share':>9}")
    for (b, s), v in sorted(tab.items(), key=lambda x: -x[1]):
        print(f"  {b:<10}{s:<10}{v:>6}{v / n:>9.1%}")
    hb = sum(v for (b, _), v in tab.items() if b == "REVIEW")
    hs = sum(v for (_, s), v in tab.items() if s == "REVIEW")
    both = sum(v for (b, s), v in tab.items()
               if b == "REVIEW" or s == "REVIEW")
    print(f"\n  needs a person: boundary {hb}/{n} ({hb / n:.1%}), "
          f"semantic {hs}/{n} ({hs / n:.1%}),\n  at least one "
          f"{both}/{n} ({both / n:.1%})  <- this is the review budget")
    if vetoed:
        print(f"\n  structural veto ({a.veto}) blocked {sum(vetoed.values())}"
              f"/{n} before any score was consulted:")
        for k, v in vetoed.most_common(5):
            print(f"    {v:>5}  {k}")
        if sum(vetoed.values()) == n:
            print(f"\n  !! the veto blocked EVERY candidate, so the boundary "
                  f"score decided nothing.\n     Under --veto full that is "
                  f"expected: AUTO_KEEP needs relation and observability,\n"
                  f"     and neither head trains. --veto morphology_only is "
                  f"the named downgrade.")
    if a.boundary_thr is None or a.semantic_thr is None:
        print(f"\n  !! at least one threshold was not given, so that arm "
              f"automated nothing.\n     Run --calibrate and choose an "
              f"operating point from its lower bounds.")
    if a.out:
        json.dump({"auditor_version": "v1", "tolerance_s": TOLERANCE_S,
                   "boundary_thr": a.boundary_thr,
                   "semantic_thr": a.semantic_thr,
                   "actions_available": {k: list(v)
                                         for k, v in ACTIONS.items()},
                   "segments": out},
                  open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nwrote {a.out}")


def self_test():
    """No input reaches an action outside ACTIONS, and no threshold automates
    what a veto blocked."""
    cases = [
        (0.99, 0.95, None, "KEEP"),
        (0.99, 0.95, "interaction_visibility=occluded blocks automation",
         "REVIEW"),
        (0.10, 0.95, None, "REVIEW"),
        (None, 0.95, None, "REVIEW"),
        (0.99, None, None, "REVIEW"),
    ]
    for score, thr, veto, want in cases:
        got, why = route_boundary(score, thr, veto)
        assert got == want, (score, thr, veto, got, want)
        assert got in ACTIONS["boundary"]
        print(f"  boundary score={score} thr={thr} veto={bool(veto)} "
              f"-> {got:<7} {why}")
    for score, thr, want in [(0.9, 0.8, "ACCEPT"), (0.5, 0.8, "REVIEW"),
                             (0.9, None, "REVIEW"), (None, 0.8, "REVIEW")]:
        got, why = route_semantic(score, thr)
        assert got == want and got in ACTIONS["semantic"]
        print(f"  semantic support={score} thr={thr} -> {got:<7} {why}")
    # The guarantee is the ACTIONS assertion above; this checks the routing
    # code itself carries no deletion literal, and reads only the functions
    # that decide -- reading the whole file would trip over this test's own
    # strings and turn a real check into one that always passes once edited.
    import inspect
    for fn in (route_boundary, route_semantic, structural_veto, run):
        body = inspect.getsource(fn)
        body = body.replace(inspect.getdoc(fn) or "\0", "")
        for banned in ("REJECT", "DELETE", "DROP_"):
            assert banned not in body, \
                f"{banned!r} appears in {fn.__name__}"
    print("\n  no code path returns a reject; ACTIONS is the whole action set.")
    print("  a veto beats any score; a missing threshold automates nothing.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--recseg")
    ap.add_argument("--gold")
    ap.add_argument("--scores")
    ap.add_argument("--boundary_scores")
    ap.add_argument("--semantic_scores")
    ap.add_argument("--boundary_field", default="morphology")
    ap.add_argument("--semantic_field", default="support")
    ap.add_argument("--truth_field", default="is_boundary")
    ap.add_argument("--boundary_thr", type=float, default=None,
                    help="NO DEFAULT. Without it every candidate is REVIEW.")
    ap.add_argument("--semantic_thr", type=float, default=None,
                    help="NO DEFAULT. Without it every label is REVIEW.")
    ap.add_argument("--veto", choices=("full", "morphology_only", "none"),
                    default="full",
                    help="full = the ontology as written, correct and "
                         "currently inert; morphology_only = keep the vetoes "
                         "backed by a head that trains. Calibrate under the "
                         "mode you deploy under.")
    ap.add_argument("--ontology",
                    default="configs/auditor/boundary_ontology_v1.yaml")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if a.calibrate:
        if not (a.gold and a.scores):
            raise SystemExit("--calibrate needs --gold and --scores")
        gold = read_any(a.gold)
        raw = read_any(a.scores)
        sc = index_scores(raw, ("event_id", "candidate_id", "id"),
                          a.boundary_field)
        preds = {str(r.get("event_id") or r.get("candidate_id")
                     or r.get("id")): r for r in raw}
        onto = None
        if a.veto != "none" and os.path.exists(a.ontology):
            from src.auditor.boundary.policy import load_ontology
            onto = load_ontology(a.ontology)
        items, nveto = [], 0
        for g in gold:
            k = str(g.get("event_id") or g.get("candidate_id") or g.get("id"))
            t = g.get(a.truth_field)
            if t is None:
                continue
            t = str(t).strip().lower() in ("1", "true", "yes", "boundary")
            # CALIBRATE UNDER THE VETO YOU DEPLOY UNDER. A vetoed candidate is
            # a REVIEW no matter its score, so it enters as unscored rather
            # than being dropped -- dropping it would inflate the coverage a
            # threshold appears to buy.
            s = sc.get(k)
            if structural_veto(preds.get(k), onto, a.veto):
                s, nveto = None, nveto + 1
            items.append((g.get("recording_id") or g.get("recording") or k,
                          s, t))
        print(f"\n  veto mode: {a.veto}   "
              f"{nveto} of {len(items)} candidates vetoed before scoring")
        if not items:
            raise SystemExit(f"no gold row carried {a.truth_field!r}")
        risk_coverage(items)
        return
    if a.run:
        if not a.recseg:
            raise SystemExit("--run needs --recseg")
        return run(a)
    ap.print_help()


if __name__ == "__main__":
    main()
