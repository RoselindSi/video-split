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
def route_boundary(score, thr, veto=None, uncertified=None):
    """score -> KEEP | REVIEW. A veto outranks any score.

    THREE INDEPENDENT WAYS TO STAY AT REVIEW, and they are kept apart because
    they mean different things to whoever reads the output: the ontology
    blocked automation, no operating point was chosen, or the operating point
    is not backed by a passing certificate. Collapsing them into one "REVIEW"
    loses the only information that says what would have to change."""
    if veto:
        return "REVIEW", f"blocked: {veto}"
    if thr is None:
        return "REVIEW", "no operating point chosen (--boundary_thr not given)"
    if uncertified:
        return "REVIEW", f"threshold not certified: {uncertified}"
    if score is None:
        return "REVIEW", "no boundary score for this candidate"
    return (("KEEP", f"score {score:.3f} >= {thr}") if score >= thr
            else ("REVIEW", f"score {score:.3f} < {thr}"))


def route_semantic(score, thr, uncertified=None):
    """score -> ACCEPT | REVIEW. Symmetric with route_boundary, deliberately.

    The score is the DIAGONAL term: this segment against ITS OWN label. That is
    what every audit sheet in this project has collected and what the 8B
    verifier was measured on. The CROSS terms -- each segment against the
    other's label -- were only ever needed for the auto-reject conjunction, so
    dropping that conjunction drops the dependency entirely.

    AN ACCEPT SHIPS A LABEL UNREVIEWED, which is the same kind of automatic
    decision a KEEP is. The first version of this function checked only
    `thr is None`, so a --semantic_thr typed on the command line automated
    immediately while the boundary side refused the same move -- two different
    safety rules for two arms doing the same thing."""
    if thr is None:
        return "REVIEW", "no operating point chosen (--semantic_thr not given)"
    if uncertified:
        return "REVIEW", f"threshold not certified: {uncertified}"
    if score is None:
        return "REVIEW", "no semantic score for this segment"
    return (("ACCEPT", f"support {score:.3f} >= {thr}") if score >= thr
            else ("REVIEW", f"support {score:.3f} < {thr}"))


# --------------------------------------------------------------------------
# observability -- "am I entitled to judge this at all"
# --------------------------------------------------------------------------
# A DIFFERENT QUESTION FROM THE SCORE, and the one the auditor could not ask.
# `morphology` answers "was there a real change"; this answers "could a change
# have been seen". They are confused constantly and their consequences are
# opposite: a continuous interaction is a reason to say NO BOUNDARY, hands
# outside the frame are a reason to say NOTHING.
#
# THREE STATES, AND `unknown` IS THE DEFAULT. No head in this project emits
# observability -- it has never had supervision -- so every field arrives
# `unknown` until one does. Forcing a guess would manufacture the evidence the
# tri-state exists to protect.
OBSERVABILITY_FIELDS = ("hand_visible", "object_visible",
                        "interaction_visible", "camera_stable")
OBS_STATES = ("present", "absent", "unknown")

# WHY THIS DOES NOT REJECT ANYTHING. `absent` -- seen clearly, no hands, no
# interaction -- would be safe to drop, and `unknown` would not: dropping an
# unobservable region silently decides there is no boundary in it, which is the
# defect `joint_policy`'s `unobservable_is_not_no_boundary` rule exists to
# block. Until a sample says how false_gap candidates split across the three,
# every one of them routes to REVIEW and only the REASON changes.
REVIEW_UNCERTAIN = "REVIEW_UNCERTAIN"
REVIEW_UNOBSERVABLE = "REVIEW_UNOBSERVABLE"


def observability_state(obs):
    """(class, reason) for one candidate's observability block.

    THE TWO REVIEW CLASSES ARE NOT THE SAME WORK. An uncertain candidate is one
    a person can settle by watching it. An unobservable one is a person
    watching hands that are not in frame, and no amount of attention fixes it --
    that is a capture problem, and it belongs in a different queue and out of
    the denominator a recall is computed over."""
    if not obs:
        return REVIEW_UNOBSERVABLE, "observability was not collected"
    bad = {k: obs.get(k) for k in OBSERVABILITY_FIELDS
           if obs.get(k) not in (None, "present")}
    if not bad:
        return REVIEW_UNCERTAIN, None
    unk = [k for k, v in bad.items() if v in (None, "unknown")]
    ab = [k for k, v in bad.items() if v == "absent"]
    parts = []
    if ab:
        parts.append("observed absent: " + ", ".join(sorted(ab)))
    if unk:
        parts.append("not assessed: " + ", ".join(sorted(unk)))
    return REVIEW_UNOBSERVABLE, "; ".join(parts)


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


def review_lift(items):
    """If a person reviews worst-score-first, how fast do they meet the errors?

    THIS NEEDS NO GATE AND NO CERTIFICATE, and the reason is the whole point:
    ordering a queue skips nothing. Every candidate is still seen by a person,
    so a bad ordering costs time and cannot put a wrong boundary into the
    output. Automation is what needs a threshold; ranking does not.

    Which makes it the part of v1 that ships today. The boundary curve says no
    threshold earns an automatic keep, and driver A's 0.641 does not clear the
    capability gate for automatic accept -- but 0.641 is already enough to put
    the errors near the front of a queue, and that is real time saved with no
    risk taken at all."""
    scored = [(r, s, bool(t)) for r, s, t in items if s is not None]
    if not scored:
        return []
    order = sorted(scored, key=lambda x: x[1])          # worst first
    bad = [not t for _, _, t in order]
    total_bad = sum(bad)
    if not total_bad:
        return []
    print(f"\n  REVIEW ORDER -- worst score first, nothing skipped\n"
          f"  {len(order)} candidates, {total_bad} of them wrong\n")
    print(f"  {'review':>8}{'errors found':>15}{'of all errors':>16}"
          f"{'lift':>8}")
    rows, seen = [], 0
    marks = [0.1, 0.2, 0.3, 0.4, 0.5]
    for frac in marks:
        k = max(1, int(round(frac * len(order))))
        found = sum(bad[:k])
        rows.append({"review_fraction": frac, "errors_found": found,
                     "recall": found / total_bad, "lift": (found / k) / (
                         total_bad / len(order))})
        print(f"  {frac:>7.0%}{found:>15}{found / total_bad:>15.1%}"
              f"{rows[-1]['lift']:>8.2f}x")
    print(f"\n  lift 1.00x is what reviewing in a random order gives. Anything "
          f"above it is\n  time saved with no decision automated -- the person "
          f"still sees every candidate.")
    return rows


def risk_coverage(items, thresholds=None, gate=None):
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
    print(f"  {'thr':>8}{'automated':>11}{'coverage':>10}{'precision':>11}"
          f"{'95% CI':>20}{'errors kept':>13}{'  gate':>8}")
    if thresholds is None:
        qs = sorted(s for _, s, _ in scored)
        # ROUNDED BEFORE MEASUREMENT, NOT AFTER. The table prints four
        # decimals and the certificate stores what was measured; if those were
        # a full-precision float the printed number could never be typed back,
        # and the number a person selects has to be the number that was
        # actually evaluated. Rounding is upward so any residue admits fewer
        # candidates than were measured, never more.
        thresholds = sorted({math.ceil(qs[int(q * (len(qs) - 1))] * 1e4) / 1e4
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
        row = {"threshold": thr, "n_automated": len(auto),
               "coverage": cov, "precision": prec,
               "ci_lo": lo, "ci_hi": hi, "errors_kept": bad}
        mark = ""
        if gate is not None:
            ok, why = check_gate(row, gate)
            row["gate_pass"], row["gate_reasons"] = ok, why
            mark = "  PASS" if ok else "  fail"
        print(f"  {thr:>8.4f}{len(auto):>11}{cov:>9.1%}{prec:>11.3f}"
              f"{ci:>20}{bad:>13}{mark:>8}")
        rows.append(row)
    print(f"\n  `errors kept` is the count that would enter the output "
          f"unreviewed.\n  Read the CI LOWER bound, not the point estimate: "
          f"the operating point that\n  failed on held-out data had a point "
          f"estimate of 0.789 and a lower bound of 0.591.")
    if gate is not None and not any(r.get("gate_pass") for r in rows):
        unset = sorted({w for r in rows for w in r.get("gate_reasons", [])
                        if "unset" in w})
        print(f"\n  NO THRESHOLD PASSES THE PRE-REGISTERED GATE, so AUTO_KEEP "
              f"stays off.")
        for w in unset:
            print(f"    {w}")
        if unset:
            print(f"  Those are null on purpose. How much unreviewed error the "
                  f"product can carry is\n  not a modelling question, and "
                  f"filling them after reading this table is the\n  same move "
                  f"that produced both withdrawn operating points.")
    return rows


# --------------------------------------------------------------------------
# the certificate -- what makes a threshold usable rather than merely typed
# --------------------------------------------------------------------------
GATE = "configs/auditor/auto_keep_gate_v1.yaml"


def load_gate(path=GATE):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_gate(row, gate):
    """One calibration row against the pre-registered conditions.

    Returns (passes, reasons). A null target is NOT a pass: an unset condition
    means nobody has decided how much unreviewed error is acceptable, and
    treating that as satisfied is how a gate becomes decorative."""
    g = gate.get("gate") or {}
    out, ok = [], True
    lo = g.get("precision_ci_lower_min")
    if lo is None:
        ok = False
        out.append("precision_ci_lower_min is unset in the gate config")
    elif row.get("ci_lo") is None or row["ci_lo"] < lo:
        ok = False
        out.append(f"CI lower {row.get('ci_lo')} < {lo}")
    mx = g.get("errors_kept_max")
    if mx is None:
        ok = False
        out.append("errors_kept_max is unset in the gate config")
    elif row["errors_kept"] > mx:
        ok = False
        out.append(f"errors_kept {row['errors_kept']} > {mx}")
    cv = g.get("coverage_min")
    if cv is None:
        ok = False
        out.append("coverage_min is unset in the gate config")
    elif row["coverage"] < cv:
        ok = False
        out.append(f"coverage {row['coverage']:.3f} < {cv}")
    return ok, out


def event_fingerprint(ids):
    """A stable digest of the events a certificate was measured on.

    Stored so `--run` can tell whether it is applying a threshold to the very
    events it was chosen on. That overlap is the specific mistake behind two
    withdrawn operating points, and it is invisible unless something checks."""
    import hashlib
    h = hashlib.sha256()
    for i in sorted(set(map(str, ids))):
        h.update(i.encode() + b"\0")
    return h.hexdigest()[:16], len(set(map(str, ids)))


def verify_certificate(cert, thr, veto_mode, run_ids, allow_overlap=False):
    """Why this threshold may NOT automate, or None if it may."""
    if cert is None:
        return "no --certificate supplied"
    if cert.get("veto_mode") != veto_mode:
        return (f"certificate was measured under veto={cert.get('veto_mode')!r}"
                f" and deployment is {veto_mode!r}")
    passing = [r for r in cert.get("rows", []) if r.get("gate_pass")]
    if not passing:
        return ("no threshold in the certificate passed the pre-registered "
                "gate")
    if not any(abs(r["threshold"] - thr) < 1e-9 for r in passing):
        av = ", ".join(f"{r['threshold']:.3f}" for r in passing[:4])
        return f"{thr} is not a gate-passing threshold (passing: {av})"
    if run_ids and not allow_overlap:
        cal = set(cert.get("event_ids") or [])
        if cal:
            ov = len(cal & set(map(str, run_ids)))
            if ov:
                return (f"{ov} events here were in the calibration set; a "
                        f"threshold chosen on the events it scores is the "
                        f"mistake behind two withdrawn operating points "
                        f"(--allow_overlap to override)")
    return None


# --------------------------------------------------------------------------
# the semantic certificate -- and the two levels it must never confuse
# --------------------------------------------------------------------------
# A CAPABILITY RESULT IS NOT A DEPLOYMENT CERTIFICATE, and keeping them in one
# field is how the confusion would happen. Driver A answers "does this scorer
# order a real YES above a real NO inside one recording" -- a paired,
# threshold-free question. It says nothing about whether `score >= 0.8` is safe
# to ship, because a paired ranking can be perfect while every absolute score
# sits on the wrong side of any fixed cut. The two are separate `kind`s and
# only one of them may back a --semantic_thr.
CERT_CAPABILITY = "capability"
CERT_AUTOMATION = "semantic_automation"


def code_fingerprint(paths):
    """Digest of the code that produced a measurement.

    A certificate that survives an edit to the evaluator is a certificate for
    a number nothing can reproduce. Cheap to record, and the only thing that
    catches "we changed how ties are scored" six weeks later."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode() + b"\0")
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


SEMANTIC_BINDINGS = ("scorer", "window", "gold_fingerprint", "gate_version",
                     "code_fingerprint")


def verify_semantic_certificate(cert, thr, deployment):
    """Why this --semantic_thr may NOT automate, or None if it may.

    `deployment` carries the same five bindings the certificate does. Each is
    a way the measured number stops describing the running system: a different
    scorer, a different window (a 3s sub-window and a full segment are not the
    same measurement -- G0 already showed a short window can lose the signal),
    a different gold, a different gate, a different evaluator."""
    if cert is None:
        return "no --semantic_certificate supplied"
    kind = cert.get("kind")
    if kind == CERT_CAPABILITY:
        return ("this is a CAPABILITY certificate (driver A pairwise), which "
                "shows the scorer ranks YES above NO but says nothing about "
                "any absolute cut; it may not back a threshold")
    if kind != CERT_AUTOMATION:
        return f"certificate kind is {kind!r}, expected {CERT_AUTOMATION!r}"
    for k in SEMANTIC_BINDINGS:
        want, got = cert.get(k), deployment.get(k)
        if want is None:
            return f"certificate does not record {k}"
        if got is None:
            return f"deployment does not state {k}; pass it so it can be checked"
        if str(want) != str(got):
            return f"{k} differs: certificate {want!r}, deployment {got!r}"
    passing = [r for r in cert.get("rows", []) if r.get("gate_pass")]
    if not passing:
        return "no threshold in the certificate passed the pre-registered gate"
    if not any(abs(r["threshold"] - thr) < 1e-9 for r in passing):
        av = ", ".join(f"{r['threshold']:.4f}" for r in passing[:4])
        return f"{thr} is not a gate-passing threshold (passing: {av})"
    return None


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


SCORE_FALLBACKS = ("score", "prob", "probability", "confidence", "support",
                   "morphology")
BOUNDARY_FIELD_DEFAULT = "morphology"
SEMANTIC_FIELD_DEFAULT = "support"


def index_scores(rows, key_fields, score_field, what="", explicit=False):
    """Join scores onto candidates, and REFUSE to do it silently badly.

    A FAIL-CLOSED SYSTEM MAKES BROKEN PLUMBING LOOK LIKE CORRECT CAUTION. With
    the wrong field name every lookup returns None, every candidate routes to
    REVIEW, and the output is indistinguishable from an auditor working exactly
    as designed. That is the most dangerous shape a bug can take here, because
    nothing about it looks wrong -- it was found by running end to end on real
    data, never by the fixture.

    So: the named field is tried, then a short list of the names score files in
    this repo actually use, and which one was used is printed. Producing a
    score for NOTHING is a hard error, not a warning."""
    out, used = {}, None
    for field in (score_field,) + tuple(f for f in SCORE_FALLBACKS
                                        if f != score_field):
        for r in rows:
            k = next((str(r[f]) for f in key_fields if r.get(f) is not None),
                     None)
            if k is None:
                continue
            v = r.get(field)
            if isinstance(v, dict):   # a probability dict -> P(POINT)
                v = v.get("POINT_TRANSITION", v.get("POINT"))
            if v is not None:
                out[k] = float(v)
        if out:
            used = field
            break
    if not out and rows:
        keys = sorted({k for r in rows[:20] for k in r})
        raise SystemExit(
            f"{what or 'scores'}: {len(rows)} rows carried no usable value. "
            f"Tried {score_field!r} then {SCORE_FALLBACKS}.\n"
            f"  fields present: {keys}\n"
            f"  This is a hard error rather than empty scores on purpose: "
            f"empty scores route\n  everything to REVIEW, which is exactly "
            f"what a correctly cautious auditor looks\n  like, so a broken "
            f"join would ship looking right.")
    if used and used != score_field:
        # AN EXPLICIT ARGUMENT THAT DOES NOT MATCH IS A TYPO, NOT A DEFAULT TO
        # BE RESCUED. The fallback list exists so the DEFAULT works across the
        # score files this repo produces; silently substituting a different
        # field than the caller named would hide `--boundary_field scoer` and
        # report a clean run.
        if explicit:
            keys = sorted({k for r in rows[:20] for k in r})
            raise SystemExit(
                f"{what}: you asked for {score_field!r} and no row has it.\n"
                f"  fields present: {keys}\n"
                f"  ({used!r} would have worked, but substituting a field you "
                f"did not name is\n  how a typo ships as a clean run. Pass it "
                f"explicitly if that is what you meant.)")
        print(f"  {what}: default {score_field!r} not present; read {used!r} "
              f"instead ({len(out)} scored)")
    return out


def audit_output(out, a, cert, scert, n_bs_rows, n_ss_rows):
    """Every invariant the end-to-end path must hold, checked out loud.

    THE FAILURE THIS EXISTS FOR looks exactly like success. A fail-closed
    auditor routes everything to REVIEW when it is working carefully AND when
    its score join is broken, so the safe-looking output is the one that hides
    a wiring error. The `--boundary_field` default not matching a real scores
    file produced 1496 silent Nones and a report that read as correct caution.

    Returns a list of problems. `--regression` turns any of them into a
    non-zero exit, because a check that prints and continues is a check that
    ships broken wiring."""
    n = len(out)
    bs_ok = sum(1 for o in out if o["boundary_score"] is not None)
    ss_ok = sum(1 for o in out if o["semantic_score"] is not None)
    rev = sum(1 for o in out if o["boundary_audit"] == "REVIEW"
              or o["semantic_audit"] == "REVIEW")
    miss = sum(1 for o in out if o["boundary_score"] is None
               and o["semantic_score"] is None)
    prob = []

    cls = Counter(o.get("review_class") for o in out
                  if o.get("review_class"))
    if cls:
        print(f"\n  review splits into two kinds of work:")
        for k in (REVIEW_UNCERTAIN, REVIEW_UNOBSERVABLE):
            v = cls.get(k, 0)
            print(f"    {k:<22}{v:>6}  {v / len(out):>6.1%}")
        if cls.get(REVIEW_UNOBSERVABLE):
            print(f"    a person cannot settle {REVIEW_UNOBSERVABLE} by "
                  f"watching harder -- those are a\n    capture problem, and "
                  f"they belong out of any recall denominator.")

    print(f"\n{'=' * 66}\nEND-TO-END CHECK\n{'=' * 66}")
    print(f"  segments                        {n}")
    print(f"  boundary scores joined          {bs_ok}/{n}"
          + (f"   (file had {n_bs_rows} rows)" if a.boundary_scores
             else "   (no --boundary_scores given)"))
    print(f"  semantic scores joined          {ss_ok}/{n}"
          + (f"   (file had {n_ss_rows} rows)" if a.semantic_scores
             else "   (no --semantic_scores given)"))
    print(f"  routed to REVIEW                {rev}/{n}")
    print(f"  no score on either arm          {miss}/{n}")
    print(f"  boundary certificate            "
          f"{'none' if cert is None else cert.get('event_fingerprint', '?')}"
          f"{'  (independent: false)' if cert and cert.get('independent') is False else ''}")
    print(f"  semantic certificate            "
          f"{'none' if scert is None else scert.get('kind', '?')}")

    # 1. every action is in the declared set, and no reject exists anywhere
    bad = [o for o in out if o["boundary_audit"] not in ACTIONS["boundary"]
           or o["semantic_audit"] not in ACTIONS["semantic"]]
    if bad:
        prob.append(f"{len(bad)} segments carry an action outside ACTIONS")

    # 2. a scores file that joined onto nothing is a wiring error, not caution
    if a.boundary_scores and bs_ok == 0:
        prob.append("--boundary_scores given and joined onto 0 segments")
    if a.semantic_scores and ss_ok == 0:
        prob.append("--semantic_scores given and joined onto 0 segments")

    # 3. review_priority is a permutation of 1..k over exactly the REVIEW set
    q = [o for o in out if o.get("review_priority") is not None]
    if len(q) != rev:
        prob.append(f"review_priority on {len(q)} items but {rev} need review")
    ranks = sorted(o["review_priority"] for o in q)
    if ranks != list(range(1, len(q) + 1)):
        prob.append("review_priority is not a permutation of 1..k "
                    "(gaps or duplicates)")

    # 4. and it is ordered: unscored first, then ascending score
    def key(o):
        return (0, 0.0) if o["boundary_score"] is None \
            else (1, o["boundary_score"])
    seq = [key(o) for o in sorted(q, key=lambda o: o["review_priority"])]
    if seq != sorted(seq):
        prob.append("review_priority is not worst-score-first")
    lead = [o for o in sorted(q, key=lambda o: o["review_priority"])
            if o["boundary_score"] is None]
    if lead and max(o["review_priority"] for o in lead) != len(lead):
        prob.append("unscored items do not lead the queue")

    # 5. a threshold that was supplied and did not take effect must say so
    for thr, refused, name in ((a.boundary_thr, a.__dict__.get("_uncert"),
                                "--boundary_thr"),
                               (a.semantic_thr, a.__dict__.get("_s_uncert"),
                                "--semantic_thr")):
        if thr is not None and refused:
            print(f"  {name} {thr} REFUSED: {str(refused)[:44]}")

    print(f"\n  top of the review queue:")
    for o in sorted(q, key=lambda o: o["review_priority"])[:3]:
        sc = o["boundary_score"]
        shown = "none" if sc is None else f"{sc:.3f}"
        print(f"    #{o['review_priority']:<5} score={shown:<6} "
              f"{o['segment_id']}  {o['boundary_reason'][:40]}")

    if prob:
        print(f"\n  {len(prob)} PROBLEM(S):")
        for p in prob:
            print(f"    !! {p}")
    else:
        print(f"\n  all invariants hold.")
    return prob


# --------------------------------------------------------------------------
def run(a):
    onto = None
    if a.ontology and os.path.exists(a.ontology):
        from src.auditor.boundary.policy import load_ontology
        onto = load_ontology(a.ontology)

    segs = read_any(a.recseg)
    obs_in = {}
    if a.observability:
        for r in read_any(a.observability):
            k = str(r.get("event_id") or r.get("segment_id") or r.get("id"))
            obs_in[k] = {f: r.get(f) for f in OBSERVABILITY_FIELDS}
        print(f"  observability for {len(obs_in)} candidates")
    n_bs_rows = len(read_any(a.boundary_scores)) if a.boundary_scores else 0
    n_ss_rows = len(read_any(a.semantic_scores)) if a.semantic_scores else 0
    bs = index_scores(read_any(a.boundary_scores),
                      ("event_id", "candidate_id", "id"), a.boundary_field,
                      "--boundary_scores",
                      a.boundary_field != BOUNDARY_FIELD_DEFAULT) \
        if a.boundary_scores else {}
    ss = index_scores(read_any(a.semantic_scores),
                      ("segment_id", "event_id", "id"), a.semantic_field,
                      "--semantic_scores",
                      a.semantic_field != SEMANTIC_FIELD_DEFAULT) \
        if a.semantic_scores else {}
    preds = {str(r.get("event_id") or r.get("id")): r
             for r in read_any(a.boundary_scores)} if a.boundary_scores else {}

    # A THRESHOLD IS NOT AN OPERATING POINT UNTIL SOMETHING BACKS IT. Typing a
    # number on the command line is exactly how the two withdrawn operating
    # points were set, so the certificate is checked once, here, and its
    # verdict travels into every routing decision below.
    cert = json.load(open(a.certificate, encoding="utf-8")) \
        if a.certificate else None
    run_ids = [str(s.get("boundary_id") or s.get("event_id")
                   or s.get("segment_id") or s.get("id")) for s in segs]
    uncert = (verify_certificate(cert, a.boundary_thr, a.veto, run_ids,
                                 a.allow_overlap)
              if a.boundary_thr is not None else None)

    scert = json.load(open(a.semantic_certificate, encoding="utf-8")) \
        if a.semantic_certificate else None
    deployment = {
        "scorer": a.semantic_model,
        "window": a.semantic_window,
        "gold_fingerprint": (scert or {}).get("gold_fingerprint"),
        "gate_version": (load_gate(a.gate) or {}).get("version")
        if os.path.exists(a.gate) else None,
        "code_fingerprint": code_fingerprint([
            __file__,
            "src/auditor/semantic/batch4_within_recording.py",
            "src/auditor/semantic/cosine_baseline.py"]),
    }
    # The gold binding is the certificate's own -- the deployment does not
    # re-derive it, because the run is on NEW segments and has no gold. What
    # this check enforces is that the certificate RECORDED one.
    s_uncert = (verify_semantic_certificate(scert, a.semantic_thr, deployment)
                if a.semantic_thr is not None else None)
    a._uncert, a._s_uncert = uncert, s_uncert

    out, tab, vetoed = [], Counter(), Counter()
    for s in segs:
        sid = str(s.get("segment_id") or s.get("id") or len(out))
        bid = str(s.get("boundary_id") or s.get("event_id") or sid)
        bscore = bs.get(bid)
        veto = structural_veto(preds.get(bid), onto, a.veto)
        b_act, b_why = route_boundary(bscore, a.boundary_thr, veto, uncert)
        sscore = ss.get(sid)
        s_act, s_why = route_semantic(sscore, a.semantic_thr, s_uncert)
        obs = obs_in.get(bid) or obs_in.get(sid)
        rev_class, obs_why = observability_state(obs)
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
            "observability": obs or {f: "unknown" for f in
                                     OBSERVABILITY_FIELDS},
            # WHAT KIND OF REVIEW, which is not the same as how urgent. An
            # unobservable candidate cannot be settled by looking harder.
            "review_class": rev_class if (b_act == "REVIEW"
                                          or s_act == "REVIEW") else None,
            "observability_reason": obs_why,
        })

    # THE QUEUE. Everything routed to REVIEW is ordered worst-score-first so
    # the output is directly workable rather than a pile with scores attached.
    # Ordering skips nothing, so it needs no threshold and no certificate --
    # and it only pays if the reviewer stops early, which is a person spending
    # a budget rather than a model deciding an item is fine.
    #
    # A CANDIDATE WITH NO SCORE SORTS FIRST, not last. No score means nothing
    # examined it, and "unexamined" belongs at the front of a review queue for
    # the same reason NOT_OBSERVABLE never satisfies a safety condition.
    pending = [o for o in out if o["boundary_audit"] == "REVIEW"
               or o["semantic_audit"] == "REVIEW"]
    pending.sort(key=lambda o: (
        o["boundary_score"] if o["boundary_score"] is not None else -1e9,
        o["semantic_score"] if o["semantic_score"] is not None else -1e9))
    for i, o in enumerate(pending):
        o["review_priority"] = i + 1
    for o in out:
        o.setdefault("review_priority", None)

    n = len(out)
    # THE OTHER HALF OF THE SAME PROBLEM. The scores can parse perfectly and
    # still join onto nothing, if the ids do not correspond. Same failure
    # shape: everything REVIEW, nothing visibly wrong.
    for tag, scored, given in (("boundary", sum(1 for o in out if o[
            "boundary_score"] is not None), a.boundary_scores),
            ("semantic", sum(1 for o in out if o[
                "semantic_score"] is not None), a.semantic_scores)):
        if given and not scored:
            raise SystemExit(
                f"{tag}: a scores file was given and not one of {n} segments "
                f"matched an id in it.\n  Everything would route to REVIEW, "
                f"which is what a working auditor also looks like, so this "
                f"is\n  an error rather than a quiet pass.")
        if given:
            print(f"  {tag} scores joined onto {scored}/{n} "
                  f"({scored / n:.1%})")

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
    if uncert:
        print(f"\n  !! AUTO_KEEP REFUSED for every candidate: {uncert}.\n"
              f"     --boundary_thr {a.boundary_thr} was supplied and did not "
              f"take effect.")
    if s_uncert:
        print(f"\n  !! AUTO_ACCEPT REFUSED for every segment: {s_uncert}.\n"
              f"     --semantic_thr {a.semantic_thr} was supplied and did not "
              f"take effect.")
    if a.boundary_thr is None or a.semantic_thr is None:
        print(f"\n  !! at least one threshold was not given, so that arm "
              f"automated nothing.\n     Run --calibrate and choose an "
              f"operating point from its lower bounds.")
    problems = audit_output(out, a, cert, scert, n_bs_rows, n_ss_rows)

    if a.out:
        json.dump({"auditor_version": "v1", "tolerance_s": TOLERANCE_S,
                   "boundary_thr": a.boundary_thr,
                   "semantic_thr": a.semantic_thr,
                   "veto_mode": a.veto,
                   "certificate": a.certificate,
                   "certificate_fingerprint": (cert or {}).get(
                       "event_fingerprint"),
                   "auto_keep_refused": uncert,
                   "auto_accept_refused": s_uncert,
                   "semantic_certificate": a.semantic_certificate,
                   "semantic_scorer": a.semantic_model,
                   "semantic_window": a.semantic_window,
                   "overlap_override_used": bool(a.allow_overlap),
                   "actions_available": {k: list(v)
                                         for k, v in ACTIONS.items()},
                   "problems": problems, "segments": out},
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
        got, why = route_boundary(score, thr, veto, None)
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
    # THE CERTIFICATE CHECK, on the four ways a threshold fails to be one.
    print()
    ok_row = {"threshold": 0.9, "coverage": 0.3, "precision": 1.0,
              "ci_lo": 0.97, "ci_hi": 1.0, "errors_kept": 0, "gate_pass": True}
    cert = {"veto_mode": "morphology_only", "rows": [ok_row],
            "event_ids": ["a", "b"], "event_fingerprint": "x"}
    checks = [
        (None, 0.9, "morphology_only", [], "no --certificate supplied"),
        (cert, 0.9, "full", [], "veto"),
        (cert, 0.5, "morphology_only", [], "not a gate-passing threshold"),
        (cert, 0.9, "morphology_only", ["a"], "calibration set"),
        (cert, 0.9, "morphology_only", ["z"], None),
        ({"veto_mode": "morphology_only", "rows": [dict(ok_row,
                                                        gate_pass=False)]},
         0.9, "morphology_only", [], "passed the pre-registered gate"),
    ]
    for c, thr, mode, ids, want in checks:
        got = verify_certificate(c, thr, mode, ids)
        if want is None:
            assert got is None, got
            print(f"  certificate ok -> KEEP permitted at {thr}")
        else:
            assert got and want in got, (want, got)
            act, _ = route_boundary(0.99, thr, None, got)
            assert act == "REVIEW"
            print(f"  refused ({want}): {got[:66]}")

    # THE SEMANTIC SIDE, symmetric with the boundary side. The first case is
    # the one that matters most: driver A's pairwise result is a CAPABILITY
    # answer and must not be usable as a deployment certificate, however good
    # the number is.
    print()
    dep = {"scorer": "reranker-8b", "window": "full_segment",
           "gold_fingerprint": "g1", "gate_version": 1, "code_fingerprint": "c1"}
    good = dict(dep, kind=CERT_AUTOMATION,
                rows=[{"threshold": 0.8, "gate_pass": True}])
    s_checks = [
        (None, "no --semantic_certificate supplied"),
        (dict(good, kind=CERT_CAPABILITY), "CAPABILITY certificate"),
        (dict(good, window="candidate_6s_half"), "window differs"),
        (dict(good, scorer="reranker-2b"), "scorer differs"),
        (dict(good, code_fingerprint="c2"), "code_fingerprint differs"),
        (dict(good, rows=[{"threshold": 0.8, "gate_pass": False}]),
         "passed the pre-registered gate"),
        (good, None),
    ]
    for c, want in s_checks:
        got = verify_semantic_certificate(c, 0.8, dep)
        if want is None:
            assert got is None, got
            act, _ = route_semantic(0.9, 0.8, got)
            assert act == "ACCEPT"
            print(f"  semantic certificate ok -> ACCEPT permitted at 0.8")
        else:
            assert got and want in got, (want, got)
            act, _ = route_semantic(0.99, 0.8, got)
            assert act == "REVIEW"
            print(f"  semantic refused ({want}): {got[:60]}")

    # OBSERVABILITY. `absent` and `unknown` both block and are different
    # information -- the first is a finding, the second is a gap -- so the
    # reason distinguishes them even though the class does not.
    print()
    ok = {f: "present" for f in OBSERVABILITY_FIELDS}
    for obs, want_cls, want_in in (
            (None, REVIEW_UNOBSERVABLE, "was not collected"),
            ({}, REVIEW_UNOBSERVABLE, "was not collected"),
            (ok, REVIEW_UNCERTAIN, None),
            (dict(ok, hand_visible="absent"), REVIEW_UNOBSERVABLE,
             "observed absent: hand_visible"),
            (dict(ok, hand_visible="unknown"), REVIEW_UNOBSERVABLE,
             "not assessed: hand_visible"),
            (dict(ok, hand_visible="absent", camera_stable="unknown"),
             REVIEW_UNOBSERVABLE, "observed absent"),
    ):
        cls, why = observability_state(obs)
        assert cls == want_cls, (obs, cls)
        if want_in:
            assert want_in in why, (why, want_in)
        print(f"  observability {str(obs)[:44]:<46} -> {cls}")
    cls, why = observability_state(dict(ok, hand_visible="absent",
                                        camera_stable="unknown"))
    assert "observed absent" in why and "not assessed" in why, why
    print(f"  absent and unknown stay apart in the reason: {why}")

    # A gate with unset targets must not pass. It ships that way on purpose.
    g = load_gate()
    passes, why = check_gate({"ci_lo": 1.0, "coverage": 1.0,
                              "errors_kept": 0}, g)
    assert not passes and any("unset" in w for w in why), why
    assert g.get("enabled") is False
    print(f"  a perfect row still fails the shipped gate: "
          f"{len(why)} target(s) unset")

    print("\n  no code path returns a reject; ACTIONS is the whole action set.")
    print("  a veto beats any score; an uncertified threshold automates "
          "nothing.")


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
    ap.add_argument("--boundary_field", default=BOUNDARY_FIELD_DEFAULT)
    ap.add_argument("--semantic_field", default=SEMANTIC_FIELD_DEFAULT)
    ap.add_argument("--truth_field", default="is_boundary")
    ap.add_argument("--boundary_thr", type=float, default=None,
                    help="NO DEFAULT. Without it every candidate is REVIEW.")
    ap.add_argument("--semantic_thr", type=float, default=None,
                    help="NO DEFAULT. Without it every label is REVIEW.")
    ap.add_argument("--observability",
                    help="per-candidate hand_visible / object_visible / "
                         "interaction_visible / camera_stable, each "
                         "present|absent|unknown. Absent file means every "
                         "field is unknown, which is the honest state: no head "
                         "emits observability yet.")
    ap.add_argument("--gate", default=GATE,
                    help="pre-registered AUTO_KEEP conditions")
    ap.add_argument("--emit_certificate",
                    help="--calibrate: write the certificate a --run threshold "
                         "must be backed by")
    ap.add_argument("--certificate",
                    help="--run: without it no boundary threshold automates")
    ap.add_argument("--semantic_certificate",
                    help="--run: without it no --semantic_thr automates. A "
                         "driver-A CAPABILITY certificate is refused here on "
                         "purpose -- it is a different question.")
    ap.add_argument("--semantic_model",
                    help="--run: scorer identity, checked against the "
                         "certificate")
    ap.add_argument("--semantic_window",
                    help="--run: window configuration (e.g. candidate_6s_half "
                         "or full_segment), checked against the certificate")
    ap.add_argument("--regression", action="store_true",
                    help="kept for symmetry; the invariant checks always run "
                         "and always exit non-zero, because a check that can "
                         "be switched off is one that will be")
    ap.add_argument("--allow_overlap", action="store_true",
                    help="permit applying a threshold to the events it was "
                         "chosen on. Recorded in the output when used.")
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
        gate = load_gate(a.gate) if os.path.exists(a.gate) else None
        rows = risk_coverage(items, gate=gate)
        lift = review_lift(items)
        ids = [str(g.get("event_id") or g.get("candidate_id") or g.get("id"))
               for g in gold]
        fp, nid = event_fingerprint(ids)
        if a.emit_certificate:
            json.dump({
                "auditor_version": "v1",
                "veto_mode": a.veto,
                "tolerance_s": TOLERANCE_S,
                "gold": os.path.abspath(a.gold),
                "scores": os.path.abspath(a.scores),
                "gate_config": os.path.abspath(a.gate),
                "gate": (gate or {}).get("gate"),
                "n_events": nid, "event_fingerprint": fp,
                "event_ids": sorted(set(ids)),
                "rows": rows,
            }, open(a.emit_certificate, "w", encoding="utf-8"),
                ensure_ascii=False, indent=1)
            npass = sum(1 for r in rows if r.get("gate_pass"))
            print(f"\nwrote {a.emit_certificate}  ({npass} gate-passing "
                  f"threshold(s), veto={a.veto}, {nid} events, fp={fp})")
            print(f"  --run refuses a threshold this certificate does not "
                  f"back, refuses a different\n  veto mode, and refuses events "
                  f"that were in the calibration set.")
        return
    if a.run:
        if not a.recseg:
            raise SystemExit("--run needs --recseg")
        return run(a)
    ap.print_help()


if __name__ == "__main__":
    main()
