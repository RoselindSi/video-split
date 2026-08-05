"""C3 decision layer: a configurable three-way selective policy, kept SEPARATE
from c3_local_eval.py so "does the representation carry signal" and "is the
decision rule deployable" are never answered by the same number.

The C3 evaluation established the representation question well enough to stop
working on it: the local branch lifts ranking everywhere it was measured
(+0.034 to +0.076 across two event sets and two coverage strata) and reliably
destroys some correct boundaries while doing it (true-positive harms 8, 7, 4,
1). Forcing that branch to overrule P1 on every event is what fails; the
remaining question is when to trust it, and when to hand the event to a human
instead.

So every event lands in exactly one of:

    AUTO_KEEP      accepted as a boundary without review
    AUTO_REJECT    rejected as same-action without review
    REVIEW         handed to a human

with a REASON attached, because "review rate 30%" is not actionable while
"22% low_local_reliability, 8% global_local_disagreement" is.

TWO MODES, and the separation is the point:

  --select   searches a SMALL, PRE-LISTED grid inside one policy family on
             development data, applies the pre-registered selection rule, and
             writes a frozen config. Development data only.
  --apply    runs a frozen config once and reports. No search, no thresholds
             derived from the data it is scoring. It refuses to run a config
             that is not marked frozen, so a held-out set cannot be scored
             with a policy that has not been committed to first.

The selection rule is precision-first and is read from the config rather than
chosen while looking at results: satisfy a minimum auto-keep precision, keep
the sharp false-reject rate under a cap, and subject to those, maximise
automatic coverage.

CLEAN-BINARY AND FULL-TAXONOMY ARE SCORED SEPARATELY. The clean binary subset
(sharp_visible_transition vs same_action_internal_motion) measures the
verifier. Everything else -- gradual_phase_transition, visibility_or_offscreen,
camera_or_viewpoint_shift, annotation_convention, ambiguous -- should not be
forced through a binary decision at all, and what matters for those is the
fraction routed to REVIEW. A policy can look excellent on the clean subset
while automatically accepting a pile of camera-motion candidates, and only the
second table shows it.

Usage:
    # develop and freeze, on 145 + batch3 only
    python -m src.boundary.c3_selective_policy --select \
        --events /workspace/tr1/results/hal/c3/local_events.csv \
        --events /workspace/tr1/results/hal/c3/local_events_batch3.csv \
        --config configs/c3_selective_policy_v1.json \
        --out_config configs/c3_selective_policy_v1.frozen.json \
        --out /workspace/tr1/results/hal/c3/policy_dev.json

    # one shot on the held-out set, no tuning
    python -m src.boundary.c3_selective_policy --apply \
        --events /workspace/tr1/results/hal/c3/local_events_batch4.csv \
        --config configs/c3_selective_policy_v1.frozen.json \
        --out /workspace/tr1/results/hal/c3/policy_batch4.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import Counter

import numpy as np

KEEP, REJECT, REVIEW = "AUTO_KEEP", "AUTO_REJECT", "REVIEW"
CLEAN_SUBTYPES = ("sharp_visible_transition", "same_action_internal_motion")


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the small counts and the near-1
    proportions this report lives at, where the normal approximation puts the
    upper bound above 1 and the lower bound below 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def load_events(paths, score_cols, reliability_col, subtype_col="subtype"):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                def _f(k, default=float("nan")):
                    v = r.get(k, "")
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return default
                e = {"event_id": r["event_id"], "recording_id": r["recording_id"],
                     "source": os.path.basename(p),
                     "y": int(r["y"]) if r.get("y") not in (None, "") else None,
                     "subtype": r.get(subtype_col) or r.get("dev_pair_subtype") or "",
                     "reliability": _f(reliability_col, 1.0)}
                for c in score_cols:
                    e[c] = _f(c)
                rows.append(e)
    return rows


def decide(e, pol):
    """-> (decision, reason). Reliability is checked FIRST: a policy that
    reads a local score it has already decided is untrustworthy is not a
    selective policy."""
    fam = pol["family"]
    if e["reliability"] < pol.get("min_reliability", 0.0):
        return REVIEW, "low_local_reliability"
    hi, lo = pol["keep_above"], pol["reject_below"]
    if fam in ("threshold", "fused"):
        s = e.get(pol["score"], float("nan"))
        if not np.isfinite(s):
            return REVIEW, "missing_score"
        if s >= hi:
            return KEEP, "high_confidence_positive"
        if s <= lo:
            return REJECT, "high_confidence_negative"
        return REVIEW, "insufficient_margin"
    if fam == "cascade":
        # Stage 1 is the agreement rule unchanged: both branches must agree at
        # confident scores. Stage 2 only sees what Stage 1 left in the middle,
        # and only at high reliability -- so the fused score never overrules a
        # confident P1, it only decides events that would otherwise have been
        # REVIEW. That is the shape the evaluation supports: fused improves
        # ranking but destroyed 8 correct boundaries on the clean-145 when it
        # was allowed to decide everything.
        a = e.get(pol["score_a"], float("nan"))
        b = e.get(pol["score_b"], float("nan"))
        fz = e.get(pol["score_fused"], float("nan"))
        if not (np.isfinite(a) and np.isfinite(b)):
            return REVIEW, "missing_score"
        if a >= hi and b >= hi:
            return KEEP, "stage1_consensus_positive"
        if a <= lo and b <= lo:
            return REJECT, "stage1_consensus_negative"
        if (a >= hi and b <= lo) or (a <= lo and b >= hi):
            return REVIEW, "global_local_disagreement"
        if e["reliability"] >= pol.get("stage2_min_reliability", 1.01) and np.isfinite(fz):
            if fz >= pol.get("stage2_keep_above", 2.0):
                return KEEP, "stage2_fused_positive"
            if fz <= pol.get("stage2_reject_below", -1.0):
                return REJECT, "stage2_fused_negative"
        return REVIEW, "insufficient_margin"

    if fam == "agreement":
        a = e.get(pol["score_a"], float("nan"))
        b = e.get(pol["score_b"], float("nan"))
        if not (np.isfinite(a) and np.isfinite(b)):
            return REVIEW, "missing_score"
        if a >= hi and b >= hi:
            return KEEP, "high_confidence_positive"
        if a <= lo and b <= lo:
            return REJECT, "high_confidence_negative"
        if (a >= hi and b <= lo) or (a <= lo and b >= hi):
            return REVIEW, "global_local_disagreement"
        return REVIEW, "insufficient_margin"
    raise SystemExit(f"unknown policy family {fam!r}")


def grouped_bootstrap_metric(rows, fn, n_boot=2000, seed=0):
    """CI by resampling RECORDINGS, matching every other CI in this project.
    Wilson is reported alongside for the precision metrics: it covers binomial
    noise at a fixed event set, while the bootstrap also covers the fact that
    a different draw of recordings would give a different event set."""
    by = {}
    for r in rows:
        by.setdefault(r["recording_id"], []).append(r)
    keys = sorted(by)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        s = [x for k in rng.choice(keys, len(keys), replace=True) for x in by[k]]
        v = fn(s)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(rows, pol, n_boot=0, seed=0):
    for e in rows:
        e["decision"], e["reason"] = decide(e, pol)
    def _is_clean(e):
        return e["subtype"] in CLEAN_SUBTYPES or e["subtype"] == ""
    clean = [e for e in rows if _is_clean(e)]
    other = [e for e in rows if not _is_clean(e)]
    n = len(clean)
    keep = [e for e in clean if e["decision"] == KEEP]
    rej = [e for e in clean if e["decision"] == REJECT]
    rev = [e for e in clean if e["decision"] == REVIEW]
    pos = [e for e in clean if e["y"] == 1]
    neg = [e for e in clean if e["y"] == 0]

    kp = sum(1 for e in keep if e["y"] == 1)
    rp = sum(1 for e in rej if e["y"] == 0)
    m = {
        "n_clean": n, "n_positive": len(pos), "n_negative": len(neg),
        "n_auto_keep": len(keep), "n_auto_reject": len(rej),
        "sharp_false_reject_count": sum(1 for e in pos if e["decision"] == REJECT),
        "same_action_false_accept_count": sum(1 for e in neg if e["decision"] == KEEP),
        "auto_keep_coverage": len(keep) / n if n else float("nan"),
        "auto_keep_precision": kp / len(keep) if keep else float("nan"),
        "auto_keep_precision_wilson": wilson(kp, len(keep)),
        "false_keep_count": len(keep) - kp,
        "same_action_false_accept_rate":
            sum(1 for e in neg if e["decision"] == KEEP) / len(neg) if neg else float("nan"),
        "auto_reject_coverage": len(rej) / n if n else float("nan"),
        "auto_reject_precision": rp / len(rej) if rej else float("nan"),
        "auto_reject_precision_wilson": wilson(rp, len(rej)),
        "sharp_false_reject_rate":
            sum(1 for e in pos if e["decision"] == REJECT) / len(pos) if pos else float("nan"),
        "review_rate": len(rev) / n if n else float("nan"),
        "automatic_coverage": (len(keep) + len(rej)) / n if n else float("nan"),
        "sharp_recall_after_automatic":
            sum(1 for e in pos if e["decision"] != REJECT) / len(pos) if pos else float("nan"),
    }
    for reason, c in Counter(e["reason"] for e in rev).items():
        m[f"abstain_{reason}_rate"] = c / n if n else float("nan")

    if n_boot and keep:
        def kprec(s):
            k = [e for e in s if e["decision"] == KEEP and
                 (e["subtype"] in CLEAN_SUBTYPES or e["subtype"] == "")]
            return (sum(1 for e in k if e["y"] == 1) / len(k)) if k else None
        m["auto_keep_precision_bootstrap"] = grouped_bootstrap_metric(rows, kprec, n_boot, seed)

    tax = {}
    if other:
        for st in sorted(set(e["subtype"] for e in other)):
            g = [e for e in other if e["subtype"] == st]
            tax[st] = {"n": len(g),
                       "review_rate": sum(1 for e in g if e["decision"] == REVIEW) / len(g),
                       "auto_keep_rate": sum(1 for e in g if e["decision"] == KEEP) / len(g),
                       "auto_reject_rate": sum(1 for e in g if e["decision"] == REJECT) / len(g)}
    return m, tax


def passes(m, sel, rule="frontier"):
    """rule='frontier': the constraints as written, which by construction lands
    on the edge of the feasible region -- the selected point spends both safety
    budgets exactly and has no event-level slack.

    rule='one_error_buffered': the same constraints, but the policy must still
    satisfy them after ONE MORE error of each kind. Expressed in event COUNTS
    rather than by tightening the rates, because that is the thing actually
    being worried about ("would one additional mistake break it?") and because
    a rate expressed as a rounded count is what the constraint reduces to at
    these sample sizes. Tightening to, say, precision >= 0.97 instead would be
    an arbitrary number; this one is derived.

    Not implemented as a Wilson lower bound: requiring the 95% lower bound to
    clear 0.95 turns "choose a policy expected to be 95% precise" into
    "certify from 313 development events that it IS", which needs 73 out of 73
    consecutive correct auto-keeps before the bound even reaches 0.95 (checked:
    61/61 gives 0.9408, 73/73 gives 0.9500). At n_auto_keep = 61 with 3 errors
    it is unreachable, so it would return 'no solution' rather than a usable
    policy."""
    if not np.isfinite(m["auto_keep_precision"]) or m["n_auto_keep"] <= 0:
        return False
    max_fk = math.floor(m["n_auto_keep"] * (1 - sel["min_auto_keep_precision"]))
    max_fr = math.floor(m["n_positive"] * sel["max_sharp_false_reject_rate"])
    if rule == "one_error_buffered":
        max_fk -= 1
        max_fr -= 1
    return (m["false_keep_count"] <= max_fk
            and m["sharp_false_reject_count"] <= max_fr)



def enumerate_policies(fams):
    """Every (family, threshold) combination in the pre-listed grids."""
    out = []
    for fam in fams:
        grid = fam.get("grid", {})
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            pol = {**{k: v for k, v in fam.items()
                      if k != "grid" and not k.startswith("_")},
                   **dict(zip(keys, combo))}
            if pol["keep_above"] > pol["reject_below"]:
                out.append(pol)
    return out


def select_one(rows, pols, sel, rule):
    """-> (policy, metrics) or (None, closest_metrics). Ties on coverage go to
    the higher auto-keep precision so the choice is deterministic rather than
    dependent on grid iteration order."""
    scored = [(p, evaluate(rows, p)[0]) for p in pols]
    ok = [(p, m) for p, m in scored if passes(m, sel, rule)]
    if not ok:
        closest = max(scored, key=lambda pm: pm[1]["auto_keep_precision"]
                      if np.isfinite(pm[1]["auto_keep_precision"]) else -1)
        return None, closest
    return max(ok, key=lambda pm: (pm[1]["automatic_coverage"],
                                   pm[1]["auto_keep_precision"]))


def nested_selection_diagnostic(rows, pols, sel, rule, k=5, seed=0):
    """How much of the selected policy's development performance is selection
    optimism?

    Outer grouped folds over RECORDINGS. Within each, the whole selection is
    re-run on the training recordings only, and the policy it picks is scored
    on the held-out recordings. This does NOT choose a policy -- it estimates
    how far a frontier point chosen this way falls when it meets recordings it
    was not selected on, which is exactly the question Batch4 will answer once
    and cannot answer twice. If the nested numbers already breach the
    constraints, the risk of a Batch4 breach has evidence behind it rather than
    being a hunch."""
    recs = sorted({r["recording_id"] for r in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(recs)
    folds = [set(recs[i::k]) for i in range(k)]
    out = []
    for i, f in enumerate(folds):
        tr = [r for r in rows if r["recording_id"] not in f]
        te = [r for r in rows if r["recording_id"] in f]
        if len(te) < 10 or len({r["y"] for r in te}) < 2:
            continue
        pol, res = select_one(tr, pols, sel, rule)
        if pol is None:
            out.append({"fold": i, "selected": None,
                        "note": "nothing satisfied the constraints on this "
                                "fold's training recordings"})
            continue
        m, _ = evaluate(te, pol)
        out.append({"fold": i, "selected": pol, "held_out": m})
    return out

def print_report(label, m, tax):
    print(f"\n=== {label} ===")
    print(f"  clean binary events: {m['n_clean']} ({m['n_positive']}+ / {m['n_negative']}-)")
    lo, hi = m["auto_keep_precision_wilson"]
    print(f"  AUTO_KEEP    coverage {m['auto_keep_coverage']:.3f}  "
          f"precision {m['auto_keep_precision']:.3f} "
          f"[Wilson {lo:.3f}, {hi:.3f}]  false keeps {m['false_keep_count']}")
    if "auto_keep_precision_bootstrap" in m:
        b = m["auto_keep_precision_bootstrap"]
        print(f"               grouped-bootstrap CI over recordings [{b[0]:.3f}, {b[1]:.3f}]")
    lo, hi = m["auto_reject_precision_wilson"]
    print(f"  AUTO_REJECT  coverage {m['auto_reject_coverage']:.3f}  "
          f"precision {m['auto_reject_precision']:.3f} [Wilson {lo:.3f}, {hi:.3f}]")
    print(f"  REVIEW       rate {m['review_rate']:.3f}")
    for k in sorted(k for k in m if k.startswith("abstain_")):
        print(f"      {k[8:]:<34} {m[k]:.3f}")
    print(f"  same-action FALSE ACCEPT rate  {m['same_action_false_accept_rate']:.3f}  "
          f"<- the primary safety number")
    print(f"  sharp FALSE REJECT rate        {m['sharp_false_reject_rate']:.3f}")
    print(f"  sharp recall after automatic   {m['sharp_recall_after_automatic']:.3f}")
    print(f"  automatic coverage             {m['automatic_coverage']:.3f}")
    if tax:
        print(f"\n  --- non-clean-binary taxonomy (should mostly be REVIEW) ---")
        print(f"  {'subtype':<32} {'n':>4} {'review':>8} {'keep':>7} {'reject':>7}")
        flagged = 0
        for st, g in sorted(tax.items(), key=lambda kv: -kv[1]["n"]):
            flag = "  !!" if g["auto_keep_rate"] > 0.10 else ""
            flagged += bool(flag)
            print(f"  {st:<32} {g['n']:>4} {g['review_rate']:>8.3f} "
                  f"{g['auto_keep_rate']:>7.3f} {g['auto_reject_rate']:>7.3f}{flag}")
        if flagged:
            print("  !! marks a class being AUTOMATICALLY ACCEPTED at more than "
                  "10% despite not being a clean boundary decision at all")
    else:
        print("\n  (no non-clean-binary events present: this input cannot show "
              "whether camera/offscreen/gradual candidates are being auto-accepted)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", action="append", required=True,
                    help="c3_local_eval --dump_events CSV(s). Repeatable; "
                         "development uses 145 + batch3 together.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--select", action="store_true",
                    help="search the config's pre-listed grid on THIS data and "
                         "freeze the winner. Development data only.")
    ap.add_argument("--apply", action="store_true",
                    help="run a frozen config once. No search.")
    ap.add_argument("--out_config", help="where to write the frozen config (--select)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--dump_decisions")
    a = ap.parse_args()
    if a.select == a.apply:
        raise SystemExit("choose exactly one of --select / --apply")

    cfg = json.load(open(a.config, encoding="utf-8"))
    if a.apply and not cfg.get("frozen"):
        raise SystemExit(
            f"{a.config} is not marked frozen. --apply exists to score data that "
            f"took no part in choosing the thresholds, so it will not run a "
            f"config that has not been committed to first. Run --select on the "
            f"development sets, commit the frozen config, then apply it.")
    if a.select and cfg.get("frozen"):
        raise SystemExit(f"{a.config} is already frozen; re-selecting on it would "
                         f"discard the point of freezing it.")

    # A frozen config holds one entry per ROLE, so the score columns to load
    # are the union across roles -- reading only the primary's would drop the
    # column the secondary needs and score it on NaNs.
    fams = cfg["families"] if a.select else [
        e["policy"] for e in cfg["policies"].values() if e]
    cols = sorted({c for f in fams for k in ("score", "score_a", "score_b")
                   if (c := f.get(k))})
    rows = load_events(a.events, cols, cfg["reliability_column"])
    print(f"{len(rows)} events from {len(a.events)} file(s), "
          f"{len(set(r['recording_id'] for r in rows))} recordings")
    print(f"  by source: {dict(Counter(r['source'] for r in rows))}")
    print(f"  reliability column {cfg['reliability_column']!r}: "
          f"median {np.median([r['reliability'] for r in rows]):.3f}")
    missing = [c for c in cols if all(not np.isfinite(r[c]) for r in rows)]
    if missing:
        raise SystemExit(f"score column(s) absent from every input row: {missing}. "
                         f"Available: {sorted(set(next(iter(rows)).keys()))}")

    n_sub = sum(1 for r in rows if r["subtype"])
    if n_sub == 0:
        print("  !! NO subtype column in the input: every event is being treated "
              "as clean-binary, so the taxonomy table cannot show whether "
              "camera/offscreen/gradual candidates are auto-accepted. Re-dump "
              "with a c3_local_eval that writes the subtype column.")
    elif n_sub < len(rows):
        print(f"  !! {len(rows) - n_sub} events have no subtype and default to "
              f"clean-binary")

    report = {"config": a.config, "n_events": len(rows),
              "sources": dict(Counter(r["source"] for r in rows))}

    if a.select:
        pols = enumerate_policies(fams)
        print(f"\n{len(pols)} threshold combinations across {len(fams)} pre-listed "
              f"families")
        frozen_policies = {}
        for role in cfg["roles"]:
            sel, rule = role["selection"], role["rule"]
            print(f"\n{'#' * 72}\n# ROLE {role['role']}  (rule: {rule})\n"
                  f"#   auto_keep_precision >= {sel['min_auto_keep_precision']}, "
                  f"sharp_false_reject_rate <= {sel['max_sharp_false_reject_rate']}"
                  + ("\n#   AND still satisfied after one additional error of each "
                     "kind" if rule == "one_error_buffered" else "")
                  + f"\n#   then MAXIMISE automatic coverage\n{'#' * 72}")
            pol, res = select_one(rows, pols, sel, rule)
            if pol is None:
                m, tax = evaluate(rows, res[0], a.n_boot, a.seed)
                print("  !! NOTHING satisfies this role's constraints. Closest by "
                      "auto-keep precision shown; do NOT relax the constraints to "
                      "make something pass -- that is the decision they exist to "
                      "prevent.")
                print(f"  closest: {json.dumps(res[0])}")
                print_report(f"CLOSEST FOR {role['role']} (NOT SELECTED)", m, tax)
                frozen_policies[role["role"]] = None
                continue
            m, tax = evaluate(rows, pol, a.n_boot, a.seed)
            print(f"selected: {json.dumps(pol)}")
            print_report(f"DEVELOPMENT ({role['role']})", m, tax)
            max_fk = math.floor(m["n_auto_keep"] * (1 - sel["min_auto_keep_precision"]))
            max_fr = math.floor(m["n_positive"] * sel["max_sharp_false_reject_rate"])
            slack_k = max_fk - m["false_keep_count"]
            slack_r = max_fr - m["sharp_false_reject_count"]
            print(f"  EVENT-LEVEL SLACK: {slack_k} more false keep(s) and "
                  f"{slack_r} more false reject(s) before a constraint breaks "
                  f"({m['false_keep_count']}/{max_fk} and "
                  f"{m['sharp_false_reject_count']}/{max_fr} used)")
            if slack_k <= 0 or slack_r <= 0:
                print("    ^ ZERO slack on at least one constraint. This is a "
                      "frontier point: maximising coverage under active "
                      "constraints always lands here, and frontier points are "
                      "the least transportable part of the feasible region. A "
                      "breach on held-out data is plausible and must NOT trigger "
                      "retuning.")
            frozen_policies[role["role"]] = {
                "policy": pol, "role": role["role"],
                "selection_rule": role.get("_rule_text", rule),
                "dev_metrics": m, "dev_taxonomy": tax,
                "dev_counts": {"auto_keep": m["n_auto_keep"],
                               "false_keep": m["false_keep_count"],
                               "sharp": m["n_positive"],
                               "sharp_false_reject": m["sharp_false_reject_count"]},
                "event_level_margin": {"additional_false_keep_tolerated": slack_k,
                                       "additional_false_reject_tolerated": slack_r},
                "expected_external_behavior": role.get("expected_external_behavior", ""),
            }

        nd = cfg.get("nested_diagnostic")
        nested = {}
        if nd:
            print(f"\n{'=' * 72}\nNESTED SELECTION DIAGNOSTIC (development only, "
                  f"selects nothing)\n{'=' * 72}")
            print("  Re-runs the ENTIRE selection inside grouped training folds and "
                  "scores the chosen policy on held-out recordings. It measures how "
                  "far a frontier point falls on recordings it was not selected on "
                  "-- the question Batch4 answers once and cannot answer twice.")
            for role in cfg["roles"]:
                res = nested_selection_diagnostic(rows, pols, role["selection"],
                                                  role["rule"], nd.get("folds", 5),
                                                  nd.get("seed", 0))
                nested[role["role"]] = res
                print(f"\n  {role['role']}")
                print(f"    {'fold':>4} {'keepP':>7} {'keepN':>6} {'sharpFR':>8} "
                      f"{'autoCov':>8} {'review':>7}")
                kp, fr = [], []
                for r in res:
                    if r.get("selected") is None:
                        print(f"    {r['fold']:>4}  {r.get('note', 'no policy')}")
                        continue
                    h = r["held_out"]
                    kp.append(h["auto_keep_precision"])
                    fr.append(h["sharp_false_reject_rate"])
                    print(f"    {r['fold']:>4} {h['auto_keep_precision']:>7.3f} "
                          f"{h['n_auto_keep']:>6} {h['sharp_false_reject_rate']:>8.3f} "
                          f"{h['automatic_coverage']:>8.3f} {h['review_rate']:>7.3f}")
                kp = [x for x in kp if np.isfinite(x)]
                fr = [x for x in fr if np.isfinite(x)]
                if kp:
                    nb = sum(1 for x in kp
                             if x < role["selection"]["min_auto_keep_precision"])
                    nr = sum(1 for x in fr
                             if x > role["selection"]["max_sharp_false_reject_rate"])
                    print(f"    median held-out auto-keep precision {np.median(kp):.3f}; "
                          f"{nb}/{len(kp)} folds below the precision constraint, "
                          f"{nr}/{len(fr)} above the false-reject constraint")
                    if nb or nr:
                        print("    ^ the constraints are ALREADY breached under "
                              "nested selection on development data, so a Batch4 "
                              "breach has evidence behind it rather than being a "
                              "prediction")

        report["selected"] = frozen_policies
        report["nested_diagnostic"] = nested
        if a.out_config and any(frozen_policies.values()):
            frozen = {"frozen": True, "policies": frozen_policies,
                      "reliability_column": cfg["reliability_column"],
                      "roles": cfg["roles"], "selected_on": a.events,
                      "n_dev_events": len(rows), "source_config": a.config,
                      "source_config_sha256": hashlib.sha256(
                          open(a.config, "rb").read()).hexdigest()[:16],
                      "nested_diagnostic": nested,
                      "primary_role": cfg["roles"][0]["role"],
                      "_reporting_note": "On held-out data report COUNTS (e.g. "
                                         "31/33) with Wilson and grouped-bootstrap "
                                         "intervals, not a bare PASS/FAIL at 0.95: "
                                         "at the auto-keep volume this policy "
                                         "produces, one event moves precision by "
                                         "about 3 points. Report BOTH roles; do not "
                                         "pick the better one afterwards."}
            with open(a.out_config, "w", encoding="utf-8") as f:
                json.dump(frozen, f, ensure_ascii=False, indent=2, default=str)
            print(f"\nwrote frozen config -> {a.out_config}")
            print("  COMMIT THIS before generating any held-out scores.")
    else:
        pols = cfg["policies"]
        print(f"\nfrozen from {cfg.get('source_config')} "
              f"(sha {cfg.get('source_config_sha256')}), selected on "
              f"{cfg.get('n_dev_events')} development events")
        print(f"primary role: {cfg.get('primary_role')}")
        applied = {}
        for role, entry in pols.items():
            if entry is None:
                print(f"\n{role}: no policy was frozen for this role")
                continue
            pol = entry["policy"]
            m, tax = evaluate(rows, pol, a.n_boot, a.seed)
            print(f"\n{'#' * 72}\n# {role}"
                  + ("  (PRIMARY)" if role == cfg.get("primary_role") else "  (secondary)")
                  + f"\n{'#' * 72}")
            print(f"policy: {json.dumps(pol)}")
            if entry.get("expected_external_behavior"):
                print(f"pre-registered expectation: {entry['expected_external_behavior']}")
            print_report("HELD-OUT (one shot, no tuning)", m, tax)
            dev = entry.get("dev_metrics") or {}
            if dev:
                print("\n  vs development, the numbers promised before this ran:")
                for k in ("auto_keep_precision", "auto_keep_coverage",
                          "same_action_false_accept_rate", "sharp_false_reject_rate",
                          "review_rate", "automatic_coverage"):
                    if k in dev and np.isfinite(m.get(k, float("nan"))):
                        print(f"    {k:<32} dev {dev[k]:.3f} -> held-out {m[k]:.3f} "
                              f"({m[k] - dev[k]:+.3f})")
                print(f"    auto-keep counts: dev "
                      f"{dev.get('n_auto_keep', 0) - dev.get('false_keep_count', 0)}"
                      f"/{dev.get('n_auto_keep', 0)} -> held-out "
                      f"{m['n_auto_keep'] - m['false_keep_count']}/{m['n_auto_keep']}")
            applied[role] = {"policy": pol, "metrics": m, "taxonomy": tax,
                             "dev_metrics": dev}
        report["applied"] = applied
        print("\nReport both roles as they stand. Choosing whichever performed "
              "better here would convert a held-out test into a selection step.")

    if a.dump_decisions:
        with open(a.dump_decisions, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "recording_id", "source", "y", "subtype",
                        "reliability", "decision", "reason"] + cols)
            for e in rows:
                w.writerow([e["event_id"], e["recording_id"], e["source"], e["y"],
                            e["subtype"], f"{e['reliability']:.3f}",
                            e.get("decision", ""), e.get("reason", "")]
                           + [f"{e[c]:.6f}" if np.isfinite(e[c]) else "" for c in cols])
        print(f"\nwrote {a.dump_decisions}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
