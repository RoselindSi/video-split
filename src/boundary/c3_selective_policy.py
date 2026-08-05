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


def passes(m, sel):
    return (np.isfinite(m["auto_keep_precision"])
            and m["auto_keep_precision"] >= sel["min_auto_keep_precision"]
            and m["auto_keep_coverage"] > 0
            and (not np.isfinite(m["sharp_false_reject_rate"])
                 or m["sharp_false_reject_rate"] <= sel["max_sharp_false_reject_rate"]))


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

    fams = cfg["families"] if a.select else [cfg["policy"]]
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
        sel = cfg["selection"]
        print(f"\nselection rule (from the config, not chosen while looking at "
              f"results):\n  auto_keep_precision >= {sel['min_auto_keep_precision']}, "
              f"sharp_false_reject_rate <= {sel['max_sharp_false_reject_rate']}, "
              f"then MAXIMISE automatic coverage")
        cands = []
        for fam in fams:
            grid = fam.get("grid", {})
            keys = sorted(grid)
            for combo in itertools.product(*(grid[k] for k in keys)):
                # underscore keys are documentation in the config; carrying
                # them into the frozen policy makes the printed rule unreadable
                pol = {**{k: v for k, v in fam.items()
                          if k != "grid" and not k.startswith("_")},
                       **dict(zip(keys, combo))}
                if pol["keep_above"] <= pol["reject_below"]:
                    continue
                m, _ = evaluate(rows, pol)
                cands.append((pol, m))
        print(f"  evaluated {len(cands)} threshold combinations across "
              f"{len(fams)} pre-listed families")
        ok = [(p, m) for p, m in cands if passes(m, sel)]
        print(f"  {len(ok)} satisfy the precision and false-reject constraints")
        if not ok:
            best = max(cands, key=lambda pm: pm[1]["auto_keep_precision"]
                       if np.isfinite(pm[1]["auto_keep_precision"]) else -1)
            print("\n  !! NOTHING satisfies the constraints. The closest by "
                  "auto-keep precision is shown below; do NOT relax the "
                  "constraints to make something pass -- that is the decision "
                  "the constraints exist to prevent.")
            m, tax = evaluate(rows, best[0], a.n_boot, a.seed)
            print(f"  closest policy: {best[0]}")
            print_report("CLOSEST (NOT SELECTED)", m, tax)
            report["selected"] = None
            report["closest"] = {"policy": best[0], "metrics": m}
        else:
            pol, _ = max(ok, key=lambda pm: pm[1]["automatic_coverage"])
            m, tax = evaluate(rows, pol, a.n_boot, a.seed)
            print(f"\nselected policy: {json.dumps(pol)}")
            print_report("DEVELOPMENT (145 + batch3)", m, tax)
            report["selected"] = {"policy": pol, "metrics": m, "taxonomy": tax}
            if a.out_config:
                frozen = {"frozen": True, "policy": pol,
                          "reliability_column": cfg["reliability_column"],
                          "selection": sel, "selected_on": a.events,
                          "n_dev_events": len(rows),
                          "source_config": a.config,
                          "source_config_sha256": hashlib.sha256(
                              open(a.config, "rb").read()).hexdigest()[:16],
                          "dev_metrics": m}
                with open(a.out_config, "w", encoding="utf-8") as f:
                    json.dump(frozen, f, ensure_ascii=False, indent=2, default=str)
                print(f"\nwrote frozen config -> {a.out_config}")
                print("  COMMIT THIS before generating any held-out scores. Its "
                      "dev metrics are recorded inside it, so a later run can be "
                      "checked against what was promised rather than against "
                      "memory.")
    else:
        pol = cfg["policy"]
        m, tax = evaluate(rows, pol, a.n_boot, a.seed)
        print(f"\napplying frozen policy: {json.dumps(pol)}")
        print(f"  frozen from {cfg.get('source_config')} "
              f"(sha {cfg.get('source_config_sha256')}), selected on "
              f"{cfg.get('n_dev_events')} development events")
        print_report("HELD-OUT (one shot, no tuning)", m, tax)
        dev = cfg.get("dev_metrics") or {}
        if dev:
            print(f"\n  vs development, the numbers that were promised:")
            for k in ("auto_keep_precision", "auto_keep_coverage",
                      "same_action_false_accept_rate", "sharp_false_reject_rate",
                      "review_rate"):
                if k in dev and np.isfinite(m.get(k, float("nan"))):
                    print(f"    {k:<32} dev {dev[k]:.3f} -> held-out {m[k]:.3f} "
                          f"({m[k] - dev[k]:+.3f})")
        report["applied"] = {"policy": pol, "metrics": m, "taxonomy": tax,
                             "dev_metrics": dev}

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
