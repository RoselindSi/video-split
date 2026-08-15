"""Is reorder_span's 0.730 the model's ceiling or the annotation's?

`reorder_span` is the lowest row anywhere in the 8B tables -- 0.730 against
0.862-1.000 everywhere else. But its ground truth IS the segment boundaries:
`A then B` is true only if A ended and B began where the annotator said. A
72-event boundary audit in this project put 34.7% of boundaries at annotation
error, and a wrong boundary makes the correct text false and scores a right
answer as wrong.

So the arm is split by whether the boundary BETWEEN the two segments was
human-confirmed, and both halves are reported. No GPU: the scores already
exist, only the subset changes.

    confirmed half clearly higher   the residual error is largely annotation,
                                    and 0.730 understates the model
    both halves equal               boundaries are not what limits it, and
                                    cross-segment temporal grounding is a real
                                    residual failure

THE BOUNDARY IN QUESTION IS THE INTERNAL ONE. A span runs [A.start, B.end] and
the benchmark row carries only those two. The boundary that decides whether the
span's order claim is true is A.end, which is recovered from the recseg files
by finding the segment that starts at the span's start.

A ZERO MATCH IS AN ERROR, NOT AN EMPTY HALF. `--exclude` once printed
"0 dropped from 0" in this project and the filter had matched nothing; a split
that silently puts every pair on one side would look like a finding.

Usage:
    python -m src.auditor.semantic.span_boundary_split \
        --scores /workspace/tr1/results/auditor/reorder_arms_8b.jsonl \
        --benchmark data/gold/reorder_arms_benchmark.jsonl \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --confirmed data/gold/<the frozen 72-event boundary audit>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np

from src.auditor.semantic.compose_supervision import resolve
from src.auditor.semantic.paired_null import boot_excess, load_extended, wins
from src.auditor.semantic.render_ontology_clips import get_segments

TIME_COLS = ("boundary_time", "corrected_time", "gt_time", "time", "t",
             "start", "boundary", "end")
KEEP_HINTS = ("tp", "true_positive", "correct", "confirmed", "ok", "keep",
              "yes")


def read_confirmed(paths, time_col, verdict_col, keep_values):
    """(recording_id, time) the audit confirmed, with the schema printed.

    The 72-event audit's column names are not known to this file, so what was
    read is shown rather than assumed -- a wrong column would silently produce
    an empty or a universal set, and both look like results."""
    out, seen_cols = defaultdict(list), Counter()
    for p in paths:
        # THREE FORMATS, because this project's gold is in all three. A .jsonl
        # handed to json.load raises on the second line, which would read as
        # "the file is broken" rather than "it is line-delimited".
        if p.lower().endswith(".csv"):
            rows = list(csv.DictReader(open(p, newline="",
                                            encoding="utf-8-sig")))
        elif p.lower().endswith(".jsonl"):
            rows = [json.loads(l) for l in open(p, encoding="utf-8-sig")
                    if l.strip()]
        else:
            blob = json.load(open(p, encoding="utf-8-sig"))
            rows = blob.get("events", blob if isinstance(blob, list)
                            else list(blob.values()))
        if not rows:
            continue
        print(f"  {p}: {len(rows)} rows, columns "
              f"{sorted(rows[0].keys())[:14]}")
        print(f"    first row: "
              f"{ {k: rows[0][k] for k in list(rows[0])[:8]} }")
        for r in rows:
            if not isinstance(r, dict):
                continue
            seen_cols.update(r.keys())
            rid = r.get("recording_id")
            if not rid:
                m = re.match(r"^(recording_\d+)", str(r.get("event_id") or ""))
                rid = m.group(1) if m else None
            tc = time_col or next((c for c in TIME_COLS if r.get(c)), None)
            if not rid or not tc or not str(r.get(tc, "")).strip():
                continue
            if verdict_col:
                v = str(r.get(verdict_col, "")).strip().lower()
                if keep_values and v not in keep_values:
                    continue
                if not keep_values and not any(h in v for h in KEEP_HINTS):
                    continue
            try:
                out[rid].append(float(r[tc]))
            except (TypeError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--confirmed", action="append", required=True)
    ap.add_argument("--time_col", help="auto-detected and printed if omitted")
    ap.add_argument("--verdict_col",
                    help="column saying whether the boundary was confirmed. "
                         "Omitted means every row in the file counts as "
                         "confirmed")
    ap.add_argument("--keep", action="append", default=[],
                    help="values of --verdict_col that count as confirmed")
    ap.add_argument("--kind", default="reorder_span")
    ap.add_argument("--tol_s", type=float, default=0.5,
                    help="how close a span's internal boundary must be to a "
                         "confirmed one to count as the same boundary")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    bench = [r for r in bench if r["kind"] == a.kind]
    if not bench:
        raise SystemExit(f"no {a.kind} pairs in {a.benchmark}")
    print(f"{len(bench)} {a.kind} pairs over "
          f"{len({r['recording_id'] for r in bench})} recordings")

    # The internal boundary: the end of the segment that starts where the span
    # starts. Recovered from recseg rather than stored, because the benchmark
    # row only carries the span's outer edges.
    inner = {}
    for path in resolve(a.recseg):
        blob = json.load(open(path, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid:
                continue
            for lab, st, en in ([str(x[0]), float(x[1]), float(x[2])]
                                for x in get_segments(r)[0]):
                inner[(rid, round(st, 3))] = en
    got = 0
    for p in bench:
        e = inner.get((p["recording_id"], round(float(p["start"]), 3)))
        p["_inner"] = e
        got += e is not None
    print(f"  internal boundary recovered for {got} of {len(bench)} pairs")
    if not got:
        raise SystemExit("no span's start matched a segment in the recseg "
                         "files; the split would put every pair on one side.")

    print(f"\nconfirmed boundaries:")
    conf = read_confirmed(a.confirmed, a.time_col, a.verdict_col,
                          {x.lower() for x in a.keep})
    n_conf = sum(len(v) for v in conf.values())
    print(f"  {n_conf} confirmed boundaries over {len(conf)} recordings")
    if not n_conf:
        raise SystemExit("read zero confirmed boundaries. Pass --time_col and "
                         "--verdict_col explicitly using the columns printed "
                         "above -- an empty set would send every pair to the "
                         "unconfirmed half and look like a finding.")

    lab = []
    for p in bench:
        t = p.get("_inner")
        ok = (t is not None and
              any(abs(t - c) <= a.tol_s
                  for c in conf.get(p["recording_id"], ())))
        lab.append(ok)
    lab = np.array(lab)
    print(f"  {int(lab.sum())} pairs sit on a confirmed boundary, "
          f"{int((~lab).sum())} do not")
    if lab.sum() == 0 or (~lab).sum() == 0:
        raise SystemExit("the split is degenerate; check --tol_s and the "
                         "columns above.")

    sc, _vid = load_extended(a.scores)
    pairings = sorted({k[0] for k in sc})
    nulls = [p for p in pairings if p != 0]

    def marg(pairing):
        out = []
        for p in bench:
            u = p["segment_uid"]
            x = sc.get((pairing, u, p["original"]))
            y = sc.get((pairing, u, p["counterfactual"]))
            out.append(np.nan if x is None or y is None else x - y)
        return np.array(out, float)

    t_m = marg(0)
    n_m = [marg(j) for j in nulls]
    rec = np.array([p["recording_id"] for p in bench], dtype=object)
    rng = np.random.default_rng(a.seed)

    print(f"\n  {'half':<26}{'n':>5}{'recs':>6}{'true':>8}{'null':>8}"
          f"{'excess':>9}{'excess 95%':>19}")
    for name, sel in (("boundary confirmed", lab),
                      ("boundary not confirmed", ~lab),
                      ("all", np.ones(len(bench), bool))):
        ok = sel & ~np.isnan(t_m)
        if not ok.any():
            continue
        t = float(np.mean(wins(t_m[ok])))
        n = float(np.nanmean(wins(np.concatenate([m[ok] for m in n_m]))))
        d = boot_excess(rec, ok, t_m, n_m, a.n_boot, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name:<26}{int(ok.sum()):>5}{len(set(rec[ok])):>6}"
              f"{t:>8.3f}{n:>8.3f}{t - n:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>19}")

    print(f"\n  A confirmed half clearly above the unconfirmed one means the "
          f"residual error is\n  largely annotation and 0.730 understates the "
          f"model. Two equal halves mean the\n  boundaries are not the limit, "
          f"and cross-segment temporal grounding is a real\n  residual "
          f"failure worth a model.")


if __name__ == "__main__":
    main()
