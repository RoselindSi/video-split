"""EXPLORATORY: does the detector fire where a human says there is no boundary?

THE POOL IS BOUNDARY-ENRICHED AND NOTHING HERE IS CONFIRMATORY. These 69
events come from audit_188_gold_v2, sampled on boundary error categories, so
an event is in the pool partly because of where the detector fired or failed
to. Every number below is descriptive of THIS pool. The 45-event peak-blind
alignment gold stays separate and is not touched.

WHAT MAKES THE COMPARISON DEFENSIBLE ANYWAY. Selection acts on the positive
and the negative arms alike -- both are events the boundary audit picked -- so
a WITHIN-POOL contrast is the one form that does not simply restate the
sampling. The question is not "how often does the detector fire near a
negative" in absolute terms; it is whether that rate differs from the rate
near a positive in the same pool.

TWO KINDS OF NEGATIVE, and this is the first data in the project that has
either:

    no_action_change    12  nothing changed here at all
    phase_change_only    8  something visibly changed -- an amplitude shift
                            inside a continuous wipe, a flip-cycle phase --
                            and the frozen ontology says it is not a task
                            boundary

They are not the same mistake. Firing in continuous idle is a plain false
positive. Firing at a phase change is the detector responding to real visual
change that the ontology has decided not to cut on, which is a disagreement
about the ontology rather than a failure to see. If the two rates come out
equal that is itself informative, and if `phase_change_only` fires much more
the detector is tracking motion structure the labels deliberately ignore.

THE WINDOW is centred on the candidate time, because the candidate is what the
annotator was asked about -- not on a segment span, which the timing sheets do
not carry. A wider window finds more peaks on both arms; the sweep is printed
so the contrast can be read against it rather than at one arbitrary width.

THE NULL is the same circular shift used everywhere else: peak times rotated
within their own recording, preserving count and spacing and destroying only
phase. It answers "would this many peaks land in this window by chance", which
is the only way to read a fire rate on 20 events.

PEAK PROVENANCE IS NOT INFERRED. --peaks states whether the head was fitted on
these recordings; the script cannot tell and a caveat that guesses is worse
than none.

Usage:
    python -m src.auditor.boundary.negative_fire_rate \
        --gold data/gold/task_timing_gold.json \
        --predictions .../predictions.jsonl --peaks in_sample \
        --window 2.0 --n_perm 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def load_peaks(path):
    peaks = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                continue
            for p in r.get("predicted_peaks") or []:
                if p.get("pred_time") is not None:
                    peaks[r["recording_id"]].append(float(p["pred_time"]))
    for v in peaks.values():
        v.sort()
    return peaks


def fired(pk, centre, w):
    return any(abs(t - centre) <= w for t in pk)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/task_timing_gold.json")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--peaks", default="unknown",
                    choices=["in_sample", "held_out", "unknown"])
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--sweep", default="0.5,1.0,2.0,3.0,5.0",
                    help="0.5 is included because it is the tolerance every "
                         "other timing measurement in this project uses, and "
                         "without it this pool cannot be compared to them")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    evs = json.load(open(a.gold, encoding="utf-8"))["events"]
    peaks = load_peaks(a.predictions)
    print(f"{len(evs)} timing-gold events; "
          f"{sum(len(v) for v in peaks.values())} peaks over {len(peaks)} "
          f"recordings in {os.path.basename(a.predictions)}")

    have = [e for e in evs if peaks.get(e["recording_id"])]
    miss = sorted({e["recording_id"] for e in evs
                   if not peaks.get(e["recording_id"])})
    print(f"\nCOVERAGE: {len(have)}/{len(evs)} events in recordings with "
          f"peaks, over {len({e['recording_id'] for e in have})} recordings")
    if miss:
        print(f"  {len(miss)} recordings absent: {miss[:6]}")
    arms = Counter()
    for e in have:
        arms["negative:" + (e["negative_kind"] or "?")
             if e["asserts_no_boundary"] else "positive"] += 1
    for k, v in sorted(arms.items()):
        print(f"  {k:<32}{v:>4}")
    if not have:
        pk_rec = sorted(peaks)
        raise SystemExit(
            f"no overlap between the pool and the peaks.\n"
            f"  the pool wants:  {sorted({e['recording_id'] for e in evs})[:6]} ...\n"
            f"  the peaks cover: {pk_rec[:6]} ... ({len(pk_rec)} recordings)\n"
            f"  These pools are split-disjoint, not mis-keyed. This 69-event "
            f"gold comes from\n  audit_188 (dev_original72 + test_batch2 -> "
            f"the VAL and part2 splits), while the\n  timing36 and OOF peak "
            f"dumps were both inferred on TRAIN recordings. The peaks\n  that "
            f"cover this pool are the original error_audit dump -- and since "
            f"the head was\n  trained on --train and those logits are the "
            f"--save_logits val dump, they are\n  HELD OUT for it: pass "
            f"--peaks held_out.")

    span = {}
    for rid in {e["recording_id"] for e in have}:
        pk = peaks[rid]
        ts = [e["candidate_time"] for e in have if e["recording_id"] == rid]
        span[rid] = max(pk + ts + [1.0])

    def rates(pk_by_rec, w):
        out = defaultdict(lambda: [0, 0])
        for e in have:
            arm = ("positive" if not e["asserts_no_boundary"]
                   else e["negative_kind"] or "negative")
            pk = pk_by_rec.get(e["recording_id"]) or []
            out[arm][1] += 1
            if fired(pk, e["candidate_time"], w):
                out[arm][0] += 1
        return out

    print(f"\n{'=' * 74}\nFIRE RATE within +/-{a.window}s of the candidate "
          f"(EXPLORATORY)\n{'=' * 74}")
    obs = rates(peaks, a.window)
    rng = random.Random(a.seed)
    nulls = defaultdict(list)
    for _ in range(a.n_perm):
        sh = {}
        for rid in span:
            d = rng.uniform(0, span[rid])
            sh[rid] = [(t + d) % span[rid] for t in peaks[rid]]
        for k, (h, n) in rates(sh, a.window).items():
            nulls[k].append(h / n if n else 0.0)

    print(f"  {'arm':<22}{'fired':>10}{'rate':>8}{'null mean':>11}"
          f"{'ratio':>8}{'p':>9}")
    res = {}
    for k in ("positive", "no_action_change", "phase_change_only"):
        if k not in obs:
            continue
        h, n = obs[k]
        r = h / n
        nl = sorted(nulls[k])
        m = sum(nl) / len(nl) if nl else 0.0
        p = (1 + sum(1 for x in nl if x >= r)) / (len(nl) + 1)
        print(f"  {k:<22}{f'{h}/{n}':>10}{r:>8.3f}{m:>11.3f}"
              f"{(r / m if m else float('nan')):>8.2f}{p:>9.4f}")
        res[k] = {"fired": h, "n": n, "rate": round(r, 4),
                  "null_mean": round(m, 4), "p_value": round(p, 5)}

    if "positive" in obs and any(k in obs for k in ("no_action_change",
                                                   "phase_change_only")):
        # SUBTRACTING RAW RATES IS WRONG HERE and the first version did it.
        # The three arms do not share a chance baseline: their null means came
        # out 0.217 / 0.304 / 0.401, because the negative events sit in
        # recordings with denser peaks. A raw difference therefore charges the
        # negative arms for peaks that would land there anyway, and understates
        # the contrast. Excess over each arm's OWN null is the comparable
        # quantity.
        def exc(k):
            h, n = obs[k]
            nl = nulls[k]
            m = sum(nl) / len(nl) if nl else 0.0
            return h / n, m, h / n - m, (h / n) / m if m else float("nan")

        print(f"\n  THE CONTRAST, on excess over each arm's OWN null:")
        print(f"  {'arm':<22}{'rate':>8}{'null':>8}{'excess':>9}{'ratio':>8}")
        for k in ("positive", "no_action_change", "phase_change_only"):
            if k in obs:
                r, m, e, ra = exc(k)
                print(f"  {k:<22}{r:>8.3f}{m:>8.3f}{e:>+9.3f}{ra:>8.2f}")
        pe = exc("positive")[2]
        for k in ("no_action_change", "phase_change_only"):
            if k in obs:
                print(f"    positive excess {pe:+.3f} vs {k} "
                      f"{exc(k)[2]:+.3f}   difference "
                      f"{pe - exc(k)[2]:+.3f}")
        print(f"  Selection acts on both arms, so a within-pool difference "
              f"restates the sampling far\n  less than either rate alone. On "
              f"{obs['positive'][1]} positives and "
              f"{sum(obs[k][1] for k in ('no_action_change', 'phase_change_only') if k in obs)}"
              f" negatives it is descriptive either way.")

    print(f"\n  window sweep, with each arm's null beside it -- a wider "
          f"window raises BOTH, so a\n  bare rate at 5s says nothing without "
          f"the chance rate at 5s. Read the EXCESS,\n  and note which width "
          f"the positive contrast is largest at:")
    print(f"  {'window':>8}" + "".join(
        f"{k[:15]:>20}" for k in ("positive", "no_action_change",
                                  "phase_change_only")))
    sweep = {}
    for w in [float(x) for x in a.sweep.split(",") if x.strip()]:
        rr = rates(peaks, w)
        rng_w = random.Random(a.seed)
        nl_w = defaultdict(list)
        for _ in range(max(200, a.n_perm // 5)):
            sh = {}
            for rid in span:
                d = rng_w.uniform(0, span[rid])
                sh[rid] = [(t + d) % span[rid] for t in peaks[rid]]
            for k, (h, n) in rates(sh, w).items():
                nl_w[k].append(h / n if n else 0.0)
        line = f"  {w:>8.1f}"
        row = {}
        for k in ("positive", "no_action_change", "phase_change_only"):
            if k in rr:
                h, n = rr[k]
                m = sum(nl_w[k]) / len(nl_w[k]) if nl_w[k] else 0.0
                line += f"{f'{h/n:.2f} (null {m:.2f})':>20}"
                row[k] = {"rate": round(h / n, 4), "null": round(m, 4)}
            else:
                line += f"{'-':>20}"
        print(line)
        sweep[str(w)] = row

    if a.peaks == "in_sample":
        print(f"\n  PEAKS ARE IN-SAMPLE: the head was fitted on these "
              f"recordings' stored annotations.\n  Those annotations DO cut at "
              f"some of the places the frozen ontology now calls\n  "
              f"phase-only, so a high phase_change_only rate is partly the "
              f"head reproducing its\n  training targets rather than "
              f"discovering motion structure.")
    elif a.peaks == "held_out":
        print(f"\n  PEAKS ARE HELD OUT: no recording contributed to the head "
              f"that scored it.")
    else:
        print(f"\n  PROVENANCE NOT STATED (--peaks unknown). Whether the head "
              f"saw these recordings\n  changes what a phase_change_only rate "
              f"means; the script cannot tell.")

    print(f"\n  EXPLORATORY, boundary-enriched pool. No claim here is "
          f"confirmatory, and the\n  peak-blind 45-event alignment gold "
          f"remains separate and untouched.")

    if a.out:
        json.dump({"window": a.window, "peaks_provenance": a.peaks,
                   "n_events": len(have), "arms": dict(arms),
                   "results": res, "sweep": sweep},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
