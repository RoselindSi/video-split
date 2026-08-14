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
    ap.add_argument("--sweep", default="1.0,2.0,3.0,5.0")
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
        raise SystemExit("no overlap between the pool and the peaks")

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
        pr = obs["positive"][0] / obs["positive"][1]
        print(f"\n  THE CONTRAST, which is what this pool can support:")
        for k in ("no_action_change", "phase_change_only"):
            if k in obs:
                r = obs[k][0] / obs[k][1]
                print(f"    positive {pr:.3f} vs {k} {r:.3f}   "
                      f"difference {pr - r:+.3f}")
        print(f"  Selection acts on both arms, so the difference restates the "
              f"sampling far less\n  than either rate alone does. On "
              f"{obs['positive'][1]} positives and "
              f"{sum(obs[k][1] for k in ('no_action_change', 'phase_change_only') if k in obs)}"
              f" negatives it is\n  descriptive either way.")

    print(f"\n  window sweep, because one width is a choice:")
    print(f"  {'window':>8}" + "".join(
        f"{k[:16]:>18}" for k in ("positive", "no_action_change",
                                  "phase_change_only")))
    sweep = {}
    for w in [float(x) for x in a.sweep.split(",") if x.strip()]:
        rr = rates(peaks, w)
        line = f"  {w:>8.1f}"
        row = {}
        for k in ("positive", "no_action_change", "phase_change_only"):
            if k in rr:
                h, n = rr[k]
                line += f"{f'{h}/{n} = {h/n:.2f}':>18}"
                row[k] = round(h / n, 4)
            else:
                line += f"{'-':>18}"
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
