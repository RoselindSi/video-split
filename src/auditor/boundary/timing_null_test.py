"""The decisive test: peak-blind human timing against detector peaks.

WHAT IS BEING DECIDED. The detector's peaks land within 0.5s of a stored
segment boundary LESS often than randomly shifted peaks do -- 0.54x against a
circular-shift null, on a boundary set carrying no peak-based selection, so
that number is clean. Against the 125 human-corrected boundaries in the audit
gold the same peaks score 6.21x. Both cannot be read at face value: the audit
corpus was built from peak-to-GT matching, so an event is in it partly BECAUSE
a peak was nearby.

These 45 events were sampled on instance_relation alone. No peak, score or
distance entered their selection. They are the only boundary set in this
project free of that selection, so this is the number that settles it:

    ratio >> 1   the detector has real timing signal and the STORED
                 annotation times are systematically displaced. That would be
                 the first evidence for the label-noise conclusion that
                 involves no human judgement in the comparison at all --
                 detector against annotations, with a permutation null.
    ratio ~ 1    the 6.21 was selection, and timing mismatch cannot be
                 attributed mainly to the stored annotations. The next step
                 would then be upstream of this file entirely.

TWO DISTANCE DEFINITIONS, RUN SIDE BY SIDE, because they already disagree.
On the candidate-to-human distances -- which need no detector at all -- 42% of
candidates are within 0.5s under midpoint and 73% under interval-aware, and 14
of the 45 events flip. A conclusion that holds under one and not the other is
a statement about the modelling of transitions, not about the video, and has
to be reported as such rather than resolved by preferring the friendlier
number.

    midpoint         every interval collapses to its centre. Comparable with
                     every earlier point-based measurement in this project.
    interval_aware   a peak inside a human-marked transition interval is at
                     distance 0; outside, the distance to the nearer edge.

TWO STATISTICS, for the same reason:

    boundary-centric  of the human boundaries, how many have a peak within
                      tol. The direct question, and the natural one at this
                      density -- roughly 1.7 human boundaries per recording.
    peak-centric      of the peaks, how many are near a human boundary. Less
                      natural here, but it is the statistic the 6.21 and 0.54
                      references were computed with, so it is what makes them
                      comparable.

THE NULL is a circular shift of every peak time within its own recording,
which preserves the number of peaks and their spacing -- including the 1.0s
NMS minimum -- and destroys only their phase relative to the boundaries. The
p-value is the permutation p-value, (1 + #{null >= observed}) / (n + 1), and
the interval reported is the null's own 2.5-97.5 percentile range, not a
bootstrap of the observation.

Usage:
    python -m src.auditor.boundary.timing_null_test \
        --gold_json data/gold/alignment_timing_gold_45.json \
        --predictions /workspace/tr1/results/boundary/error_audit/predictions.jsonl \
        --n_perm 2000 --out /workspace/tr1/results/auditor/timing_null.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

REFERENCES = [
    ("audited old sample (peak-selected)", 6.21,
     "contaminated: the audit corpus was built from peak-to-GT matching"),
    ("stored GT annotation", 0.54,
     "unbiased wrt peaks, and BELOW chance"),
]


def load_peaks(path):
    peaks = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict):   # a json array, not jsonl
                continue
            for p in r.get("predicted_peaks") or []:
                if p.get("pred_time") is not None:
                    peaks[r["recording_id"]].append(float(p["pred_time"]))
    for v in peaks.values():
        v.sort()
    if not peaks:
        print(f"  !! no predicted_peaks found in {path}. It should be JSONL "
              f"with one object per\n     recording carrying "
              f"`recording_id` and `predicted_peaks`.")
    return peaks


def boundaries_for(ev, mode):
    """Each human boundary as (kind, a, b). Points have a == b."""
    out = [("point", p, p) for p in ev["human_points"]]
    for a, b in ev["human_intervals"]:
        if mode == "midpoint":
            m = (a + b) / 2.0
            out.append(("point", m, m))
        else:
            out.append(("interval", min(a, b), max(a, b)))
    return out


def dist(t, b):
    _k, lo, hi = b
    if lo <= t <= hi:
        return 0.0
    return lo - t if t < lo else t - hi


def statistics(peaks, by_rec, tol):
    """(boundaries with a peak within tol, peaks within tol of a boundary)."""
    n_b = n_p = 0
    for rid, bs in by_rec.items():
        pk = peaks.get(rid) or []
        if not pk:
            continue
        for b in bs:
            if any(dist(t, b) <= tol for t in pk):
                n_b += 1
        for t in pk:
            if any(dist(t, b) <= tol for b in bs):
                n_p += 1
    return n_b, n_p


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold_json",
                    default="data/gold/alignment_timing_gold_45.json")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--recseg_train", action="append", default=[],
                    help="recseg json(s) for the TRAINING split. The head "
                         "having been fitted on a recording changes what a "
                         "peak on it means, and that has to be stated")
    ap.add_argument("--recseg_val", action="append", default=[])
    ap.add_argument("--paired_recseg", action="append", default=[],
                    help="recseg json(s) to build STORED boundaries for the "
                         "same recordings, so human and stored times can be "
                         "compared against each other on the same peaks. "
                         "This is the test the hypothesis actually names")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    evs = json.load(open(a.gold_json, encoding="utf-8"))["events"]
    peaks = load_peaks(a.predictions)
    print(f"{len(evs)} timing-gold events over "
          f"{len({e['recording_id'] for e in evs})} recordings")
    print(f"{sum(len(v) for v in peaks.values())} peaks over "
          f"{len(peaks)} recordings in {os.path.basename(a.predictions)}")

    have = [e for e in evs if peaks.get(e["recording_id"])]
    miss = sorted({e["recording_id"] for e in evs
                   if not peaks.get(e["recording_id"])})
    print(f"\nCOVERAGE -- the test can only use events whose recording has "
          f"peaks:")
    print(f"  usable {len(have)}/{len(evs)} events over "
          f"{len({e['recording_id'] for e in have})} recordings")
    if miss:
        print(f"  {len(miss)} recordings absent from predictions: "
              f"{miss[:6]}{' ...' if len(miss) > 6 else ''}")
        print(f"  every number below is on the usable subset. If that subset "
              f"is small the test is\n  underpowered and the honest output is "
              f"a wide interval, not a verdict.")
    # WHICH SPLIT. A peak on a recording the head was fitted on is not the
    # same measurement as a peak on a held-out one, and the bias does not run
    # in one obvious direction: the head was trained to hit the STORED
    # annotation times, which on this hypothesis are displaced, so training
    # exposure pushes its peaks toward the stored times and AWAY from the
    # human-corrected ones -- against the hypothesis. It also makes peaks
    # sharper and more numerous there, which inflates alignment with anything
    # nearby -- for it. Both are real and neither is quantified here. The
    # split is reported so the number is never read without it.
    def rids(paths):
        out = set()
        for p in paths:
            if not os.path.exists(p):
                print(f"  !! {p} not found")
                continue
            blob = json.load(open(p, encoding="utf-8"))
            if isinstance(blob, dict):
                blob = blob.get("recordings") or blob.get("data") or []
            out |= {r.get("recording_id") for r in blob if r.get("recording_id")}
        return out

    if a.recseg_train or a.recseg_val:
        tr, va = rids(a.recseg_train), rids(a.recseg_val)
        gold_r = {e["recording_id"] for e in evs}
        n_tr, n_va = len(gold_r & tr), len(gold_r & va)
        print(f"\n  SPLIT of the 45 events' {len(gold_r)} recordings: "
              f"{n_tr} in train, {n_va} in val, "
              f"{len(gold_r - tr - va)} in neither")
        if n_tr and not n_va:
            print(f"  !! ALL of them are TRAINING recordings. The head was "
                  f"fitted on their stored\n     annotations, so a peak here "
                  f"is not evidence of generalisation. State this\n     "
                  f"wherever the ratio is quoted, or infer with a checkpoint "
                  f"that held them out.")

    if not have:
        raise SystemExit(
            "no overlap between the timing gold and the peaks.\n"
            "  This is what peak-blind sampling does: selecting on "
            "instance_relation alone\n  picked recordings the detector was "
            "never dumped on. Nothing is wrong with the\n  gold -- the "
            "detector output has to be extended to cover it:\n"
            "    python -m src.boundary.train_head_multi ... "
            "--infer_extra <feats covering these recordings> \\\n"
            "        --infer_extra_out <logits.pt>\n"
            "    python -m src.boundary.boundary_error_audit --logits "
            "<logits.pt> --out_dir <dir>")

    span = {}
    for rid in {e["recording_id"] for e in have}:
        pk = peaks[rid]
        bs = [x for e in have if e["recording_id"] == rid
              for _k, lo, hi in boundaries_for(e, "interval_aware")
              for x in (lo, hi)]
        span[rid] = max(pk + bs + [1.0])

    results = {}
    for mode in ("midpoint", "interval_aware"):
        by_rec = defaultdict(list)
        for e in have:
            by_rec[e["recording_id"]] += boundaries_for(e, mode)
        n_bounds = sum(len(v) for v in by_rec.values())
        obs_b, obs_p = statistics(peaks, by_rec, a.tol)

        rng = random.Random(a.seed)
        nb, npk = [], []
        for _ in range(a.n_perm):
            sh = {}
            for rid, pk in peaks.items():
                if rid not in by_rec:
                    continue
                d = rng.uniform(0, span[rid])
                sh[rid] = [(t + d) % span[rid] for t in pk]
            x, y = statistics(sh, by_rec, a.tol)
            nb.append(x)
            npk.append(y)

        def summarise(obs, null):
            null = sorted(null)
            m = sum(null) / len(null)
            lo = null[int(0.025 * len(null))]
            hi = null[min(int(0.975 * len(null)), len(null) - 1)]
            p = (1 + sum(1 for x in null if x >= obs)) / (len(null) + 1)
            return {"observed": obs, "null_mean": round(m, 2),
                    "null_95": [lo, hi],
                    "ratio": round(obs / m, 2) if m else None,
                    "p_value": round(p, 5)}

        results[mode] = {
            "n_events": len(have), "n_boundaries": n_bounds,
            "n_recordings": len(by_rec),
            "boundary_centric": summarise(obs_b, nb),
            "peak_centric": summarise(obs_p, npk)}

        print(f"\n{'=' * 74}\n{mode.upper()}   {n_bounds} human boundaries, "
              f"{len(have)} events, {len(by_rec)} recordings\n{'=' * 74}")
        for name, key in (("boundary-centric (the direct question)",
                           "boundary_centric"),
                          ("peak-centric (comparable to 6.21 / 0.54)",
                           "peak_centric")):
            r = results[mode][key]
            print(f"  {name}")
            print(f"    observed {r['observed']:>6}   null mean "
                  f"{r['null_mean']:>7}   null 95% [{r['null_95'][0]}, "
                  f"{r['null_95'][1]}]")
            print(f"    ratio {r['ratio']}   permutation p = {r['p_value']}"
                  + ("   (floor: p cannot go below 1/(n_perm+1))"
                     if r["p_value"] <= 1.5 / (a.n_perm + 1) else ""))

    # ------------------------------------------------------------- PAIRED
    #
    # THE HYPOTHESIS IS A COMPARISON AND I TESTED TWO ABSOLUTE LEVELS.
    #
    # "the stored annotation times are displaced" says human times are BETTER
    # located than stored ones. Testing each against chance separately throws
    # away the pairing and most of the power: 58 human boundaries can only
    # detect a ratio above about 2, so the run below could reject the audited
    # 6.21 and could say nothing at all about 1.3-1.5.
    #
    # The paired form uses the SAME peaks and the SAME shift for both, and the
    # statistic is the difference in per-boundary hit RATE, which is what the
    # claim is about. Rates rather than counts because there are roughly two
    # human boundaries per recording against seventy stored ones.
    if a.paired_recseg:
        from src.auditor.boundary.alignment_from_peaks import gt_boundaries
        rid_have = {e["recording_id"] for e in have}
        stored = gt_boundaries(a.paired_recseg, keep_recordings=rid_have,
                               mode="gap_midpoint")
        stored_b = {r: [("point", t, t) for t in ts]
                    for r, ts in stored.items()}
        n_stored = sum(len(v) for v in stored_b.values())
        print(f"\n{'=' * 74}\nPAIRED -- human timing against STORED "
              f"annotation, same peaks, same null\n{'=' * 74}")
        print(f"  {n_stored} stored boundaries (gap_midpoint) over "
              f"{len(stored_b)} recordings, against\n  the human boundaries "
              f"on those same recordings.")

        def rate(pk, by_rec):
            n = hit = 0
            for rid, bs in by_rec.items():
                p_ = pk.get(rid) or []
                for b in bs:
                    n += 1
                    if p_ and any(dist(t, b) <= a.tol for t in p_):
                        hit += 1
            return hit / n if n else 0.0

        for mode in ("midpoint", "interval_aware"):
            by_rec = defaultdict(list)
            for e in have:
                by_rec[e["recording_id"]] += boundaries_for(e, mode)
            obs = rate(peaks, by_rec) - rate(peaks, stored_b)
            rng = random.Random(a.seed)
            nulls = []
            for _ in range(a.n_perm):
                sh = {}
                for rid in rid_have:
                    pk = peaks.get(rid) or []
                    d = rng.uniform(0, span[rid])
                    sh[rid] = [(t + d) % span[rid] for t in pk]
                nulls.append(rate(sh, by_rec) - rate(sh, stored_b))
            nulls.sort()
            m = sum(nulls) / len(nulls)
            lo = nulls[int(0.025 * len(nulls))]
            hi = nulls[min(int(0.975 * len(nulls)), len(nulls) - 1)]
            pv = (1 + sum(1 for x in nulls if x >= obs)) / (len(nulls) + 1)
            rh, rs = rate(peaks, by_rec), rate(peaks, stored_b)
            print(f"\n  {mode}")
            print(f"    human hit rate {rh:.3f}   stored hit rate {rs:.3f}   "
                  f"difference {obs:+.3f}")
            print(f"    null mean {m:+.3f}   null 95% [{lo:+.3f}, {hi:+.3f}]"
                  f"   permutation p = {(pv):.4f}")
            results[mode]["paired_vs_stored"] = {
                "human_rate": round(rh, 4), "stored_rate": round(rs, 4),
                "difference": round(obs, 4), "null_mean": round(m, 4),
                "null_95": [round(lo, 4), round(hi, 4)],
                "p_value": round(pv, 5)}
        print(f"\n  A difference above the null band means human times are "
              f"better located than the\n  stored ones ON THE SAME PEAKS -- "
              f"the claim, tested directly. Inside the band\n  means the two "
              f"annotations are equally (un)related to where the detector "
              f"fires.")

    print(f"\n{'=' * 74}\nTHE THREE NUMBERS SIDE BY SIDE (peak-centric, "
          f"tol {a.tol}s)\n{'=' * 74}")
    for lab, val, note in REFERENCES:
        print(f"  {lab:<38}{val:>6}   {note}")
    for mode in ("midpoint", "interval_aware"):
        r = results[mode]["peak_centric"]
        print(f"  {'new timing audit (' + mode + ')':<38}{r['ratio']:>6}   "
              f"peak-blind sampling, p = {r['p_value']}")
    print(f"\n  The two references were computed on this same statistic and "
          f"null, so they are\n  comparable. They were NOT recomputed in this "
          f"run -- they are quoted, and if the\n  predictions file or the "
          f"tolerance changed they would have to be rerun to stay so.")

    if a.out:
        json.dump({"tol": a.tol, "n_perm": a.n_perm,
                   "usable_events": len(have), "missing_recordings": miss,
                   "references": {k: v for k, v, _ in REFERENCES},
                   "results": results},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
