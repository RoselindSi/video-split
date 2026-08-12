"""Build the alignment set from detector peaks. The offsets need no annotator.

THE PREVIOUS PLAN WAS HALF WRONG AND THIS FILE REPLACES THAT HALF. A targeted
audit cannot supply misaligned candidates, because the corpus does not contain
them and relabelling does not create them. Three measurements, in order:

1.  `cand_time(event_id)` MEANS TWO DIFFERENT THINGS. 94 of the 188 audited
    events are GT-centred -- exact / early / late / missed_* -- and for those
    the id's time is the ANNOTATION time, with the model's prediction
    somewhere else or absent entirely. The other 94 are prediction-centred.
    So the existing alignment derivation measures how far the ANNOTATION moved
    on half the corpus and how far the CANDIDATE was off on the other half,
    and reports them in one column. That is the same conflation the two-field
    split was made to remove, one level further down.

2.  ONLY 38 EVENTS ASK THE QUESTION AT ALL -- prediction-centred, boundary
    valid, corrected time known. Of those, 7 are misaligned at tol 0.5s and 13
    at tol 0.25s, over 6 and 10 recordings. That is the whole pool in the whole
    gold, so a targeted audit of it cannot reach 30.

3.  THE CANDIDATE GRID IS 0.5s AND THE TOLERANCE IS 0.5s. Every pred_time in
    the corpus falls on a half-second, so one grid step of error sits exactly
    at the tolerance and reads as EXACT. The audit's own `early` and `late`
    categories are all one step -- median AND max |pred - gt| is 0.50s across
    all 27 of them. The categories name an error the tolerance then absorbs.

WHAT ACTUALLY GENERATES THE DATA. Each audited event kept ONE candidate. The
detector produced many, and predictions.jsonl still has them. A corrected
boundary time plus the detector's peaks in a window around it gives one
labelled example per peak, with the offset known exactly and no annotator
involved -- the human already did the only part that needs a human, which is
saying where the boundary is.

    ALIGNED       nearest peak, |offset| <= tol
    EARLY / LATE  nearest peak, tol < |offset| <= max_assoc_s
    DUPLICATE     a further peak assigned to a boundary another peak owns
    (a boundary with no peak in the window is a MISS -- an existence failure,
     not an alignment case, and it is reported separately rather than dropped)

WHAT THIS SET IS NOT. The offsets are this detector's offsets. Retrain the
detector and the dataset changes, so a verifier fitted here is calibrated to
one upstream model and its threshold does not transfer to another. That is a
real limitation and it is the price of not waiting for an annotation round
that would take weeks and still land on the same 0.5s grid.

Usage:
    python -m src.auditor.boundary.alignment_from_peaks \
        --predictions /workspace/tr1/results/boundary/error_audit/predictions.jsonl \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --out /workspace/tr1/results/auditor/alignment_from_peaks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from src.auditor.boundary.labels import TOL, cand_time, nearest_corrected
from src.auditor.semantic.render_ontology_clips import get_segments

POSITIVE = ("new_action", "same_action_new_instance")


def segment_lists(paths, keep_recordings=None):
    """recording -> sorted [[label, start, end]], for the gap analysis."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid or (keep_recordings and rid not in keep_recordings):
                continue
            segs, _ = get_segments(r)
            if segs:
                out[rid] = sorted(([sg[0], float(sg[1]), float(sg[2])]
                                   for sg in segs), key=lambda x: x[1])
    return out


def gt_boundaries(paths, keep_recordings=None, mode="edges"):
    """Segment starts and ends as boundaries -- the large, NOISY source.

    125 audited boundaries is the whole clean supply and it is not enough. The
    recseg annotations carry thousands, and for the ALIGNMENT question a
    boundary location is all that is needed. The cost is that these are the
    stored labels, which this project has repeatedly measured as the dominant
    error source -- an offset computed against a mislocated boundary is a
    mislocated offset.

    So they are tagged `gt_annotation` and never merged into the audited set
    without the tag. The intended use is a large noisy TRAIN pool with the 125
    audited boundaries held out as the clean evaluation, not one pooled set."""
    out = defaultdict(set)
    for p in paths:
        if not os.path.exists(p):
            print(f"  !! {p} not found")
            continue
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if not rid or (keep_recordings and rid not in keep_recordings):
                continue
            segs, _ = get_segments(r)
            segs = sorted(([sg[0], float(sg[1]), float(sg[2])]
                           for sg in segs), key=lambda x: x[1])
            if mode == "edges":
                for _l, st, en in segs:
                    out[rid].add(round(st, 1))
                    out[rid].add(round(en, 1))
            else:
                # gap_midpoint: segments are NOT contiguous in this dataset.
                # Between the end of one labelled action and the start of the
                # next there is unlabelled time, and the transition is inside
                # it. Emitting both edges puts two boundaries around the
                # transition and none at it.
                for i, (_l, st, en) in enumerate(segs):
                    if i + 1 < len(segs):
                        nxt = segs[i + 1][1]
                        out[rid].add(round((en + nxt) / 2.0, 1)
                                     if nxt > en else round(en, 1))
                    else:
                        out[rid].add(round(en, 1))
                if segs:
                    out[rid].add(round(segs[0][1], 1))
    return out


def corrected_boundaries(gold, mig, positives_only):
    """One entry per (recording, corrected boundary time), deduplicated.

    Several audited events can point at the same boundary, so the boundary --
    not the event -- is the unit here. That is also why DUPLICATE becomes
    computable: it is a property of the boundary's peak list."""
    by = defaultdict(set)
    prov = {}
    for eid, g in gold.items():
        if g.get("no_valid_boundary") or g.get("boundary_time_unresolved"):
            continue
        if g.get("temporal_truth") != "valid":
            continue
        rel = mig.get(eid)
        if positives_only and rel not in POSITIVE:
            continue
        times = [float(x) for x in (g.get("corrected_boundary_times_json")
                                    or [])]
        p = g.get("primary_corrected_boundary_time")
        if p is not None:
            times.append(float(p))
        for t in times:
            key = (g["recording_id"], round(t, 1))
            by[g["recording_id"]].add(round(t, 1))
            prov.setdefault(key, []).append((eid, rel))
    return by, prov


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="boundary_error_audit.py output; needs "
                         "predicted_peaks per recording")
    ap.add_argument("--gold", action="append")
    ap.add_argument("--migrated",
                    help="pair_schema_v2_migrated.csv. Without it the "
                         "ontology filter cannot be applied and the set will "
                         "include boundaries the ontology denies")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max_assoc_s", type=float, default=2.0,
                    help="beyond this a peak is not this boundary's candidate")
    ap.add_argument("--positives_only", action="store_true",
                    help="restrict to new_action / same_action_new_instance")
    ap.add_argument("--recseg", action="append", default=[],
                    help="recseg json(s). Adds SEGMENT boundaries as a large "
                         "but NOISY source, tagged separately -- intended as "
                         "a train pool with the audited boundaries held out")
    ap.add_argument("--null_shift", type=int, default=200,
                    help="permutations of the CHANCE-ASSOCIATION null: peak "
                         "times circularly shifted within each recording. 0 "
                         "disables it")
    ap.add_argument("--boundary_mode", default="edges",
                    choices=["edges", "gap_midpoint"],
                    help="`edges` treats every segment start and end as a "
                         "boundary; `gap_midpoint` puts one boundary in the "
                         "middle of the unlabelled time between two segments")
    ap.add_argument("--sweep", default="0.25,0.5,1.0")
    ap.add_argument("--out")
    a = ap.parse_args()
    # argparse APPENDS to an action="append" default rather than
    # replacing it, so passing --gold once would silently load the
    # default file as well. Defaults are applied here instead.
    a.gold = a.gold or ["data/gold/audit_188_gold_v2.jsonl"]

    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r
    mig = {}
    if a.migrated:
        with open(a.migrated, newline="", encoding="utf-8-sig") as f:
            mig = {r["event_id"]: r["instance_relation"]
                   for r in csv.DictReader(f)}
    elif a.positives_only:
        print("!! --positives_only needs --migrated; ignoring the filter")
        a.positives_only = False

    peaks = defaultdict(list)
    n_rec = 0
    with open(a.predictions, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rec += 1
            for p in r.get("predicted_peaks") or []:
                if p.get("pred_time") is not None:
                    peaks[r["recording_id"]].append(
                        (float(p["pred_time"]), p.get("score")))
    for v in peaks.values():
        v.sort()
    print(f"{n_rec} recordings in predictions; "
          f"{sum(len(v) for v in peaks.values())} peaks over "
          f"{len(peaks)} of them")
    print(f"  median peaks per recording: "
          f"{sorted(len(v) for v in peaks.values())[len(peaks)//2]}")

    bounds, prov = corrected_boundaries(gold, mig, a.positives_only)
    source = {rid: "audited" for rid in bounds}
    n_b = sum(len(v) for v in bounds.values())
    print(f"{n_b} distinct corrected boundaries over {len(bounds)} recordings"
          + ("  (ontology positives only)" if a.positives_only else ""))

    # THE FUNNEL, split. "no peak" was one number covering two causes with
    # different fixes: a recording absent from the logits dump is recoverable
    # by running inference, a boundary the detector genuinely did not fire
    # near is not.
    no_rec = {r: v for r, v in bounds.items() if r not in peaks}
    n_no_rec = sum(len(v) for v in no_rec.values())
    print(f"  lost, recording absent from predictions : {n_no_rec:>4} "
          f"boundaries over {len(no_rec)} recordings  <- fix by running "
          f"inference")
    covered = {r: v for r, v in bounds.items() if r in peaks}
    n_no_peak = sum(1 for r, ts in covered.items() for t in ts
                    if not any(abs(pt - t) <= a.max_assoc_s
                               for pt, _ in peaks[r]))
    print(f"  lost, covered but no peak within {a.max_assoc_s}s : "
          f"{n_no_peak:>4} boundaries  <- the detector genuinely missed "
          f"these")
    print(f"  scorable                                : "
          f"{n_b - n_no_rec - n_no_peak:>4} boundaries")

    if a.recseg:
        # THE GAPS. A hole at zero with symmetric enrichment at +/-1s is what
        # you get when the thing being detected sits BETWEEN the two
        # boundaries rather than at either -- segments here are not
        # contiguous, so the end of one action and the start of the next
        # bracket an unlabelled interval, and a peak in the middle of it is
        # about a gap-width/2 from both edges. That is a property of how the
        # boundary set was derived, not of the detector.
        seglists = segment_lists(a.recseg, keep_recordings=set(peaks))
        gaps = [(sg[i][2], sg[i + 1][1]) for sg in seglists.values()
                for i in range(len(sg) - 1) if sg[i + 1][1] > sg[i][2]]
        if seglists:
            n_seg = sum(len(v) for v in seglists.values())
            w = sorted(b - a_ for a_, b in gaps)
            print(f"\n  segment structure: {n_seg} segments, {len(gaps)} "
                  f"non-contiguous joins "
                  f"({100 * len(gaps) / max(n_seg - len(seglists), 1):.0f}% "
                  f"of joins)")
            if w:
                print(f"    gap width  median {w[len(w)//2]:.2f}s   "
                      f"p25 {w[len(w)//4]:.2f}s   p75 {w[3*len(w)//4]:.2f}s")
                print(f"    half-width median {w[len(w)//2]/2:.2f}s -- compare "
                      f"this to where the signed-offset\n    mass sits. If "
                      f"they match, `edges` is putting two boundaries around "
                      f"the\n    transition and none at it; rerun with "
                      f"--boundary_mode gap_midpoint.")
        gtb = gt_boundaries(a.recseg, keep_recordings=set(peaks),
                            mode=a.boundary_mode)
        n_g = sum(len(v) for v in gtb.values())
        # UNITS FIRST. "the detector avoids GT boundaries" and "the two time
        # bases disagree" produce the same below-chance alignment, and the
        # second is far more likely. A 10fps frame index looks exactly like a
        # timestamp until the ranges are compared.
        bad = []
        for rid, ts in gtb.items():
            pmax = max(t for t, _ in peaks[rid])
            gmax = max(ts)
            if pmax > 0 and (gmax / pmax > 2.0 or gmax / pmax < 0.5):
                bad.append((rid, gmax, pmax, gmax / pmax))
        print(f"\n  time-base check, GT boundary range against peak range:")
        rr = sorted((max(ts) / max(t for t, _ in peaks[rid]))
                    for rid, ts in gtb.items()
                    if max(t for t, _ in peaks[rid]) > 0)
        print(f"    ratio max(GT)/max(peak) over {len(rr)} recordings: "
              f"min {rr[0]:.2f}  median {rr[len(rr)//2]:.2f}  "
              f"max {rr[-1]:.2f}")
        if bad:
            print(f"    !! {len(bad)} recordings out of the 0.5-2.0 band. If "
                  f"the median is near 10 these are\n       frame indices at "
                  f"10fps, not seconds, and every count below is meaningless.")
            for rid, gm, pm, r_ in bad[:4]:
                print(f"       {rid}: GT max {gm:.1f}, peak max {pm:.1f}, "
                      f"ratio {r_:.2f}")
        else:
            print(f"    all recordings within 0.5-2.0x; the two are on the "
                  f"same time base")
        print(f"\n+ {n_g} GT segment boundaries over {len(gtb)} recordings "
              f"that have peaks (NOISY source)")
        for rid, ts in gtb.items():
            for t in ts:
                if t not in bounds.get(rid, set()):
                    bounds[rid].add(t)
                    source.setdefault(rid, "gt_annotation")
        print(f"  combined pool: {sum(len(v) for v in bounds.values())} "
              f"boundaries over {len(bounds)} recordings. Rows carry "
              f"`boundary_source`;\n  the audited ones are the clean "
              f"evaluation and must not be trained on.")

    def assign(tol):
        rows, miss = [], 0
        for rid, times in bounds.items():
            pk = peaks.get(rid, [])
            for bt in sorted(times):
                near = [(abs(t - bt), t, s) for t, s in pk
                        if abs(t - bt) <= a.max_assoc_s]
                if not near:
                    miss += 1
                    continue
                near.sort()
                for i, (d, t, s) in enumerate(near):
                    cls = ("DUPLICATE" if i else
                           "ALIGNED" if d <= tol else
                           ("EARLY" if t < bt else "LATE"))
                    rows.append({"recording_id": rid, "boundary_time": bt,
                                 "peak_time": t, "offset_s": round(bt - t, 3),
                                 "score": s, "label": cls,
                                 "rank_at_boundary": i,
                                 "boundary_source": (
                                     "audited" if (rid, bt) in prov
                                     else "gt_annotation"),
                                 "relation": sorted(
                                     {r for _, r in prov.get((rid, bt), [])
                                      if r})})
        return rows, miss

    rows, miss = assign(a.tol)
    print(f"\nat tol {a.tol}s, association window {a.max_assoc_s}s:")
    c = Counter(r["label"] for r in rows)
    for k in ("ALIGNED", "EARLY", "LATE", "DUPLICATE"):
        sel = [r for r in rows if r["label"] == k]
        print(f"  {k:<12} {c.get(k, 0):>5} over "
              f"{len({r['recording_id'] for r in sel})} recordings")
    print(f"  boundaries with NO peak within {a.max_assoc_s}s: {miss}"
          f"  <- existence failures, not alignment cases")
    mis_n = sum(c.get(k, 0) for k in ("EARLY", "LATE", "DUPLICATE"))
    mis_r = len({r["recording_id"] for r in rows if r["label"] != "ALIGNED"})
    print(f"\n  ALIGNED    {c.get('ALIGNED', 0):>5} over "
          f"{len({r['recording_id'] for r in rows if r['label'] == 'ALIGNED'})}"
          f" recordings")
    print(f"  MISALIGNED {mis_n:>5} over {mis_r} recordings")

    # BY SOURCE. Pooling them was the point of the tag and the summary was
    # not honouring it.
    print(f"\nby boundary_source -- these are not one population:")
    print(f"  {'source':<16}{'ALIGNED':>9}{'EARLY':>8}{'LATE':>7}"
          f"{'DUP':>7}{'<=0.5s of nearest':>20}")
    for src in ("audited", "gt_annotation"):
        sel = [r for r in rows if r["boundary_source"] == src]
        if not sel:
            continue
        cc = Counter(r["label"] for r in sel)
        nr = [r for r in sel if r["rank_at_boundary"] == 0]
        close = sum(1 for r in nr if abs(r["offset_s"]) <= 0.5)
        print(f"  {src:<16}{cc.get('ALIGNED', 0):>9}{cc.get('EARLY', 0):>8}"
              f"{cc.get('LATE', 0):>7}{cc.get('DUPLICATE', 0):>7}"
              f"{close:>10}/{len(nr):<9}"
              f" ({100 * close / max(len(nr), 1):.0f}%)")

    print(f"\n|offset| distribution of the nearest peak per boundary:")
    nearest = [r for r in rows if r["rank_at_boundary"] == 0]
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        n = sum(1 for r in nearest if lo <= abs(r["offset_s"]) < hi)
        print(f"  {lo:>4.2f}-{hi:>4.2f}s  {n:>5}")
    print(f"  the grid is 0.5s, so these bins are not equally reachable -- "
          f"0.25-0.50 and\n  0.00-0.25 are one grid step and zero steps, and "
          f"nothing lands between them.")

    print(f"\nthe threshold is the whole labelling, so:")
    print(f"  {'tol':>6}{'ALIGNED':>10}{'MISALIGN':>10}{'mis recs':>10}")
    sweep = {}
    for t in [float(x) for x in a.sweep.split(",") if x.strip()]:
        rr, _ = assign(t)
        cc = Counter(r["label"] for r in rr)
        mr = len({r["recording_id"] for r in rr if r["label"] != "ALIGNED"})
        m = sum(cc.get(k, 0) for k in ("EARLY", "LATE", "DUPLICATE"))
        print(f"  {t:>6.2f}{cc.get('ALIGNED', 0):>10}{m:>10}{mr:>10}")
        sweep[str(t)] = {"aligned": cc.get("ALIGNED", 0), "misaligned": m,
                         "mis_recordings": mr}

    # ------------------------------------------------- chance association
    #
    # TWO STATISTICS, AND THE FIRST ONE ANSWERS THE WRONG QUESTION.
    #
    # Counting BOUNDARIES with a peak nearby is recall -- "was this boundary
    # detected". It is also density-inflated under the null: a shifted peak
    # that lands in a cluster of boundaries scores every one of them at once,
    # and random placement samples dense regions more evenly than a detector
    # does, so the null can exceed the observation without the detector doing
    # anything wrong. That is most of what a below-1.0 ALIGNED ratio means.
    #
    # Counting PEAKS with a boundary nearby is the alignment question, and it
    # is the one the dataset rows are about: each row IS a peak. One peak
    # contributes one count whatever the local boundary density.
    def peak_centric(pk_by_rec, tol):
        al = mis = un = 0
        for rid, pk in pk_by_rec.items():
            bs = sorted(bounds.get(rid, ()))
            for t, _ in pk:
                if not bs:
                    un += 1
                    continue
                d = min(abs(b - t) for b in bs)
                if d <= tol:
                    al += 1
                elif d <= a.max_assoc_s:
                    mis += 1
                else:
                    un += 1
        return al, mis, un

    if a.null_shift:
        import random as _r
        rng = _r.Random(0)
        span = {rid: max([t for t, _ in pk] + list(bounds.get(rid, {0.0})))
                for rid, pk in peaks.items()}
        real = Counter(r["label"] for r in rows)
        nulls = []
        for _ in range(a.null_shift):
            shifted = {}
            for rid, pk in peaks.items():
                d = rng.uniform(0, span[rid] or 1.0)
                shifted[rid] = sorted(((t + d) % (span[rid] or 1.0), s_)
                                      for t, s_ in pk)
            n_al = n_mis = 0
            for rid, times in bounds.items():
                pk = shifted.get(rid, [])
                for bt in sorted(times):
                    near = sorted((abs(t - bt), t) for t, _ in pk
                                  if abs(t - bt) <= a.max_assoc_s)
                    if not near:
                        continue
                    n_al += 1 if near[0][0] <= a.tol else 0
                    n_mis += (1 if near[0][0] > a.tol else 0) + len(near) - 1
            nulls.append((n_al, n_mis))
        m_al = sum(x for x, _ in nulls) / len(nulls)
        m_mis = sum(y for _, y in nulls) / len(nulls)
        print(f"\n{'=' * 74}\nCHANCE ASSOCIATION -- peaks circularly shifted "
              f"within each recording, {a.null_shift}x\n{'=' * 74}")
        print(f"  {'':<14}{'observed':>10}{'null mean':>12}{'ratio':>9}")
        print(f"  {'ALIGNED':<14}{real.get('ALIGNED', 0):>10}{m_al:>12.1f}"
              f"{real.get('ALIGNED', 0) / max(m_al, 1e-9):>9.2f}")
        obs_mis = sum(real.get(k, 0) for k in ("EARLY", "LATE", "DUPLICATE"))
        print(f"  {'MISALIGNED':<14}{obs_mis:>10}{m_mis:>12.1f}"
              f"{obs_mis / max(m_mis, 1e-9):>9.2f}")
        # the peak-centric null, on the same shifts
        rng2 = _r.Random(0)
        p_obs = peak_centric(peaks, a.tol)
        p_null = []
        for _ in range(a.null_shift):
            sh = {}
            for rid, pk in peaks.items():
                d = rng2.uniform(0, span[rid] or 1.0)
                sh[rid] = [((t + d) % (span[rid] or 1.0), s_) for t, s_ in pk]
            p_null.append(peak_centric(sh, a.tol))
        m = [sum(x[i] for x in p_null) / len(p_null) for i in range(3)]
        # PER SOURCE. The audited boundaries are human-verified times; the
        # GT ones are the stored annotations this project has repeatedly
        # measured as the dominant error source. If peaks beat chance against
        # the audited set and lose against the annotations, the offset is in
        # the LABELS. If they lose against both, it is in the detector or in
        # the peak extraction. Those have nothing in common as next steps, and
        # no further hypothesis from me discriminates them -- this does.
        def peak_centric_sub(pk_by_rec, tol, keep):
            al = mis = 0
            for rid, pk in pk_by_rec.items():
                bs = sorted(b for b in bounds.get(rid, ())
                            if ((rid, b) in prov) == keep)
                for t, _ in pk:
                    if not bs:
                        continue
                    d = min(abs(b - t) for b in bs)
                    if d <= tol:
                        al += 1
                    elif d <= a.max_assoc_s:
                        mis += 1
            return al, mis

        print(f"\n  by boundary source, against the SAME shift null:")
        print(f"  {'':<30}{'observed':>10}{'null mean':>12}{'ratio':>9}")
        for keep, lab in ((True, "audited (human-verified)"),
                          (False, "gt_annotation (stored)")):
            o = peak_centric_sub(peaks, a.tol, keep)
            rng4 = _r.Random(0)
            acc = [0.0, 0.0]
            reps = max(1, a.null_shift // 5)
            for _ in range(reps):
                sh = {}
                for rid, pk in peaks.items():
                    d = rng4.uniform(0, span[rid] or 1.0)
                    sh[rid] = [((t + d) % (span[rid] or 1.0), s_)
                               for t, s_ in pk]
                r_ = peak_centric_sub(sh, a.tol, keep)
                acc[0] += r_[0]; acc[1] += r_[1]
            mn = acc[0] / reps
            print(f"  {lab:<30}{o[0]:>10}{mn:>12.1f}"
                  f"{o[0] / max(mn, 1e-9):>9.2f}   within tol")

        print(f"\n  PEAK-CENTRIC -- the statistic the dataset rows actually "
              f"are:")
        print(f"  {'':<28}{'observed':>10}{'null mean':>12}{'ratio':>9}")
        for i, lab in enumerate(("peak within tol of a boundary",
                                 "peak within assoc, not tol",
                                 "peak near no boundary")):
            print(f"  {lab:<28}{p_obs[i]:>10}{m[i]:>12.1f}"
                  f"{p_obs[i] / max(m[i], 1e-9):>9.2f}")
        # SIGNED offsets. |offset| cannot tell a systematic lag from noise,
        # and the three peak-centric rows have the signature of one: the
        # 0-0.5s band is DEPLETED (0.35) while 0.5-2.0s is ENRICHED (1.57) and
        # "near nothing" is slightly depleted (0.89). Peaks are closer to
        # boundaries than chance overall, with the mass in the wrong place.
        # A constant lag does exactly that; noise does not.
        def signed(pk_by_rec):
            out = []
            for rid, pk in pk_by_rec.items():
                bs = sorted(bounds.get(rid, ()))
                for t, _ in pk:
                    if not bs:
                        continue
                    b = min(bs, key=lambda x: abs(x - t))
                    if abs(b - t) <= a.max_assoc_s:
                        out.append(b - t)   # >0 : the peak fired EARLY
            return out
        obs_s = signed(peaks)
        rng3 = _r.Random(0)
        nul_s = []
        for _ in range(max(1, a.null_shift // 10)):
            sh = {}
            for rid, pk in peaks.items():
                d = rng3.uniform(0, span[rid] or 1.0)
                sh[rid] = [((t + d) % (span[rid] or 1.0), s_) for t, s_ in pk]
            nul_s += signed(sh)
        scale = len(nul_s) / max(len(obs_s), 1)
        edges = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0]
        print(f"\n  SIGNED offset (boundary - peak; POSITIVE = the peak "
              f"fired EARLY), nearest boundary:")
        print(f"  {'band':>16}{'observed':>10}{'null/scale':>12}{'ratio':>8}")
        for lo, hi in zip(edges, edges[1:]):
            o = sum(1 for x in obs_s if lo <= x < hi)
            n = sum(1 for x in nul_s if lo <= x < hi) / max(scale, 1e-9)
            print(f"  {lo:>6.2f}..{hi:<8.2f}{o:>10}{n:>12.1f}"
                  f"{o / max(n, 1e-9):>8.2f}")
        if obs_s:
            med = sorted(obs_s)[len(obs_s) // 2]
            pos = sum(1 for x in obs_s if x > 0) / len(obs_s)
            print(f"\n  median signed offset {med:+.2f}s; "
                  f"{100 * pos:.0f}% of peaks fire early.")
            print(f"  A median far from zero with one side dominant is a "
                  f"SYSTEMATIC LAG, not scatter --\n  a window-centring or "
                  f"receptive-field offset upstream, and a bug to find rather "
                  f"than\n  a property of the data. A median near zero with "
                  f"the mass spread means the peaks\n  are simply not "
                  f"locating these boundaries.")

        print(f"\n  This is the arm to read. The boundary-centric table above "
              f"is recall and is\n  inflated under the null by boundary "
              f"density -- one shifted peak landing in a\n  cluster scores "
              f"every boundary in it.")

        print(f"\n  A MISALIGNED ratio near 1.0 means the association is "
              f"chance: at this boundary\n  density a peak lands within "
              f"{a.max_assoc_s}s of SOME boundary whether or not it detected\n"
              f"  one, and `EARLY by 1.5s` is then not a fact about the "
              f"candidate. ALIGNED is the\n  arm that has to beat chance for "
              f"the pool to carry any alignment signal at all.")

    print(f"\nWHAT THIS SET IS NOT: these are THIS detector's offsets. Retrain "
          f"it and the\ndataset changes, so a verifier fitted here is "
          f"calibrated to one upstream model.\nIts threshold does not "
          f"transfer, and that has to be restated wherever it is used.")

    if a.out:
        json.dump({"tol": a.tol, "max_assoc_s": a.max_assoc_s,
                   "positives_only": a.positives_only, "sweep": sweep,
                   "no_peak_boundaries": miss, "rows": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out} ({len(rows)} labelled candidates)")


if __name__ == "__main__":
    main()
