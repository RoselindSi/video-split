"""The exact segment windows the 48 audited events were judged on.

WHY THIS IS THE FIRST STEP AND NOT A CHORE. The naming arm has never run on
the audited population -- 8 of 186 events joined last time -- and the fix is
not a bigger model, it is running naming on the right windows. The right
window is the SEGMENT, with the bounds the pipeline actually uses, not a
window centred on a candidate. Sampling the enrichment batch before this is
resolved would sample candidate-centred events again and produce a second gold
in the wrong unit.

WHAT THE AUDITED UNIT IS, established rather than assumed. The sheet showed
each event's previous, containing and next segment LABELS, and the audit notes
refer to them as seg0 / seg1 / seg2 in time order -- "Seg0 cannot be fully
judged; seg1 is wrong; seg2 is supported". So an event's unit is that ordered
set of one to three segments: 20 events have all three, 19 have two, 9 have
one, because segments in this dataset are not contiguous and a candidate on a
boundary has no containing segment.

RECOVER BY TIME, VERIFY BY LABEL. The segments are located the same way the
context file built them -- previous, containing and next by time around the
candidate -- and then their labels are CHECKED against the labels the
annotator actually saw. Locating by label instead would fail exactly where it
matters: `Rinse and seat sink strainer` occurs four times in one recording,
and matching on the string would pick an arbitrary one. A label mismatch means
the recseg file is not the one the audit was built from, and it is reported
per event rather than counted.

THE MANIFEST IS DEDUPLICATED. Several events share a segment -- 431/t427.5 and
431/t440.5 overlap -- and naming should run once per segment, not once per
(event, segment). The event-to-segment mapping is kept alongside so the
results can be joined back without rerunning anything.

Usage:
    python -m src.auditor.semantic.naming_targets \
        --gold_json data/gold/semantic_ontology_gold_48.json \
        --context data/gold/audit_188_context.jsonl \
        --recseg /workspace/tr1/data_recseg/recseg_val.json \
        --recseg /workspace/tr1/data_recseg/recseg_train.json \
        --out /workspace/tr1/results/auditor/naming_targets_48.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

from src.auditor.semantic.render_ontology_clips import get_segments, get_video

TIME = re.compile(r"_t(\d+(?:\.\d+)?)$")


def cand_time(eid):
    m = TIME.search(eid)
    return float(m.group(1)) if m else None


def around(segs, t):
    """previous / containing / next by TIME, in time order, deduplicated.

    The same rule the context file used. `containing` can be None while both
    neighbours exist -- that is the non-contiguous case, not an error."""
    segs = sorted(segs, key=lambda s: float(s[1]))
    contain = next((s for s in segs if float(s[1]) <= t <= float(s[2])), None)
    prev = [s for s in segs if float(s[2]) <= t]
    nxt = [s for s in segs if float(s[1]) >= t]
    out, seen = [], set()
    for s in ([prev[-1]] if prev else []) + ([contain] if contain else []) \
            + ([nxt[0]] if nxt else []):
        key = (round(float(s[1]), 2), round(float(s[2]), 2))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return sorted(out, key=lambda s: float(s[1]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold_json",
                    default="data/gold/semantic_ontology_gold_48.json",
                    help="the audited events. JSON with an `events` list, or "
                         "a CSV carrying an `event_id` column -- the "
                         "enrichment sheet is a CSV and this flag used to "
                         "accept only JSON")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--recseg", action="append", required=True,
                    help="recseg json path, DIRECTORY, or glob. Globs are "
                         "expanded here because hand-typed absolute paths to "
                         "this dataset have been wrong four times in a row; "
                         "quote them so the shell does not expand first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.gold_json.lower().endswith(".csv"):
        with open(a.gold_json, newline="", encoding="utf-8-sig") as f:
            evs = [r for r in csv.DictReader(f) if (r.get("event_id") or
                                                    "").strip()]
        if not evs:
            raise SystemExit(
                f"{a.gold_json} has no usable `event_id` column. A sheet "
                f"whose keys are only
  `audit_key` cannot be resolved to "
                f"segments without the event ids.")
        for e in evs:
            e.setdefault("audit_key", e["event_id"])
    else:
        evs = json.load(open(a.gold_json, encoding="utf-8-sig"))["events"]
    ctx = {}
    for line in open(a.context, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            ctx[r["event_id"]] = r

    # ONE RECORDING CAN APPEAR IN SEVERAL recseg FILES. First-wins is fine
    # when the segmentations agree and silently picks one when they do not,
    # which would make the recovered windows depend on flag order. So the
    # conflicts are found and named.
    # A path, a directory, or a glob. `recseg_part2_combined.json` does not
    # live where the other recseg files do, and typing its absolute path from
    # memory has produced a silent 19/48 more than once -- the run looked
    # complete because the missing file only printed one line.
    paths, unresolved = [], []
    for pat in a.recseg:
        if os.path.isdir(pat):
            hits = sorted(glob.glob(os.path.join(pat, "*.json")))
        elif os.path.exists(pat):
            hits = [pat]
        else:
            hits = sorted(glob.glob(pat))
        if not hits:
            unresolved.append(pat)
        for h in hits:
            if h not in paths and not h.endswith(".manifest.json"):
                paths.append(h)
    if unresolved:
        print(f"  !! matched nothing: {unresolved}")
    print(f"--recseg resolved to {len(paths)} file(s)")

    # basename alone is ambiguous here: data_recseg/ and data_recseg_part2/
    # both hold a recseg_val.json, and the run printed the same name twice
    # against different record counts
    def short(p):
        return os.path.join(os.path.basename(os.path.dirname(p)),
                            os.path.basename(p))

    recs, src_of, conflicts = {}, {}, []
    for p in paths:
        if not os.path.exists(p):
            print(f"  !! {p} not found")
            continue
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        n_new = 0
        for r in blob:
            rid = r.get("recording_id")
            if not rid:
                continue
            if rid in recs:
                a_ = [(str(x[0]), round(float(x[1]), 2), round(float(x[2]), 2))
                      for x in get_segments(recs[rid])[0]]
                b_ = [(str(x[0]), round(float(x[1]), 2), round(float(x[2]), 2))
                      for x in get_segments(r)[0]]
                if a_ != b_:
                    conflicts.append((rid, src_of[rid], short(p),
                                      len(a_), len(b_)))
                continue
            recs[rid] = r
            src_of[rid] = short(p)
            n_new += 1
        print(f"  {short(p):<44} {len(blob):>4} records, {n_new:>4} new")
    print(f"{len(evs)} audited events; {len(recs)} recordings loaded")
    if conflicts:
        print(f"  !! {len(conflicts)} recordings appear in more than one file "
              f"with DIFFERENT segments.\n     The first file wins, so the "
              f"windows depend on --recseg order. Resolve before\n     "
              f"trusting them:")
        for rid, first, other, na, nb in conflicts[:6]:
            print(f"     {rid}: {first} ({na} segs) kept over {other} "
                  f"({nb} segs)")

    rows, per_event = [], {}
    seg_id = {}
    no_rec, no_seg, label_mismatch = [], [], []
    for e in evs:
        eid = e.get("event_id")
        m = re.match(r"^(recording_\d+)", eid or "")
        rid = m.group(1) if m else None
        rec = recs.get(rid)
        if rec is None:
            no_rec.append(e["audit_key"])
            continue
        segs, _k = get_segments(rec)
        t = cand_time(eid)
        got = around(segs, t) if (segs and t is not None) else []
        if not got:
            no_seg.append(e["audit_key"])
            continue

        c = ctx.get(eid, {})
        expected = [x for x in (
            c.get("prev_segment_label")
            or c.get("nearest_previous_segment_label"),
            c.get("containing_segment_label"),
            c.get("next_segment_label")
            or c.get("nearest_next_segment_label")) if x]
        recovered = [str(s[0]) for s in got]
        # VERIFY: every label the annotator saw must be among the recovered
        # ones. Not the reverse -- `around` can legitimately return a segment
        # the context file did not name.
        unseen = [x for x in expected if x not in recovered]
        if unseen:
            label_mismatch.append((e["audit_key"], unseen, recovered))

        idxs = []
        for i, s in enumerate(got):
            key = (rid, round(float(s[1]), 2), round(float(s[2]), 2))
            if key not in seg_id:
                seg_id[key] = f"{rid}_s{round(float(s[1]), 2)}"
                rows.append({"segment_uid": seg_id[key], "recording_id": rid,
                             "start": round(float(s[1]), 2),
                             "end": round(float(s[2]), 2),
                             "duration": round(float(s[2]) - float(s[1]), 2),
                             "stored_label": str(s[0]),
                             "video": get_video(rec)})
            # WAS THIS SEGMENT ON THE SHEET? `around` recovers by time and
            # can return a neighbour the context file left null, so the
            # targets are a SUPERSET of what the annotator judged. Attaching
            # an event's claim_support to a segment nobody looked at would
            # attach a judgement to something it was not about, and that would
            # be invisible after the join.
            idxs.append({"seg_index": i, "segment_uid": seg_id[key],
                         "stored_label": str(s[0]),
                         "shown_in_sheet": str(s[0]) in expected})
        per_event[e["audit_key"]] = {"event_id": eid, "recording_id": rid,
                                     "candidate_time": t, "segments": idxs}

    ok = len(per_event)
    print(f"\nCOVERAGE: {ok}/{len(evs)} events resolved to segment windows")
    if no_rec:
        print(f"  !! recording not in --recseg: {len(no_rec)} {no_rec[:5]}")
    if no_seg:
        print(f"  !! no segment near the candidate: {len(no_seg)} "
              f"{no_seg[:5]}")
    print(f"  {len(rows)} DISTINCT segments after deduplication "
          f"(events reference "
          f"{sum(len(v['segments']) for v in per_event.values())})")
    n_seg = Counter(len(v["segments"]) for v in per_event.values())
    print(f"  segments per event: {dict(sorted(n_seg.items()))}")
    shown = sum(1 for v in per_event.values() for x in v["segments"]
                if x["shown_in_sheet"])
    total = sum(len(v["segments"]) for v in per_event.values())
    extra_ev = [k for k, v in per_event.items()
                if any(not x["shown_in_sheet"] for x in v["segments"])]
    print(f"\n  AUDITED vs RECOVERED: {shown}/{total} (event, segment) pairs "
          f"carry a label the\n  annotator was shown. The other "
          f"{total - shown} are neighbours the context file left\n  null and "
          f"`around` found by time -- real segments, but nobody judged them.")
    print(f"  {len(extra_ev)} events have at least one such segment: "
          f"{extra_ev[:8]}")
    print(f"  Run naming on all of them; JOIN claim_support only through "
          f"`shown_in_sheet: true`,\n  or a judgement gets attached to a "
          f"segment it was never about.")

    print(f"\nLABEL VERIFICATION -- recovered by time, checked by label:")
    if label_mismatch:
        print(f"  !! {len(label_mismatch)} events where a label the annotator "
              f"saw is NOT among the\n     segments recovered here. That means "
              f"this recseg is not the file the audit was\n     built from, "
              f"and the windows below are for different segments.")
        for k, unseen, rec_ in label_mismatch[:5]:
            print(f"     {k}: missing {unseen}")
            print(f"        recovered {rec_}")
    else:
        print(f"  every label the annotator saw appears among the recovered "
              f"segments, on all\n  {ok} events. The recseg join is the right "
              f"one.")

    d = [r["duration"] for r in rows]
    if d:
        d.sort()
        print(f"\n  segment duration: median {d[len(d)//2]:.1f}s   "
              f"p10 {d[len(d)//10]:.1f}s   p90 {d[9*len(d)//10]:.1f}s   "
              f"max {d[-1]:.1f}s")
        print(f"  these are the windows naming must receive. A candidate-"
              f"centred window of a few\n  seconds is a different input and "
              f"would produce a different experiment.")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mp = a.out.replace(".jsonl", "_event_map.json")
    json.dump(per_event, open(mp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nwrote {len(rows)} segment targets -> {a.out}")
    print(f"wrote the event-to-segment map -> {mp}")
    print(f"  run naming on `start`..`end` of each row, keep `segment_uid`, "
          f"and join back\n  through the map. Coverage of the audited "
          f"population is then {ok}/{len(evs)} by\n  construction rather than "
          f"by luck.")


if __name__ == "__main__":
    main()
