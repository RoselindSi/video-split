"""Turn the segment targets into a naming `--data` file, plus the join key.

THE JOIN KEY IS THE POINT. eval_naming_decoupled feeds a whole video, asks for
one name per GT segment, and writes `pred_names` as a POSITIONAL list aligned
to `_as_segs(r["solution"])`. The segment manifest identifies segments by
(recording, start, end). Those do not join, and running naming before noticing
would produce predictions nobody could attach to an audited segment -- the
same coverage failure as before, one level down.

So this emits both:

    <out>.json        the naming --data file: one row per recording, `video`
                      plus `solution` as a LIST of [name, start, end] triples,
                      which is the shape _as_segs expects. recseg stores its
                      segments under a `solution.segments` wrapper and
                      _as_segs would index straight into that and fail.
    <out>_join.json   segment_uid -> {recording_id, position, start, end}

Position AND bounds are both written so a mismatch is detectable rather than
silent: if the naming output ever changes order, the bounds disagree and the
join fails loudly instead of attaching the wrong name.

THE FULL SEGMENT LIST GOES IN, NOT JUST THE TARGETS. Naming is asked to
produce one name per segment of the recording; handing it only the audited
segments would be a different task with a different prompt length and a
different error rate, and the number that came out would not be the number the
pipeline produces. The targets are selected FROM the output afterwards.

WHAT CAN STILL GO WRONG, and is reported before the run rather than after:
`pred_names` is truncated to min(len(pred), len(gt)), and this decoder is
known to degenerate on long lists -- the eval's own comment records a
147-segment recording returning 36 identical names. A target sitting late in a
long recording may simply have no prediction. The per-recording segment counts
and where the targets sit in them are printed, so that is a known risk with a
size rather than a surprise in the results.

Usage:
    python -m src.auditor.semantic.naming_run_spec \
        --targets /workspace/tr1/results/auditor/naming_targets_48.jsonl \
        --targets /workspace/tr1/results/auditor/naming_targets_enrichment.jsonl \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --out /workspace/tr1/results/auditor/naming_run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from src.auditor.semantic.render_ontology_clips import get_segments, get_video


def resolve(patterns):
    paths = []
    for pat in patterns:
        hits = (sorted(glob.glob(os.path.join(pat, "*.json")))
                if os.path.isdir(pat) else
                ([pat] if os.path.exists(pat) else sorted(glob.glob(pat))))
        if not hits:
            print(f"  !! matched nothing: {pat}")
        for h in hits:
            if h not in paths and not h.endswith(".manifest.json"):
                paths.append(h)
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", action="append", required=True,
                    help="naming_targets_*.jsonl from naming_targets.py")
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="how close a stored segment must be to a target's "
                         "bounds to be that segment")
    ap.add_argument("--out", required=True, help="path prefix")
    a = ap.parse_args()

    targets = []
    for p in resolve(a.targets):
        with open(p, encoding="utf-8") as f:
            n = 0
            for line in f:
                if line.strip():
                    targets.append(json.loads(line))
                    n += 1
        print(f"  {os.path.basename(p):<44} {n:>4} target segments")
    want = defaultdict(list)
    for t in targets:
        want[t["recording_id"]].append(t)
    print(f"{len(targets)} target segments over {len(want)} recordings")

    recs = {}
    for p in resolve(a.recseg):
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):
            blob = blob.get("recordings") or blob.get("data") or []
        for r in blob:
            rid = r.get("recording_id")
            if rid and rid not in recs and rid in want:
                recs[rid] = r
    missing = [r for r in want if r not in recs]
    if missing:
        print(f"  !! {len(missing)} target recordings not in --recseg: "
              f"{missing[:5]}")

    rows, join, unresolved = [], {}, []
    counts = []
    for rid, rec in recs.items():
        segs = get_segments(rec)[0]          # stored order, NOT re-sorted
        triples = [[str(s[0]), float(s[1]), float(s[2])] for s in segs]
        rows.append({"recording_id": rid, "video": get_video(rec),
                     "solution": triples})
        counts.append(len(triples))
        for t in want[rid]:
            hit = [i for i, (_l, st, en) in enumerate(triples)
                   if abs(st - t["start"]) <= a.eps
                   and abs(en - t["end"]) <= a.eps]
            if len(hit) == 1:
                join[t["segment_uid"]] = {
                    "recording_id": rid, "position": hit[0],
                    "start": triples[hit[0]][1], "end": triples[hit[0]][2],
                    "stored_label": triples[hit[0]][0],
                    "n_segments_in_recording": len(triples)}
            else:
                unresolved.append((t["segment_uid"], len(hit)))

    print(f"\nJOIN KEY: {len(join)}/{len(targets)} target segments located by "
          f"position")
    if unresolved:
        amb = sum(1 for _u, n in unresolved if n > 1)
        print(f"  !! {len(unresolved)} unresolved ({amb} matched more than "
              f"one stored segment,\n     {len(unresolved) - amb} matched "
              f"none). Those cannot be joined back and would be\n     "
              f"silently absent from the diagnostic: {unresolved[:5]}")

    if counts:
        counts.sort()
        print(f"\n  segments per recording: median {counts[len(counts)//2]}  "
              f"max {counts[-1]}  (total {sum(counts)} names requested for "
              f"{len(join)} needed)")
        deep = [v for v in join.values() if v["position"] > 40]
        print(f"  {len(deep)} targets sit beyond position 40 in their "
              f"recording. `pred_names` is\n  truncated to "
              f"min(len(pred), len(gt)) and this decoder is documented "
              f"degenerating on\n  long lists, so those are the ones most "
              f"likely to come back without a prediction.")

    dp, jp = a.out + ".json", a.out + "_join.json"
    json.dump(rows, open(dp, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(join, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nwrote {dp}  ({len(rows)} recordings)")
    print(f"wrote {jp}  ({len(join)} segments)")
    print(f"\nthen:\n  python -m src.eval.eval_naming_decoupled "
          f"--model_base <ckpt> \\\n    --data {dp} --out "
          f"{a.out}_pred.jsonl")
    print(f"\n  join `pred_names[position]` through {os.path.basename(jp)}, "
          f"and CHECK the bounds\n  alongside the position -- if the naming "
          f"output ever reorders, the bounds catch it\n  and the position "
          f"silently would not.")


if __name__ == "__main__":
    main()
