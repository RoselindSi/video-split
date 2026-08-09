"""Blind re-annotation of the 37 INTERVAL events into transition topology.

WHY THESE 37 AND NOTHING ELSE. Boundary v1 preserved the legacy signal --
POINT vs NONE 0.797 against the frozen fused 0.821, paired delta -0.024
[-0.075, +0.023] -- and produced no new morphology discrimination:
POINT vs INTERVAL 0.709 where local-alone reads 0.745, INTERVAL vs NONE 0.626.
The score geometry says why it might be unlearnable rather than unlearned. On
P(POINT), 7 of the 37 sit at or above the POINT median, 18 at or below the
NONE median and 12 between. A single class that is simply "a wider transition"
would sit in the middle; mass at both ends is what a semantic umbrella label
looks like.

THAT IS A HYPOTHESIS AND NOT A FINDING. A bimodal score is equally consistent
with the model failing systematically on some gradual events, or with the
folds differing in composition -- the per-fold POINT-vs-INTERVAL AUROC spans
0.558 to 0.871, a spread of 0.313, which is far too large to read as noise.
Only a person watching the video separates those explanations, which is why
this is the next step and why no architecture changes until it returns.

BLIND MEANS BLIND TO THE MODEL. The sheet carries the clip and the
neighbouring segment labels. P(POINT), the frozen scores, the fold and the
score plot all stay out -- `score_plot_path` in particular is a picture of the
model's own opinion, and pair_labels_v1.csv carries one for most of these
events. An annotator who can see the score will call the high ones point_like
and the low ones smooth, and the table afterwards will confirm exactly that.

The annotator does know every event is currently `gradual`; that is
unavoidable, since sub-typing them is the task.

THE VOCABULARY, and `point_like` is in it on purpose:

  smooth_ramp             one interaction state changes continuously with no
                          clear local break; any cut is arbitrary
  overlapping_transition  the old interaction has not ended when the new one
                          begins -- common when the two hands are out of step
  multi_step_transition   several discrete micro-events (release, reach, idle,
                          new contact) and no single one owns the boundary
  point_like              on review there IS a compact, repeatably locatable
                          switch, even if motion continues on both sides
  ambiguous_interval      none of the above hold stably

Without `point_like` the 7 point-like scores would be forced into some gradual
subtype and label noise would be relabelled as morphology. It is not an
accusation of a bad original call: `gradual` was assigned against a different
question than the one asked here.

Usage:
    python -m src.auditor.boundary.interval_audit_sheet \
        --labels data/gold/boundary_v1_labels.json \
        --pair_labels data/gold/pair_labels_v1.csv \
        --data .../recseg_train.json --data .../recseg_val.json \
        --out_dir /workspace/tr1/results/auditor/interval_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

SUBTYPES = ["smooth_ramp", "overlapping_transition", "multi_step_transition",
            "point_like", "ambiguous_interval"]
INTERVAL = "INTERVAL_TRANSITION"
# never written into the blind sheet: each is the model's own opinion
LEAKS = ("score_plot_path", "pred_score", "score", "fold", "priority",
         "priority_reason", "raw_left_right_distance")

GUIDELINES = """# Sub-typing the 37 gradual events

You are re-watching 37 events that were all labelled `gradual_phase_transition`.
The question is NOT whether that was right. It is what KIND of extended
transition each one is, because "gradual" may be covering several visually
different things, and a model cannot learn one class out of several shapes.

Watch the clip. Do not look at any model output. Fill `your_call` first, then
`confidence`, then one line of `why`.

## smooth_ramp
One interaction state changes continuously into another. No clear local break,
no new contact event that stands out. Cutting anywhere feels arbitrary.

    press -> press-lighter -> lift-slightly -> lift

## overlapping_transition
The old interaction has not finished when the new one starts. Very common when
the two hands are doing different things: one is still holding while the other
has already reached for the next object. There is no single instant where "old
ended" and "new began" coincide, because they did not.

    left hand:   holding box ----------------
    right hand:        --------- opening bag

## multi_step_transition
Several discrete little events happen in a row -- release, reach, a pause,
a new contact -- and no single one of them is THE boundary. It is not smooth;
it is several steps.

    release cup -> hand travels -> pauses -> grasps cloth

## point_like
On this viewing there IS a compact switch you could point at and would point at
again tomorrow, even if the hands keep moving before and after. Say so. This is
not a complaint about the original label -- `gradual` was assigned against a
different question.

## ambiguous_interval
None of the above holds stably. Use it rather than forcing a choice; a
reluctant guess is worse than a recorded uncertainty.

## confidence
    1  guess
    2  lean
    3  sure

## What NOT to do
Do not try to be consistent with what you think the model would say, and do not
go back to change earlier rows once you notice a pattern. The point of the
exercise is the distribution, and a distribution you smoothed by hand answers
nothing.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True,
                    help="boundary_v1_labels.json")
    ap.add_argument("--pair_labels", action="append", default=[],
                    help="for the neighbouring segment labels and clip paths")
    ap.add_argument("--data", action="append", default=[],
                    help="recseg json(s), to write the clip commands")
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    lab = json.load(open(a.labels, encoding="utf-8"))["events"]
    ev = [e for e in lab if e.get("morphology") == INTERVAL]
    print(f"{len(ev)} INTERVAL events over "
          f"{len({e['recording_id'] for e in ev})} recordings")
    print(f"  by source: audited "
          f"{sum(1 for e in ev if e.get('audited'))}, batch3 "
          f"{sum(1 for e in ev if not e.get('audited'))}")

    ctx = {}
    for p in a.pair_labels:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ctx.setdefault(r.get("event_id"), r)
    video = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video[r["recording_id"]] = r.get("video")

    # shuffled, so the order carries no information about source or recording
    import random
    rng = random.Random(a.seed)
    rng.shuffle(ev)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "GUIDELINES.md"), "w",
              encoding="utf-8") as f:
        f.write(GUIDELINES)

    sheet = os.path.join(a.out_dir, "interval_audit_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"your_call({'|'.join(SUBTYPES)})",
                    "confidence(1_guess|2_lean|3_sure)", "why_one_line",
                    "event_id", "clip", "prev_segment_label",
                    "next_segment_label", "containing_segment_label"])
        for e in ev:
            c = ctx.get(e["event_id"], {})
            w.writerow(["", "", "", e["event_id"],
                        f"{e['event_id']}.mp4",
                        c.get("prev_segment_label", ""),
                        c.get("next_segment_label", ""),
                        c.get("containing_segment_label", "")])
    # the answer columns come first so they are what the eye lands on, and the
    # key is a separate file so the sheet can be handed over whole
    key = os.path.join(a.out_dir, "interval_audit_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "recording_id", "candidate_time", "audited",
                    "candidate_relation", "offset_s"])
        for e in ev:
            w.writerow([e["event_id"], e["recording_id"], e["candidate_time"],
                        e.get("audited"), e.get("candidate_relation"),
                        e.get("offset_s")])

    man = os.path.join(a.out_dir, "interval_audit_manifest.jsonl")
    with open(man, "w", encoding="utf-8") as f:
        for e in ev:
            f.write(json.dumps({"event_id": e["event_id"],
                                "recording_id": e["recording_id"],
                                "t": e["candidate_time"]},
                               ensure_ascii=False) + "\n")

    sh = os.path.join(a.out_dir, "make_clips.sh")
    miss = []
    with open(sh, "w", encoding="utf-8") as f:
        # NO `set -e`. One unreadable source must not abort the other 36 and
        # leave a silently short folder; the failures are collected and named
        # at the end instead.
        f.write("#!/bin/sh\n# clips centred on the candidate; the candidate "
                "is the MIDDLE of each clip and is not marked,\n# because a "
                "marker would tell the annotator where the answer is "
                "supposed to be.\n")
        f.write('D="$(dirname "$0")"\nmkdir -p "$D/clips"\nfail=0\n')
        n_cmd = 0
        for e in ev:
            v = video.get(e["recording_id"])
            if not v:
                miss.append(e["event_id"])
                continue
            n_cmd += 1
            t0 = max(0.0, float(e["candidate_time"]) - a.half_s)
            out = f'$D/clips/{e["event_id"]}.mp4'
            f.write(f'if [ ! -f "{v}" ]; then echo "MISSING SOURCE '
                    f'{e["event_id"]}: {v}"; fail=$((fail+1)); else\n')
            f.write(f'  ffmpeg -nostdin -loglevel error -y -ss {t0:.2f} '
                    f'-i "{v}" -t {2 * a.half_s:.2f} '
                    f'-c:v libx264 -crf 23 -an "{out}" '
                    f'|| {{ echo "FFMPEG FAILED {e["event_id"]}"; '
                    f'fail=$((fail+1)); }}\n')
            f.write("fi\n")
        # the sheet has one row per event; a folder with fewer clips than rows
        # is the failure mode that prompted this, so the script counts for you
        f.write(f'\nn=$(ls "$D/clips" | wc -l)\n'
                f'echo "clips written: $n of {len(ev)} sheet rows '
                f'({n_cmd} had a source path, {len(miss)} had none)"\n'
                f'if [ "$n" -lt {len(ev)} ]; then\n'
                f'  echo "MISSING, in sheet order:"\n'
                f'  while IFS=, read -r c1 c2 c3 eid rest; do\n'
                f'    [ "$eid" = "event_id" ] && continue\n'
                f'    [ -n "$eid" ] && [ ! -f "$D/clips/$eid.mp4" ] '
                f'&& echo "  $eid"\n'
                f'  done < "$D/interval_audit_sheet.csv"\n'
                f'fi\n')
    os.chmod(sh, 0o755)
    if miss:
        with open(os.path.join(a.out_dir, "no_video_path.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(miss) + "\n")

    print(f"\nwrote {a.out_dir}/")
    print(f"  GUIDELINES.md                 the five subtypes, with examples")
    print(f"  interval_audit_sheet.csv      {len(ev)} rows, answer columns "
          f"first, shuffled")
    print(f"  interval_audit_key.csv        event -> recording and timing, "
          f"SEPARATE so the sheet can be handed over whole")
    print(f"  make_clips.sh                 +/-{a.half_s}s clips, candidate "
          f"unmarked at the centre")
    if miss:
        rec = sorted({e["recording_id"] for e in ev
                      if e["event_id"] in set(miss)})
        print(f"\n  !! {len(miss)} of {len(ev)} events have NO VIDEO PATH and "
              f"got no clip command. The sheet still has {len(ev)} rows, so "
              f"the clip\n     folder will be short by exactly that many and "
              f"the sheet cannot be filled in full.")
        print(f"     They span {len(rec)} recordings not present in any "
              f"--data file: {rec[:6]}")
        print(f"     Written to {a.out_dir}/no_video_path.txt. Add the "
              f"--data json that holds those recordings and rerun; the batch3 "
              f"recordings\n     are usually in a different file from the "
              f"dev ones.")
    leaked = [c for c in LEAKS if any(c in (ctx.get(e["event_id"]) or {})
                                      for e in ev)]
    print(f"\n  columns deliberately NOT carried into the sheet: {LEAKS}")
    if leaked:
        print(f"  ({', '.join(leaked)} exist in the pair-label file and were "
              f"dropped; score_plot_path is a picture of the model's own\n   "
              f"opinion and would decide the answer before the video is "
              f"watched)")
    print(f"\n  Fill the sheet, then freeze it, THEN join it to the scores "
          f"with interval_audit_join. Joining first, or looking at\n  the "
          f"scores while labelling, produces the table it was meant to test.")


if __name__ == "__main__":
    main()
