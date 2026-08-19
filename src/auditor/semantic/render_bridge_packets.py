"""Render the bridge-audit packets: clips the auditor can judge, and a blank sheet.

`bridge_audit_targets.json` names what to audit. This turns each packet into
one directory of clips plus one CSV with the frozen schema's columns empty, so
a packet is a self-contained hour of work.

WHAT MAY APPEAR ON SCREEN IS AN ALLOWLIST, NOT A DENYLIST. The packet file
carries `join_status`, `why_selected`, `sem_gap`, `tier` -- all of which say
what this batch expects the answer to be. Burning any of them into a frame
would let the gold be written by the hypothesis it exists to test, and a
renderer that prints "everything except the score" leaks the moment a field is
added. Only these reach the video:

    semantic   the candidate label, and the segment's clock time
    span       both clause labels, the clock time, and where the internal join
               falls

Nothing else. `--show_selection_reason` exists so that decision is visible as a
flag someone has to set, and it prints a warning when it is on.

THE JOIN IS MARKED IN TIME, NOT NAMED. A span clip runs A then B and the
question is whether the moment between them is a task boundary. The banner
switches from clause A's label to clause B's at the join, so the auditor sees
WHEN the annotation thinks the action changed without being told whether that
was judged correct.

Usage:
    python -m src.auditor.semantic.render_bridge_packets \
        --packets data/gold/bridge_audit_targets.json \
        --recseg '/workspace/tr1/data_recseg*/recseg_*.json' \
        --out_dir /workspace/tr1/results/auditor/bridge_packets \
        --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

from src.auditor.semantic.render_ontology_clips import (
    banner_png, get_video, have_filters, load_recordings, video_width)

SEM_COLUMNS = ["target_id", "recording_id", "start", "end", "candidate_label",
               "claim_support", "granularity", "major_action_missing",
               "action_presence", "segment_structure",
               "upstream_timing_issue", "auditor", "notes"]
# SPAN sheet v2. The first three packets returned as free text and every
# useful distinction in them had to be read out of a notes column, so the
# distinctions are columns now.
#
# `boundary_exists` IS NOT HERE ON PURPOSE. It is derived from `join_relation`
# by instance_relation_policy_v2. Asking an auditor for both the relation and
# its consequence puts the policy layer back inside the annotation sheet, and
# the two would disagree the first time the policy changed.
SPAN_COLUMNS = ["target_id", "recording_id", "start", "end", "internal_join",
                "clause_a", "clause_b",
                # the frozen ontology; boundary existence follows from this
                "join_relation",
                # timing, kept SEPARATE from existence: 242/span4 is a real
                # boundary whose join is 1.5s late, and collapsing it into one
                # task_boundary=YES loses the only EARLY/LATE evidence in a
                # class that has ten events in total
                "candidate_relation", "boundary_time",
                "boundary_interval_start", "boundary_interval_end",
                # the distinction the first three packets exposed: 8 of 12
                # joins were NOT task boundaries while the phase split itself
                # was valid. "the annotation saw a real change, and the change
                # is inside one action" is not the same finding as "the
                # annotation saw nothing", and a single NO cannot carry both
                "semantic_phase_split",
                # the evidence joint_policy_v1's first precedence block turns
                # on. Four values, not three: "looked but unsure" and "could
                # not see" are different information for later analysis and
                # identical for the policy, which licenses a reject only on
                # observed_absent
                "release_reset_restart",
                "auditor", "notes"]

JOIN_RELATIONS = ("new_action", "same_action_new_instance", "same_instance",
                  "cannot_determine")
CANDIDATE_RELATIONS = ("exact", "early", "late", "not_applicable")
PHASE_SPLIT = ("valid", "valid_but_labels_poor", "weak_or_incorrect",
               "not_applicable")
RESET = ("observed_present", "observed_absent", "uncertain", "not_observable")


def cut(video, lo, hi, banners, out_path, tmp_dir, ffmpeg_bin, filters):
    """One clip with time-switched banners composited over it.

    `banners` is [(start_offset, end_offset, text)] in clip-relative seconds.
    A span clip carries two: clause A until the join, clause B after it, which
    is how the join is shown without being described."""
    if "overlay" not in filters:
        raise SystemExit("this ffmpeg has no `overlay`; nothing can be "
                         "labelled and an unlabelled clip is not auditable.")
    w = video_width(video, ffmpeg_bin)
    inputs, filt, prev = ["-ss", f"{lo:.3f}", "-to", f"{hi:.3f}", "-i", video], [], "[0:v]"
    for i, (s, e, text) in enumerate(banners):
        png = os.path.join(tmp_dir, f"_b{i}.png")
        banner_png(text, w, png)
        inputs += ["-i", png]
        nxt = f"[v{i}]"
        filt.append(f"{prev}[{i + 1}:v]overlay=0:0:"
                    f"enable='between(t,{s:.3f},{e:.3f})'{nxt}")
        prev = nxt
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"] + inputs + [
        "-filter_complex", ";".join(filt), "-map", prev, "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", out_path]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets",
                    default="data/gold/bridge_audit_targets.json")
    ap.add_argument("--recseg", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad_s", type=float, default=3.0,
                    help="context before and after. The auditor needs to see "
                         "what led into the segment to judge whether the "
                         "label describes it")
    ap.add_argument("--limit", type=int, default=0,
                    help="render only the first N packets. They are already "
                         "in priority order")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--show_selection_reason", action="store_true",
                    help="burn why_selected into the clip. OFF by default: it "
                         "says what this batch expects the answer to be, and "
                         "the auditor's job is to not know that")
    ap.add_argument("--arms", default="both",
                    choices=["both", "sem", "span"],
                    help="which arm to render. `both` is the original "
                         "behaviour and stays the default. `span` stops the "
                         "semantic arm without touching the frozen packet "
                         "selection -- the sampling design and the remaining "
                         "packet order are unchanged, only the evidence "
                         "requested per packet")
    ap.add_argument("--sheets_only", action="store_true")
    a = ap.parse_args()

    blob = json.load(open(a.packets, encoding="utf-8"))
    packs = blob["packets"][:a.limit] if a.limit else blob["packets"]
    # load_recordings ALREADY returns {recording_id: record}. Re-keying it
    # iterated the dict, which yields its keys, and every "record" was a
    # string.
    recs = load_recordings(a.recseg)
    filters = have_filters(a.ffmpeg)
    print(f"{len(packs)} packets; arms={a.arms}; "
          f"ffmpeg filters available: {sorted(filters)}")
    if a.arms != "both":
        print(f"  the packet SELECTION is untouched -- same frozen file, same "
              f"order.\n  Only the evidence requested per packet changes, so "
              f"no adaptive sampling\n  bias is introduced by having seen the "
              f"first three results.")
    if a.show_selection_reason:
        print("  !! --show_selection_reason is ON. The clip will state why "
              "the target was\n     picked, which tells the auditor what this "
              "batch is hoping for.")
    os.makedirs(a.out_dir, exist_ok=True)

    missing, made = [], 0
    for pk in packs:
        rid = pk["recording_id"]
        rec = recs.get(rid)
        if not rec:
            missing.append(rid)
            continue
        video = get_video(rec)
        d = os.path.join(a.out_dir, rid)
        os.makedirs(d, exist_ok=True)
        sem_rows, span_rows = [], []

        for i, t in enumerate(pk.get("sem_targets", []) if a.arms in
                              ("both", "sem") else [], 1):
            tid = f"{rid}_SEM{i}"
            lo = max(0.0, float(t["start"]) - a.pad_s)
            hi = float(t["end"]) + a.pad_s
            txt = f"{t['label']}    [{t['start']:.1f}-{t['end']:.1f}s]"
            if a.show_selection_reason:
                txt += f"    ({'; '.join(t.get('why_selected', []))})"
            # THE LABEL SHOWS ONLY DURING THE LABELLED SPAN. A single banner
            # over the whole clip covered the padding too, so a 20s segment
            # rendered with 3s either side put the label on screen for all 26
            # seconds with nothing marking where the segment actually starts.
            # The frozen rule says claim_support is judged against the exact
            # segment the naming model saw; an auditor judging the padded clip
            # is judging a 6s-wider window, and the median segment is 9-10s.
            # The span arm already switches banners at the join -- this is the
            # same mechanism, and its absence here was an asymmetry, not a
            # decision.
            s0, s1 = float(t["start"]) - lo, float(t["end"]) - lo
            bans = []
            if s0 > 0.05:
                bans.append((0.0, s0, "— context before —"))
            bans.append((s0, s1, txt))
            if hi - float(t["end"]) > 0.05:
                bans.append((s1, hi - lo, "— context after —"))
            if not a.sheets_only:
                cut(video, lo, hi, bans,
                    os.path.join(d, f"{tid}.mp4"), d, a.ffmpeg, filters)
                made += 1
            sem_rows.append({"target_id": tid, "recording_id": rid,
                             "start": t["start"], "end": t["end"],
                             "candidate_label": t["label"]})

        for i, t in enumerate(pk.get("span_targets", []) if a.arms in
                               ("both", "span") else [], 1):
            tid = f"{rid}_SPAN{i}"
            lo = max(0.0, float(t["start"]) - a.pad_s)
            hi = float(t["end"]) + a.pad_s
            j = float(t["internal_join"]) - lo
            # TWO BANNERS, SWITCHING AT THE JOIN. The auditor sees when the
            # annotation says the action changed, and is never told whether
            # anyone has judged that correct.
            bans = [(0.0, j, f"A: {t['clause_a']}"),
                    (j, hi - lo, f"B: {t['clause_b']}")]
            if not a.sheets_only:
                cut(video, lo, hi, bans,
                    os.path.join(d, f"{tid}.mp4"), d, a.ffmpeg, filters)
                made += 1
            span_rows.append({"target_id": tid, "recording_id": rid,
                              "start": t["start"], "end": t["end"],
                              "internal_join": t["internal_join"],
                              "clause_a": t["clause_a"],
                              "clause_b": t["clause_b"]})

        want = {"both": ("sem", "span"), "sem": ("sem",),
                "span": ("span",)}[a.arms]
        for name, cols, rows in [x for x in
                                 (("sem_sheet.csv", SEM_COLUMNS, sem_rows),
                                  ("span_sheet.csv", SPAN_COLUMNS, span_rows))
                                 if x[0].split("_")[0] in want]:
            with open(os.path.join(d, name), "w", newline="",
                      encoding="utf-8") as f:
                wri = csv.DictWriter(f, fieldnames=cols)
                wri.writeheader()
                for r in rows:
                    wri.writerow({c: r.get(c, "") for c in cols})

    print(f"\nwrote {made} clips and {2 * (len(packs) - len(missing))} sheets "
          f"-> {a.out_dir}")
    if missing:
        print(f"  !! {len(missing)} packets had no recording in --recseg: "
              f"{missing[:6]}")
    print(f"\n  sem_sheet.csv: fill `claim_support` with yes / partial / no / "
          f"uncertain.\n    Audit ALL FOUR even after a `no` appears -- one "
          f"yes and one no in a recording\n    is ONE pair, and the target is "
          f"50 within-recording pairs.")
    print(f"  span_sheet.csv:")
    print(f"    join_relation        {' / '.join(JOIN_RELATIONS)}")
    print(f"    candidate_relation   {' / '.join(CANDIDATE_RELATIONS)}")
    print(f"    semantic_phase_split {' / '.join(PHASE_SPLIT)}")
    print(f"    release_reset_restart {' / '.join(RESET)}")
    print(f"    same_instance means NO boundary. A change of motion direction "
          f"is not a\n    boundary -- wiping left then wiping right is one "
          f"instance of wiping.")
    print(f"    `candidate_relation` is asked SEPARATELY from `join_relation`: "
          f"a boundary can\n    exist and its join still be late, and that "
          f"pair is the scarcest supervision\n    in the project.")
    print(f"    `semantic_phase_split` is what makes a NO informative. Eight of "
          f"the first\n    twelve joins were not task boundaries while the "
          f"phase split was valid -- the\n    annotation saw a real change "
          f"that sits inside one action.")
    print(f"    `release_reset_restart`: only `observed_absent` is an "
          f"observation of absence.\n    `uncertain` and `not_observable` both "
          f"block, and are kept apart because they\n    are different "
          f"information even though the policy treats them alike.")
    # UPDATED AFTER THE FIRST THREE PACKETS. This used to say the verdict had
    # never once been recorded; recordings 176/242/250 returned ten of them.
    # Leaving the old sentence up would have told the next auditor they were
    # hunting something already found.
    print(f"  As of packets 176/242/250 the `same_instance` verdict is no "
          f"longer missing: ten of\n  twelve joins came back NOT a task "
          f"boundary and two of three recordings hold both\n  verdicts. The "
          f"target is now RECORDINGS holding both -- 20 of them -- not more "
          f"NOs.")


if __name__ == "__main__":
    main()
