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
SPAN_COLUMNS = ["target_id", "recording_id", "start", "end", "internal_join",
                "clause_a", "clause_b", "join_relation", "boundary_time",
                "boundary_interval_start", "boundary_interval_end",
                "auditor", "notes"]

JOIN_RELATIONS = ("new_action", "same_action_new_instance", "same_instance",
                  "cannot_determine")


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
    ap.add_argument("--sheets_only", action="store_true")
    a = ap.parse_args()

    blob = json.load(open(a.packets, encoding="utf-8"))
    packs = blob["packets"][:a.limit] if a.limit else blob["packets"]
    recs = {r.get("recording_id"): r for r in load_recordings(a.recseg)}
    filters = have_filters(a.ffmpeg)
    print(f"{len(packs)} packets; ffmpeg filters available: {sorted(filters)}")
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

        for i, t in enumerate(pk.get("sem_targets", []), 1):
            tid = f"{rid}_SEM{i}"
            lo = max(0.0, float(t["start"]) - a.pad_s)
            hi = float(t["end"]) + a.pad_s
            txt = f"{t['label']}    [{t['start']:.1f}-{t['end']:.1f}s]"
            if a.show_selection_reason:
                txt += f"    ({'; '.join(t.get('why_selected', []))})"
            if not a.sheets_only:
                cut(video, lo, hi, [(0.0, hi - lo, txt)],
                    os.path.join(d, f"{tid}.mp4"), d, a.ffmpeg, filters)
                made += 1
            sem_rows.append({"target_id": tid, "recording_id": rid,
                             "start": t["start"], "end": t["end"],
                             "candidate_label": t["label"]})

        for i, t in enumerate(pk.get("span_targets", []), 1):
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

        for name, cols, rows in (("sem_sheet.csv", SEM_COLUMNS, sem_rows),
                                 ("span_sheet.csv", SPAN_COLUMNS, span_rows)):
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
    print(f"  span_sheet.csv: fill `join_relation` with one of "
          f"{'/'.join(JOIN_RELATIONS)}.\n    same_instance means NO boundary. "
          f"A change of motion direction is not a\n    boundary -- wiping left "
          f"then wiping right is one instance of wiping.")
    print(f"  The scarce verdict is `same_instance`: zero recordings currently "
          f"hold a join\n  judged not to be a boundary, and a confirmed one "
          f"has no counterpart without it.")


if __name__ == "__main__":
    main()
