"""Render clip + contact_sheet ONLY (no score_plot) for a batch3 sample.

No score_plot is a deliberate, structural enforcement of blind review, not
just an instruction to the reviewer: a probability curve would show the
frozen artifact's confidence and where it crosses threshold, which is
exactly the anchoring the review's protocol says to avoid ("no model score,
no provisional decision visible"). If a score plot existed, "don't look at
it" relies on the reviewer's discipline; not generating it at all doesn't.

The contact sheet and clip caption are built from a row with `category`,
`gt_time`, `pred_time`, `offset`, `pred_score` all blanked -- render_audit_
media.py's make_contact_sheet prints those fields directly onto the image,
so passing real values would bake the candidate_type/score into the picture
even if the CSV column is absent. Only segment-label context (legitimate,
not a model output) is shown.

Usage (server):
    python -m src.boundary.render_batch3_media \
        --manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --blind_csv /workspace/tr1/results/hal/batch3/batch3_blind_review.csv \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg_part2/recseg_train.json \
        --data /workspace/tr1/data_recseg_part2/recseg_val.json \
        --out_dir /workspace/tr1/results/hal/batch3/media \
        --ffmpeg_bin /usr/bin/ffmpeg
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from decord import VideoReader

from src.boundary.render_audit_media import make_clip, make_contact_sheet, label_pair_text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="batch3_manifest.jsonl (for recording_id/t only)")
    ap.add_argument("--blind_csv", required=True, help="batch3_blind_review.csv (for segment-label context)")
    ap.add_argument("--data", action="append", required=True,
                    help="recseg json(s) for video paths -- repeat for train/val/part2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_s", type=float, default=3.0)
    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    a = ap.parse_args()

    video_path = {}
    for path in a.data:
        for r in json.load(open(path, encoding="utf-8")):
            video_path[r["recording_id"]] = r["video"]
    print(f"video paths available: {len(video_path)} recordings")

    ctx = {}
    with open(a.blind_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            ctx[row["event_id"]] = row

    media_dir = os.path.join(a.out_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    events = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    n_ok = n_missing_video = n_clip_fail = 0
    for e in events:
        eid, rid, t = e["event_id"], e["recording_id"], e["t"]
        vp = video_path.get(rid)
        if vp is None:
            print(f"  !! {eid}: no video path for recording {rid}")
            n_missing_video += 1
            continue
        c = ctx.get(eid, {})
        # BLIND row: only legitimate segment-label context, everything that
        # would reveal candidate_type/score/decision is left blank.
        blind_row = {
            "category": "", "gt_time": "", "pred_time": "", "offset": "", "pred_score": "",
            "prev_segment_label": c.get("prev_segment_label", ""),
            "next_segment_label": c.get("next_segment_label", ""),
            "containing_segment_label": c.get("containing_segment_label", ""),
            "nearest_next_label": "", "nearest_next_gap_s": "",
            # label_pair_text()'s fallback branch (prev/next/nearest_next all
            # empty but containing_segment_label set -- happens for candidates
            # near a recording's start/end) reads these two keys directly;
            # batch3_blind_review.csv never populates them, so they must still
            # be PRESENT (as "?") or the KeyError crashes the whole render run.
            "nearest_previous_segment_label": "", "nearest_next_segment_label": "",
            "recording_id": rid,
        }
        caption = label_pair_text(blind_row)
        clip_path = os.path.join(media_dir, f"{eid}.mp4")
        contact_path = os.path.join(media_dir, f"{eid}_contact_sheet.png")
        clip_ok = make_clip(vp, t, a.window_s, clip_path, caption, a.ffmpeg_bin)
        contact_ok = False
        try:
            vr = VideoReader(vp, num_threads=1)
            sheet = make_contact_sheet(vr, vr.get_avg_fps(), t, a.window_s, [], [], blind_row)
            sheet.save(contact_path)
            contact_ok = True
            del vr
        except Exception as ex:
            print(f"  !! {eid}: contact sheet failed ({ex})")
        if clip_ok and contact_ok:
            n_ok += 1
        else:
            n_clip_fail += 1
        c["clip_path"] = clip_path if clip_ok else ""
        c["contact_sheet_path"] = contact_path if contact_ok else ""
        c["event_id"] = eid
        ctx[eid] = c
        print(f"{eid}: clip={'ok' if clip_ok else 'FAILED'} contact_sheet={'ok' if contact_ok else 'FAILED'}")

    out_csv = a.blind_csv.replace(".csv", "_with_media.csv") if not a.blind_csv.endswith("_with_media.csv") else a.blind_csv
    fieldnames = list(csv.DictReader(open(a.blind_csv, encoding="utf-8", errors="replace")).fieldnames)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in events:
            w.writerow(ctx.get(e["event_id"], {}))
    print(f"\nrendered {n_ok}/{len(events)} events "
          f"({n_missing_video} missing video, {n_clip_fail} render failures)")
    print(f"wrote {out_csv} (blind review sheet with clip/contact_sheet paths filled in)")


if __name__ == "__main__":
    main()
