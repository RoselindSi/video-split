"""Segment-span media for the semantic ontology audit.

The existing error_audit clips are +/-3s around the candidate. Four of the
eight questions on the sheet cannot be answered inside six seconds:

    q1  what does the video actually show      needs the whole action
    q6  is the label accurate but too coarse   needs what it is coarse OVER
    q7  does the segment contain more than
        one phase                              needs the segment, by definition

A six-second window centred on a boundary shows the end of one action and the
start of another, which is exactly the evidence for a boundary question and
exactly the wrong evidence for a segment question. Rendering the segment span
is not a nicety here; the sheet is unanswerable without it.

WHAT IS RENDERED, per event:

    {event_id}_span.mp4     from the start of the previous segment to the end
                            of the next one, with the CURRENT segment's label
                            burned in and changing as the clip crosses each
                            boundary, plus a red bar at the candidate time
    {event_id}_span.png     a contact sheet across the same span, each frame
                            stamped with its time and its segment index, with
                            a rule drawn where the segment changes

THE LABEL OVERLAY IS THE POINT. Burning the label in and letting it change
mid-clip is what makes "the two neighbours are the same label" visible --
41 of the 188 events have neighbour labels sharing most of their words, and
on those the overlay will read the same before and after the candidate. That
is the finding the audit needs to see rather than be told.

THE SPAN IS CAPPED at --max_span_s and, when it has to be cut, is cut
symmetrically around the candidate with the truncation stated in the overlay,
because a segment that ran 400 seconds would otherwise produce a clip nobody
watches and a contact sheet at 40s per frame.

Usage (server):
    python -m src.auditor.semantic.render_ontology_clips \
        --sheet data/gold/semantic_ontology_audit.csv \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --out_dir /workspace/tr1/results/auditor/semantic_ontology_media
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter

TIME = re.compile(r"_t(\d+(?:\.\d+)?)$")


def cand_time(eid):
    m = TIME.search(eid)
    return float(m.group(1)) if m else None


def esc(t):
    """ffmpeg drawtext is parsed twice; colons and quotes must survive both."""
    return (str(t).replace("\\", "\\\\").replace(":", "\\:")
            .replace("'", "").replace("%", ""))


# The segment list is not always under `segments`, and a recording whose
# segments are under another key looks exactly like a recording with no
# segments. That cost a whole render pass, so the key is detected and the
# detection is printed.
SEG_KEYS = ("segments", "gt_segments", "segs", "labels", "annotations",
            "boundaries")
VIDEO_KEYS = ("video", "video_path", "path", "mp4")


def get_segments(rec):
    """First key holding a list of [label, start, end] triples."""
    for k in SEG_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)) \
                and len(v[0]) >= 3:
            return list(v), k
        if isinstance(v, list) and v and isinstance(v[0], dict) \
                and {"start", "end"} <= set(v[0]):
            lab = next((c for c in ("label", "name", "text", "caption")
                        if c in v[0]), None)
            return ([[d.get(lab, ""), d["start"], d["end"]] for d in v], k)
    return [], None


def get_video(rec):
    for k in VIDEO_KEYS:
        if rec.get(k):
            return rec[k]
    return None


def load_recordings(paths):
    """recseg json: records with recording_id, video, segments [[label,s,e]]."""
    recs = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  !! {p} not found")
            continue
        blob = json.load(open(p, encoding="utf-8"))
        if isinstance(blob, dict):  # {rid: rec} or {"recordings": [...]}
            blob = (blob.get("recordings") or blob.get("data")
                    or [dict(v, recording_id=v.get("recording_id", k))
                        for k, v in blob.items() if isinstance(v, dict)])
        n_seg = 0
        keys = Counter()
        for r in blob:
            rid = r.get("recording_id") or r.get("id") or r.get("name")
            if not rid:
                continue
            segs, k = get_segments(r)
            keys[k] += 1
            if segs:
                n_seg += 1
            if rid not in recs:
                recs[rid] = r
        print(f"  {os.path.basename(p)}: {len(blob)} records, {n_seg} with "
              f"segments, key(s) {dict(keys)}")
        if blob and not n_seg:
            print(f"    !! no segment list found. Keys on the first record: "
                  f"{sorted(blob[0])[:20]}")
    return recs


def span_for(segs, t, max_span):
    """Previous, containing and next segment around t -- and the cut span.

    Segments are not contiguous in this dataset, so `containing` may be None
    while both neighbours exist. That is the 105-event case the audit is
    about, and it is passed through rather than smoothed over."""
    segs = sorted(segs, key=lambda s: float(s[1]))
    contain = next((s for s in segs if float(s[1]) <= t <= float(s[2])), None)
    prev = [s for s in segs if float(s[2]) <= t]
    nxt = [s for s in segs if float(s[1]) >= t]
    prev = prev[-1] if prev else None
    nxt = nxt[0] if nxt else None
    parts = [s for s in (prev, contain, nxt) if s]
    if not parts:
        return None, None, (prev, contain, nxt), False
    lo = min(float(s[1]) for s in parts)
    hi = max(float(s[2]) for s in parts)
    cut = False
    if hi - lo > max_span:  # keep the candidate centred and say so
        lo, hi = max(0.0, t - max_span / 2), t + max_span / 2
        cut = True
    return lo, hi, (prev, contain, nxt), cut


def build_filter(segs, lo, hi, t, cut, font_h=28):
    """drawtext per segment, enabled over that segment's slice of the clip."""
    f = []
    for s in sorted(segs, key=lambda s: float(s[1])):
        a, b = max(float(s[1]), lo) - lo, min(float(s[2]), hi) - lo
        if b <= 0 or a >= hi - lo:
            continue
        f.append(f"drawtext=text='{esc(s[0])}':x=10:y=10:fontsize={font_h}:"
                 f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6:"
                 f"enable='between(t,{a:.2f},{b:.2f})'")
    f.append(f"drawtext=text='t\\={t:.1f}s':x=10:y=h-40:fontsize=22:"
             f"fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=4")
    f.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.8:t=12:"
             f"enable='between(t,{max(0.0, t - lo - 0.2):.2f},"
             f"{t - lo + 0.2:.2f})'")
    if cut:
        f.append("drawtext=text='SPAN TRUNCATED':x=10:y=h-75:fontsize=20:"
                 "fontcolor=orange:box=1:boxcolor=black@0.6:boxborderw=4")
    return ",".join(f)


def contact_sheet(video, lo, hi, t, segs, out_path, n=12, thumb_w=200):
    from decord import VideoReader
    from PIL import Image, ImageDraw
    vr = VideoReader(video)
    fps = vr.get_avg_fps()
    times = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
    order = {id(s): i for i, s in
             enumerate(sorted(segs, key=lambda s: float(s[1])))}

    def seg_at(x):
        for s in segs:
            if float(s[1]) <= x <= float(s[2]):
                return s
        return None

    imgs, tags = [], []
    for x in times:
        idx = max(0, min(len(vr) - 1, int(x * fps)))
        im = Image.fromarray(vr[idx].asnumpy())
        w, h = im.size
        imgs.append(im.resize((thumb_w, int(h * thumb_w / w))))
        s = seg_at(x)
        tags.append(order[id(s)] if s is not None else None)

    cols = 6
    rows = (n + cols - 1) // cols
    H = max(i.height for i in imgs)
    sheet = Image.new("RGB", (thumb_w * cols, (H + 46) * rows + 60), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, x, tg) in enumerate(zip(imgs, times, tags)):
        cx, cy = (i % cols) * thumb_w, (i // cols) * (H + 46)
        sheet.paste(im, (cx, cy + 22))
        d.text((cx + 4, cy + 4),
               f"{x:.1f}s  seg{tg if tg is not None else '-'}", fill="black")
        if i and tags[i] != tags[i - 1] and i % cols:  # segment changed here
            d.line([(cx, cy + 22), (cx, cy + 22 + H)], fill="red", width=5)
        if abs(x - t) <= (hi - lo) / (2 * (n - 1)):
            d.rectangle([cx, cy + 22, cx + thumb_w - 2, cy + 22 + H],
                        outline="orange", width=5)
    y = (H + 46) * rows + 6
    for i, s in enumerate(sorted(segs, key=lambda s: float(s[1]))):
        d.text((6, y), f"seg{i}: [{float(s[1]):.1f}-{float(s[2]):.1f}] {s[0]}",
               fill="darkblue")
        y += 16
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="recseg json(s). Repeat -- this flag APPENDS, and a "
                         "single --data covering only one split silently "
                         "drops the other half of the sample")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_span_s", type=float, default=120.0)
    ap.add_argument("--no_sheet", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="dump the schema of --data and which sheet "
                         "recordings it covers, then exit without rendering")
    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.sheet, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("recording_id")]
    recs = load_recordings(a.data)
    print(f"{len(rows)} sheet rows; {len(recs)} recordings loaded")

    want = {r["recording_id"] for r in rows}
    have = want & set(recs)
    print(f"  sheet recordings covered: {len(have)}/{len(want)}")
    if len(have) < len(want):
        print(f"    missing e.g. {sorted(want - set(recs))[:5]}")
    if a.inspect:
        for rid in sorted(have)[:2]:
            rec = recs[rid]
            segs, k = get_segments(rec)
            print(f"\n  {rid}: keys {sorted(rec)[:14]}")
            print(f"    video={get_video(rec)}")
            print(f"    segments under {k!r}, n={len(segs)}; "
                  f"first {segs[:2]}")
        return

    miss = Counter()
    done = 0
    for r in rows:
        eid, rid = r["event_id"], r["recording_id"]
        t = cand_time(eid)
        rec = recs.get(rid)
        if rec is None:
            miss["no recording in --data"] += 1
            continue
        video = get_video(rec)
        if not video:
            miss["no video field on the record"] += 1
            continue
        if not os.path.exists(video):
            miss["video path does not exist"] += 1
            continue
        segs, _ = get_segments(rec)
        if not segs:
            miss["record has no segment list"] += 1
            continue
        if t is None:
            miss["no _t<time> in the event id"] += 1
            continue
        lo, hi, (p, c, n_), cut = span_for(segs, t, a.max_span_s)
        if lo is None:
            miss["no segment near the candidate"] += 1
            continue
        keep = [s for s in segs
                if float(s[2]) >= lo and float(s[1]) <= hi]
        mp4 = os.path.join(a.out_dir, f"{eid}_span.mp4")
        cmd = [a.ffmpeg_bin, "-y", "-loglevel", "error",
               "-ss", f"{lo:.2f}", "-i", video, "-t", f"{hi - lo:.2f}",
               "-vf", build_filter(keep, lo, hi, t, cut),
               "-an", "-c:v", "libx264", "-preset", "veryfast", mp4]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            miss["ffmpeg failed"] += 1
            if miss["ffmpeg failed"] == 1:
                print(f"  first ffmpeg failure on {eid}:\n    "
                      f"{rc.stderr.strip()[:300]}")
            continue
        if not a.no_sheet:
            try:
                contact_sheet(video, lo, hi, t, keep,
                              os.path.join(a.out_dir, f"{eid}_span.png"))
            except Exception as e:
                miss[f"contact sheet: {type(e).__name__}"] += 1
        done += 1
        if done % 10 == 0:
            print(f"  {done} rendered", flush=True)

    print(f"\n{done}/{len(rows)} rendered -> {a.out_dir}")
    for k, v in miss.most_common():
        print(f"  skipped, {k}: {v}")
    if done < len(rows):
        print("  a skipped event cannot be audited; report it rather than "
              "letting the sample quietly shrink")


if __name__ == "__main__":
    main()
