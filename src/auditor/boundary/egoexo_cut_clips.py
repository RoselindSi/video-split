"""Cut the sampled Ego-Exo4D windows out of the RGB ego stream.

WHICH STREAM, AND WHY IT WAS CHECKED RATHER THAN NAMED. Each take ships nine
videos and the names mislead. `ego_preview.mp4` sounds like the ego view and
is actually a grid of several panels including the greyscale SLAM pair and the
exocentric cameras. `aria*_1201-*` are the SLAM cameras -- monochrome fisheye,
useless for labelling -- and `aria*_211-1` came back entirely black. The RGB
ego view is `aria*_214-1`, confirmed by pulling a frame from each and looking
at it, not by reading the Aria stream id that happens to agree.

The Aria number is not fixed either: across 18 takes it is aria01, 02, 03, 05
and 06, so the glob matters. Hardcoding `aria01` would silently drop five
takes, which is the kind of mistake that shows up as a smaller sample nobody
questions.

The windows come from `egoexo_sample_windows`, which draws uniformly inside
the cooking family. The clip boundaries are the keystep times from the
annotation, padded, because a keyframe seek is not frame-accurate.

Every clip is verified with ffprobe rather than trusted because a file
appeared: a zero-length or truncated output is what a failed cut produces and
it looks like success to `ls`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

PAD = 2.0


EGO_RE = re.compile(r"^aria.*214-1\.mp4$", re.I)


def ego_stream(takes_dir, take_name):
    """-> path of the RGB ego stream, or None.

    NOT A GLOB, AND NOT CASE-SENSITIVE. The filename was `aria01_214-1.mp4`
    across every take seen at first, so the obvious `aria*_214-1.mp4` glob
    looked safe. It is not: `indiana_bike_02_3` ships `Aria_214-1.mp4` --
    capital A, no index at all -- and `glob` matches with fnmatch, which is
    case-sensitive whatever the filesystem does. That take was silently
    dropped, and a take that goes missing shows up only as a smaller sample
    nobody questions. Matching the listing against a case-insensitive pattern
    with no required separator covers every spelling observed.
    """
    d = os.path.join(takes_dir, take_name,
                     "frame_aligned_videos", "downscaled", "448")
    if not os.path.isdir(d):
        return None
    hits = sorted(f for f in os.listdir(d) if EGO_RE.match(f))
    return os.path.join(d, hits[0]) if hits else None


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "format=duration:stream=codec_name",
                        "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    vals = [x for x in r.stdout.split() if x]
    if r.returncode or len(vals) < 2:
        return None, None
    dur, codec = None, None
    for v in vals:
        try:
            dur = float(v)
        except ValueError:
            codec = codec or v
    return dur, codec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--takes", required=True, help="egoexo_data/takes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=PAD)
    a = ap.parse_args()

    doc = json.load(open(a.sample, encoding="utf-8"))
    wins = doc["windows"]
    os.makedirs(a.out, exist_ok=True)
    ok, bad = [], []
    for i, w in enumerate(wins, 1):
        wid = "%s_w%06.0f" % (w["take_name"], w["start_s"])
        src = ego_stream(a.takes, w["take_name"])
        if not src:
            bad.append((wid, "找不到 aria*_214-1"))
            print("  [%2d/%2d] %-34s 找不到 ego 流" % (i, len(wins), wid[:34]),
                  flush=True)
            continue
        dst = os.path.join(a.out, wid + ".mp4")
        start = max(0.0, w["start_s"] - a.pad)
        want = max(1.0, w["end_s"] - w["start_s"]) + 2 * a.pad
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", "%.2f" % start, "-t", "%.2f" % want, "-i", src,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", "-y", dst],
            capture_output=True, text=True)
        dur, codec = probe(dst) if os.path.exists(dst) else (None, None)
        if r.returncode or not dur or dur < 1.0:
            bad.append((wid, "rc=%d dur=%s" % (r.returncode, dur)))
            print("  [%2d/%2d] %-34s 失败 %s" % (i, len(wins), wid[:34],
                                               (r.stderr or "").strip()[:40]),
                  flush=True)
            if os.path.exists(dst):
                os.remove(dst)
            continue
        print("  [%2d/%2d] %-34s %6.1fs %s %5.1fMB"
              % (i, len(wins), wid[:34], dur, codec,
                 os.path.getsize(dst) / 1e6), flush=True)
        ok.append({"window_id": wid, "video": os.path.abspath(dst),
                   "duration_s": round(dur, 3), "scenario": w["scenario"],
                   "take_name": w["take_name"], "take_uid": w["take_uid"],
                   "src_start_s": w["start_s"], "n_keysteps": w["n_keysteps"]})

    print("\n成功 %d / 失败 %d" % (len(ok), len(bad)))
    for wid, why in bad:
        print("  失败 %s（%s）" % (wid, why))
    json.dump({"videos": [{"video_id": c["window_id"], "video": c["video"],
                           "duration_s": c["duration_s"]} for c in ok],
               "detail": ok, "failed": bad},
              open(os.path.join(a.out, "manifest.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-> %s/manifest.json" % a.out)


if __name__ == "__main__":
    main()
