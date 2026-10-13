"""Pull just the sampled windows out of EPIC's videos, without downloading them.

WHY NOT DOWNLOAD THE VIDEOS. One of them is 4.5 GB and twenty would be on the
order of ninety, for sixty minutes of content. The host sends
`Accept-Ranges: bytes`, so ffmpeg can seek over HTTP and fetch only the bytes
covering the window. The cost of a mistake here is bandwidth and hours, which
is why the extraction is verified per clip with ffprobe rather than trusted
because a file appeared -- a zero-length or moov-less file is the failure mode
this host produces when a seek goes wrong, and it looks like success to `ls`.

TWO URL LAYOUTS, because EPIC-100 is EPIC-55 plus an extension and the videos
did not move:

    index >= 100   <base100>/<PID>/videos/<video_id>.MP4
    index <  100   <base55>/videos/<split>/<PID>/<video_id>.MP4

The split for the older videos is not in the annotation CSVs, so it is probed:
`train` first, then `test`. Guessing wrong gives a 404, not a corrupt clip.

Clips are named by window, not by video, because two windows can come from one
recording and the coarse page keys its state on the file it is given.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

BASE_100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
BASE_55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"


def head_ok(url, timeout=40):
    r = subprocess.run(["curl", "-sI", "-L", "-o", "/dev/null",
                        "-w", "%{http_code}", "--max-time", str(timeout), url],
                       capture_output=True, text=True)
    return r.stdout.strip() == "200"


def url_for(video_id, cache):
    pid, idx = video_id.split("_")
    if int(idx) >= 100:
        return "%s/%s/videos/%s.MP4" % (BASE_100, pid, video_id)
    if video_id in cache:
        return cache[video_id]
    for split in ("train", "test"):
        u = "%s/videos/%s/%s/%s.MP4" % (BASE_55, split, pid, video_id)
        if head_ok(u):
            cache[video_id] = u
            return u
    return None


def probe(path):
    """-> (duration, codec) or (None, None). 文件存在不等于片子是好的。

    两个 section 必须写在**同一个** `-show_entries` 里。给两次 ffprobe 只保留
    后一个，于是只回 codec_name 一个值 —— 第一版那么写，判「少于两个值就算
    失败」，结果每一条下完都被判失败删掉重取，会永远跑不完。
    """
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "format=duration:stream=codec_name",
                        "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    vals = [x for x in r.stdout.split() if x]
    if r.returncode or len(vals) < 2:
        return None, None
    # 顺序由 ffprobe 决定（stream 在 format 之前），所以按能否转成浮点来认，
    # 不按位置认。
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
    ap.add_argument("--sample", required=True, help="epic_sample_windows 的 json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=2.0,
                    help="两端各多取几秒；关键帧定位不是精确到帧的")
    ap.add_argument("--only", action="append", help="只取这些 window_id")
    a = ap.parse_args()

    doc = json.load(open(a.sample, encoding="utf-8"))
    wins = doc["windows"]
    if a.only:
        want = set(a.only)
        wins = [w for w in wins if w["window_id"] in want]
    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, "_urls.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    ok, bad = [], []
    for i, w in enumerate(wins, 1):
        dst = os.path.join(a.out, w["window_id"] + ".mp4")
        want_dur = max(1.0, w["end_s"] - w["start_s"]) + 2 * a.pad
        if os.path.exists(dst):
            dur, codec = probe(dst)
            if dur and dur > 0.5 * want_dur:
                print("  [%2d/%2d] %-22s 已有 %.0fs" % (i, len(wins),
                                                      w["window_id"], dur),
                      flush=True)
                ok.append(w["window_id"])
                continue
            os.remove(dst)                 # 半成品，重取
        url = url_for(w["video_id"], cache)
        json.dump(cache, open(cache_path, "w"))
        if not url:
            print("  [%2d/%2d] %-22s 找不到 URL" % (i, len(wins), w["window_id"]),
                  flush=True)
            bad.append((w["window_id"], "no url"))
            continue
        t0 = time.time()
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", "%.2f" % max(0.0, w["start_s"] - a.pad),
             "-t", "%.2f" % want_dur,
             "-i", url, "-c", "copy", "-y", dst],
            capture_output=True, text=True)
        dur, codec = probe(dst) if os.path.exists(dst) else (None, None)
        if r.returncode or not dur:
            print("  [%2d/%2d] %-22s 失败 rc=%d %s"
                  % (i, len(wins), w["window_id"], r.returncode,
                     (r.stderr or "").strip()[:60]), flush=True)
            bad.append((w["window_id"], "rc=%d" % r.returncode))
            if os.path.exists(dst):
                os.remove(dst)
            continue
        print("  [%2d/%2d] %-22s %.0fs %s %.0fMB %.0fs 墙钟"
              % (i, len(wins), w["window_id"], dur, codec,
                 os.path.getsize(dst) / 1e6, time.time() - t0), flush=True)
        ok.append(w["window_id"])

    print("\n成功 %d / 失败 %d -> %s" % (len(ok), len(bad), a.out))
    for wid, why in bad:
        print("  失败 %s（%s）" % (wid, why))
    json.dump({"ok": ok, "failed": bad},
              open(os.path.join(a.out, "_fetch_report.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
