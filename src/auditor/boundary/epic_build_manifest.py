"""Turn the fetched EPIC clips into the manifest the offline packer expects.

The packer is reused rather than replaced -- it already takes the page as a
parameter, hashes every video, writes a label-blind `public_cases` list and a
checksum file, and those properties are what make a packet auditable. All that
is missing is a manifest in its shape:

    {"videos": [{"video_id", "video", "duration_s"}, ...]}

Durations come from ffprobe rather than from the sampled window span, because
the extraction seeks to a keyframe and the clip is therefore a second or two
longer or shorter than the span asked for. The page draws its timeline from
`duration_s`, so a value copied from the request rather than measured from the
file would put the rail and the video out of step for the whole batch.

Clips that failed to fetch are left out and named, so the packet's count is
whatever actually exists instead of a hole discovered by an annotator.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess


def duration_of(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--sample", help="抽样 json，用来对照哪些窗口没取到")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(a.clips, "*.mp4"))
                   if os.path.basename(p) != "probe.mp4")
    videos, bad = [], []
    for p in paths:
        wid = os.path.splitext(os.path.basename(p))[0]
        dur = duration_of(p)
        if not dur or dur < 5:
            bad.append((wid, "时长 %s" % dur))
            continue
        videos.append({"video_id": wid, "video": os.path.abspath(p),
                       "duration_s": round(dur, 3)})
        print("  %-26s %7.1fs  %6.0f MB"
              % (wid, dur, os.path.getsize(p) / 1e6))

    missing = []
    if a.sample:
        want = {w["window_id"] for w in
                json.load(open(a.sample, encoding="utf-8"))["windows"]}
        have = {v["video_id"] for v in videos}
        missing = sorted(want - have)

    json.dump({"videos": videos}, open(a.out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n%d 个片子 -> %s" % (len(videos), a.out))
    if bad:
        print("  剔除（时长无效）：%s" % ", ".join(w for w, _ in bad))
    if missing:
        print("  抽样里没取到的 %d 个：%s" % (len(missing), ", ".join(missing)))
    print("\n下一步（打包器的页面是参数，所以直接指粗标页）：")
    print("  python -m src.auditor.boundary.build_graph_timeline_offline_packet \\")
    print("    --manifest %s --out <包目录> \\" % a.out)
    print("    --reviewer-id <标注者> --batch-id epic_b2_coarse \\")
    print("    --title 'EPIC 粗标：状态序列与分叉原因' \\")
    print("    --expected-count %d --page b2_coarse_page.html" % len(videos))


if __name__ == "__main__":
    main()
