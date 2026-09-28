"""Pick the densest three-minute window of each recording, not the recording.

WHY THE UNIT CHANGED. A ten-minute recording with eighty segments is a far
bigger annotation job than the five-to-seven-segment batches this contract was
written against, and the reviewer asked for units of at most three minutes. A
clip is not simply the first three minutes, though: the decisions the study
needs require a node to be visited twice and then go somewhere different, and
in a recording that switches between two activities six times over ten minutes
an arbitrary third of it may contain no switch at all.

SO THE WINDOW IS CHOSEN FOR STRUCTURE. A window slides over the recording and
is scored by how many times the object family changes inside it. The chosen
window is the best-scoring one, and windows that hold fewer than two switches
or fewer than three distinct labels are not offered at all -- inside such a
window there is nothing for a transition prior to arbitrate, so the clip would
cost annotation time and produce no usable decision.

THE WINDOW SNAPS TO SEGMENT BOUNDARIES of the existing segmentation, so a clip
never opens or closes in the middle of an action. That segmentation is someone
else's contract and is used here only to place the cut; whatever the clip gets
annotated as is decided from scratch.

SCANNING ALL 471 RATHER THAN THE 23. The earlier screen kept recordings that
switch four or more times across their whole length. With a three-minute unit
the question is different -- a recording that switches rarely overall can still
hold one dense window -- so every recording is scanned and ranked on its best
window instead of on its total.

HAND VISIBILITY IS MEASURED INSIDE THE WINDOW, separately, because it is a
property of the clip and not of the recording: the first batch was chosen on
structure alone and seven of its eight recordings turned out to keep the hands
too far from the camera to read.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

from src.auditor.boundary.branch_screen import norm, STOP

VERB = set("""remove install attach reattach detach place put take pick open close seal
unseal screw unscrew tighten loosen clean wash rinse scrub wipe squeeze pour apply press
tap push pull inspect check examine assemble disassemble prepare finish reset adjust
manipulate move set arrange fold unfold insert extract pack unpack retrieve replace
return start begin end complete cycle iteration partial first second third again handling
operation""".split())


def objs(lab):
    return frozenset(t for t in re.findall(r"[a-z]+", norm(lab))
                     if t not in STOP and t not in VERB and len(t) > 2)


def families(labels):
    U = sorted({norm(x) for x in labels if norm(x)})
    par = {u: u for u in U}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    O = {u: objs(u) for u in U}
    for i, a in enumerate(U):
        for b in U[i + 1:]:
            if O[a] and O[b] and (O[a] & O[b]):
                ra, rb = f(a), f(b)
                if ra != rb:
                    par[rb] = ra
    return {u: f(u) for u in U}


def best_window(segs, fam, win, min_sw, min_lab):
    """-> (start_s, end_s, switches, n_labels) 卡在段边界上的最佳窗口。"""
    rows = [(float(s["start_s"]), float(s["end_s"]),
             norm(s.get("label_en") or s.get("label_zh") or ""))
            for s in segs]
    rows = [r for r in rows if r[2]]
    best = None
    for i in range(len(rows)):
        j = i
        while j + 1 < len(rows) and rows[j + 1][1] - rows[i][0] <= win:
            j += 1
        if j <= i:
            continue
        sub = rows[i:j + 1]
        fseq = [fam.get(x[2], x[2]) for x in sub]
        fseq = [x for k, x in enumerate(fseq) if k == 0 or x != fseq[k - 1]]
        sw = len(fseq) - 1
        nl = len({x[2] for x in sub})
        key = (sw, nl, -(sub[-1][1] - sub[0][0]))
        if sw >= min_sw and nl >= min_lab and (best is None or key > best[0]):
            best = (key, sub[0][0], sub[-1][1], sw, nl, len(sub))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--win", type=float, default=180.0)
    ap.add_argument("--min_switches", type=int, default=2)
    ap.add_argument("--min_labels", type=int, default=3)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seen, paths = set(), []
    for r in a.roots:
        for p in sorted(glob.glob(os.path.join(r, "**", "segments.json"),
                                  recursive=True)):
            rid = os.path.basename(os.path.dirname(p))
            if rid not in seen:
                seen.add(rid)
                paths.append(p)

    out = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        segs = d.get("segments") or []
        labs = [s.get("label_en") or s.get("label_zh") or "" for s in segs]
        b = best_window(segs, families(labs), a.win, a.min_switches, a.min_labels)
        if not b:
            continue
        _, s0, s1, sw, nl, nseg = b
        vids = [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".mp4")]
        if not vids:
            continue
        out.append({"rid": d.get("recording_id"),
                    "src": os.path.join(os.path.dirname(p), vids[0]),
                    "start_s": round(s0, 2), "end_s": round(s1, 2),
                    "dur_s": round(s1 - s0, 2), "switches": sw,
                    "n_labels": nl, "n_seg_in_window": nseg})

    out.sort(key=lambda r: (-r["switches"], -r["n_labels"]))
    print("扫过 %d 条录像；能切出 %.0fs 内、>=%d 次族切换且 >=%d 个不同标签的窗口：%d 条"
          % (len(paths), a.win, a.min_switches, a.min_labels, len(out)))
    if out:
        sw = collections.Counter(r["switches"] for r in out)
        print("  窗口内族切换次数分布 %s" % dict(sorted(sw.items())))
        print("\n%-20s %8s %8s %7s %7s %8s"
              % ("recording", "起", "止", "时长s", "切换", "窗内段数"))
        for r in out[:a.top]:
            print("%-20s %8.1f %8.1f %7.1f %7d %8d"
                  % (r["rid"], r["start_s"], r["end_s"], r["dur_s"],
                     r["switches"], r["n_seg_in_window"]))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
