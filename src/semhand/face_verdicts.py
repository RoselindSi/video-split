"""Turn episode-level face answers into the per-box table the render reads.

WHY A SEPARATE FILE AND NOT A DICT IN MEMORY. The question is asked once per
episode -- a box held across twelve frames is one decision seen twelve times,
and eighty seconds of cam3 comes to thirteen questions over the cap instead of
three hundred. The render, though, meets boxes one frame at a time. So the
episode's answer is expanded back over every frame its box appears on, and
written down, so the two passes of the render are reading the same thing and a
mismatch is visible rather than inferred.

KEYED ON THE BOX, NOT ON THE FRAME. A held box keeps the coordinates of the
detection that created it, so the same four integers recur for as long as the
hold lasts and the verdict follows them without needing to know which frame is
which. Keying on the frame as well would miss every held frame, which is most
of them.

ONLY BOXES OVER THE CAP ARE WRITTEN. Below it nothing is asked and nothing
changes: the probe's not-a-face verdict is right 74 times in 100 down there,
against 13 of 13 above, and a gate built on the first number would trade a
measured privacy leak for an unmeasured one. What the small boxes did buy is a
number -- about 59% of them are not faces -- and that is a fact about the
detector, not a licence to act on this model's opinion of them.

THE DEFAULT IS TO KEEP THE MOSAIC. A box with no answer, or an answer that is
not confidently negative, stays covered. Dropping one wrongly puts a
recognisable person in the delivered video; keeping one wrongly costs a patch
of bench.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from src.semhand.faceness import episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", default="/workspace/cam3_faces",
                    help="the run directory whose <rec>.faces.csv enumerated the boxes")
    ap.add_argument("--scores", default="/workspace/faceness.jsonl")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--cap", type=float, default=0.18)
    ap.add_argument("--thr", type=float, default=0.50,
                    help="below this the box is dropped; everything else is kept")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])
    P = {}
    for line in open(a.scores):
        d = json.loads(line)
        P[d["stem"]] = float(d["p"])
    print("face-ness 打分 %d 段" % len(P))

    os.makedirs(a.out, exist_ok=True)
    tot = kept = dropped = 0
    for rec in recs:
        p = os.path.join(a.faces, rec + ".faces.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        eps = episodes(rows)
        # The stems were built from the SAME episode grouping, ordered by the
        # episode's middle frame, so the index rebuilds identically here.
        out = []
        for j, e in enumerate(sorted(eps, key=lambda x: x["frame"])):
            if e["w_frac"] <= a.cap:
                continue
            stem = "%s_f%06d_h%d" % (rec, e["frame"], j)
            pf = P.get(stem)
            is_face = 1 if (pf is None or pf > a.thr) else 0
            # Every frame this box appears on, not only the middle one.
            seen = set()
            for r in rows:
                b = tuple(int(r[c]) for c in ("x0", "y0", "x1", "y1"))
                if b in seen:
                    continue
                if abs(b[2] - b[0] - (e["box"][2] - e["box"][0])) > 2:
                    continue
                if b != tuple(e["box"]):
                    continue
                seen.add(b)
                out.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                            "is_face": is_face,
                            "p_face": "" if pf is None else round(pf, 4),
                            "w_frac": round(e["w_frac"], 4),
                            "n_frames": len(e["frames"]), "stem": stem})
            tot += 1
            kept += is_face
            dropped += 1 - is_face
        # A FILE FOR EVERY RECORDING, EMPTY OR NOT. Writing only the ones
        # with something in them makes "no over-cap boxes here" and "the path
        # is wrong" look identical to the render, and the render would then
        # have to guess which. An empty file says the first; a missing one is
        # then unambiguously the second, and the loader raises on it.
        cols = ["x0", "y0", "x1", "y1", "is_face", "p_face", "w_frac",
                "n_frames", "stem"]
        with open(os.path.join(a.out, rec + ".faceverdict.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(out)
        if out:
            print("  %-16s 超过 cap 的 %d 段 -> 保留 %d、撤掉 %d"
                  % (rec, len(out), sum(r["is_face"] for r in out),
                     sum(1 - r["is_face"] for r in out)))
        else:
            print("  %-16s 没有超过 cap 的框（写了空表）" % rec)
    print("\n合计 %d 段超过 cap：判为脸保留 %d，判为不是脸撤掉 %d" % (tot, kept, dropped))
    print("-> %s/<rec>.faceverdict.csv" % a.out)


if __name__ == "__main__":
    main()
