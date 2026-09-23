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

BELOW THE CAP, ONLY THE CONFIDENT REJECTIONS ARE WRITTEN, and that threshold
was earned rather than picked. Aggregated over every score, face-ness on small
boxes is right 74 times in 100 and unusable. Split by score, every error it
made sits at p >= 0.060, and below 0.05 it is 78 for 78 across two sheets --
lower bound 95.3%. The nearest error at 0.060 is what makes 0.05 a boundary
instead of a round number.

So a small box gets a row only when p <= SMALL_THR, and the row says "not a
face". Everything else down there gets no row and keeps its mosaic, which is
the old behaviour. Half the small-box mosaic comes off: 2,382 frame instances
of 4,406, from tables, food and bench.

THE SAMPLE IS SIX RECORDINGS. 78 of 78 says this region is clean on cam3, not
that it is clean anywhere; the threshold needs re-confirming on new material
before it travels.

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
                    help="over the cap: below this the box is dropped")
    ap.add_argument("--small_thr", type=float, default=0.05,
                    help="under the cap: a box is dropped only below this, "
                         "where the verdict was 78/78; 0 disables")
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
        n_small = 0
        for j, e in enumerate(sorted(eps, key=lambda x: x["frame"])):
            stem = "%s_f%06d_h%d" % (rec, e["frame"], j)
            pf = P.get(stem)
            if e["w_frac"] <= a.cap:
                # No row unless the model was confident; no row means the
                # mosaic stays, which is what happened before this existed.
                if not a.small_thr or pf is None or pf > a.small_thr:
                    continue
                is_face = 0
                n_small += 1
            else:
                is_face = 1 if (pf is None or pf > a.thr) else 0
            # EVERY COORDINATE THE EPISODE TOUCHED. One row per episode left
            # the rest unjudged, and an unjudged box is covered -- so the cap
            # came off for 46 of 53 boxes in one recording and the delivered
            # mask got BIGGER than with no second opinion at all.
            for b in sorted(e["boxes"]):
                out.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                            "is_face": is_face,
                            "p_face": "" if pf is None else round(pf, 4),
                            "w_frac": round(e["w_frac"], 4),
                            "n_frames": len(e["frames"]), "stem": stem})
            if e["w_frac"] > a.cap:
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
            print("  %-16s %d 行（大框保留 %d、大框撤掉 %d、小框撤掉 %d 段）"
                  % (rec, len(out), sum(r["is_face"] for r in out),
                     sum(1 for r in out if not r["is_face"]
                         and float(r["w_frac"]) > a.cap), n_small))
        else:
            print("  %-16s 无可撤的框（写了空表）" % rec)
    print("\n超过 cap 的 %d 段：判为脸保留 %d，判为不是脸撤掉 %d" % (tot, kept, dropped))
    print("小框（<=%.2f 宽）里 p<=%.2f 的段也写了撤掉的行" % (a.cap, a.small_thr))
    print("-> %s/<rec>.faceverdict.csv" % a.out)


if __name__ == "__main__":
    main()
