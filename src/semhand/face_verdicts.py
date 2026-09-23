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

BELOW THE CAP NOTHING IS DROPPED, AND THAT DECISION WAS MADE TWICE. Split by
score, face-ness on small boxes looked usable: every error in two sheets sat
at p >= 0.060 and below 0.05 it was 78 for 78. The gate was turned on at 0.05
and 136 episodes of mosaic came off.

Two of them were faces. A woman at p=1.7e-05 and a man at p=0.0, the second on
a box the detector had scored 0.81, both found by watching the delivered video
rather than by any number. They were among the 58 episodes the sheets never
reached, and missing both by chance had probability 0.18 -- the sample was not
unlucky, the reading of it was wrong.

78 of 78 has a lower bound of 95.3%, which over 136 episodes permits about six
wrong drops. Precision is simply the wrong quantity to gate a privacy decision
on. What is needed is a bound on faces uncovered, and no sample this size can
give a tight one; the region would have to be clean over several hundred
judged episodes before it is worth the trade, and it would still be six
recordings.

The mechanism stays behind `--small_thr` because the measurement is real and
the finding -- about 59% of small mosaics are not faces -- is worth acting on
one day. The default does not use it.

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
    # OFF. 78 of 78 was not enough and the bound said so: its lower limit is
    # 95.3%, which over 136 episodes permits about six wrong drops, and two
    # were found by watching the output -- a woman's face at p=1.7e-05 and a
    # man's at 0.0, the second on a box the detector scored 0.81. Both were
    # among the 58 episodes the sheets did not reach; missing both by chance
    # had probability 0.18, so the sample was not unlucky, the reading of it
    # was wrong. Precision is the wrong quantity to gate a privacy decision
    # on: what is needed is a bound on faces uncovered, and 78 of 78 does not
    # give a tight one. The mechanism stays; the default does not use it.
    ap.add_argument("--small_thr", type=float, default=0.0,
                    help="under the cap: drop a box only below this. 0 = off, "
                         "which is the shipped setting -- see the note above")
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
