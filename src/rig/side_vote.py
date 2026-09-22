"""One answer per track for which hand it is, instead of one per frame.

WHY A VOTE AT ALL. The detector's two classes are `left` and `right` and it
gets them right on 91.8% of judged boxes, which sounds usable and is not: at
that rate two consecutive frames disagree about one time in six from noise
alone, and what leaves the pipeline is a label that flickers. A hand does not
change which hand it is, so every disagreement inside one track is an error
by construction and the majority is the answer.

MEASURED, NOT ASSUMED, AND ITS PRECONDITION IS MEASURED TOO. A vote over a
track is only meaningful if the track holds one hand. 52 sustained flip
events were put to a person and all 52 are the same hand -- the label moves,
the hand does not -- so there is nothing for the vote to average across. On
the 134-box gold, voting takes the shipped configuration from 91.8% to 95.5%
and the lower continue threshold from 92.5% to 99.2%; C1 gains more because
its tracks are longer, which is the third benefit of that change nobody
predicted.

BY TRACK ID, NOT BY OVERLAP CHAIN, AND IT IS BETTER. The earlier figures came
from chains built by IoU between consecutive frames, because that is what the
analysis had. The pipeline has track ids, and 49 of the 52 flip events sit
inside a single id, so they nearly agree -- but not quite, and the difference
runs the useful way. On the shipped configuration the IoU chains gave 95.5%
and track ids give 97.8% over 81 tracks, with no tie anywhere. The tracker
decides with motion, scale, pose and edge cost rather than overlap alone, so
it holds a hand together through the fast movements where overlap chains
break, and a longer track is a better-supported vote. Track ids are also what
the pipeline actually commits to.

A POST-PASS, BECAUSE THE VOTE NEEDS THE WHOLE TRACK. A causal version would
have to answer while the track is still running and would be wrong at exactly
the moment a track is short, which is when the label is least reliable. The
runs already write a CSV per recording and the deliverable is built from it,
so the honest implementation reads that file after the run rather than
pretending the decision can be made early.

IT DOES NOT TOUCH OWNERSHIP OR THE MASK. Only the `side` column changes. A
vote that also moved `own` would put an identity decision behind a question
about handedness, and those two have separate evidence and separate failures.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

SIDES = ("left", "right")


def vote_tracks(rows):
    """rows: dicts with `tid`, `side`, `own`. -> {tid: side}

    Only own hands vote. A foreign hand's side is not used anywhere and its
    labels would dilute a track id the tracker later reuses."""
    tally = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("own") not in ("1", 1, True):
            continue
        t = str(r.get("tid") or "")
        s = (r.get("side") or "").strip()
        if t and s in SIDES:
            tally[t][s] += 1
    out = {}
    for t, c in tally.items():
        l, r_ = c["left"], c["right"]
        # A tie is not a coin toss: it means the track carries no majority and
        # saying so is better than inventing one from the iteration order.
        out[t] = "left" if l > r_ else ("right" if r_ > l else "")
    return out


def apply_vote(rows):
    """-> (rows with `side_voted`, n_changed, n_tracks, n_tied)"""
    votes = vote_tracks(rows)
    changed = 0
    for r in rows:
        t = str(r.get("tid") or "")
        v = votes.get(t, "")
        r["side_voted"] = v
        if v and (r.get("side") or "") in SIDES and v != r["side"]:
            changed += 1
    tied = sum(1 for v in votes.values() if not v)
    return rows, changed, len(votes), tied


def _load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="a run directory whose CSVs carry `side` and `tid`")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--gold", default="/workspace/handedpkg/hands.csv")
    ap.add_argument("--gold_map", default="owner:left,other:right",
                    help="how the sheet's stored values map to sides")
    ap.add_argument("--write", action="store_true",
                    help="write the voted column back into the run CSVs")
    a = ap.parse_args()

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])

    gmap = dict(kv.split(":") for kv in a.gold_map.split(","))
    gold = {}
    if os.path.exists(a.gold):
        for r in _load(a.gold):
            s = gmap.get(r["label"])
            if not s:
                continue
            rec, rest = r["stem"].rsplit("_f", 1)
            # BY GEOMETRY, NOT BY (rec, frame). A frame carries two hands and
            # joining on the frame alone has silently mixed them up before.
            gold[(rec, int(rest.split("_h")[0]),
                  round(float(r["cx_frac"]), 3),
                  round(float(r["cy_frac"]), 3))] = s
        print("gold 里有 side 的框 %d 个" % len(gold))

    for d in a.arm:
        per, voted_per = [0, 0], [0, 0]
        n_changed = n_tracks = n_tied = 0
        unmatched = 0
        for rec in recs:
            p = os.path.join(d, rec + ".csv")
            if not os.path.exists(p):
                continue
            rows = _load(p)
            if "tid" not in rows[0] or "side" not in rows[0]:
                print("  !! %s 没有 side/tid 列" % p)
                break
            rows, ch, nt, ti = apply_vote(rows)
            n_changed += ch
            n_tracks += nt
            n_tied += ti
            if a.write:
                with open(p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            for r in rows:
                if r.get("own") != "1":
                    continue
                W, H = 1920.0, 1520.0
                k = (rec, int(r["frame"]),
                     round((float(r["x0"]) + float(r["x1"])) / 2 / W, 3),
                     round((float(r["y0"]) + float(r["y1"])) / 2 / H, 3))
                g = gold.get(k)
                if g is None:
                    continue
                if (r.get("side") or "") in SIDES:
                    per[0] += r["side"] == g
                    per[1] += 1
                if r["side_voted"] in SIDES:
                    voted_per[0] += r["side_voted"] == g
                    voted_per[1] += 1
                unmatched += 0
        name = os.path.basename(d)
        print("== %s" % name)
        print("   轨迹 %d 条，投票后改掉 %d 个框的 side，平票的轨迹 %d 条"
              % (n_tracks, n_changed, n_tied))
        if per[1]:
            print("   对 gold：逐框 %d/%d = %.1f%%   投票后 %d/%d = %.1f%%"
                  % (per[0], per[1], 100.0 * per[0] / per[1],
                     voted_per[0], voted_per[1],
                     100.0 * voted_per[0] / max(1, voted_per[1])))
        else:
            print("   gold 一个都没对上——检查几何 join")


if __name__ == "__main__":
    main()
