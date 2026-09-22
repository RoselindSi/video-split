"""One ownership answer per track, and the rule that a vote must never uncover.

THE SAME ARGUMENT AS HANDEDNESS AND A DIFFERENT RISK. A hand does not change
whose it is inside one track, so disagreement within a track is error and the
majority is the answer. But the two questions fail in opposite directions. A
wrong handedness label mislabels a deliverable; a wrong ownership label leaves
a colleague's hand in the video. So the vote here is not symmetric and the
asymmetry is the point of the file.

MAJORITY DECIDES `own=1`, ANY EVIDENCE DECIDES `own=0`. A track the pipeline
called the wearer's on most frames becomes the wearer's on all of them. A
track it called somebody else's on most frames becomes somebody else's on all
of them -- and a track that is merely ambiguous stays covered rather than
being rounded to the wearer. On human gold this is what took exposure events
from 15 to zero; the earlier arm that voted symmetrically left four tracks
exposed end to end, which is why `max_owner` could not be removed then.

TIES COVER. Anything that is not a strict majority for `own=1` is covered.
That is a deliberate loss of some of the wearer's own pixels, and it is the
cheap side of a trade whose other side is a person's face or hands going out.

IT IS A POST-PASS AND CANNOT BE OTHERWISE. The vote needs the whole track.
Applied live it would have to answer while a track is two frames old, which
is exactly when the ownership head is least reliable -- and a wrong early
`own=1` would then be propagated over the rest of the track by the vote
itself, turning one bad frame into a whole exposed track.

WHAT IT DOES NOT DO. It never reads hand-ness, never reads `side`, and never
looks outside a single track id. Joining those decisions would put three
different kinds of evidence behind one flag.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def vote_tracks(rows, min_frac=0.5):
    """-> {tid: 1 or 0}. `own=1` needs a strict majority; everything else is 0."""
    tally = collections.defaultdict(lambda: [0, 0])     # tid -> [own, not]
    for r in rows:
        t = str(r.get("tid") or "")
        if not t:
            continue
        tally[t][0 if str(r.get("own")) == "1" else 1] += 1
    out = {}
    for t, (o, n) in tally.items():
        out[t] = 1 if o > (o + n) * float(min_frac) else 0
    return out


def apply_vote(rows, min_frac=0.5):
    """-> (rows with `own_voted`, n_to_own, n_to_other, n_tracks)"""
    votes = vote_tracks(rows, min_frac)
    up = down = 0
    for r in rows:
        t = str(r.get("tid") or "")
        v = votes.get(t)
        if v is None:
            r["own_voted"] = r.get("own")
            continue
        r["own_voted"] = str(v)
        was = str(r.get("own")) == "1"
        if v == 1 and not was:
            up += 1
        elif v == 0 and was:
            down += 1
    return rows, up, down, len(votes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--min_frac", type=float, default=0.5)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])

    for d in a.arm:
        up = down = tracks = 0
        mixed = 0
        for rec in recs:
            p = os.path.join(d, rec + ".csv")
            if not os.path.exists(p):
                continue
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            if not rows or "tid" not in rows[0]:
                print("  !! %s 没有 tid 列" % p)
                break
            by_t = collections.defaultdict(set)
            for r in rows:
                by_t[str(r.get("tid") or "")].add(str(r.get("own")))
            mixed += sum(1 for t, s in by_t.items() if t and len(s) > 1)
            rows, u, dn, nt = apply_vote(rows, a.min_frac)
            up += u
            down += dn
            tracks += nt
            if a.write:
                with open(p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        print("== %s" % os.path.basename(d))
        print("   轨迹 %d 条，其中归属在轨迹内翻过的 %d 条（%.1f%%）"
              % (tracks, mixed, 100.0 * mixed / max(1, tracks)))
        print("   投票把 %d 个框改判为自己的手，%d 个改判为别人的手"
              % (up, down))


if __name__ == "__main__":
    main()
