"""The third outcome: a box that is not a hand, which is not the same as not covered.

WHY THE PIPELINE NEEDS A THIRD ANSWER AT ALL. `own` is one bit, so every box
leaves through one of two doors: kept and delivered as the wearer's hand, or
mosaicked as somebody else's. A machine part on the bench is neither. Given
the first door it becomes a phantom hand in the training stream, anchors the
grace rule, lengthens a track and inflates every continuity figure computed
over own boxes. 7.7% of large detector boxes are exactly that.

AND WHY THE VETO IS ASYMMETRIC, WHICH IS THE WHOLE FILE. The obvious design
is "not a hand, so do not cover it", and it is unsafe. The probe that answers
hand-ness rejects 15% of other people's hands at 0.60, and on 30 of those put
to a person 29 were real hands. Wire that to the mask and one in seven
foreign hands walks out of the pipeline uncovered -- a failure that does not
exist today. So:

    own=1 and a hand      delivered and protected, as now
    own=1 and NOT a hand  dropped from OWNER: not counted, not protected,
                          not an anchor -- and not covered either, because
                          it is not a person
    own=0                 covered, whatever hand-ness says. Always.

The last line is the invariant. `apply` raises rather than returns if a veto
would change a covered box, because a rule that silently uncovers is worth
more as a crash than as a result.

AND IT ONLY ASKS ABOUT BOXES THE PROBE CAN ANSWER. Below 150 px its non-hand
verdict was right 3 times in 55; above, 98 times in 100 and 40 in 40 on two
separate blind sheets. Boxes under the floor are left alone -- unvetoed, still
delivered, still counted. That is not a safe default so much as an honest one:
the small end is unmeasured and pretending otherwise would put the instrument
exactly where it is known to break.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

MIN_PX = 150


class WouldUncover(RuntimeError):
    """A veto tried to change a box that was being covered."""


def apply(rows, verdict, thr=0.10, min_px=MIN_PX):
    """rows: run CSV rows. verdict: {key: p_hand}. -> (rows, n_vetoed, n_skipped)

    Adds `is_hand` (1 / 0 / '' when not asked) and `own_kept`, which is `own`
    with the vetoed boxes removed from OWNER. `own=0` is never touched."""
    vetoed = skipped = 0
    for r in rows:
        w = float(r["x1"]) - float(r["x0"])
        key = (r.get("rec", ""), int(r["frame"]), int(round(float(r["x0"]))))
        p = verdict.get(key)
        own = str(r.get("own")) == "1"
        if p is None or w < min_px:
            r["is_hand"] = ""
            r["own_kept"] = "1" if own else "0"
            skipped += p is None or w < min_px
            continue
        is_hand = p > thr
        r["is_hand"] = "1" if is_hand else "0"
        if not own:
            # THE INVARIANT. A covered box stays covered whatever the probe
            # thinks, because the probe's mistakes on foreign hands are the
            # expensive ones and they are frequent.
            r["own_kept"] = "0"
            if not is_hand:
                vetoed += 0        # counted below only when it changes OWNER
            continue
        if is_hand:
            r["own_kept"] = "1"
        else:
            r["own_kept"] = "0-nothand"
            vetoed += 1
    return rows, vetoed, skipped


def check_never_uncovers(rows):
    """-> True, or raise. Every `own=0` box must still be covered."""
    for r in rows:
        if str(r.get("own")) == "0" and r.get("own_kept") not in ("0", None):
            raise WouldUncover(
                "frame %s: a covered box was changed to %r"
                % (r.get("frame"), r.get("own_kept")))
    return True


def load_verdict(paths, arm_rows=None):
    """Probe output -> {(rec, frame, x0): p}. Stems are `<rec>_f<6d>_b<k>`."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            rec = d.get("rec")
            if rec is None:
                continue
            out[(rec, int(d["frame"]), d.get("x0"))] = float(d["p"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--scores", action="append", required=True,
                    help="hand-ness jsonl for this arm")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--iou", type=float, default=0.10)
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--min_px", type=int, default=MIN_PX)
    a = ap.parse_args()

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])

    # the probe was run on the disjoint-kept boxes, indexed by rank per frame
    P = {}
    for p in a.scores:
        for line in open(p):
            d = json.loads(line)
            P[d["stem"]] = float(d["p"])

    def iou(x, y):
        x0, y0 = max(x[0], y[0]), max(x[1], y[1])
        x1, y1 = min(x[2], y[2]), min(x[3], y[3])
        i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        u = ((x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - i)
        return i / u if u > 0 else 0.0

    tot = vetoed = own_tot = 0
    frames_lost = 0
    for rec in recs:
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        by_f = collections.defaultdict(list)
        for r in rows:
            by_f[int(r["frame"])].append(r)
        for f, rs in by_f.items():
            own = [r for r in rs if str(r.get("own")) == "1"]
            keep = sorted(own, key=lambda r: -(float(r["x1"]) - float(r["x0"]))
                          * (float(r["y1"]) - float(r["y0"])))
            picked = []
            for r in keep:
                b = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
                if all(iou(b, k) < a.iou for k in picked):
                    picked.append(b)
                    k = len(picked) - 1
                    pr = P.get("%s_f%06d_b%d" % (rec, f, k))
                    w = b[2] - b[0]
                    own_tot += 1
                    if pr is not None and w >= a.min_px:
                        tot += 1
                        if pr <= a.thr:
                            vetoed += 1
            live = [r for r in own]
            if own and all(
                    (P.get("%s_f%06d_b%d" % (rec, f, i)) or 1.0) <= a.thr
                    for i in range(len(picked))) and picked:
                frames_lost += 1
    print("== %s" % os.path.basename(a.arm))
    print("   自己的手框 %d 个，其中 >=%dpx 且有判读的 %d 个"
          % (own_tot, a.min_px, tot))
    print("   被否决（判为不是手）%d 个 = %.1f%%"
          % (vetoed, 100.0 * vetoed / max(1, tot)))
    print("   因此整帧不再有自己的手的帧 %d" % frames_lost)
    print("   别人的手：一个都没动（不变量）")


if __name__ == "__main__":
    main()
