"""What does the track buy by lowering the detector's floor?

THE ONLY LIVE USE OF THE TRACK THAT HAS NEVER BEEN SCORED. A detection at 0.60
or above may start a track; one between 0.25 and 0.60 may only continue an
existing one, and if it matches nothing it is discarded and never appears in
any dump. So every measurement this project has made was taken downstream of a
decision the measurements could not see. Turning the continuation floor up to
the admission floor -- and only that, leaving the detector's own floor at 0.25
so the frame still contains the same boxes for `max_owner` to rank and for the
geometric prior to read -- is the one-variable form of the question.

THE TWO RUNS ARE JOINED BY THE BOX, NOT BY THE TRACK ID. Track ids are assigned
in order of appearance and both runs invent their own, but the detections are
identical: same frames, same model, same floor. `(recording, frame, box)` is
therefore an exact key, and it is what carries a person's ownership verdict
from the labelled run onto the ablated one. Joining on the id would silently
compare different hands.

FRAGMENTATION IS ITS OWN COST. A hand that survives as three short tracks
instead of one long one is still detected and still blurred, but every one of
the three starts its ownership belief from scratch and consolidates over a
third of the evidence. That does not show up in a frame count, so the number
of distinct ablated tracks per labelled hand is reported beside the frames.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics

COAST = 2
FPS = 30.0


def read(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        by[(r["rec"], r["tid"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame"]))
    return by


def key_of(r):
    return (r["rec"], int(r["frame"]),
            int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))


def runs(flags):
    out, cur = [], 0
    for x in flags:
        if not x:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a0", required=True, help="deployed dump")
    ap.add_argument("--a1", required=True, help="continue_conf = new_track_conf")
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--precision", nargs="*")
    a = ap.parse_args()

    A0, A1 = read(a.a0), read(a.a1)
    gold = {}
    for f in a.gold:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            gold[(r["rec"], r["tid"])] = r["human_ownership"]
    hand = {}
    for f in (a.precision or []):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            hand[(r["rec"], r["tid"])] = r["verdict"]

    a1_by_key = {}
    for k, v in A1.items():
        for r in v:
            a1_by_key[key_of(r)] = (k, r)
    n0 = sum(len(v) for v in A0.values())
    n1 = sum(len(v) for v in A1.values())
    print(f"\n  === A0 现役 (continue .25) vs A1 (continue .60) ===")
    print(f"  帧-手  A0 {n0}   A1 {n1}   A0 多出 {n0 - n1} "
          f"({(n0 - n1) / n0:.1%})")
    print(f"  轨迹    A0 {len(A0)}   A1 {len(A1)}")

    # What the extra admissions are, judged by the label the hand already has.
    extra = collections.Counter()
    for k, v in A0.items():
        for r in v:
            if key_of(r) not in a1_by_key:
                extra[gold.get(k, hand.get(k, "未标"))] += 1
    print(f"\n  A0 多接住的 {n0 - n1} 帧，按人工判定归属：")
    for name, n in extra.most_common():
        print(f"    {name:<16}{n:>6}  {n/(n0-n1):>6.1%}")

    D = {k: v for k, v in A0.items() if gold.get(k) in ("owner", "other")}
    maj0 = {k: sum(1 for r in v if int(r["final_owner_post_cap"])) * 2 > len(v)
            for k, v in D.items()}
    maj1 = {k: sum(1 for r in v if int(r["final_owner_post_cap"])) * 2 > len(v)
            for k, v in A1.items()}

    def cover(v, arm):
        """Per frame over the labelled hand's span: is it covered?"""
        seen = {}
        for r in v:
            f = int(r["frame"])
            if arm == "A0":
                seen[f] = maj0[(r["rec"], r["tid"])]
            else:
                hit = a1_by_key.get(key_of(r))
                if hit is None:
                    continue          # this detection was never admitted
                seen[f] = maj1[hit[0]]
        lo, hi = int(v[0]["frame"]), int(v[-1]["frame"])
        out, last = [], None
        for f in range(lo, hi + 1):
            if f in seen:
                out.append(not seen[f])
                last = f
            else:
                out.append(last is not None and f - last <= COAST)
        return out

    print(f"\n  === 别人的手的暴露（整轨多数票 + coast2，两臂同口径）===")
    print(f"  {'':<8}{'整轨漏':>7}{'暴露事件':>9}{'未覆盖帧':>9}{'秒':>7}"
          f"{'最长s':>8}{'录像':>6}")
    for arm in ("A0", "A1"):
        allr, recs, miss = [], set(), 0
        for k, v in D.items():
            if gold[k] != "other":
                continue
            c = cover(v, arm)
            if not any(c):
                miss += 1
            rr = runs(c)
            allr += rr
            if rr:
                recs.add(k[0])
        allr.sort()
        print(f"  {arm:<8}{miss:>7}{len(allr):>9}{sum(allr):>9}"
              f"{sum(allr)/FPS:>7.1f}{(allr[-1]/FPS if allr else 0):>8.2f}"
              f"{len(recs):>6}")
    print("  『整轨漏』这里是『整条一帧都没盖住』，比多数票判反更严格。")

    print(f"\n  === 碎片化：一只被标注的手在 A1 里裂成几条 ===")
    for want in ("other", "owner"):
        ks = [k for k in D if gold[k] == want]
        n_tid, gone = [], 0
        for k in ks:
            tids = {a1_by_key[key_of(r)][0] for r in D[k]
                    if key_of(r) in a1_by_key}
            if not tids:
                gone += 1
            n_tid.append(len(tids))
        n_tid.sort()
        print(f"    {want:<8} n={len(ks)}   A1 轨迹数中位 "
              f"{n_tid[len(n_tid)//2]}   裂成 >1 条的 "
              f"{sum(1 for x in n_tid if x > 1)}/{len(ks)}"
              f"   在 A1 里完全消失的 {gone}")

    own = [k for k in D if gold[k] == "owner"]
    for arm in ("A0", "A1"):
        blur = sum(1 for k in own for x in cover(D[k], arm) if x)
        print(f"  {arm} 自己的手被糊 {blur} 帧 = {blur/FPS:.1f}s")

    if hand:
        nh = [k for k in A0 if hand.get(k) == "nohand"]
        kept = sum(1 for k in nh for r in A0[k] if key_of(r) in a1_by_key)
        tot = sum(len(A0[k]) for k in nh)
        print(f"\n  非手轨迹的帧：A0 {tot}   A1 仍留下 {kept} "
              f"({kept/max(1,tot):.0%})   低置信续轨多喂进来 {tot-kept} 帧")


if __name__ == "__main__":
    main()
