"""What the reacquire reset buys, and what it costs, over every recording with gold.

THE RULE. `OwnHold` throws away a track's history when the tracker loses it
and finds it again, if that track was the wearer's, and holds it foreign until
two frames confirm. The reason is a real failure mode: a hand leaves, someone
else's arrives in the same place, the tracker joins them, and an inherited
"self" leaves a stranger uncovered. The cost is one covered frame on the
wearer's hand every time the detector merely blinks.

HOW IT IS MEASURED. Both replays, on the same dump: the shipped post-
processing with the reset and without it. Every hand-frame where they differ
is an effect of the rule, and the track's human label says which kind:

    gold = other  -> the cover stayed on a foreign hand      (what it is for)
    gold = owner  -> the cover landed on the wearer's hand   (the flicker)

Counted per recording and pooled, for V1's classifier and for the student,
because which tracks are "self" before the gap is the classifier's doing.
Nothing here changes the pipeline; the switch already exists
(`--inherit_self_on_reacquire`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from src.semhand.distil_eval import read_zone_gold
from src.semhand.holdreplay import invert_prior, key, replay


def gaps_of(rows):
    """-> (reacquisitions, tracks with any gap). A reacquisition is a track
    appearing again after at least one frame without a detection."""
    seen = collections.defaultdict(list)
    for r in rows:
        seen[(r["rec"], str(r["tid"]))].append(int(r["frame"]))
    n = t = 0
    for v in seen.values():
        v.sort()
        g = sum(1 for i in range(1, len(v)) if v[i] != v[i - 1] + 1)
        n += g
        t += g > 0
    return n, t, len(seen)


def full_rate(rows):
    """True when the rows are consecutive frames rather than a sampled subset.

    A stride-3 prediction file makes every pair look like a gap, which would
    fire the reset on almost every hand and produce a table that means
    nothing. Refuse instead of printing it."""
    seen = collections.defaultdict(list)
    for r in rows:
        seen[(r["rec"], str(r["tid"]))].append(int(r["frame"]))
    step1 = tot = 0
    for v in seen.values():
        v.sort()
        for i in range(1, len(v)):
            tot += 1
            step1 += v[i] == v[i - 1] + 1
    return tot == 0 or step1 / tot > 0.5


def shape_of(rows, gold, off, on, p, prior, geom_w, min_gap):
    """A count of covered frames says nothing about how they sit in time.

    One wrong frame at the moment a hand returns is a blink; the same hand
    held foreign for ten frames afterwards is the EMA having been wiped and
    having to climb back over the 0.70 hysteresis. These are different faults
    with different fixes, so they are separated: for every own-hand frame the
    rule changed, is it the frame the hand came back on, or the tail after
    it -- and what was the blended score on the frame that went wrong."""
    seen, by_track = {}, collections.defaultdict(list)
    ret, low = set(), []
    for r in sorted(rows, key=lambda r: (r["rec"], int(r["frame"]))):
        t, f = (r["rec"], str(r["tid"])), int(r["frame"])
        if f - seen.get(t, f) - 1 >= min_gap:
            ret.add((t, f))
        seen[t] = f
        if gold.get(t) == "owner" and off[key(r)] and not on[key(r)]:
            by_track[t].append(f)
            if (t, f) in ret:
                low.append(min(1.0, max(0.0, (1 - geom_w) * p[key(r)]
                                        + geom_w * prior[key(r)])))
    runs, first = [], 0
    for t, fs in by_track.items():
        fs.sort()
        cur = 1
        for i in range(1, len(fs)):
            if fs[i] == fs[i - 1] + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        first += sum(1 for f in fs if (t, f) in ret)
    runs.sort()
    return {"frames": sum(runs), "runs": len(runs),
            "median_run": runs[len(runs) // 2] if runs else 0,
            "max_run": runs[-1] if runs else 0,
            "on_the_return_frame": first, "in_the_tail": sum(runs) - first,
            "return_blend_median": sorted(low)[len(low) // 2] if low else None,
            "return_blend_over_half": sum(1 for v in low if v >= 0.5),
            "return_frames_wrong": len(low)}


def one(dump_path, labels_path, pred_path, arm, geom_w, cap, policies=((1, 2),)):
    dump = list(csv.DictReader(open(dump_path, encoding="utf-8")))
    gold = read_zone_gold(labels_path)
    ident = lambda r: f"{r['rec']}|{r['frame']}|{r['tid']}"
    if arm == "V1":
        p = {key(r): float(r["p_owner_raw"]) for r in dump}
        rows = [r for r in dump if (r["rec"], str(r["tid"])) in gold]
    else:
        pred = {}
        for r in csv.DictReader(open(pred_path)):
            if r.get(arm):
                pred[r["id"]] = float(r[arm])
        rows = [r for r in dump if (r["rec"], str(r["tid"])) in gold and ident(r) in pred]
        p = {key(r): pred[ident(r)] for r in rows}
    if not rows:
        return None
    if not full_rate(rows):
        return {"skipped": "预测不是逐帧的（stride 采样），重获门槛无法测量"}
    prior = invert_prior(dump)
    off = replay(rows, p, prior, geom_w=geom_w, cap=cap)
    by_gap = {}
    for mg, rc in policies:
        on = replay(rows, p, prior, geom_w=geom_w, cap=cap, reacquire=mg, reconfirm=rc)
        per = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            k = key(r)
            if on[k] == off[k]:
                continue
            g = gold[(r["rec"], str(r["tid"]))]
            # The reset can only turn `self` into `other`; the other direction
            # is the EMA recovering afterwards and is counted apart rather
            # than netted off, because on screen both are a change.
            d = "covered" if off[k] and not on[k] else "uncovered"
            per[r["rec"]][f"{g}_{d}"] += 1
        tot = collections.Counter()
        for c in per.values():
            tot.update(c)
        by_gap[f"{mg}|{rc}"] = {"totals": dict(tot),
                      "per_recording": {k: dict(v) for k, v in per.items()},
                      "shape": shape_of(rows, gold, off, on, p, prior, geom_w, mg)}
    nre, ntr_gap, ntr = gaps_of(rows)
    return {"hand_frames": len(rows), "tracks": ntr, "tracks_with_gap": ntr_gap,
            "reacquisitions": nre, "by_min_gap": by_gap,
            "totals": by_gap[f"{policies[0][0]}|{policies[0][1]}"]["totals"],
            "own_frames": sum(1 for r in rows if gold[(r["rec"], str(r["tid"]))] == "owner"),
            "other_frames": sum(1 for r in rows if gold[(r["rec"], str(r["tid"]))] == "other")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True,
                    help="name:dump:labels[:pred]")
    ap.add_argument("--arm", action="append", default=None,
                    help="V1, or a column in pred.csv (default: V1 and S_wide_g6)")
    ap.add_argument("--policy", default="1/2,2/2,3/2,1/1",
                    help="<最少丢帧数>/<恢复自己需要的确认帧数>; 1/2 是上线的规则")
    ap.add_argument("--out", default="/workspace/distil/student/reacquire.json")
    a = ap.parse_args()
    arms = a.arm or ["V1", "S_wide_g6"]
    gaps = [tuple(int(v) for v in x.split("/")) for x in a.policy.split(",")]
    report = {}
    head = "".join(f"{'丢'+str(g)+'帧起/确认'+str(c)+'帧':>22}" for g, c in gaps)
    print(f"  {'':<10}{'':<12}{'手帧':>8}{'重获':>7}{head}")
    print(f"  {'':<10}{'':<12}{'':>8}{'':>7}"
          + "".join(f"{'保住别人':>11}{'误盖自己':>11}" for _ in gaps))
    for spec in a.batch:
        parts = spec.split(":")
        name, dump, labels = parts[0], parts[1], parts[2]
        pred = parts[3] if len(parts) > 3 else None
        for arm in arms:
            if arm != "V1" and not (pred and os.path.exists(pred)):
                continue
            geom_w, cap = (0.5, 2) if arm == "V1" else (0.0, None)
            r = one(dump, labels, pred, arm, geom_w, cap, gaps)
            if r is None:
                continue
            report[f"{name}|{arm}"] = r
            if "skipped" in r:
                print(f"  {name:<10}{arm:<12}  {r['skipped']}")
                continue
            cells = ""
            for g, c in gaps:
                t = r["by_min_gap"][f"{g}|{c}"]["totals"]
                cells += f"{t.get('other_covered', 0):>11}{t.get('owner_covered', 0):>11}"
            print(f"  {name:<10}{arm:<12}{r['hand_frames']:>8}{r['reacquisitions']:>7}{cells}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print()
    for g, c in gaps:
        good = sum(v["by_min_gap"][f"{g}|{c}"]["totals"].get("other_covered", 0)
                   for k, v in report.items() if k.endswith("|V1") and "by_min_gap" in v)
        bad = sum(v["by_min_gap"][f"{g}|{c}"]["totals"].get("owner_covered", 0)
                  for k, v in report.items() if k.endswith("|V1") and "by_min_gap" in v)
        print(f"V1 合计（丢 {g} 帧起复位，确认 {c} 帧）：保住别人的手 {good} 帧，"
              f"误盖自己的手 {bad} 帧  = 1 : {bad / max(1, good):.1f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
