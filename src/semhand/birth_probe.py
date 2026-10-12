"""The tracks that were called the wearer's on their first frame, put to both teachers and to a person.

THE QUESTION, WHICH IS NOT "IS THE TEACHER ACCURATE". At the moment a track is
born, a hand that is the wearer's and a hand that is not are indistinguishable
from the track's own evidence: there is one probability and no history.
`OwnHold` is a machine for using history, so it has nothing to work with here,
and no setting of `lo`, `hi`, `fast` or `slow` changes that. Evidence has to
come from outside the probability sequence.

The obvious outside source is the semantic layer, and the obvious objection is
that it is not outside at all: `S_wide_g6` was distilled from Qwen Q1, so
asking Q1 again may return the student's own answer with more confidence and
no new information. That objection is testable before anything is built on
it, and testing it is what this file is for.

WHAT THE ANSWER LOOKS LIKE. Three outcomes, three different next steps:

  both teachers agree with the student   the witness is an echo; this line
                                         closes and the cap or the birth
                                         threshold is the only lever left
  the teachers disagree with each other  the student was trained only where
                                         they agreed, so these boxes are out
                                         of its distribution -- which is
                                         itself the explanation for a birth
                                         score of 0.567
  teachers agree, student is opposite    distillation lost something, and the
                                         teacher is worth putting back in the
                                         loop for these cases

SELECTION IS PRE-REGISTERED AND USES NO MODEL THE PROBE WILL BE SCORED
AGAINST. Two criteria, both from the shipped run's own CSV:

  A  born owner on a score in [0.50, 0.70). A new track enters `owner` at
     0.50 while a track already called `other` needs 0.70 to get there, so
     these verdicts could not have been earned one frame later.
  B  a strict majority of frames delivered as `owner` while the median
     smoothed score sits below 0.5 -- a verdict carried by hysteresis rather
     than by evidence.
  C  born `other` and later promoted to `owner`. The mirror of A: the system
     got the first frame right and then changed its mind, which it can only
     do by the belief crossing `hi` (0.70). Added after the first batch came
     back, because A and B between them account for 8% of the wrong-looking
     `owner` boxes in the half of one recording where the wearer is away from
     their bench, and 27% belong to two tracks of this shape.

Both are computed on `p`, which is OwnHold's smoothed value and not the raw
student score; the raw score is overwritten in place by `demo_video` and has
never been written down. B is therefore a proxy for the statistic it is named
after, and says so here so that nobody reads it as that statistic later.

FIVE FRAMES PER TRACK, AT FIXED POSITIONS, CHOSEN BEFORE LOOKING. At 10, 30,
50, 70 and 90 per cent of the track's length: not the frames where the model
is least sure, which would select for the answer.

AND THE QUESTION IS ASKED PER HAND PER FRAME, NOT PER TRACK. "Is this whole
track the wearer's" folds four jobs into one -- recognising the same hand
across frames, reading several moments, judging ownership, and summarising --
and Q1's measured 95.7% belongs to the single-hand single-frame form. The five
answers are aggregated outside the model, by a rule fixed in advance.

THE SHEET FOR THE PERSON CARRIES NO SCORES. Not the student's, not either
teacher's, not which criterion selected the track. A reader told "the model
says 0.567" is being asked to agree rather than to look.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import statistics

BIRTH_LO, BIRTH_HI = 0.50, 0.70
MIN_FRAMES = 5
PICKS = (0.10, 0.30, 0.50, 0.70, 0.90)


def tracks_in(path):
    """-> {tid: [row, ...]} in frame order, hand-ness rejects dropped."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("not_hand") == "1":
            continue
        by[r["tid"]].append(r)
    for tid in by:
        by[tid].sort(key=lambda r: int(r["frame"]))
    return by


def candidates(arm):
    """-> [(rec, tid, rows, why)] for both pre-registered criteria."""
    out = []
    for p in sorted(glob.glob(os.path.join(arm, "*.csv"))):
        if p.endswith((".faces.csv", ".decisions.csv")):
            continue
        rec = os.path.basename(p)[:-4]
        for tid, fr in tracks_in(p).items():
            if len(fr) < MIN_FRAMES:
                continue
            p0 = float(fr[0]["p"])
            own_n = sum(int(r["own"]) for r in fr)
            med = statistics.median(float(r["p"]) for r in fr)
            why = []
            if int(fr[0]["own"]) == 1 and BIRTH_LO <= p0 < BIRTH_HI:
                why.append("birth")
            if own_n * 2 > len(fr) and med < 0.5:
                why.append("deadband")
            if int(fr[0]["own"]) == 0 and own_n > 0:
                why.append("promoted")
            if why:
                out.append((rec, tid, fr, "+".join(why)))
    return out


def scan(arms):
    """Count what each criterion selects, without building anything.

    A criterion whose population turns out to be large needs its sampling
    rule written down before any track is looked at, and a rule cannot be
    written before the size is known. So the count comes first and on its
    own."""
    rows = []
    for arm in arms:
        for rec, tid, fr, why in candidates(arm):
            own_n = sum(int(r["own"]) for r in fr)
            rows.append({"arm": os.path.basename(arm), "rec": rec, "tid": tid,
                         "why": why, "n": len(fr), "own_n": own_n,
                         "birth_p": float(fr[0]["p"]),
                         "max_p": max(float(r["p"]) for r in fr),
                         "w": int(fr[0]["x1"]) - int(fr[0]["x0"])})
    return rows


def picks(fr):
    """The five rows to ask about. Fixed positions, not model-selected."""
    n = len(fr)
    seen, out = set(), []
    for f in PICKS:
        i = min(n - 1, max(0, int(round(f * (n - 1)))))
        if i not in seen:
            seen.add(i)
            out.append(fr[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+",
                    default=["/workspace/ten_base", "/workspace/cam3_ship2"])
    ap.add_argument("--jobs", nargs="+",
                    default=["/workspace/cam3_jobs10.txt", "/workspace/cam3_jobs.txt"])
    ap.add_argument("--out", help="the blind package")
    ap.add_argument("--frames", help="where the clean frames go")
    ap.add_argument("--scan", action="store_true",
                    help="count the criteria and stop; build nothing")
    a = ap.parse_args()
    if a.scan:
        rows = scan(a.arms)
        by = collections.Counter(r["why"] for r in rows)
        print("%-22s %6s %8s %9s %10s" % ("判据", "轨迹", "帧合计", "判自己帧", "中位框宽"))
        for why in sorted(by):
            g = [r for r in rows if r["why"] == why]
            ws = sorted(r["w"] for r in g)
            print("%-22s %6d %8d %9d %10d"
                  % (why, len(g), sum(r["n"] for r in g),
                     sum(r["own_n"] for r in g), ws[len(ws) // 2]))
        print()
        pro = [r for r in rows if "promoted" in r["why"]]
        print("promoted 这一类，按「被提升的帧数」分布：")
        for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 30), (31, 10 ** 9)):
            g = [r for r in pro if lo <= r["own_n"] <= hi]
            if g:
                print("   %-8s %3d 条轨迹，合计 %5d 帧判自己"
                      % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else ">=31",
                         len(g), sum(r["own_n"] for r in g)))
        print()
        print("   出生分中位 %.3f，之后最高分中位 %.3f"
              % (sorted(r["birth_p"] for r in pro)[len(pro) // 2],
                 sorted(r["max_p"] for r in pro)[len(pro) // 2]))
        print()
        print("按录像：")
        rec = collections.Counter(r["rec"] for r in pro)
        for k, n in rec.most_common():
            print("   %-22s %d 条" % (k, n))
        return
    if not (a.out and a.frames):
        raise SystemExit("--out 和 --frames 是建包必需的（或者用 --scan）")
    import cv2
    from src.rig.seam_fix import RawCameraReader

    jobs = {}
    for jp in a.jobs:
        for line in open(jp):
            f = line.strip().split("|")
            if len(f) >= 4 and not line.startswith("#"):
                jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    cands = []
    for arm in a.arms:
        got = candidates(arm)
        print("%-24s %d 条候选轨迹" % (arm, len(got)))
        cands += got
    if not cands:
        raise SystemExit("没有候选")

    os.makedirs(a.frames, exist_ok=True)
    os.makedirs(os.path.join(a.out, "context"), exist_ok=True)

    want = collections.defaultdict(set)
    for rec, tid, fr, why in cands:
        for r in picks(fr):
            want[rec].add(int(r["frame"]))

    # THE CLEAN NATIVE FRAME, NOT THE RENDER'S PANEL. The panel has boxes and
    # labels drawn on it and is half size; asking either teacher about that is
    # asking about a different picture than the one Q1 was measured on. Cost
    # of getting this wrong, once already: a held-out run that reported 100%
    # not-a-hand because it was handed the drawn copy.
    for rec in sorted(want):
        bag = jobs[rec][0]
        vids = {k: os.path.join(bag, f"{k}.mp4")
                for k in ("cam12", "cam34", "cam56")}
        need = sorted(want[rec])
        rd = RawCameraReader(vids, "cam3", need[0])
        cur = need[0] - 1
        for f in need:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            cv2.imwrite(os.path.join(a.frames, "%s_f%06d.jpg" % (rec, f)),
                        img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rd.close()
        print("  %-20s %d 帧" % (rec, len(need)), flush=True)

    rows, key = [], []
    for rec, tid, fr, why in sorted(cands, key=lambda c: (c[0], int(c[1]))):
        p0 = float(fr[0]["p"])
        own_n = sum(int(r["own"]) for r in fr)
        med = statistics.median(float(r["p"]) for r in fr)
        for r in picks(fr):
            f = int(r["frame"])
            img = os.path.join(a.frames, "%s_f%06d.jpg" % (rec, f))
            if not os.path.exists(img):
                continue
            box = [int(r[c]) for c in ("x0", "y0", "x1", "y1")]
            stem = "%s_t%s_f%06d" % (rec, tid, f)
            from src.semhand import qwen
            frame, _ = qwen.views({"image": img, "box": box})
            frame.save(os.path.join(a.out, "context", stem + ".jpg"), quality=86)
            rows.append({"stem": stem, "rec": rec, "tid": tid, "frame": f,
                         "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                         "w": box[2] - box[0], "image": img,
                         "label": "", "label_mode": ""})
            key.append({"stem": stem, "rec": rec, "tid": tid, "frame": f,
                        "why": why, "n_frames": len(fr), "own_frames": own_n,
                        "birth_p": round(p0, 4), "median_p": round(med, 4),
                        "p_here": round(float(r["p"]), 4),
                        "own_here": int(r["own"]), "w": box[2] - box[0]})

    with open(os.path.join(a.out, "hands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write("问题：绿框里的手是佩戴者自己的，还是别人的？\n"
                 "佩戴者 -> owner   别人 -> other   不是手 -> nothand   "
                 "看不清 -> unsure\n"
                 "同一条轨迹的五帧文件名里 tid 相同，可以一起看。\n"
                 "模型的答案和挑中这条轨迹的理由都不在这张表里。\n")
    n_tr = len({(r["rec"], r["tid"]) for r in rows})
    print("-> %s  %d 轨迹 / %d 框" % (a.out, n_tr, len(rows)))
    print("   两个判据的分布：%s"
          % dict(collections.Counter(k["why"] for k in key)))


if __name__ == "__main__":
    main()
