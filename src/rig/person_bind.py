"""Bind each hand to a visible body, then ask whether the binding survived.

THE IDEA. In an egocentric video the wearer's body is the one body never in
shot, so a hand inside somebody's person box is somebody's, and a hand inside
nobody's is the wearer's. That is a STRUCTURAL signal, independent of the
appearance features the ownership student reads, and unlike a binary verdict
it yields a person INDEX -- which is what binding a hand to an identity
actually needs, and what makes an ID switch computable instead of a matter of
opinion.

WHAT THIS TEST IS. The 180 pairs already judged by hand -- same physical hand
or not, across a gap -- scored again by the person binding alone: which body
held the hand before, which body held it after, and are they the same body.
Agreement with the human verdict is the measurement. The person boxes are
matched between the two frames by overlap, since nothing tracks them.

PERSON DETECTION IS FREE HERE: the CrowdHuman checkpoint already on disk has
two classes and the project had only ever used the head one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def inside(box, p):
    ix = max(0, min(box[2], p[2]) - max(box[0], p[0]))
    iy = max(0, min(box[3], p[3]) - max(box[1], p[1]))
    return ix * iy / max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="/workspace/idswitchpkg")
    ap.add_argument("--arm", default="/workspace/cam3_c1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--inside", type=float, default=0.5)
    ap.add_argument("--model",
                    default="/shared/models/yolov5-crowdhuman/crowdhuman_yolov5m.pt")
    a = ap.parse_args()
    from src.rig import face_mask
    from src.rig.seam_fix import RawCameraReader

    det = face_mask.load_detector(a.model, a.conf)
    det.CLASS = 0                                   # 0 = person, 1 = head

    jobs = {}
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    # the pairs, rebuilt exactly as the sheet was: same arm, same rule
    def own(path):
        out = collections.defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["own"] == "1":
                out[int(r["frame"])].append(
                    ([int(float(r[c])) for c in ("x0", "y0", "x1", "y1")],
                     float(r["conf"] or 0)))
        return out

    labels = {}
    for r in csv.DictReader(open(os.path.join(a.pkg, "hands.csv"))):
        if r["label"] in ("owner", "other", "nothand", "unsure"):
            labels[r["stem"]] = (r["label"], r["model"])
    want = collections.defaultdict(list)
    for stem in labels:
        rec, rest = stem.rsplit("_f", 1)
        f, h = rest.split("_h")
        want[rec].append((int(f), int(h), stem))

    tally = collections.Counter()
    rows = []
    for rec in sorted(want):
        bag, start, n = jobs[rec]
        N = own(os.path.join(a.arm, rec + ".csv"))
        vids = {k: os.path.join(bag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
        pairs = []
        for f, h, stem in sorted(want[rec]):
            prior = [g for g in N if g < f and N[g]]
            if not prior or f not in N:
                continue
            pf = max(prior)
            box = max(N[f], key=lambda t: t[1])[0] if len(N[f]) == 1 else None
            # the sheet paired the hand with the closest box in the prior
            # frame, so rebuild that rather than guess
            for bx, cf in N[f]:
                pbox = max(N[pf], key=lambda t: iou(t[0], bx))[0]
                pairs.append((pf, pbox, f, bx, stem))
                break
        frames = sorted({x for p in pairs for x in (p[0], p[2])})
        if not frames:
            continue
        rd = RawCameraReader(vids, "cam3", frames[0])
        cur = frames[0] - 1
        persons = {}
        for f in frames:
            img = None
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            persons[f] = [[int(v) for v in b[:4]]
                          for b in face_mask.detect_faces(det, img)]
        rd.close()
        for pf, pbox, f, bx, stem in pairs:
            if pf not in persons or f not in persons:
                continue
            def who(box, ps):
                best, bi = 0.0, None
                for i, p in enumerate(ps):
                    v = inside(box, p)
                    if v > best:
                        best, bi = v, i
                return (bi if best >= a.inside else None), best
            i0, v0 = who(pbox, persons[pf])
            i1, v1 = who(bx, persons[f])
            if i0 is None and i1 is None:
                verdict = "都不属于任何人 -> 同一只(佩戴者)"
            elif i0 is None or i1 is None:
                verdict = "一端属于某人一端不属于 -> 换了"
            else:
                same = iou(persons[pf][i0], persons[f][i1]) >= 0.3
                verdict = "同一个人 -> 同一只" if same else "不同的人 -> 换了"
            lab, grp = labels[stem]
            tally[(lab, verdict.split(" -> ")[1])] += 1
            rows.append({"stem": stem, "human": lab, "group": grp,
                         "person_before": i0, "person_after": i1,
                         "in0": round(v0, 2), "in1": round(v1, 2),
                         "verdict": verdict})
        print("  %-16s %d 对" % (rec, len(pairs)), flush=True)

    NAME = {"owner": "同一只手", "other": "换成别的手了",
            "nothand": "不是手", "unsure": "看不出来"}
    print("\n  %-14s %14s %10s" % ("人工判定", "person 也说同一只", "person 说换了"))
    for lab in ("owner", "other", "nothand", "unsure"):
        s = tally[(lab, "同一只(佩戴者)")] + tally[(lab, "同一只")]
        d = tally[(lab, "换了")]
        if s + d == 0:
            continue
        print("  %-14s %10d %14d" % (NAME[lab], s, d))
    agree = tally[("owner", "同一只(佩戴者)")] + tally[("owner", "同一只")] \
        + tally[("other", "换了")]
    total = sum(v for (l, _), v in tally.items() if l in ("owner", "other"))
    if total:
        print("\n  在『同一只/换了』这 %d 对上，person 绑定与人工一致 %d 个 = %.1f%%"
              % (total, agree, 100.0 * agree / total))
    with open("/workspace/person_bind.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  -> /workspace/person_bind.csv")


if __name__ == "__main__":
    main()
