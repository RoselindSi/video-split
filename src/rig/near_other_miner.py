"""Find the frames where somebody else's arm is IN the workspace.

The blind census answered how often that happens -- 2 of 180 instants, both
from one recording -- and in doing so made clear that a blind sample can never
supply enough of them to train on. 180 instants is 0.04% of the footage, and a
colleague reaching across lasts a few seconds, so blind sampling misses these
almost by construction. This finds them on purpose.

RECALL ONLY. Nothing here decides anything. The score orders candidates for a
person to confirm or reject, so a false positive costs a glance and a false
negative costs a training example -- which is why the threshold is set loose
and the output is a contact sheet rather than a label. The same asymmetry is
why this must NEVER touch the census: a mined sample can train a head, only a
blind sample can say what the deployment rate is, and a prevalence computed
over mined frames would be the miner's own bias read back as a fact.

THE SCORE IS THE THIRD-LARGEST SKIN COMPONENT. A wearer has two arms. Their
two hands often merge into one blob when held together at the centre, so one
or two large skin regions is the normal state of every frame in this corpus; a
THIRD is someone else. That formulation needs no entry direction, which
matters because the direction differs per module -- cam1 is mounted rolled, so
the owner's arm enters cam12 from the lower left, cam34 from the bottom and
cam56 from the right -- and a hardcoded 'arms come from the bottom' would mine
one module correctly and two wrongly.

WHY NOT DEPTH. Depth would separate near from far, which is exactly the
distinction wanted. It is not used because depth needs calibration and only
15 of 40 databags have it, and because the near/far separation is already
carried by component AREA under this rig's fixed geometry: an arm within reach
is large, a colleague across the bench is small. Area costs nothing and covers
the whole corpus.

WHAT THE SCORE ORDERS, AND THE GAP THAT LEAVES. The third-largest area is the
SMALLEST of the three, so it measures confidence that three arms are really
present, not how near the intruder is -- enlarging an intruder makes it the
largest component and does not move the score at all. The consequence is a
real recall gap: a frame where a colleague's arm dominates while both of the
wearer's arms are small or out of shot ranks low. That case is rare on this
rig, because the wearer's own forearm is the nearest object to the camera and
therefore usually the largest thing in the frame, but it is a gap and not an
argument, and it is why the candidate list is reviewed rather than trusted.

WHAT IT WILL ALSO CATCH, stated in advance so the reviewer is not surprised:
bare-wood work surfaces, cardboard, skin-toned gloves and the wearer's own
knee or apron when they lean back. All of those are skin-coloured and large.
They are why a person confirms every candidate.
"""
from __future__ import annotations

import csv
import os

import numpy as np

# Classic YCrCb skin gate. Chroma-only, so it survives the illumination swing
# between a lit bench and the shadow under a shelf far better than an RGB or
# HSV rule would.
CR = (133, 175)
CB = (77, 128)
Y_MIN = 40          # below this it is shadow, and chroma is meaningless there

# A component smaller than this fraction of the frame is a colleague across
# the line, a face, or noise -- not an arm within reach. The census says
# 45.6% of frames contain a distant person, so without a floor this returns
# nearly every frame and orders nothing.
MIN_AREA_FRAC = 0.012

# Sampled every this many frames. At 30 fps this is one look every half
# second, and a reach across a bench lasts several.
STRIDE = 15

VIEWS = ("cam12", "cam34", "cam56")


def skin_mask(bgr):
    import cv2
    y, cr, cb = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))
    m = ((cr >= CR[0]) & (cr <= CR[1]) & (cb >= CB[0]) & (cb <= CB[1])
         & (y >= Y_MIN))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def large_components(mask, min_area_frac=MIN_AREA_FRAC):
    """Areas of components above the floor, largest first, as frame fractions."""
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = mask.size
    a = sorted((stats[i, cv2.CC_STAT_AREA] / total for i in range(1, n)),
               reverse=True)
    return [x for x in a if x >= min_area_frac]


def frame_score(bgr, min_area_frac=MIN_AREA_FRAC):
    """-> (score, areas). Score is the THIRD largest area, 0 if there are
    fewer than three. Two arms are the wearer's; a third is not."""
    a = large_components(skin_mask(bgr), min_area_frac)
    return (a[2] if len(a) >= 3 else 0.0), a


def scan(path, stride=STRIDE, min_area_frac=MIN_AREA_FRAC, half="left"):
    """-> [(frame_index, score, n_large)] over one video."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    out, i = [], 0
    # Sequential reads, not seeks. A seek costs a decode from the nearest
    # keyframe and there are tens of thousands of these; reading straight
    # through and discarding is far cheaper than seeking to every stride.
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, img = cap.retrieve()
            if ok:
                w = img.shape[1] // 2
                img = img[:, :w] if half == "left" else img[:, w:]
                s, a = frame_score(img, min_area_frac)
                out.append((i, s, len(a)))
        i += 1
    cap.release()
    return out


def _self_test():
    """Synthetic frames, because the thing being tested is the counting rule."""
    import cv2
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    H = W = 400
    SKIN = (110, 150, 200)      # BGR that lands inside the YCrCb gate

    def frame(*boxes):
        f = np.full((H, W, 3), (40, 90, 60), np.uint8)   # green bench
        for (x0, y0, x1, y1) in boxes:
            f[y0:y1, x0:x1] = SKIN
        return f

    big = 70    # 70x70 = 3.1% of the frame, above the floor
    sml = 20    # 20x20 = 0.25%, below it

    chk(frame_score(frame((10, 10, 10 + big, 10 + big)))[0] == 0,
        "one arm scores zero")
    chk(frame_score(frame((10, 10, 10 + big, 10 + big),
                          (200, 10, 200 + big, 10 + big)))[0] == 0,
        "two arms score zero -- a wearer has two")
    s, a = frame_score(frame((10, 10, 10 + big, 10 + big),
                             (150, 10, 150 + big, 10 + big),
                             (290, 10, 290 + big, 10 + big)))
    chk(s > 0 and len(a) == 3, "a third arm scores above zero")
    chk(frame_score(frame((10, 10, 10 + big, 10 + big),
                          (150, 10, 150 + big, 10 + big),
                          (290, 10, 290 + sml, 10 + sml)))[0] == 0,
        "a distant third person is below the area floor and does not score")
    chk(frame_score(frame())[0] == 0 and frame_score(frame())[1] == [],
        "an empty bench scores zero and lists nothing")
    # The bench itself must not read as skin, or every frame ranks equally.
    chk(len(large_components(skin_mask(
        np.full((H, W, 3), (40, 90, 60), np.uint8)))) == 0,
        "the green bench is not skin")
    # Two arms that touch merge into one component -- the normal centre pose.
    s, a = frame_score(frame((100, 10, 100 + big, 10 + big),
                             (100 + big, 10, 100 + 2 * big, 10 + big)))
    chk(s == 0 and len(a) == 1, "two hands held together merge, still zero")
    # What the score actually orders, stated as a test so it cannot drift.
    s3, _ = frame_score(frame((10, 10, 80, 80), (150, 10, 220, 80),
                              (290, 10, 360, 80)))
    s_big, _ = frame_score(frame((10, 10, 80, 80), (150, 10, 220, 80),
                                 (250, 10, 390, 150)))
    chk(s_big == s3,
        "enlarging the intruder does NOT raise the score -- it becomes the "
        "largest\n       and the third place is still an owner arm")
    s_all, _ = frame_score(frame((10, 10, 110, 110), (150, 10, 250, 110),
                                 (280, 10, 380, 110)))
    chk(s_all > s3,
        "...the score rises only when ALL THREE are large, which is what "
        "'three\n       arms are really present' means")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  The rule is a count, not a direction: cam1 is mounted rolled and\n"
          "  the owner's arm enters each module from a different edge.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--root", help="directory searched for cam*.mp4")
    ap.add_argument("--out", help="output directory. The SAN is /workspace.")
    ap.add_argument("--recordings", default="",
                    help="comma-separated substrings; default is all")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--top", type=int, default=8,
                    help="candidates kept per recording")
    ap.add_argument("--min_gap_sec", type=float, default=20.0,
                    help="candidates closer than this collapse to one, so a "
                         "single long reach does not fill the whole list")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.root or not a.out:
        ap.error("--root and --out are required without --self_test")

    import cv2
    from src.rig.class2_census import find_recordings, _check_space

    free = _check_space(a.out)
    recs = [r for r in find_recordings(a.root) if len(r[1]) == len(VIEWS)]
    if a.recordings:
        want = [s.strip() for s in a.recordings.split(",") if s.strip()]
        recs = [r for r in recs if any(w in r[0] for w in want)]
    if not recs:
        raise SystemExit("no recording matched")

    print(f"{len(recs)} recordings, stride {a.stride} "
          f"({a.stride/30:.1f}s), top {a.top} each\n"
          f"  -> {a.out}  ({free:.0f} GB free)\n")

    rows, sheets = [], []
    for i, (rid, views, _) in enumerate(recs, 1):
        per_view = {}
        for v in VIEWS:
            per_view[v] = {f: (s, n) for f, s, n in scan(views[v], a.stride)}
        frames = sorted(set().union(*(set(d) for d in per_view.values())))
        # A frame's score is the best any module saw. A colleague reaching in
        # from the left is in cam12 and absent from cam56, and taking the max
        # is what keeps a one-module event from being averaged away.
        scored = [(f, max(per_view[v].get(f, (0, 0))[0] for v in VIEWS))
                  for f in frames]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        kept = []
        for f, s in scored:
            if all(abs(f - g) >= a.min_gap_sec * 30 for g, _ in kept):
                kept.append((f, s))
            if len(kept) >= a.top:
                break
        print(f"  [{i}/{len(recs)}] {rid}: {len(scored)} scoring frames, "
              f"{len(kept)} kept")
        if not kept:
            continue
        cap = {v: cv2.VideoCapture(views[v]) for v in VIEWS}
        T = 400
        sheet = np.zeros((len(kept) * T, len(VIEWS) * T, 3), np.uint8)
        for r, (f, s) in enumerate(kept):
            for c, v in enumerate(VIEWS):
                cap[v].set(cv2.CAP_PROP_POS_FRAMES, int(f))
                got, img = cap[v].read()
                if not got:
                    continue
                t = cv2.resize(img[:, :img.shape[1] // 2], (T, T))
                cv2.putText(t, f"{v} f{f} {f/30:.0f}s s={s:.3f}", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                sheet[r * T:(r + 1) * T, c * T:(c + 1) * T] = t
            rows.append({"recording": rid, "frame": f,
                         "t_sec": round(f / 30, 1), "score": round(s, 4),
                         "confirmed": "", "note": ""})
        for v in cap.values():
            v.release()
        p = os.path.join(a.out, f"mined_{rid}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        sheets.append(p)

    if not rows:
        print("\n  nothing scored above zero anywhere. Either no third arm "
              "ever enters,\n  or MIN_AREA_FRAC is too high for this rig's "
              "framing -- check one\n  known positive before believing the "
              "first.")
        return
    csv_path = os.path.join(a.out, "mined_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot = sum(os.path.getsize(p) for p in sheets) + os.path.getsize(csv_path)
    print(f"\n  {len(rows)} candidates, {len(sheets)} sheets, "
          f"{tot/1e6:.1f} MB\n  {csv_path}")
    print("\n  'confirmed' takes yes/no. These are CANDIDATES -- the score "
          "orders them,\n  it does not decide, and a rejected one costs a "
          "glance while a missed one\n  costs a training example.")
    print("  Do NOT compute a prevalence from this file. That is what the "
          "blind census\n  is for, and it is already frozen.")


if __name__ == "__main__":
    main()
