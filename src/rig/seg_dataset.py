"""Training samples for the 3-class ownership head: RGB + range + validity.

    0  background and objects
    1  the wearer's own hand and arm
    2  anyone else's hand and arm

WHY A LEARNED HEAD AT ALL. Depth alone cannot do this, and that is measured
rather than assumed: on this rig the bench's front edge sits at 0.21-0.25 m
and the wearer's forearm at 0.23-0.30 m, so every hand-written rule that asked
depth to separate them took the bench too (see `owner_arm`). Depth is kept as
an INPUT here, not a decision -- one signal among appearance, arm shape, scale
and geometry, with the head free to weigh it.

RENDERED IN THE WIDE FRAME, FOR TWO REASONS. It is the production frame, so a
mask trained here needs no transfer. And it is three times cheaper to annotate:
one wide image per instant instead of one per module. The second reason is the
decisive one, because drawing masks is the entire cost of this dataset.

It also normalises geometry across modules. cam1 is mounted rolled, so in raw
frames the wearer's arm enters cam12 from the lower left, cam34 from the bottom
and cam56 from the right. A head trained on raw modules would have to learn
that prior three times from a few hundred frames; rendered into the fan's own
frame there is one prior, and the arm enters from the bottom.

RANGE IS STORED AS ONE FILE, NOT TWO. A uint16 PNG in millimetres, where 0
means nothing was measured. Zero is not a legal depth -- the matcher's floor is
0.12 m -- so the encoding is unambiguous, and a single file cannot fall out of
step with itself the way a range map and a separate validity mask can. The
loader derives validity; `channels()` hands the model both planes and refuses
to hand it one.

THE CENSUS FRAMES ARE THE EVAL SET AND MUST NOT BE TRAINED ON. Those 180
instants are blind, stratified across 25 recordings and apportioned by
footage, which makes them the only sample in this project that can state a
deployment rate. Mined frames are the opposite -- selected because a score
liked them -- and are training material only. This module stamps `split` on
every row from its source and refuses to write a mixed directory, because the
one thing that would waste all of it is discovering later that the eval frames
were in the training set.

RECORDINGS WITHOUT CALIBRATION STILL CONTRIBUTE. Only 15 of 40 databags carry
real calibration, and the wide render needs it. Rather than drop two thirds of
the corpus, an uncalibrated recording can supply raw per-module RGB with an
all-zero range plane: the head must already cope with 40-50% unmeasured depth,
and all-unmeasured is that same case at its limit. Those rows are marked
`has_depth=0` so the effect of depth can be read off directly instead of being
assumed.
"""
from __future__ import annotations

import csv
import os

import numpy as np

# Millimetres in a uint16. The matcher's own bounds are 0.12-8.0 m, so the
# range fits with three orders of magnitude to spare and 1 mm quantisation is
# far below SGBM's error.
MM_MAX = 65535
INVALID = 0

SPLITS = ("eval", "train")


def encode_range(range_m):
    """metres float (NaN where unmeasured) -> uint16 millimetres, 0 = unmeasured."""
    # Mask BEFORE scaling. nan_to_num turns an inf into a huge finite value,
    # which then overflows on the multiply -- the clip below would still land
    # on the right answer, but by accident rather than by construction.
    ok = np.isfinite(range_m)
    mm = np.where(ok, range_m, 0.0).astype(np.float64) * 1000.0
    return np.clip(mm, 0, MM_MAX).astype(np.uint16)


def decode_range(png16):
    """uint16 millimetres -> (metres float with NaN, validity bool)."""
    valid = png16 > INVALID
    m = np.where(valid, png16.astype(np.float32) / 1000.0, np.nan)
    return m, valid


def channels(png16):
    """The two input planes, together. Range in metres with unmeasured set to
    zero, and the validity that says which zeros are measurements -- neither is
    returned without the other, because a model handed only the first would
    read half the bench as touching the lens."""
    m, valid = decode_range(png16)
    return np.nan_to_num(m, nan=0.0), valid.astype(np.float32)


def find_calibration(databag_dir, cache=None):
    """The rig calibration for one databag, or None. Parse decides, not the
    filename: a yaml that RigCalibration refuses is the same as absent."""
    from pathlib import Path
    from src.rig.calibration import RigCalibration, CalibrationError
    cache = {} if cache is None else cache
    if databag_dir in cache:
        return cache[databag_dir]
    got = None
    for p in sorted(Path(databag_dir).rglob("*.yaml"))[:40]:
        try:
            got = RigCalibration(str(p))
            break
        except CalibrationError:
            continue                 # a yaml that is not a rig calibration
        except (OSError, ValueError, KeyError, TypeError):
            continue                 # malformed, truncated, or not yaml
        # Anything else -- an ImportError, a typo in RigCalibration -- is a
        # bug and must not be swallowed as "this databag has no calibration".
    cache[databag_dir] = got
    return got


def survey_calibration(root):
    """Which databags can produce a wide render at all. -> [(rid, status)]

    This is the number that bounds everything downstream: the wide frame is
    the production frame and the cheap one to annotate, and it needs
    calibration. A databag whose file says `uncalibrated` with zeroed
    intrinsics is not a parsing problem to be worked around -- there is no
    calibration in it."""
    from pathlib import Path
    from src.rig.calibration import RigCalibration, CalibrationError
    from src.rig.class2_census import find_recordings
    out = []
    for rid, views, d in find_recordings(root):
        ys = sorted(Path(d).rglob("*.yaml"))[:40]
        if not ys:
            out.append((rid, "no yaml"))
            continue
        why = "no rig yaml"
        for p in ys:
            try:
                RigCalibration(str(p))
                why = "OK"
                break
            except CalibrationError as e:
                why = str(e).split("\n")[0].split(": ")[-1][:60]
            except Exception as e:
                why = f"{type(e).__name__}: {e}"[:60]
        out.append((rid, why))
    return out


def export_instant(rig, vcam, views, frame, out_dir, rid, rect_cache=None):
    """One instant -> (rgb path, range path). Range is None without depth."""
    import cv2
    from src.rig.render_wide import read_frame, split_halves, render
    from src.rig.wide_depth import wide_depth

    sources = {}
    for m in rig.modules:
        key = f"cam{m.left.name[-1]}{m.right.name[-1]}"
        if key not in views:
            continue
        l, r = split_halves(read_frame(views[key], frame))
        sources[m.left.name], sources[m.right.name] = l, r
    if not sources:
        return None, None

    rgb, _, _, _ = render(rig, vcam, sources, 0.6)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "range"), exist_ok=True)
    stem = f"{rid}_f{frame:06d}"
    p_rgb = os.path.join(out_dir, "images", stem + ".png")
    cv2.imwrite(p_rgb, rgb)

    wd = wide_depth(rig, vcam, sources, rect_cache=rect_cache)
    p_rng = os.path.join(out_dir, "range", stem + ".png")
    cv2.imwrite(p_rng, encode_range(wd.range_m))
    return p_rgb, p_rng


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--survey_calib", action="store_true",
                    help="report which databags carry a usable calibration "
                         "and stop. This bounds everything downstream.")
    ap.add_argument("--frames",
                    help="CSV with recording,frame columns -- the labelled "
                         "census for eval, the mined candidates for train")
    ap.add_argument("--split", choices=SPLITS,
                    help="stamped on every row and on the directory. The "
                         "census is eval; anything mined is train.")
    ap.add_argument("--out", help="output directory. The SAN is /workspace.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.survey_calib:
        res = survey_calibration(a.root)
        ok = [r for r in res if r[1] == "OK"]
        print(f"{len(ok)}/{len(res)} databags carry a usable calibration\n")
        from collections import Counter
        for why, n in Counter(w for _, w in res).most_common():
            print(f"  {n:3d}  {why}")
        print()
        for rid, why in sorted(res, key=lambda r: (r[1] != "OK", r[0])):
            print(f"    {'OK ' if why == 'OK' else '-- '} {rid}"
                  + ("" if why == "OK" else f"   {why}"))
        print("\n  Only 'OK' databags can produce a wide render, which is the "
              "production\n  frame and the one that is cheap to annotate. The "
              "rest can still supply\n  raw per-module RGB with an all-zero "
              "range plane -- three times the\n  annotation cost per instant, "
              "and no depth.")
        return
    if not a.frames or not a.split or not a.out:
        ap.error("--frames, --split and --out are required "
                 "unless --survey_calib")

    from src.rig.class2_census import find_recordings, _check_space
    from src.rig.geometry import VirtualWideCamera

    free = _check_space(a.out)
    marker = os.path.join(a.out, "SPLIT")
    if os.path.exists(marker):
        was = open(marker).read().strip()
        if was != a.split:
            raise SystemExit(
                f"{a.out} already holds the '{was}' split and this is "
                f"'{a.split}'.\n  Mixing them silently is the one mistake "
                f"that would waste the whole dataset:\n  a head evaluated on "
                f"frames it trained on reports a number about nothing.\n"
                f"  Use a separate directory.")
    os.makedirs(a.out, exist_ok=True)
    open(marker, "w").write(a.split)

    by_id = {rid: (views, d) for rid, views, d in find_recordings(a.root)}
    want = list(csv.DictReader(open(a.frames, encoding="utf-8-sig")))
    if a.limit:
        want = want[:a.limit]

    print(f"{len(want)} instants, split '{a.split}'\n"
          f"  -> {a.out}  ({free:.0f} GB free)\n")

    calib_cache, rect_cache, rows = {}, {}, []
    n_depth = n_skip = 0
    for i, w in enumerate(want, 1):
        rid = w["recording"]
        if rid not in by_id:
            n_skip += 1
            continue
        views, d = by_id[rid]
        rig = find_calibration(d, calib_cache)
        if rig is None:
            n_skip += 1
            rows.append({"recording": rid, "frame": int(w["frame"]),
                         "split": a.split, "has_depth": 0,
                         "census_label": w.get("label", ""),
                         "rgb": "", "range": "", "note": "no calibration"})
            continue
        key = id(rig)
        if key not in rect_cache:
            rect_cache[key] = ({}, VirtualWideCamera.from_rig(rig))
        rc, vcam = rect_cache[key]
        p_rgb, p_rng = export_instant(rig, vcam, views, int(w["frame"]),
                                      a.out, rid, rect_cache=rc)
        if p_rgb is None:
            n_skip += 1
            continue
        n_depth += 1
        rows.append({"recording": rid, "frame": int(w["frame"]),
                     "split": a.split, "has_depth": 1,
                     "census_label": w.get("label", ""),
                     "rgb": os.path.relpath(p_rgb, a.out),
                     "range": os.path.relpath(p_rng, a.out), "note": ""})
        if i % 20 == 0 or i == len(want):
            print(f"  [{i}/{len(want)}] {n_depth} with depth, {n_skip} skipped")

    man = os.path.join(a.out, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    assert os.path.getsize(man) > 0
    tot = sum(os.path.getsize(os.path.join(dp, fn))
              for dp, _, fns in os.walk(a.out) for fn in fns)
    print(f"\n  {len(rows)} rows, {n_depth} rendered with depth, "
          f"{tot/1e6:.1f} MB\n  {man}")
    n_cal = sum(1 for v in calib_cache.values() if v is not None)
    print(f"  calibration found for {n_cal}/{len(calib_cache)} databags")
    print(f"\n  Annotate images/ into three classes: 0 background, "
          f"1 owner arm, 2 other\n  arm. range/ is uint16 millimetres with 0 "
          f"meaning unmeasured -- never\n  treat a 0 as a distance.")
    if a.split == "eval":
        print("\n  This is the EVAL split. It came from the blind census, it "
              "carries the\n  real prevalence, and nothing here may enter "
              "training.")


if __name__ == "__main__":
    main()
