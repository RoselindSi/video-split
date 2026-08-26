"""Annotation packages: one keyframe to draw, six seconds to propagate through.

Each of the 180 blind census instants becomes an anchor:

    sample_<recording>_f<frame>/
        keyframe_rgb.png        the frame a person draws three classes on
        keyframe_range.png      uint16 mm, 0 = unmeasured
        clip_rgb.mp4            t-3s .. t+3s at full rate, for SAM2
        clip_range.mp4          the same window, subsampled, for human eyes
        range/f######.png       the frames that will become training rows
        meta.json               provenance, the window, and the break hints

WHY SIX SECONDS AND NOT TWO OR TWENTY. Long enough to contain a complete
reach-and-place; short enough that a mask survives it. Past a few seconds the
things that break propagation start arriving -- a hand leaves and re-enters,
an occlusion swaps two identities, the wearer's arm crosses a colleague's --
and repairing a drifted mask costs more than drawing a fresh keyframe. The
window is a MAXIMUM, not a quota.

THE BREAK HINTS ARE THE POINT OF THIS TOOL, not the clips. An annotator told
to 'stop propagating at an occlusion, a cut, a new hand or heavy blur' has to
watch 181 frames per sample to find them, 180 times over. Two of those four
conditions are measurable and are measured here for every frame:

    blur        variance of the Laplacian; a drop is motion blur or a cut
    n_skin      count of large skin components; a rise from 2 to 3 is a new
                arm arriving, which is exactly a propagation break

They are HINTS and are labelled as such. They will miss a slow occlusion and
they will flag a bright reflection, so meta.json carries the raw series as
well as the suggested break frames -- a person judges, this narrows where they
look.

FULL RATE FOR RGB, SUBSAMPLED FOR RANGE, AND THE REASON IS COST. SAM2 tracks
through consecutive frames, so clip_rgb cannot be thinned. Depth costs about
1.6 s per frame across three modules, which at every frame of every clip is
15 hours; the frames that actually become training rows are one in five, and
those are the only ones that need it. The manifest says which frames have
range so nothing downstream has to guess.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

FPS = 30.0
HALF_WINDOW_SEC = 3.0
RANGE_EVERY = 5              # training rows: 6 fps out of 30
CLIP_RANGE_FPS = 6.0

VIEWS = ("cam12", "cam34", "cam56")

# A frame whose Laplacian variance falls below this share of the clip's own
# median is blurred or cut. Relative, because absolute sharpness varies with
# the bench, the lighting and the lens.
BLUR_REL = 0.45

README = """\
ANNOTATION PACKAGE -- 3-class ownership masks

  0  background and objects
  1  the WEARER's own hand and arm
  2  ANYONE ELSE's hand and arm

Draw on keyframe_rgb.png. Then propagate through clip_rgb.mp4, which runs
from 3 seconds before the keyframe to 3 seconds after at the full frame rate.

THE WINDOW IS A MAXIMUM, NOT A QUOTA. Stop propagating and re-seed whenever:

  - the wearer's hand leaves the frame entirely
  - a new person's hand enters
  - a strong occlusion passes over either arm
  - a camera cut, or blur heavy enough to lose the boundary

A wrong mask carried forward costs more to repair than a fresh keyframe costs
to draw. Six seconds of good propagation is worth more than 181 frames of
drift.

meta.json lists SUGGESTED break frames under "breaks", found from image blur
and from the number of large skin regions. They are hints. They miss a slow
occlusion and they fire on a bright reflection -- they say where to look, not
what to do.

range/ holds uint16 PNGs in millimetres for one frame in five. ZERO MEANS NOT
MEASURED, never zero distance: the stereo pair leaves the textureless bench
top unmeasured over much of the frame, and it is left that way rather than
filled in.
"""


def _laplacian_var(bgr):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def suggest_breaks(series, blur_rel=BLUR_REL):
    """-> (break frame indices, reasons). Hints, and named as hints.

    A break is where propagation should restart: sharpness collapsing (a cut
    or heavy motion blur) or the number of large skin regions RISING, which is
    an arm arriving. A falling count is an arm leaving, which propagation
    handles by itself -- flagging it would bury the real ones."""
    if not series:
        return [], {}
    blur = np.array([s["blur"] for s in series], float)
    nsk = np.array([s["n_skin"] for s in series], int)
    med = np.median(blur[blur > 0]) if (blur > 0).any() else 0.0
    out, why = [], {}
    for i, s in enumerate(series):
        r = []
        if med > 0 and blur[i] < blur_rel * med:
            r.append("blur")
        if i > 0 and nsk[i] > nsk[i - 1]:
            r.append(f"skin_regions {nsk[i-1]}->{nsk[i]}")
        if r:
            out.append(s["frame"])
            why[str(s["frame"])] = r
    return out, why


def build_sample(rig, vcam, views, anchor, out_dir, rid, label="",
                 half_sec=HALF_WINDOW_SEC, range_every=RANGE_EVERY,
                 rect_cache=None, fps=FPS):
    """One anchor -> one directory. Returns its meta dict."""
    import cv2
    from src.rig.render_wide import split_halves, render
    from src.rig.wide_depth import wide_depth
    from src.rig.near_other_miner import large_components, skin_mask
    from src.rig.seg_dataset import encode_range

    caps = {v: cv2.VideoCapture(views[v]) for v in VIEWS if v in views}
    if len(caps) != len(VIEWS):
        for c in caps.values():
            c.release()
        return None
    n_total = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    half = int(round(half_sec * fps))
    f0 = max(0, anchor - half)
    f1 = min(n_total - 1, anchor + half)
    clamped = (f0 != anchor - half) or (f1 != anchor + half)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "range"), exist_ok=True)
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, f0)

    writer = None
    range_frames, series, rng_vis = [], [], []
    for f in range(f0, f1 + 1):
        srcs = {}
        got = True
        for v in VIEWS:
            ok, img = caps[v].read()          # sequential, never re-seeking
            if not ok:
                got = False
                break
            mod = next(m for m in rig.modules
                       if f"cam{m.left.name[-1]}{m.right.name[-1]}" == v)
            l, r = split_halves(img)
            srcs[mod.left.name], srcs[mod.right.name] = l, r
        if not got:
            break
        rgb, _, _, _ = render(rig, vcam, srcs, 0.6)
        if writer is None:
            h, w = rgb.shape[:2]
            writer = cv2.VideoWriter(os.path.join(out_dir, "clip_rgb.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (w, h))
        writer.write(rgb)
        series.append({"frame": f,
                       "blur": round(_laplacian_var(rgb), 1),
                       "n_skin": len(large_components(skin_mask(rgb)))})
        if f == anchor:
            cv2.imwrite(os.path.join(out_dir, "keyframe_rgb.png"), rgb)
        if (f - f0) % range_every == 0 or f == anchor:
            wd = wide_depth(rig, vcam, srcs, rect_cache=rect_cache)
            enc = encode_range(wd.range_m)
            cv2.imwrite(os.path.join(out_dir, "range", f"f{f:06d}.png"), enc)
            range_frames.append(f)
            if f == anchor:
                cv2.imwrite(os.path.join(out_dir, "keyframe_range.png"), enc)
            t = np.nan_to_num(np.clip((wd.range_m - 0.2) / 2.3, 0, 1))
            vis = cv2.applyColorMap((t * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
            vis[~wd.valid] = 0
            rng_vis.append(vis)
    if writer is not None:
        writer.release()
    for c in caps.values():
        c.release()

    if rng_vis:
        h, w = rng_vis[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(out_dir, "clip_range.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), CLIP_RANGE_FPS,
                             (w, h))
        for v in rng_vis:
            vw.write(v)
        vw.release()

    breaks, why = suggest_breaks(series)
    meta = {
        "recording_id": rid, "census_label": label,
        "keyframe_frame": anchor, "keyframe_time_sec": round(anchor / fps, 3),
        "clip_start_frame": f0, "clip_end_frame": f1,
        "clip_start_sec": round(f0 / fps, 3), "clip_end_sec": round(f1 / fps, 3),
        "window_clamped_by_recording_bounds": bool(clamped),
        "fps": fps, "source_cameras": list(VIEWS),
        "range_frames": range_frames, "range_every": range_every,
        "range_units": "uint16 millimetres, 0 = NOT MEASURED",
        "classes": {"0": "background", "1": "owner_arm", "2": "other_arm"},
        "suggested_breaks": breaks, "break_reasons": why,
        "breaks_are_hints": ("blur and skin-region count only; they miss a "
                             "slow occlusion and fire on reflections"),
        "series": series,
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def _one(job):
    """Worker entry. Rebuilds the rig per process -- cv2 remap tables and
    calibration objects do not survive a fork cleanly on every platform, and
    a corrupt map would produce a plausible-looking wrong render."""
    (rid, views, dbdir, anchor, label, out_root, half, every) = job
    try:
        from src.rig.seg_dataset import find_calibration
        from src.rig.geometry import VirtualWideCamera
        from src.rig.depth import rectify_maps
        rig = find_calibration(dbdir)
        if rig is None:
            return {"recording_id": rid, "keyframe_frame": anchor,
                    "error": "no calibration"}
        vcam = VirtualWideCamera.from_rig(rig)
        rc = {m.name: rectify_maps(rig, m) for m in rig.modules}
        d = os.path.join(out_root, f"sample_{rid}_f{anchor:06d}")
        m = build_sample(rig, vcam, views, anchor, d, rid, label,
                         half_sec=half, range_every=every, rect_cache=rc)
        return m or {"recording_id": rid, "keyframe_frame": anchor,
                     "error": "render failed"}
    except Exception as e:                       # one bad sample, not the run
        return {"recording_id": rid, "keyframe_frame": anchor,
                "error": f"{type(e).__name__}: {e}"}


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", required=True,
                    help="labelled census CSV: recording, frame, label")
    ap.add_argument("--out", required=True, help="the SAN is /workspace")
    ap.add_argument("--half_sec", type=float, default=HALF_WINDOW_SEC)
    ap.add_argument("--range_every", type=int, default=RANGE_EVERY)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 picks cpu_count-4")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from src.rig.class2_census import find_recordings, _check_space
    free = _check_space(a.out)
    by_id = {rid: (views, d) for rid, views, d in find_recordings(a.root)}
    want = list(csv.DictReader(open(a.frames, encoding="utf-8-sig")))
    if a.limit:
        want = want[:a.limit]

    # Calibration is checked ONCE per recording, here, rather than inside a
    # worker per anchor. On this corpus 104 of the 180 census instants sit in
    # databags whose calibration file says `uncalibrated`, and letting each
    # spawn a process to rediscover that would cost 104 process starts and
    # print a wall of identical errors over the real ones.
    from src.rig.seg_dataset import find_calibration
    cal, jobs, skipped = {}, [], {}
    for w in want:
        rid = w["recording"]
        if rid not in by_id:
            skipped[rid] = "not under --root"
            continue
        views, d = by_id[rid]
        if rid not in cal:
            cal[rid] = find_calibration(d) is not None
        if not cal[rid]:
            skipped[rid] = "no usable calibration"
            continue
        jobs.append((rid, views, d, int(w["frame"]), w.get("label", ""),
                     a.out, a.half_sec, a.range_every))
    if skipped:
        n = sum(1 for w in want if w["recording"] in skipped)
        print(f"  skipping {n} anchors in {len(skipped)} recordings:")
        for rid, why in sorted(skipped.items()):
            print(f"    {rid}  {why}")
        print()
    if not jobs:
        raise SystemExit("no requested recording has a usable calibration")

    nw = a.workers or max(1, (os.cpu_count() or 8) - 4)
    span = int(round(2 * a.half_sec * FPS)) + 1
    print(f"{len(jobs)} anchors x {span} frames "
          f"(+/-{a.half_sec:g}s), range every {a.range_every}\n"
          f"  {nw} workers -> {a.out}  ({free:.0f} GB free)\n")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "README.txt"), "w").write(README)

    t0 = time.time()
    metas, errs = [], []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(nw) as pool:
        for i, m in enumerate(pool.imap_unordered(_one, jobs), 1):
            if m.get("error"):
                errs.append(m)
            else:
                metas.append(m)
            if i == 1 or i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                eta = el / i * (len(jobs) - i)
                print(f"  [{i}/{len(jobs)}] {len(errs)} failed, "
                      f"{el/60:.1f} min elapsed, {eta/60:.1f} min left")

    rows = []
    for m in metas:
        for f in m["range_frames"]:
            rows.append({"recording": m["recording_id"], "frame": f,
                         "anchor": m["keyframe_frame"],
                         "is_anchor": int(f == m["keyframe_frame"]),
                         "census_label": m["census_label"],
                         "sample_dir": f"sample_{m['recording_id']}"
                                       f"_f{m['keyframe_frame']:06d}"})
    man = os.path.join(a.out, "frames.csv")
    if rows:
        with open(man, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)

    tot = sum(os.path.getsize(os.path.join(dp, fn))
              for dp, _, fns in os.walk(a.out) for fn in fns)
    nb = sum(len(m["suggested_breaks"]) for m in metas)
    ncl = sum(1 for m in metas
              if m["window_clamped_by_recording_bounds"])
    print(f"\n  {len(metas)} packages, {len(errs)} failed, "
          f"{tot/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min")
    print(f"  {len(rows)} frames carry range and become training rows")
    print(f"  {nb} suggested break points across all clips "
          f"({nb/max(len(metas),1):.1f} per clip)")
    print(f"  {ncl} clips clamped by the start or end of their recording")
    for e in errs[:6]:
        print(f"    FAILED {e['recording_id']} f{e['keyframe_frame']}: "
              f"{e['error']}")
    print(f"\n  {man if rows else '(no manifest -- nothing rendered)'}")
    print("\n  The breaks are hints. Six seconds of good propagation is worth "
          "more than\n  181 frames of drift, and the window is a maximum "
          "rather than a quota.")


if __name__ == "__main__":
    main()
