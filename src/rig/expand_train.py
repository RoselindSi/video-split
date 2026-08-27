"""More pseudo-labelled training frames, from the recordings already in train.

The gold set answered the question this exists to act on. On seven blind
hand-drawn frames the head scored 0.764 against truth where its own training
labels scored 0.717 -- a paired difference of +0.047, CI [+0.017, +0.098],
winning on 7 of 7 frames. It is not reproducing GrabCut; it is denoising it.
That is the condition under which more noisy labels are worth generating, and
it was worth measuring rather than assuming, because the opposite result would
have said to fix the labeller instead.

So: sample more frames from the SAME recordings that are already in train,
label them the same way, and add them.

THE HELD-OUT RECORDINGS ARE REFUSED BY NAME. Every recording in the eval
manifest is excluded here, and the exclusion is checked rather than trusted to
the caller getting a path right. Four recordings are all that stands between
this project and a number about nothing, and the gold frames are inside them.

SAMPLING IS BLIND AND STRATIFIED OVER TIME. One random frame from each of N
equal blocks of the recording, seeded by the recording's name. Not by a model
score -- a training set mined by what the current head finds uncertain teaches
it its own blind spots and nothing else -- and not evenly spaced, because the
work is a repeating assembly cycle a few seconds long and an even grid can
sample the same phase of it every time.

WHAT THIS DOES NOT ADD is any `other_near` frame, because none exists in the
calibrated half of the corpus. This makes the owner mask better. It cannot
make ownership testable.
"""
from __future__ import annotations

import csv
import os

import numpy as np

# Per recording. The GrabCut call is the cost -- about eleven seconds a frame
# at full resolution -- so this is the knob that decides whether a run is
# twenty minutes or three hours.
PER_RECORDING = 40


def sample_frames(n_frames, k, rec_id):
    """Stratified random over time, seeded by the recording's own name."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(rec_id.encode()) & 0xFFFFFFFF)
    edges = np.linspace(0, n_frames, k + 1).astype(int)
    return [int(rng.integers(a, b)) for a, b in zip(edges[:-1], edges[1:])
            if b > a]


def _one(job):
    """Render and label one frame in a worker. -> manifest row or None."""
    (rid, views, dbdir, frame, out_dir) = job
    try:
        import cv2
        from src.rig.seg_dataset import find_calibration, export_instant
        from src.rig.geometry import VirtualWideCamera
        from src.rig.depth import rectify_maps
        from src.rig.auto_label import Segmenter, label_frame, overlay

        rig = find_calibration(dbdir)
        if rig is None:
            return None
        vcam = VirtualWideCamera.from_rig(rig)
        rc = {m.name: rectify_maps(rig, m) for m in rig.modules}
        p_rgb, p_rng = export_instant(rig, vcam, views, frame, out_dir, rid,
                                      rect_cache=rc, map_cache={})
        if p_rgb is None:
            return None
        rgb = cv2.imread(p_rgb)
        # census_label is unknown for a frame the census never saw, so the
        # suppression that rule provides is unavailable. It does not matter
        # here: class 2 is off, and every non-owner region is background.
        mask, picks = label_frame(Segmenter("grabcut"), rgb, "")
        stem = f"{rid}_f{frame:06d}.png"
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "masks", stem), mask)
        cv2.imwrite(os.path.join(out_dir, "overlays",
                                 stem.replace(".png", ".jpg")),
                    overlay(rgb, mask), [cv2.IMWRITE_JPEG_QUALITY, 82])
        return {"recording": rid, "frame": frame, "split": "train",
                "has_depth": int(bool(p_rng)), "census_label": "",
                "label_source": "auto",
                "n_owner": sum(1 for c, _ in picks if c == 1),
                "n_other": sum(1 for c, _ in picks if c == 2),
                "owner_px": int((mask == 1).sum()),
                "other_px": int((mask == 2).sum())}
    except Exception as e:
        return {"recording": rid, "frame": frame, "error":
                f"{type(e).__name__}: {e}"}


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--train", required=True,
                    help="the existing seg_auto/train directory; new frames "
                         "are added into it")
    ap.add_argument("--eval", required=True,
                    help="the seg_auto/eval directory. Its recordings are "
                         "REFUSED -- the gold frames live in them.")
    ap.add_argument("--per_recording", type=int, default=PER_RECORDING)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import cv2
    from src.rig.class2_census import find_recordings, _check_space

    def recs_of(d):
        p = os.path.join(d, "manifest.csv")
        return {r["recording"] for r in
                csv.DictReader(open(p, encoding="utf-8-sig"))}

    train_recs, eval_recs = recs_of(a.train), recs_of(a.eval)
    leak = train_recs & eval_recs
    if leak:
        raise SystemExit(f"train and eval already share {sorted(leak)}. "
                         f"Refusing to make it worse.")
    _check_space(a.train)
    have = {(r["recording"], int(r["frame"])) for r in
            csv.DictReader(open(os.path.join(a.train, "manifest.csv"),
                                encoding="utf-8-sig"))}
    by_id = {rid: (v, d) for rid, v, d in find_recordings(a.root)}

    jobs = []
    for rid in sorted(train_recs):
        if rid in eval_recs or rid not in by_id:
            continue
        views, d = by_id[rid]
        cap = cv2.VideoCapture(views["cam12"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for f in sample_frames(n, a.per_recording, rid):
            if (rid, f) not in have:
                jobs.append((rid, views, d, f, a.train))
    if not jobs:
        raise SystemExit("nothing new to render")

    print(f"{len(jobs)} new frames over {len(train_recs)} training "
          f"recordings\n  {len(eval_recs)} held-out recordings refused: "
          f"{sorted(eval_recs)}\n  {a.workers} workers -> {a.train}\n")

    import multiprocessing as mp
    t0, rows, errs = time.time(), [], []
    try:
        pool = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8)
    except (BlockingIOError, OSError) as e:
        raise SystemExit(f"could not start {a.workers} workers ({e}). "
                         f"Rerun with a smaller --workers.")
    with pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs), 1):
            if r is None:
                continue
            (errs if r.get("error") else rows).append(r)
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {el/60:.1f} min, "
                      f"{el/i*(len(jobs)-i)/60:.1f} left, {len(errs)} failed")

    man = os.path.join(a.train, "manifest.csv")
    old = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    fields = list(old[0].keys())
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])
    print(f"\n  {len(old)} + {len(rows)} = {len(old)+len(rows)} training "
          f"frames, {len(errs)} failed")
    n_owner = sum(1 for r in rows if r["n_owner"] > 0)
    print(f"  {n_owner}/{len(rows)} new frames have an owner region "
          f"({len(rows)-n_owner} are empty -- no arm in shot, or the "
          f"detector missed it)")
    for e in errs[:5]:
        print(f"    FAILED {e['recording']} f{e['frame']}: {e['error']}")
    print("\n  The eval split is untouched, and so are the gold frames "
          "inside it. Retrain\n  against the same eval to see whether more "
          "noisy labels bought anything.")


if __name__ == "__main__":
    main()
