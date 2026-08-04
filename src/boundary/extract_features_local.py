"""C3-lite step 1: extract frozen-ViT features from a LOCAL hand-region crop.

The global branch (extract_features_recseg.py) encodes the whole frame, so a
hand occupying ~10% of frame width survives as ~100 px and every fine contact
cue -- fingertip touching, releasing, re-grasping, switching object -- is
averaged into a scene-level vector. C3-lite's premise is that the errors P1
cannot fix (regrasp_reposition, direction_reversal) are exactly the ones that
need those cues. This script produces the local half of that two-branch input,
in the same cache format as the global features so every existing consumer
(state_adapter.build_events, pairwise_verifier.build_matrices, the probes)
works on it unchanged.

THREE DESIGN DECISIONS, each forced by something already established:

1. Crops are taken WITHIN ONE EYE, never across the frame. The source is
   1280x480 side-by-side stereo (two 640x480 cameras; confirmed by
   cross-correlation and by eye). A box in whole-frame coordinates can
   straddle the seam and produce a crop that is part of one camera spliced to
   part of the other -- a picture of nothing that exists. That is exactly what
   pool_patch's `center` block does, and the block probe measured it as the
   worst-performing pool. --eye selects which camera to work in; everything
   downstream operates in that eye's 640x480 coordinate frame.

2. Pooling is `global` (one 1152-d mean per crop), NOT `multi`. The block
   probe compared the five-pool 5760-d representation against the 1152-d
   global pool on two different event sets: +0.054 for global-only on the
   clean-145 (CI excluding 0) but only +0.007 on batch3 (CI straddling 0), so
   the ADVANTAGE did not replicate -- but nothing anywhere suggested the extra
   pools help, and inside a tight hand crop the spatial sub-pools would be
   near-duplicates of each other. Global keeps the local branch cheap and
   avoids re-importing a question that has already been answered as "no
   evidence either way".

3. NO new trained encoder. C2 established what happens to a
   trained-from-scratch sequence encoder at this data scale: train AUROC 0.99
   against OOF 0.77, a +0.21 generalization gap on 110-121 training events.
   The local branch therefore produces FROZEN features to be consumed by the
   existing PCA+logreg pipeline, exactly as the global ones are.

CROP STABILITY over crop resolution: a detector that jitters frame to frame
injects apparent visual change that has nothing to do with the action, which
is precisely the nuisance signal this branch exists to remove. So boxes are
temporally smoothed (--smooth_s) and the per-recording jitter is reported, and
--detector none (a fixed lower-centre box, the ego-video prior, zero jitter by
construction) is a legitimate arm to compare against rather than a fallback to
apologise for.

Usage (server, needs GPU; --detector mediapipe additionally needs mediapipe):
    python -m src.boundary.extract_features_local \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --out /workspace/tr1/data_recseg/feat_local_train.pt \
        --fps 2 --eye left --detector mediapipe --margin 0.30
"""
import argparse
import json
import os

import numpy as np
import torch

from src.boundary.extract_features_recseg import gray, lap_var, encode_frames


def eye_slice(frames, eye):
    """frames: [N,H,W,3]. Returns the chosen camera's half of a packed
    side-by-side stereo frame, or the whole frame if --eye full."""
    if eye == "full":
        return frames
    W = frames.shape[2]
    return frames[:, :, :W // 2] if eye == "left" else frames[:, :, W // 2:]


class HandBoxer:
    """Per-frame hand box in the eye's coordinate frame, as (x0,y0,x1,y1) in
    PIXELS, or None when nothing was found.

    'mediapipe' detects hands and returns the UNION of all hands found, which
    is the box the review recommended: a hand-only crop can miss what the hand
    is interacting with, and an object-only crop cannot show contact state.
    'none' returns a fixed lower-centre box -- in egocentric manipulation
    footage the hands are overwhelmingly there, and it has zero jitter by
    construction, which makes it the honest control for "did the detector
    actually buy anything"."""

    def __init__(self, mode, margin=0.30, fixed=(0.20, 0.35, 0.80, 1.00),
                 hand_model=None):
        self.mode = mode
        self.margin = margin
        self.fixed = fixed
        self.det = None
        if mode == "mediapipe":
            try:
                from src.boundary.hand_detect import HandDetector
            except ImportError:
                raise SystemExit(
                    "--detector mediapipe needs the mediapipe package "
                    "(python -m pip install mediapipe -- note `python -m pip`, "
                    "since on this host bare `pip` resolves to a different "
                    "interpreter). Use --detector none for the fixed "
                    "lower-centre crop instead; that arm is a control worth "
                    "running regardless.")
            self.det = HandDetector(hand_model)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if self.mode == "none":
            x0, y0, x1, y1 = self.fixed
            return (x0 * w, y0 * h, x1 * w, y1 * h)
        b = self.det.union_box(frame)
        if b is None:
            return None
        x0, y0, x1, y1 = b
        mx, my = self.margin * (x1 - x0), self.margin * (y1 - y0)
        return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def smooth_boxes(boxes, times, smooth_s):
    """Fill detection gaps, smooth, and report honest stability numbers.

    Returns (boxes, smoothed_jitter, raw_jitter, stats).

    GAPS ARE INTERPOLATED, not carried forward. Carrying the last valid box
    through a gap freezes the crop while the hand keeps moving, so on a
    recording where the detector misses 45% of frames the crop follows a stale
    box for long stretches; linear interpolation between the surrounding
    detections at least tracks toward where the hand actually reappears.

    RAW JITTER IS MEASURED ON DETECTED BOXES ONLY. Measuring it on the filled
    sequence made a worse detector look more stable: carried-forward runs have
    zero frame-to-frame displacement, which drags the median down, and
    smoothing then spreads the accumulated jump across neighbours and pushes
    it back up. That produced the impossible reading "jitter 14.3 -> 30.1 px"
    -- smoothing appearing to make things worse -- which was an artefact of
    the measurement, not of the smoother."""
    n = len(boxes)
    t = np.asarray(times, dtype=float)
    valid = [i for i, b in enumerate(boxes) if b is not None]

    def _jitter(a, at=None):
        """a: [k,4] boxes ALREADY in the order they occur. `at` are their
        timestamps; when given, each displacement is divided by how many
        frame-intervals it spans, so a jump across a detection gap is not
        counted as if it happened in one frame."""
        if len(a) < 2:
            return 0.0
        c = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], 1)
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)
        if at is not None and len(t) > 1:
            step = float(np.median(np.diff(t)))
            d = d / np.maximum(np.diff(at) / max(step, 1e-6), 1.0)
        return float(np.median(d))

    stats = {"n_frames": n, "n_detected": len(valid),
             "detected_frac": len(valid) / max(1, n)}
    if not valid:
        return [None] * n, 0.0, 0.0, stats

    arr_v = np.array([boxes[i] for i in valid], dtype=float)
    raw_jit = _jitter(arr_v, t[valid])

    gaps = np.diff(valid)
    stats["longest_gap_s"] = float((gaps.max() if len(gaps) else 0)
                                   * np.median(np.diff(t)) if n > 1 else 0.0)

    arr = np.empty((n, 4), dtype=float)
    for k in range(4):
        arr[:, k] = np.interp(t, t[valid], arr_v[:, k])

    if smooth_s > 0:
        out = np.empty_like(arr)
        for i in range(n):
            m = np.abs(t - t[i]) <= smooth_s
            out[i] = arr[m].mean(0)
        arr = out
    return [tuple(b) for b in arr], _jitter(arr), raw_jit, stats


def upscale_crop(crop, mode, max_pixels, cap=3.0):
    """Enlarge a crop so it actually fills the ViT's token budget.

    Without this the local branch is strictly worse than the global one.
    Qwen's smart_resize only DOWNSCALES to fit max_pixels and never upscales
    to fill it, and the source frames (1280x480) barely exceed that budget, so
    the global branch already runs at 0.984x linear. Measured token counts for
    one eye:

        global branch, per eye      360 tokens   853 native px per token
        fixed crop 384x312          154 tokens   778 px per token (1.10x)
        tighter crop 200x150         35 tokens   857 px per token (no gain)
        the 384x312 crop at 2x      594 tokens   202 px per token (4.2x)

    So cropping alone buys nothing, and cropping TIGHTER makes it worse --
    fewer pixels under a fixed patch size is fewer patches. Upscaling adds no
    real detail, but it does put several times more ViT patches on the hand,
    which is the whole point of a local branch.

    'auto' scales to fill max_pixels, capped at `cap` because beyond a few
    times the source resolution the extra patches are interpolating between
    pixels that were never measured."""
    if mode in (None, "none", "1", "1.0"):
        return crop
    h, w = crop.shape[:2]
    k = min(cap, (max_pixels / max(1, h * w)) ** 0.5) if mode == "auto" else float(mode)
    if k <= 1.0:
        return crop
    from PIL import Image
    return np.asarray(Image.fromarray(crop).resize(
        (max(1, int(w * k)), max(1, int(h * k))), Image.BICUBIC))


def crop_frame(frame, box, min_side=64):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(w - min_side, x0)))
    y0 = int(max(0, min(h - min_side, y0)))
    x1 = int(min(w, max(x0 + min_side, x1)))
    y1 = int(min(h, max(y0 + min_side, y1)))
    return frame[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="recseg json(s). REPEATABLE -- passing --data twice keeps "
                         "both files (extract_features_recseg's single --data has "
                         "burned a run before).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recordings_from", action="append",
                    help="CSV with a recording_id column (e.g. a --dump_events "
                         "file), or a manifest .jsonl. Repeatable. Restricts "
                         "extraction to those recordings -- the clean-145 dev set "
                         "is 46 recordings out of several hundred in the recseg "
                         "jsons, so without this the run spends most of its GPU "
                         "time on recordings no evaluation will ever read.")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--eye", choices=["left", "right", "full"], default="left",
                    help="which camera of the packed stereo pair to work inside. "
                         "'full' would let a box straddle the seam and is only "
                         "here for non-stereo sources.")
    ap.add_argument("--detector", choices=["mediapipe", "none"], default="mediapipe")
    ap.add_argument("--hand_model",
                    help="hand_landmarker.task, required by mediapipe >= 1.0")
    ap.add_argument("--margin", type=float, default=0.30,
                    help="context margin around the hand union box, as a fraction "
                         "of its size -- a hand with no surroundings cannot show "
                         "WHAT it is interacting with")
    ap.add_argument("--smooth_s", type=float, default=0.5)
    ap.add_argument("--upscale", default="auto",
                    help="'auto' (fill the token budget, capped at 3x), 'none', "
                         "or a float. Without this a crop gets FEWER ViT patches "
                         "on the hand than the global branch does -- see "
                         "upscale_crop()'s docstring for the measured numbers.")
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--enc_batch", type=int, default=48)
    ap.add_argument("--dec_chunk", type=int, default=200)
    ap.add_argument("--th_black", type=float, default=20.0)
    ap.add_argument("--th_blur", type=float, default=0.0,
                    help="default 0 (off), matching the *_noblur_* global caches "
                         "the local branch has to align with event-for-event")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint_every", type=int, default=20)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    boxer = HandBoxer(a.detector, a.margin, hand_model=a.hand_model)

    from decord import VideoReader
    rows = []
    for p in a.data:
        rows.extend(json.load(open(p)))
    if a.recordings_from:
        want = set()
        for p in a.recordings_from:
            if p.endswith(".jsonl"):
                for line in open(p, encoding="utf-8"):
                    if line.strip():
                        rid = json.loads(line).get("recording_id")
                        if rid:
                            want.add(rid)
            else:
                import csv as _csv
                with open(p, newline="", encoding="utf-8", errors="replace") as f:
                    for r in _csv.DictReader(f):
                        if r.get("recording_id"):
                            want.add(r["recording_id"])
        before = len(rows)
        rows = [r for r in rows if r.get("recording_id") in want]
        missing = want - {r.get("recording_id") for r in rows}
        print(f"--recordings_from: {len(want)} wanted, {len(rows)} matched "
              f"(of {before} in --data)")
        if missing:
            # Loud, because a silently-short extraction produces a cache that
            # looks fine and then drops events at evaluation time, where the
            # loss is much harder to trace back to here.
            print(f"  !! {len(missing)} wanted recordings are NOT in any --data "
                  f"file and will be missing from the cache: "
                  f"{sorted(missing)[:5]}")
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} recordings, eye={a.eye} detector={a.detector} "
          f"margin={a.margin} smooth={a.smooth_s}s")

    cache, stats = [], []
    for ri, r in enumerate(rows):
        vr = VideoReader(r["video"], num_threads=1)
        n, vfps = len(vr), vr.get_avg_fps()
        step = max(1, int(round(vfps / a.fps)))
        cand = list(range(0, n, step))

        kept, times, boxes, n_black, n_blur, n_nodet = [], [], [], 0, 0, 0
        for c0 in range(0, len(cand), a.dec_chunk):
            idx = cand[c0:c0 + a.dec_chunk]
            frames = eye_slice(vr.get_batch(idx).asnumpy(), a.eye)
            for j, f in enumerate(frames):
                g = gray(f)
                if float(g.mean()) < a.th_black:
                    n_black += 1
                    continue
                if a.th_blur > 0 and lap_var(g) < a.th_blur:
                    n_blur += 1
                    continue
                b = boxer(f)
                if b is None:
                    n_nodet += 1
                kept.append(f)
                times.append(idx[j] / vfps)
                boxes.append(b)

        det_rate = 1.0 - (n_nodet / max(1, len(kept)))
        sboxes, jitter, raw_jitter, bstats = smooth_boxes(boxes, times, a.smooth_s)
        crops = [upscale_crop(crop_frame(f, b), a.upscale, a.max_pixels)
                 for f, b in zip(kept, sboxes)] if sboxes[0] else []

        feats = []
        for b0 in range(0, len(crops), a.enc_batch):
            feats.append(encode_frames(crops[b0:b0 + a.enc_batch], proc, model,
                                       a.max_pixels, "global"))
            torch.cuda.empty_cache()
        feats = torch.cat(feats, 0) if feats else torch.zeros(0)
        segs = [(s[0], float(s[1]), float(s[2])) for s in r["solution"]]
        cache.append({"video": r["video"], "recording_id": r.get("recording_id"),
                      "feats": feats, "times": torch.tensor(times[:len(feats)]),
                      "duration": float(r["duration"]), "segments": segs,
                      "detection_rate": det_rate, "box_jitter_px": jitter,
                      "box_jitter_raw_px": raw_jitter, "box_stats": bstats})
        stats.append((det_rate, jitter, raw_jitter))
        # Both sides measured over the SAME crops. An earlier version averaged
        # the raw side over the first 20 frames and the upscaled side over all
        # of them, so "125 -> 432" compared two different populations and the
        # upscale factor it implied was fiction.
        mean_side = float(np.mean([min(c.shape[0], c.shape[1]) for c in crops])) if crops else 0.0
        raw_side = float(np.mean([min(crop_frame(f, b).shape[:2])
                                  for f, b in zip(kept, sboxes)])) if crops else 0.0
        print(f"[{ri+1}/{len(rows)}] {r.get('recording_id')} kept {len(kept)} "
              f"det {det_rate:.2f} jitter {raw_jitter:.1f}->{jitter:.1f}px "
              f"gap_max {bstats.get('longest_gap_s', 0):.1f}s "
              f"crop_side {raw_side:.0f}->{mean_side:.0f} (x{mean_side/max(1,raw_side):.1f}) "
              f"feats {tuple(feats.shape)}", flush=True)

        del vr, kept, crops
        if (ri + 1) % a.checkpoint_every == 0 or ri == len(rows) - 1:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save(cache, a.out)
            print(f"  [checkpoint] {len(cache)}/{len(rows)} -> {a.out}", flush=True)

    if stats:
        d = np.array([s[0] for s in stats]); j = np.array([s[1] for s in stats])
        rj = np.array([s[2] for s in stats])
        print(f"\ndetection rate: median {np.median(d):.3f} "
              f"min {d.min():.3f} ({int((d < 0.5).sum())} recordings below 0.5)")
        print(f"box jitter px:  raw median {np.median(rj):.1f} (max {rj.max():.1f})  "
              f"-> smoothed median {np.median(j):.1f} (max {j.max():.1f})")
        print("  raw is the detector's own stability; smoothed is what reaches the "
              "crops. A large gap means the smoother is carrying the branch, and "
              "the crop follows the smoother more than it follows the hand.")
        print("  a low detection rate or a high jitter means the crop branch is "
              "feeding the model an unstable window, and any AUROC it produces "
              "is measuring the detector as much as the hypothesis -- compare "
              "against --detector none before reading anything into it.")
    print(f"\nwrote {len(cache)} recordings -> {a.out}")


if __name__ == "__main__":
    main()
