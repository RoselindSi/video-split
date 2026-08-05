"""Event-centred 10 fps MediaPipe hand trajectories, raw and per frame.

Decodes ONLY the candidate windows, not whole recordings: 412 events x 41
frames is about 17k frames against the millions a full 10 fps re-extraction of
173 recordings would touch. No Qwen features are computed -- this is a probe
for whether direction reversal, periodicity, stop-start and visibility exist
as observables at all, before anything expensive is built on them.

ONE DETECTOR PER EVENT, deliberately. MediaPipe's VIDEO running mode carries
tracking state between calls and requires strictly increasing timestamps, so a
single detector walked across 412 disjoint windows would let one event's hand
track continue into the next and manufacture continuity that was never
observed -- precisely the artefact the trajectory features are meant to
measure. The cost is a model load per event; the alternative is silently
fabricated tracks.

RAW OUTPUT IS PRESERVED. Landmarks, world landmarks, boxes and validity are
stored per frame exactly as detected. Interpolation (at most 2 frames, 0.2 s)
and smoothing are applied to DERIVED copies, with the interpolated fraction
recorded, so a later analysis can always tell an observation from a fill. The
local-crop work showed why: a 64-second interpolated stretch scored as the
most stable trajectory in that set, because a straight line has no jitter.

Boxes are stored BOTH raw and margin-expanded. Detection runs on the full
left-eye frame, not on a crop, so the margin never affects what was detected;
it exists only for a later crop step. Landmark coordinates are mediapipe's
normalised ones, which are scale-invariant, so --upscale changes detection
sensitivity without changing the stored geometry.

Windows are clipped only by the video's own start and end. They are NOT
clipped at neighbouring annotation boundaries: doing so would let the label
decide how much context a candidate gets.

Usage:
    python -m src.boundary.extract_hand_trajectory \
        --events /workspace/tr1/results/hal/c3/local_events.csv \
        --events /workspace/tr1/results/hal/c3/local_events_batch3.csv \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg_part2/recseg_train.json \
        --data /workspace/tr1/data_recseg_part2/recseg_val.json \
        --hand_model /workspace/tr1/ckpts/hand_landmarker.task \
        --fps 10 --eye left --window_before 2.0 --window_after 2.0 \
        --max_hands 2 --upscale auto --margin 0.30 --max_interp_gap_frames 2 \
        --out /workspace/tr1/data_recseg/hand_trajectory_dev_10fps.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch

from src.boundary.hand_trajectory import (
    box_from_landmarks, expand_box, edge_touch, associate,
)


def event_time(eid):
    m = re.search(r"_t(\d+(?:\.\d+)?)$", eid)
    return float(m.group(1)) if m else None


def eye_slice(frames, eye):
    """Half of a packed side-by-side stereo frame. Accepts a single frame
    [H,W,3] or a batch [N,H,W,3] and always slices the WIDTH axis.

    The first version took shape[1] as the width and sliced axis 1. Called with
    a BATCH that is the height axis, so it returned the TOP HALF of the packed
    frame -- 1280x240, the upper portion of both cameras -- and since hands
    enter an egocentric frame from below, they were cropped away entirely.
    That is what produced a 0.000 detection rate, and the size was printed on
    the first line of every run before anyone read it."""
    if eye == "full":
        return frames
    ax = 2 if frames.ndim == 4 else 1
    W = frames.shape[ax]
    sl = slice(0, W // 2) if eye == "left" else slice(W // 2, W)
    out = frames[:, :, sl] if frames.ndim == 4 else frames[:, sl]
    h, w = (out.shape[1], out.shape[2]) if out.ndim == 4 else out.shape[:2]
    if not 0.8 <= w / max(h, 1) <= 2.2:
        print(f"  !! eye slice is {w}x{h}, aspect {w / max(h, 1):.2f} -- a single "
              f"camera should be near 1.33. A wrong slice axis produces exactly "
              f"this and silently removes the hands.")
    return out


def upscale_frame(f, mode, target_long=960, cap=2.0):
    """Enlarge before detection only. MediaPipe's hand landmarker was trained
    on larger inputs than a 640x480 eye, and its coordinates come back
    normalised, so this changes detection sensitivity without touching the
    stored geometry."""
    if mode in (None, "none", "1", "1.0"):
        return f
    h, w = f.shape[:2]
    k = min(cap, target_long / max(h, w)) if mode == "auto" else float(mode)
    if k <= 1.0:
        return f
    from PIL import Image
    return np.ascontiguousarray(
        np.asarray(Image.fromarray(f).resize((int(w * k), int(h * k)), Image.BICUBIC)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", action="append", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--hand_model", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--eye", choices=["left", "right", "full"], default="left")
    ap.add_argument("--window_before", type=float, default=2.0)
    ap.add_argument("--window_after", type=float, default=2.0)
    ap.add_argument("--max_hands", type=int, default=2)
    ap.add_argument("--min_confidence", type=float, default=0.3)
    ap.add_argument("--upscale", default="auto")
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--max_interp_gap_frames", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    video_path = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video_path[r["recording_id"]] = r["video"]

    seen, events = set(), []
    for p in a.events:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                eid = r["event_id"]
                if eid in seen:
                    continue
                seen.add(eid)
                t = event_time(eid)
                if t is None or r["recording_id"] not in video_path:
                    continue
                events.append({"event_id": eid, "recording_id": r["recording_id"],
                               "t": t, "subtype": r.get("subtype", ""),
                               "y": r.get("y", "")})
    if a.limit:
        events = events[:a.limit]
    print(f"{len(events)} events from {len(a.events)} file(s), "
          f"{len({e['recording_id'] for e in events})} recordings")
    print(f"  by subtype: {dict(Counter(e['subtype'] or '(none)' for e in events))}")
    n_frames = int(round((a.window_before + a.window_after) * a.fps)) + 1
    print(f"  window [-{a.window_before}, +{a.window_after}]s at {a.fps} fps "
          f"= {n_frames} frames/event, ~{len(events) * n_frames} frames total")

    from src.boundary.hand_detect import HandDetector
    from decord import VideoReader

    cache, stats = {}, []
    n_det_exc, first_exc = 0, None
    t0 = time.time()
    for ei, e in enumerate(events):
        try:
            vr = VideoReader(video_path[e["recording_id"]], num_threads=1)
        except Exception as ex:
            print(f"  !! {e['event_id']}: {type(ex).__name__}")
            continue
        vfps, n = vr.get_avg_fps(), len(vr)
        want = e["t"] + np.linspace(-a.window_before, a.window_after, n_frames)
        idx = np.clip(np.round(want * vfps).astype(int), 0, n - 1)
        frames = eye_slice(vr.get_batch(idx.tolist()).asnumpy(), a.eye)
        H, W = frames.shape[1:3]

        # one detector per event -- see module docstring
        det = HandDetector(a.hand_model, max_hands=a.max_hands,
                           min_confidence=a.min_confidence, running_mode="video")
        prev, rec = {}, []
        for fi in range(n_frames):
            f = np.ascontiguousarray(frames[fi])
            up = upscale_frame(f, a.upscale)
            try:
                r = det.detect_full(up, timestamp_ms=int(fi * 1000 / a.fps))
                ok = True
            except Exception as ex:
                # Counted and SURFACED. The first version swallowed this into a
                # per-frame flag that nothing printed, so a detector raising on
                # every call was indistinguishable from a detector finding no
                # hands -- and a 0.000 coverage run gave no way to tell which.
                r, ok = {"landmarks": [], "world": [], "handedness": [], "score": []}, False
                n_det_exc += 1
                if first_exc is None:
                    first_exc = f"{type(ex).__name__}: {ex}"[:200]
            dets = []
            for hi, lm in enumerate(r["landmarks"]):
                pts = [(p.x, p.y, getattr(p, "z", 0.0)) for p in lm]
                box = box_from_landmarks(pts, W, H)
                dets.append({
                    "box": box, "landmarks": pts,
                    "world": ([(p.x, p.y, p.z) for p in r["world"][hi]]
                              if hi < len(r["world"]) else None),
                    "handedness": r["handedness"][hi] if hi < len(r["handedness"]) else None,
                    "score": float(r["score"][hi]) if hi < len(r["score"]) else float("nan"),
                    "expanded_box": expand_box(box, a.margin),
                    "edge_touch": edge_touch(box, W, H),
                })
            tracks = associate(prev, dets)
            prev = tracks
            rec.append({"rel_t": float(want[fi] - e["t"]),
                        "abs_t": float(idx[fi] / vfps),
                        "decode_success": True, "detector_success": ok,
                        "n_hands": len(dets), "tracks": tracks})
        det.close()
        del vr

        cov = float(np.mean([f["n_hands"] > 0 for f in rec]))
        cache[e["event_id"]] = {
            "event_id": e["event_id"], "recording_id": e["recording_id"],
            "candidate_time": e["t"], "subtype": e["subtype"], "y": e["y"],
            "frame_w": int(W), "frame_h": int(H), "fps": a.fps,
            "n_frames": n_frames, "frames": rec,
            "config": {"eye": a.eye, "window": [a.window_before, a.window_after],
                       "max_hands": a.max_hands, "upscale": a.upscale,
                       "margin": a.margin,
                       "max_interp_gap_frames": a.max_interp_gap_frames,
                       "min_confidence": a.min_confidence, "running_mode": "video"},
        }
        stats.append(cov)
        if (ei + 1) % 10 == 0 or ei == len(events) - 1:
            el = time.time() - t0
            print(f"[{ei+1}/{len(events)}] {e['event_id'][:44]:<44} "
                  f"detected {cov:.2f}  {el/(ei+1):.1f}s/event  "
                  f"eta {(len(events)-ei-1)*el/(ei+1)/60:.0f}min", flush=True)
        if (ei + 1) % a.checkpoint_every == 0 or ei == len(events) - 1:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save(cache, a.out)
            print(f"  [checkpoint] {len(cache)}/{len(events)} -> {a.out}", flush=True)

    if n_det_exc:
        print(f"\n  !! the detector RAISED on {n_det_exc} frame(s); first was: "
              f"{first_exc}\n     zero coverage caused by exceptions is a broken "
              f"call, not an absent hand -- do not read it as a detection rate")
    if stats:
        s = np.array(stats)
        print(f"\nper-event detection coverage: median {np.median(s):.3f}  "
              f"min {s.min():.3f}  below 0.5: {int((s < 0.5).sum())}/{len(s)}")
        print("  coverage here is over the CANDIDATE WINDOW only, so it is not "
              "comparable to the whole-recording rates the 2 fps local extraction "
              "reported -- those were dragged down by idle stretches this never "
              "decodes.")
    print(f"wrote {len(cache)} events -> {a.out}")


if __name__ == "__main__":
    main()
