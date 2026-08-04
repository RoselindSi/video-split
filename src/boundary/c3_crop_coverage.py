"""C3-lite step 0: does the local crop actually contain the hands, and how much
of it is wasted? Run BEFORE any local encoder, per the review's instruction to
check 20-30 random recordings first -- if crop recall is poor, every number the
local branch produces afterwards is measuring the crop, not the hypothesis.
configs/local_gate_c3.json encodes that as an INVALID (not negative) outcome.

Two questions, and the second one turns out to matter more:

  COVERAGE -- what fraction of frames have the hands inside the crop, and how
  often are they clipped by an edge? Reported per failure direction (left,
  right, top, bottom) because "hands on the left" and "tool extending out of
  frame" need different fixes to the box, and a single scalar recall hides
  which one is happening.

  OCCUPANCY -- what fraction of the crop's AREA is hand. This is the number
  that sizes the box, and it is easy to overlook because a crop can score
  perfect coverage while being mostly empty worktop.

Why occupancy is the load-bearing number here. Qwen's smart_resize only ever
DOWNSCALES to fit max_pixels; it never upscales to fill it. The source frames
are 1280x480, which barely exceeds the 768*28*28 budget, so the global branch
already runs at 0.984x linear -- near native. Cropping therefore buys almost
no extra spatial detail:

    global branch, per eye     360 tokens   853 native px per token
    fixed crop 384x312         154 tokens   778 px per token  (1.10x denser)
    tighter crop 200x150        35 tokens   857 px per token  (no gain at all)
    the 384x312 crop at 2x     594 tokens   202 px per token  (4.2x denser)

A tighter crop makes it WORSE, not better: fewer pixels under a fixed patch
size means fewer patches. The only way a local branch gets more ViT capacity
onto the hand is to crop tight AND upscale the crop to fill the token budget.
Occupancy is what tells you how tight you can go without clipping, and the
suggested box printed at the end is derived from the measured hand boxes
rather than guessed.

Needs mediapipe for the quantitative part. Without it, --contact_sheet still
writes a grid of sampled crops for the eyeball check the review asked for,
which is the part that cannot be automated anyway.

Usage (server, no GPU):
    python -m src.boundary.c3_crop_coverage \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --recordings_from /workspace/tr1/results/hal/slow_latent_c2/events_v2_final.csv \
        --n_recordings 25 --n_frames 12 \
        --contact_sheet /workspace/tr1/results/hal/c3/crop_grid.png \
        --out /workspace/tr1/results/hal/c3/crop_coverage.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

FIXED = (0.20, 0.35, 0.80, 1.00)


def load_wanted(paths):
    want = set()
    for p in paths or []:
        if p.endswith(".jsonl"):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    rid = json.loads(line).get("recording_id")
                    if rid:
                        want.add(rid)
        else:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    if r.get("recording_id"):
                        want.add(r["recording_id"])
    return want


def hand_box(det, frame):
    """Union box of all detected hands, in pixels, or None."""
    res = det.process(frame)
    if not res.multi_hand_landmarks:
        return None
    h, w = frame.shape[:2]
    xs = [p.x * w for lm in res.multi_hand_landmarks for p in lm.landmark]
    ys = [p.y * h for lm in res.multi_hand_landmarks for p in lm.landmark]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--recordings_from", action="append")
    ap.add_argument("--n_recordings", type=int, default=25)
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--eye", choices=["left", "right", "full"], default="left")
    ap.add_argument("--box", type=float, nargs=4, default=list(FIXED),
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="fixed crop as fractions of the EYE frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contact_sheet", help="PNG grid of sampled crops for the "
                                            "eyeball check")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for p in a.data:
        rows.extend(json.load(open(p, encoding="utf-8")))
    want = load_wanted(a.recordings_from)
    if want:
        rows = [r for r in rows if r.get("recording_id") in want]
        print(f"--recordings_from: {len(want)} wanted, {len(rows)} matched")
    rng = np.random.RandomState(a.seed)
    if len(rows) > a.n_recordings:
        rows = [rows[i] for i in rng.choice(len(rows), a.n_recordings, replace=False)]
    print(f"sampling {a.n_frames} frames from each of {len(rows)} recordings, "
          f"eye={a.eye} box={tuple(a.box)}")

    try:
        import mediapipe as mp
        det = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                       min_detection_confidence=0.3)
        print("mediapipe available: coverage and occupancy will be measured")
    except ImportError:
        det = None
        print("mediapipe NOT available: only the contact sheet will be produced. "
              "Coverage/occupancy cannot be measured and the gate's "
              "min_crop_contains_hand_rate stays unverified.")

    from decord import VideoReader
    sheet, per_rec = [], []
    n_det = n_in = n_clip = 0
    clip_dir = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    occ, hand_frac_eye = [], []
    x0f, y0f, x1f, y1f = a.box

    for r in rows:
        try:
            vr = VideoReader(r["video"], num_threads=1)
        except Exception as e:
            print(f"  !! {r.get('recording_id')}: {type(e).__name__}")
            continue
        n = len(vr)
        idx = sorted(rng.choice(np.arange(int(0.05 * n), int(0.95 * n)),
                                min(a.n_frames, max(1, int(0.9 * n))), replace=False).tolist())
        frames = vr.get_batch(idx).asnumpy()
        if a.eye != "full":
            W = frames.shape[2]
            frames = frames[:, :, :W // 2] if a.eye == "left" else frames[:, :, W // 2:]
        h, w = frames.shape[1:3]
        cx0, cy0, cx1, cy1 = int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h)
        rec_det = rec_in = 0
        for f in frames:
            if len(sheet) < 60:
                sheet.append(f[cy0:cy1, cx0:cx1])
            if det is None:
                continue
            b = hand_box(det, f)
            if b is None:
                continue
            n_det += 1
            rec_det += 1
            bx0, by0, bx1, by1 = b
            inside = (bx0 >= cx0 and by0 >= cy0 and bx1 <= cx1 and by1 <= cy1)
            if inside:
                n_in += 1
                rec_in += 1
            else:
                n_clip += 1
                if bx0 < cx0: clip_dir["left"] += 1
                if bx1 > cx1: clip_dir["right"] += 1
                if by0 < cy0: clip_dir["top"] += 1
                if by1 > cy1: clip_dir["bottom"] += 1
            # occupancy: intersection of the hand box with the crop, over crop area
            ix0, iy0 = max(bx0, cx0), max(by0, cy0)
            ix1, iy1 = min(bx1, cx1), min(by1, cy1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            occ.append(inter / max(1.0, (cx1 - cx0) * (cy1 - cy0)))
            hand_frac_eye.append(((bx1 - bx0) * (by1 - by0)) / (w * h))
            per_rec.append((r.get("recording_id"), bx0 / w, by0 / h, bx1 / w, by1 / h))
        if det is not None:
            print(f"  {r.get('recording_id')}: detected {rec_det}/{len(frames)}  "
                  f"fully inside crop {rec_in}/{max(1, rec_det)}")
        del vr

    report = {"box": list(a.box), "eye": a.eye,
              "n_recordings": len(rows), "n_frames_each": a.n_frames}
    if det is not None and n_det:
        inside_rate = n_in / n_det
        occ = np.array(occ)
        print(f"\n=== coverage (of frames WITH a detection, n={n_det}) ===")
        print(f"  hands fully inside the crop: {inside_rate:.3f}")
        print(f"  clipped by an edge:          {n_clip / n_det:.3f}  {clip_dir}")
        print(f"    ^ which edge dominates says how to fix the box; a single "
              f"recall number would hide that")
        print(f"\n=== occupancy ===")
        print(f"  hand area / crop area: median {np.median(occ):.3f}  "
              f"p25 {np.percentile(occ, 25):.3f}  p75 {np.percentile(occ, 75):.3f}")
        print(f"  hand area / EYE area:  median {np.median(hand_frac_eye):.3f}")
        tok_crop = int((a.box[2] - a.box[0]) * 640 / 28) * int((a.box[3] - a.box[1]) * 480 / 28)
        print(f"  the crop is ~{tok_crop} ViT tokens, so ~{tok_crop * np.median(occ):.0f} "
              f"of them land on hand; the global branch gives ~360 tokens per eye, "
              f"~{360 * np.median(hand_frac_eye):.0f} on hand")
        if tok_crop * np.median(occ) < 360 * np.median(hand_frac_eye):
            print("  !! the crop puts FEWER tokens on the hand than the global "
                  "branch does. Cropping alone cannot help; the crop must be "
                  "upscaled to fill the token budget, and tightened only "
                  "together with that.")
        if per_rec:
            arr = np.array([p[1:] for p in per_rec], dtype=float)
            sug = (float(np.percentile(arr[:, 0], 2)), float(np.percentile(arr[:, 1], 2)),
                   float(np.percentile(arr[:, 2], 98)), float(np.percentile(arr[:, 3], 98)))
            print(f"\n  box covering 96% of observed hand extents: "
                  f"({sug[0]:.2f}, {sug[1]:.2f}, {sug[2]:.2f}, {sug[3]:.2f})  "
                  f"vs current ({a.box[0]:.2f}, {a.box[1]:.2f}, {a.box[2]:.2f}, {a.box[3]:.2f})")
            report["suggested_box"] = list(sug)
        report.update({"n_detected": n_det, "inside_rate": inside_rate,
                       "clip_rate": n_clip / n_det, "clip_directions": clip_dir,
                       "occupancy_median": float(np.median(occ)),
                       "hand_frac_eye_median": float(np.median(hand_frac_eye))})
    elif det is not None:
        print("\n  !! ZERO detections across every sampled frame. mediapipe is not "
              "working on this footage; the detector arm is not viable and the "
              "fixed-box arm cannot be validated this way -- fall back to the "
              "contact sheet.")

    if a.contact_sheet and sheet:
        from PIL import Image
        k = min(len(sheet), 60)
        cols = 10
        rowsn = (k + cols - 1) // cols
        th, tw = 96, 120
        grid = Image.new("RGB", (cols * tw, rowsn * th), (20, 20, 20))
        for i, c in enumerate(sheet[:k]):
            grid.paste(Image.fromarray(c).resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        os.makedirs(os.path.dirname(os.path.abspath(a.contact_sheet)) or ".", exist_ok=True)
        grid.save(a.contact_sheet)
        print(f"\nwrote {a.contact_sheet} ({k} crops) -- the eyeball check the "
              f"review asked for: look for hands on the left, objects at the "
              f"worktop edge, two-handed interaction, and tools leaving the crop")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
