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


def load_sample_points(paths):
    """{recording_id: [times]} from an events CSV or manifest jsonl.

    Coverage is checked AT THE CANDIDATE MOMENTS, not at random times: the
    crop only has to work where a boundary decision is actually made, and a
    box that covers idle stretches but fails during the manipulation itself
    would look fine under uniform sampling. `t` is read from a column if
    present, otherwise parsed from the event_id's trailing _t<float>."""
    import re
    pts = {}
    for p in paths or []:
        if p.endswith(".jsonl"):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    m = json.loads(line)
                    if m.get("recording_id"):
                        pts.setdefault(m["recording_id"], []).append(
                            float(m["t"]) if m.get("t") is not None else None)
        else:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    rid = r.get("recording_id")
                    if not rid:
                        continue
                    t = r.get("t")
                    if t in (None, ""):
                        mm = re.search(r"_t(\d+(?:\.\d+)?)$", r.get("event_id", ""))
                        t = mm.group(1) if mm else None
                    pts.setdefault(rid, []).append(float(t) if t else None)
    return pts


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
    """Union box of all detected hands, in pixels, or None. Version handling
    lives in hand_detect.HandDetector -- mediapipe 1.0 dropped mp.solutions."""
    return det.union_box(frame)


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
    ap.add_argument("--hand_model",
                    help="path to hand_landmarker.task, required by mediapipe >= 1.0 "
                         "(the Tasks API takes an external model bundle; 0.10.x "
                         "packaged one inside the wheel). See hand_detect.py.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet_per_rec", type=int, default=3,
                    help="panels per recording on the contact sheet. Sampling is "
                         "per-recording, NOT first-N-frames: an earlier version "
                         "filled a 60-panel sheet in recording order and so showed "
                         "only the first 5 of 25 recordings while printing '25 "
                         "recordings', which made the sheet unable to answer the "
                         "question it existed for.")
    ap.add_argument("--contact_sheet", help="PNG grid. Each panel shows the FULL "
                                            "eye frame with the crop rectangle "
                                            "drawn NEXT TO the crop itself -- the "
                                            "crop alone cannot show whether a "
                                            "second hand or the key object was "
                                            "cut off, which is the whole point.")
    ap.add_argument("--rating_csv", help="write a pre-filled sheet for the manual "
                                         "full / partial_but_usable / failed rating")
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
    ev_pts = load_sample_points(a.recordings_from)
    n_with_t = sum(1 for r in rows if any(t is not None for t in ev_pts.get(r.get("recording_id"), [])))
    print(f"sampling {a.n_frames} frames from each of {len(rows)} recordings, "
          f"eye={a.eye} box={tuple(a.box)}")
    print(f"  candidate times available for {n_with_t}/{len(rows)} recordings "
          f"-> those are sampled AT the candidates (t-1s, t, t+1s); the rest "
          f"fall back to uniform random times")

    try:
        from src.boundary.hand_detect import HandDetector
        det = HandDetector(a.hand_model)
        print(f"mediapipe {det.version} ({det.api} API): coverage and occupancy "
              f"will be measured")
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
        n, vfps = len(vr), vr.get_avg_fps()
        ts = [t for t in ev_pts.get(r.get("recording_id"), []) if t is not None]
        if ts:
            rng.shuffle(ts)
            idx = []
            for t in ts[:max(1, a.n_frames // 3)]:
                for dt in (-1.0, 0.0, 1.0):
                    j = int(round((t + dt) * vfps))
                    if 0 <= j < n:
                        idx.append(j)
            idx = sorted(set(idx))[:a.n_frames]
        else:
            idx = []
        if len(idx) < 2:
            idx = sorted(rng.choice(np.arange(int(0.05 * n), int(0.95 * n)),
                                    min(a.n_frames, max(1, int(0.9 * n))),
                                    replace=False).tolist())
        frames = vr.get_batch(idx).asnumpy()
        ftimes = [j / vfps for j in idx]
        if a.eye != "full":
            W = frames.shape[2]
            frames = frames[:, :, :W // 2] if a.eye == "left" else frames[:, :, W // 2:]
        h, w = frames.shape[1:3]
        cx0, cy0, cx1, cy1 = int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h)
        rec_det = rec_in = 0
        # PER-RECORDING sampling for the sheet. The previous version appended
        # in frame order until a global cap of 60, which with 12 frames each
        # meant the sheet showed the first 5 recordings out of 25 while the
        # log said "25 recordings" -- it could not answer the coverage
        # question it existed for.
        pick = set(np.linspace(0, len(frames) - 1,
                               min(a.sheet_per_rec, len(frames))).round().astype(int).tolist())
        for fi, f in enumerate(frames):
            if fi in pick:
                sheet.append({"eye": f, "crop_box": (cx0, cy0, cx1, cy1),
                              "recording_id": r.get("recording_id"),
                              "t": ftimes[fi] if fi < len(ftimes) else None})
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
        from PIL import Image, ImageDraw
        # Each panel is FULL EYE (with the crop rectangle drawn) beside the
        # CROP. Showing the crop alone makes it impossible to see whether a
        # second hand or the key object was cut off -- which is exactly the
        # judgement the sheet is for.
        # A gutter between panels, and the crop rendered at its true aspect
        # ratio. Without the gutter adjacent panels butt together and are hard
        # to read one at a time, which is the whole task; squashing the crop to
        # a square distorts exactly the thing being judged.
        ew, eh, gut = 176, 132, 10
        b = sheet[0]["crop_box"]
        cw = max(40, int(eh * (b[2] - b[0]) / max(1, b[3] - b[1])))
        pw, ph = ew + cw + 6 + gut, eh + 16 + gut
        cols = 5
        rowsn = (len(sheet) + cols - 1) // cols
        grid = Image.new("RGB", (cols * pw, rowsn * ph), (18, 18, 18))
        dr = ImageDraw.Draw(grid)
        for i, sm in enumerate(sheet):
            ox, oy = (i % cols) * pw, (i // cols) * ph
            f = sm["eye"]
            h0, w0 = f.shape[:2]
            cx0, cy0, cx1, cy1 = sm["crop_box"]
            eim = Image.fromarray(f).resize((ew, eh))
            grid.paste(eim, (ox, oy + 14))
            sx, sy = ew / w0, eh / h0
            dr.rectangle([ox + cx0 * sx, oy + 14 + cy0 * sy,
                          ox + cx1 * sx, oy + 14 + cy1 * sy], outline=(255, 90, 90), width=2)
            grid.paste(Image.fromarray(f[cy0:cy1, cx0:cx1]).resize((cw, eh)),
                       (ox + ew + 6, oy + 14))
            tt = f"{i}  {sm['recording_id'] or '?'}" + (
                f"  t={sm['t']:.1f}" if sm.get("t") is not None else "")
            dr.text((ox + 3, oy + 3), tt[:44], fill=(210, 210, 210))
        os.makedirs(os.path.dirname(os.path.abspath(a.contact_sheet)) or ".", exist_ok=True)
        grid.save(a.contact_sheet)
        nrec = len({sm["recording_id"] for sm in sheet})
        print(f"\nwrote {a.contact_sheet}: {len(sheet)} panels spanning {nrec} "
              f"recordings (left = full eye + crop rectangle, right = the crop)")
        print("  look for: hands entering from the left, objects at the worktop "
              "edge, two-handed interaction split by the box, tools leaving the "
              "crop -- all of which are invisible if you only see the crop")
        if a.rating_csv:
            with open(a.rating_csv, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["panel", "recording_id", "t", "rating", "note"])
                for i, sm in enumerate(sheet):
                    wcsv.writerow([i, sm["recording_id"],
                                   f"{sm['t']:.2f}" if sm.get("t") is not None else "", "", ""])
            print(f"wrote {a.rating_csv} -- rate each panel full / "
                  f"partial_but_usable / failed. Suggested gate: full >= 0.75, "
                  f"full+partial >= 0.90, failed <= 0.10, AND no single task "
                  f"family concentrating the failures (an overall 0.90 with a "
                  f"long-tool family at 0.40 is not a pass).")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
