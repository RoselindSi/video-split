"""36-event hand-object pilot: can the relation be observed at all?

Runs on the audit sample ONLY. The audit found 21 of 36 REVIEW events needing
object-relative evidence, which is a reason to test feasibility, not a reason
to spend a detector pass on all 412 -- a stratified sample of 36 does not carry
the band's natural proportions, and nothing yet shows a detector can recover
the relation on even one of those 21.

The output is built to be LOOKED AT, not scored. There is no classifier here
and no AUROC. A contact sheet per event carries the hand box, the active-object
box, the contact state and the object track id, and the report groups events by
the audit's own answer, because the question is not what fraction of objects a
detector finds. It is whether the 21 events where a human needed object
evidence are the ones where the machine can supply it. A backend with excellent
average recall that fails on exactly those 21 has answered the pilot in the
negative, and an aggregate number would hide that.

BACKENDS ARE PLUGGABLE AND NONE IS BUILT IN. `--backend hoi100doh` loads the
hand_object_detector released with "Understanding Human Hands in Contact at
Internet Scale", which emits a contact class per hand and a box for the object
in contact -- the layer-2 quantity, directly. It needs the repo and checkpoint
on disk; see --hoi_repo. `--backend synthetic` fabricates observations from
scripted geometry and exists to verify this pipeline end to end (tracking,
schema, overlays, report) without a detector, so a failure in the real run can
be attributed to the detector rather than to the plumbing.

OBJECT IDENTITY IS ASSIGNED HERE, NOT READ FROM A LABEL. IoU across frames,
broken ties by a coarse colour signature. Two bowls share a label and are not
the same instance, and layer 2's discriminating quantity is whether the SAME
instance returns after a release. Track ids therefore survive short gaps
(--track_max_gap) but never bridge a long one.

Usage:
    python -m src.boundary.extract_hand_object_pilot \
        --manifest /workspace/tr1/results/hal/c3/observable_audit/audit_manifest.jsonl \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg_part2/recseg_train.json \
        --data /workspace/tr1/data_recseg_part2/recseg_val.json \
        --backend hoi100doh --hoi_repo /workspace/tr1/third_party/hand_object_detector \
        --hoi_ckpt /workspace/tr1/ckpts/faster_rcnn_1_8_132028.pth \
        --audit_sheet /workspace/tr1/results/hal/c3/observable_audit/audit_sheet.csv \
        --out_dir /workspace/tr1/results/hal/c3/observable_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from src.boundary.extract_hand_trajectory import eye_slice, upscale_frame
from src.boundary.hand_trajectory import associate as associate_hands, edge_touch
from src.boundary.hand_object_state import contact_state as geom_contact, _iou
from src.boundary.hand_object_observations import (
    HandObservation, RawInteraction, CONTACT, NEAR, FREE, UNKNOWN,
    DOH_CONTACT, DOH_OBJECT_KIND, colour_signature, signature_similarity,
    frame_summary, series_for_hand, dominant_hand,
)


# ----------------------------------------------------------------- backends
def backend_synthetic(seed=0):
    """Scripted geometry: a hand holding an object, releasing it, and picking
    up a DIFFERENT one. Verifies the pipeline produces two object tracks and a
    release, which is the thing the pilot has to be able to show."""
    rng = np.random.RandomState(seed)

    def run(img, fi, n):
        j = rng.randn(2) * 2.0
        hx = 200 + 60 * np.sin(fi / n * 3.14)
        hand = (hx - 40 + j[0], 240 + j[1], hx + 40 + j[0], 320 + j[1])
        if fi < n * 0.4:
            obj, lab = (hx - 20, 250, hx + 60, 330), "cup"
        elif fi < n * 0.55:
            obj, lab = None, None                      # released
        else:
            obj, lab = (420, 200, 520, 300), "bowl"     # a different object
        return [RawInteraction(hand_box=hand, hand_score=0.9,
                               handedness="Right", object_box=obj,
                               object_score=0.8 if obj else float("nan"),
                               object_label=lab,
                               contact_state=CONTACT if obj else FREE)]
    return run


def backend_hoi100doh(repo, ckpt, thresh_hand=0.5, thresh_obj=0.5):
    import sys
    if not os.path.isdir(repo):
        raise SystemExit(
            f"--hoi_repo {repo} not found. The pilot needs the hand_object_"
            f"detector repo and its checkpoint:\n"
            f"  git clone https://github.com/ddshan/hand_object_detector "
            f"{repo}\n"
            f"  # then build its layers and place faster_rcnn_1_8_132028.pth\n"
            f"Run --backend synthetic first to confirm the rest of this "
            f"pipeline works, so a failure here is attributable.")
    sys.path.insert(0, repo)
    from model.utils.config import cfg           # noqa: E402
    from model.faster_rcnn.resnet import resnet  # noqa: E402
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes  # noqa
    from model.roi_layers import nms             # noqa: E402

    net = resnet(['__background__', 'targetobject', 'hand'], 101,
                 pretrained=False, class_agnostic=False)
    net.create_architecture()
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["model"])
    net.eval()
    if torch.cuda.is_available():
        net.cuda()
    cfg.CUDA = torch.cuda.is_available()

    def run(img, fi, n):
        raise SystemExit(
            "the hoi100doh forward pass is a thin wrapper over the repo's "
            "demo.py and must be pinned to the repo revision actually "
            "installed -- fill it in against that checkout rather than "
            "against a guess at its tensor layout, which is how a silently "
            "wrong box reaches a report that looks fine")
    return run


# ------------------------------------------------------------------ tracking
class ObjectTracker:
    """Object track ids across frames, with ids that are NEVER reused.

    A track survives `max_gap` unseen frames and is then retired. Short gaps
    are occlusion by the hand doing the manipulating, constant in egocentric
    footage; a long gap is a different object, and bridging it would
    manufacture the "same object returned" evidence layer 2 exists to measure.

    THE COUNTER IS MONOTONIC BECAUSE THE FIRST VERSION WAS NOT. It allocated
    `max(prev) + 1 if prev else 0`, so once every track had been retired the
    next object seen took id 0 again -- and the synthetic fixture, which
    scripts a hand releasing a cup and picking up a DIFFERENT bowl, reported
    one object track and `n_recontact_same=1`. That is the exact inversion of
    layer 2's discriminating quantity: a boundary recorded as one interrupted
    action, on every event where the free interval outlasted the gap
    tolerance. Reusing an id is not a cosmetic defect when identity IS the
    measurement."""

    def __init__(self, iou_min=0.2, sim_min=0.5, max_gap=3):
        self.tracks, self.next_id = {}, 0
        self.iou_min, self.sim_min, self.max_gap = iou_min, sim_min, max_gap

    def step(self, det, sig):
        for t in self.tracks.values():
            t["missed"] += 1
        oid = None
        if det is not None:
            best, best_s = None, -1.0
            for tid, t in self.tracks.items():
                iou = _iou(t["box"], det)
                sim = signature_similarity(t["sig"], sig)
                if iou >= self.iou_min or sim >= self.sim_min:
                    if iou + 0.5 * sim > best_s:
                        best, best_s = tid, iou + 0.5 * sim
            if best is None:
                best = self.next_id
                self.next_id += 1
            self.tracks[best] = {"box": det, "sig": sig, "missed": 0}
            oid = best
        self.tracks = {k: v for k, v in self.tracks.items()
                       if v["missed"] <= self.max_gap}
        return oid


# ------------------------------------------------------------------- overlay
def contact_sheet(frames, obs, path, cand_idx, n_cols=6, n_show=12, scale=0.5):
    from PIL import Image, ImageDraw
    idx = sorted(set(np.linspace(0, len(frames) - 1, n_show).astype(int).tolist()
                     + [cand_idx]))
    tiles = []
    for i in idx:
        im = Image.fromarray(np.ascontiguousarray(frames[i]))
        im = im.resize((int(im.width * scale), int(im.height * scale)))
        d = ImageDraw.Draw(im)
        for o in obs[i]:
            if o.hand_box:
                d.rectangle([v * scale for v in o.hand_box], outline=(0, 255, 0), width=2)
            if o.object_box:
                d.rectangle([v * scale for v in o.object_box], outline=(255, 160, 0), width=2)
                d.text((o.object_box[0] * scale + 2, o.object_box[1] * scale + 2),
                       f"obj{o.object_track_id}", fill=(255, 160, 0))
            d.text((3, 3), f"{o.rel_t:+.1f}s {o.contact_state}",
                   fill=(255, 255, 0))
        if i == cand_idx:
            d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(255, 0, 0), width=4)
        tiles.append(im)
    if not tiles:
        return
    w, h = tiles[0].size
    rows = (len(tiles) + n_cols - 1) // n_cols
    sheet = Image.new("RGB", (w * n_cols, h * rows), (16, 16, 16))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % n_cols) * w, (k // n_cols) * h))
    sheet.save(path)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--backend", choices=["hoi100doh", "synthetic"], required=True)
    ap.add_argument("--hoi_repo")
    ap.add_argument("--hoi_ckpt")
    ap.add_argument("--hand_model", help="mediapipe .task, for hand tracks when "
                                         "the backend supplies no handedness")
    ap.add_argument("--audit_sheet", help="group the report by the audit answer")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--eye", default="left")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--upscale", default="auto")
    ap.add_argument("--track_max_gap", type=int, default=3)
    ap.add_argument("--no_media", action="store_true")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.backend == "synthetic":
        run = backend_synthetic()
        print("!! SYNTHETIC BACKEND -- this verifies the pipeline, not the "
              "detector. No conclusion about C3.2 may be drawn from its output.")
    else:
        run = backend_hoi100doh(a.hoi_repo, a.hoi_ckpt)

    video_path = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video_path[r["recording_id"]] = r["video"]
    events = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    print(f"{len(events)} pilot events, {len(video_path)} video paths known")

    answers = {}
    if a.audit_sheet:
        with open(a.audit_sheet, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                col = next((k for k in r if k.startswith("answer")), None)
                if col and r[col].strip():
                    answers[r["event_id"]] = r[col].strip()
        print(f"audit answers joined for {len(answers)}/{len(events)} events: "
              f"{dict(Counter(answers.values()))}")

    n_frames = int(round(2 * a.window * a.fps)) + 1
    media_dir = os.path.join(a.out_dir, "hand_object_pilot_media")
    os.makedirs(media_dir, exist_ok=True)
    from decord import VideoReader

    cache, rows = {}, []
    for ei, e in enumerate(events):
        vp = video_path.get(e["recording_id"])
        if vp is None:
            print(f"  !! {e['event_id']}: no video path")
            continue
        vr = VideoReader(vp, num_threads=1)
        vfps, n = vr.get_avg_fps(), len(vr)
        want = e["t"] + np.linspace(-a.window, a.window, n_frames)
        idx = np.clip(np.round(want * vfps).astype(int), 0, n - 1)
        frames = eye_slice(vr.get_batch(idx.tolist()).asnumpy(), a.eye)
        H, W = frames.shape[1:3]

        prev_hands = {}
        otracks = ObjectTracker(max_gap=a.track_max_gap)
        per_frame = []
        for fi in range(n_frames):
            img = np.ascontiguousarray(frames[fi])
            try:
                raw = run(upscale_frame(img, a.upscale), fi, n_frames)
            except SystemExit:
                raise
            except Exception as ex:
                print(f"  !! {e['event_id']} frame {fi}: "
                      f"{type(ex).__name__}: {ex}")
                raw = []
            dets = [{"box": r.hand_box, "handedness": r.handedness, "raw": r}
                    for r in raw if r.hand_box]
            tracks = associate_hands(prev_hands, dets)
            prev_hands = tracks

            obs = []
            for hid, d in tracks.items():
                r: RawInteraction = d["raw"]
                sig = colour_signature(img, r.object_box)
                oid = otracks.step(r.object_box, sig)
                if r.contact_state is not None:
                    st, src, ev = r.contact_state, "detector", {}
                elif r.object_box is not None:
                    st, ev = geom_contact({"box": r.hand_box},
                                          {"box": r.object_box})
                    src = "geometry"
                else:
                    st, src, ev = FREE, "geometry", {}
                # the schema refuses contact without a visible object, and it
                # is right to: a detector class of "in contact" with no object
                # box is an assertion with nothing behind it
                if st in (CONTACT, NEAR) and r.object_box is None:
                    st, src = UNKNOWN, "none"
                obs.append(HandObservation(
                    rel_t=float(want[fi] - e["t"]), abs_t=float(idx[fi] / vfps),
                    hand_track_id=hid, hand_visible=True, hand_box=r.hand_box,
                    hand_score=r.hand_score, handedness=r.handedness,
                    object_visible=r.object_box is not None,
                    object_box=r.object_box, object_score=r.object_score,
                    object_track_id=oid if r.object_box is not None else None,
                    object_label=r.object_label, object_kind=r.object_kind,
                    contact_state=st, contact_source=src, contact_evidence=ev,
                    detector_confidence=r.hand_score))
            if not obs:
                obs = [HandObservation(rel_t=float(want[fi] - e["t"]),
                                       abs_t=float(idx[fi] / vfps))]
                otracks.step(None, None)
            per_frame.append(obs)

        hid = dominant_hand(per_frame)
        rel = series_for_hand(per_frame, hid) if hid is not None else \
            [None] * n_frames
        from src.boundary.hand_object_state import reset_events, continuity_evidence
        res = reset_events(rel, 1.0 / a.fps)
        con = continuity_evidence(rel, 1.0 / a.fps)
        seen_obj = {o.object_track_id for fr in per_frame for o in fr
                    if o.object_visible and o.object_track_id is not None}
        row = {"event_id": e["event_id"], "recording_id": e["recording_id"],
               "audit_answer": answers.get(e["event_id"], ""),
               "frames": n_frames,
               "hand_visible_frac": float(np.mean(
                   [any(o.hand_visible for o in fr) for fr in per_frame])),
               "object_visible_frac": float(np.mean(
                   [any(o.object_visible for o in fr) for fr in per_frame])),
               "contact_frac": float(np.mean(
                   [any(o.contact_state == CONTACT for o in fr) for fr in per_frame])),
               "n_object_tracks": len(seen_obj), **res, **con}
        rows.append(row)
        cache[e["event_id"]] = {"event_id": e["event_id"], "frames": per_frame,
                               "frame_w": int(W), "frame_h": int(H),
                               "candidate_time": e["t"], "fps": a.fps,
                               "summary": row}
        if not a.no_media:
            contact_sheet(frames, per_frame,
                          os.path.join(media_dir, f"{e['event_id']}.jpg"),
                          n_frames // 2)
        print(f"[{ei+1}/{len(events)}] {e['event_id'][:40]:<40} "
              f"hand {row['hand_visible_frac']:.2f} obj "
              f"{row['object_visible_frac']:.2f} contact {row['contact_frac']:.2f} "
              f"tracks {row['n_object_tracks']} rel {res['n_release']}", flush=True)

    torch.save(cache, os.path.join(a.out_dir, "hand_object_pilot.pt"))
    if rows:
        with open(os.path.join(a.out_dir, "hand_object_pilot.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'=' * 72}\nQA, GROUPED BY THE AUDIT'S OWN ANSWER\n{'=' * 72}")
    print("Aggregate recall is not the question. The question is whether the "
          "events where a HUMAN needed object evidence are the events where "
          "the detector supplies it.")
    by = defaultdict(list)
    for r in rows:
        by[r["audit_answer"] or "(unanswered)"].append(r)
    print(f"\n  {'group':<22} {'n':>3} {'hand':>6} {'obj':>6} {'contact':>8} "
          f"{'tracks':>7} {'release':>8} {'held':>6}")
    for k in sorted(by):
        g = by[k]
        def m(f):
            v = [x[f] for x in g if np.isfinite(x[f])] if g else []
            return float(np.mean(v)) if v else float("nan")
        print(f"  {k:<22} {len(g):>3} {m('hand_visible_frac'):>6.2f} "
              f"{m('object_visible_frac'):>6.2f} {m('contact_frac'):>8.2f} "
              f"{m('n_object_tracks'):>7.1f} {m('n_release'):>8.2f} "
              f"{m('held_fraction'):>6.2f}")
    obj_rel = by.get("2_object_relative", [])
    if obj_rel:
        bad = [r for r in obj_rel if r["object_visible_frac"] < 0.5]
        print(f"\n  Of the {len(obj_rel)} object-relative events, "
              f"{len(bad)} have an active object visible in under half the "
              f"window. Those are the pilot's verdict: on the events a human "
              f"said need object evidence, the detector did not produce it.")
        for r in bad:
            print(f"    {r['event_id']:<44} obj "
                  f"{r['object_visible_frac']:.2f}  tracks {r['n_object_tracks']}")
    print(f"\nNext step is to LOOK at {media_dir} for the object-relative "
          f"events before any number here is believed -- active-object recall, "
          f"contact validity, track continuity and false interactions with the "
          f"table or the arm are all visible there and none is measurable "
          f"without ground truth this pilot does not have.")


if __name__ == "__main__":
    main()
