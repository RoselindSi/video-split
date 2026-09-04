"""Four ways to let a hand into the pipeline, scored on hands people labelled.

THE FAILURE BEING ABLATED. A detection scoring under 0.60 cannot start a track
and is dropped entirely -- no box, no ownership, no cover. That is ByteTrack's
policy and it is not a bug: low scores rescue an EXISTING track, and new tracks
still need a high one. The assumption underneath it is that a real object gets
at least one confident frame to anchor on. A colleague's hand across the bench,
fifteen pixels wide after the detector's resize, never does: on three clips the
audited runs sat at 0.53-0.58 for up to sixty-seven frames and were never
admitted. Sixteen of the seventeen long ones a person looked at were real
hands, and all sixteen were somebody else's.

FOUR POLICIES, ONE ASSOCIATION. Every arm below replays the SAME cached
detections through the SAME `Tracker`, so what differs is admission alone and
nothing else can explain a gap:

    P0  baseline      a track needs one frame at 0.60
    P1  offline       chain everything from 0.50, admit a chain that lasted,
                      and admit it FROM ITS FIRST FRAME -- the clip is already
                      recorded, so there is no reason to pay a confirmation
                      delay in the output
    P2  tentative     DeepSORT's lifecycle: start on any unmatched detection
                      from 0.50, require n_init consecutive hits, output only
                      from confirmation onward, delete on a miss before then
    P3  resolution    P0's policy on detections from a larger detector input.
                      Orthogonal to the other three: it attacks why the score
                      is 0.55 rather than what to do about a 0.55

WHY THE OFFLINE ONE IS NOT JUST TENTATIVE WITH A TIME MACHINE. They confirm on
the same evidence and differ in what reaches the output. Tentative starts
covering at the confirming frame, so the first n_init frames of every real hand
render uncovered -- in a privacy pipeline those are exactly the frames that
leak. Offline admission covers from frame one. The metric below reports that
gap directly as `latency`.

THE GROUND TRUTH IS THE RUN AUDIT AND IT IS NARROW. The labelled runs are
those that peaked in [0.50, 0.60) and lasted at least six frames, on three
clips chosen for rendering rather than at random. So this measures how well a
policy recovers THAT population -- which is the population the failure is made
of -- and says nothing about the base rate of such hands in the corpus. The
random census answered that separately, and the answer was: rarer than these
three clips suggest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

from src.rig.hand_track import box_iou

# Where a chain is allowed to start, for P1 and P2. Not 0.25: the census put
# the median run under 0.25 at one to two frames and essentially motionless,
# which is furniture, and admitting it would drown the real signal.
ADMIT_FLOOR = 0.50

# Frames of evidence before a chain is believed. 12 is where the audit put
# precision at 0.944 and no `nothand` run reached it; 6 is where the sheet's
# own cut-off was and precision there is 0.765. Both are swept.
CONFIRM_N = (3, 6, 12)

# Matching an admitted detection to a labelled run.
HIT_IOU = 0.30


def cache_detections(databag, start, n, stride, weights, imgszs, out):
    """Decode once, detect at every input size, write them all to disk.

    Decoding is the expensive half and it does not depend on `imgsz`, so
    running the sizes in the same pass costs one decode instead of three."""
    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch
    from src.rig.hand_detect import detect
    import time

    cal = os.path.join(databag, "calibration.yaml")
    vids = {k: os.path.join(databag, f"{k}.mp4")
            for k in ("cam12", "cam34", "cam56")}
    rig = RigCalibration(cal)
    vcam = VirtualWideCamera.from_rig(rig)
    model = YOLO(weights)
    rd = Prefetch(ClipReader(rig, vids, start), skip=max(0, stride - 1))
    mc = {}
    got = {int(s): [] for s in imgszs}
    cost = {int(s): 0.0 for s in imgszs}
    shape = None
    for k in range(n):
        src = rd.next()
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        shape = list(rgb.shape)
        for s in imgszs:
            t0 = time.time()
            dets = detect(model, rgb, imgsz=int(s), min_conf=0.25)
            cost[int(s)] += time.time() - t0
            got[int(s)].append([
                {"box": [float(v) for v in d["box"]],
                 "conf": float(d.get("conf", 1.0)),
                 "kp": np.asarray(d["kp"], float).tolist()}
                for d in dets])
        if (k + 1) % 25 == 0:
            print(f"    [{k + 1}/{n}]", flush=True)
    rd.close()
    with open(out, "w") as f:
        json.dump({"shape": shape, "start": start, "stride": stride,
                   "dets": {str(s): got[int(s)] for s in imgszs},
                   "detect_seconds": {str(s): round(cost[int(s)], 1)
                                      for s in imgszs}}, f)
    print(f"  cached -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    for s in imgszs:
        print(f"    imgsz {s}: {sum(len(v) for v in got[int(s)])} detections, "
              f"{cost[int(s)]:.0f}s of inference")


def _run_tracker(frames, shape, new_conf, continue_conf=0.25):
    """Replay one policy's association. -> (ids per frame, tracker)

    The real `Tracker`, not a reimplementation: association, gating and the
    two-stage matching have to be identical across arms or the comparison is
    between trackers rather than between admission rules."""
    from src.rig.hand_track import Tracker
    tr = Tracker()
    out = []
    for dets in frames:
        ids = tr.update(dets, shape, new_track_conf=new_conf,
                        continue_conf=continue_conf)
        out.append(list(ids))
    return out, tr


def policy_baseline(frames, shape, high=0.60):
    """P0. A track needs one frame at `high`; admitted from that frame."""
    ids, _ = _run_tracker(frames, shape, high)
    return [[i for i, t in enumerate(f) if t is not None] for f in ids]


def policy_offline(frames, shape, n_confirm, floor=ADMIT_FLOOR):
    """P1. Chain from `floor`, keep chains that lasted, admit RETROACTIVELY.

    The whole clip is available, so a chain that turns out to be real is
    covered from its first frame rather than from the frame that convinced
    us. That is the entire difference from P2 in the rendered output."""
    ids, _ = _run_tracker(frames, shape, floor)
    hits = {}
    for f in ids:
        for t in f:
            if t is not None:
                hits[t] = hits.get(t, 0) + 1
    keep = {t for t, c in hits.items() if c >= n_confirm}
    return [[i for i, t in enumerate(f) if t in keep] for f in ids]


def policy_tentative(frames, shape, n_init, floor=ADMIT_FLOOR):
    """P2. DeepSORT's lifecycle, online: confirm, then output.

    A track begins tentative on any unmatched detection at or above `floor`,
    needs `n_init` CONSECUTIVE hits, and is discarded if it misses one before
    reaching them. Nothing tentative reaches the output, so the frames before
    confirmation render uncovered."""
    ids, _ = _run_tracker(frames, shape, floor)
    streak, confirmed, dead = {}, set(), set()
    out = []
    for f in ids:
        seen = {t for t in f if t is not None}
        for t in seen:
            if t in dead:
                continue
            if t not in confirmed:
                streak[t] = streak.get(t, 0) + 1
                if streak[t] >= n_init:
                    confirmed.add(t)
        for t in list(streak):
            if t not in seen and t not in confirmed:
                dead.add(t)          # missed while still tentative
                streak.pop(t, None)
        out.append([i for i, t in enumerate(f)
                    if t is not None and t in confirmed])
    return out


def load_runs(cache_frames, audit_csv, tag, chain_iou=0.25):
    """Rebuild the audited runs and attach their labels. -> [run]

    The audit sheet stored a label against a run id and a length, not the
    boxes. Re-chaining the same detections with the same rule reproduces the
    runs; the join key is (first frame, length), because two runs really can
    start on the same frame and the id alone was ambiguous -- which is how two
    rows of the first batch got a label nobody gave them."""
    from src.rig.det_audit import Runs
    r = Runs(chain_iou)
    for k, dets in enumerate(cache_frames):
        r.update(k, dets)
    runs = r.all()
    by = {}
    for x in runs:
        by.setdefault((x["first_k"], x["n"]), []).append(x)
    out, unmatched = [], 0
    for row in csv.DictReader(open(audit_csv, encoding="utf-8-sig")):
        m = re.search(r"_r(\d+)", row["run_id"])
        if not m:
            continue
        key = (int(m.group(1)), int(row["n_frames"]))
        cand = by.get(key)
        if not cand:
            unmatched += 1
            continue
        x = dict(cand.pop(0))
        x["label"] = row["label"]
        out.append(x)
    if unmatched:
        print(f"    !! {unmatched} audited rows in {os.path.basename(audit_csv)}"
              f" did not rejoin a run")
    return out


def score(admitted, boxes, runs, hit_iou=HIT_IOU):
    """-> dict. Did each labelled run get in, and on which frame?

    `boxes` MUST be the box list of the same detection set `admitted` indexes
    into. An earlier version kept one global table and the high-resolution arm
    indexed the 512px boxes with 1024px indices: no error, no crash, and a
    policy scored against boxes belonging to other detections."""
    real = [r for r in runs if r["label"].startswith("hand")]
    junk = [r for r in runs if r["label"] == "nothand"]

    def first_admit(run, frames):
        for i, k in enumerate(run["frames"]):
            if k >= len(frames):
                break
            for j in frames[k]:
                if box_iou(run["boxes"][i], boxes[k][j]) >= hit_iou:
                    return k
        return None

    got_real = [(r, first_admit(r, admitted)) for r in real]
    got_junk = [(r, first_admit(r, admitted)) for r in junk]
    in_real = [(r, f) for r, f in got_real if f is not None]
    lat = sorted(f - r["first_k"] for r, f in in_real)
    return {
        "real": len(real), "real_admitted": len(in_real),
        "junk": len(junk),
        "junk_admitted": sum(1 for _, f in got_junk if f is not None),
        "lat_med": (lat[len(lat) // 2] if lat else float("nan")),
        "lat_max": (lat[-1] if lat else float("nan")),
        "lat_zero": sum(1 for x in lat if x == 0),
        "admitted_per_frame": (sum(len(f) for f in admitted)
                               / max(len(admitted), 1))}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", action="append", required=True,
                    help="clip spec tag:start:cachefile:auditcsv")
    ap.add_argument("--build", action="store_true",
                    help="run the detector and write the caches first")
    ap.add_argument("--databag_list", default="/workspace/cand_fresh.txt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--imgsz", action="append", type=int, default=[])
    a = ap.parse_args()
    imgszs = a.imgsz or [512, 768, 1024]

    specs = []
    for c in a.cache:
        tag, start, cache, audit = c.split(":")
        specs.append((tag, int(start), cache, audit))

    if a.build:
        for tag, start, cache, _ in specs:
            import subprocess
            d = subprocess.run(
                ["bash", "-c",
                 f"grep -h 'databag-26_{tag[1:]}' {a.databag_list} "
                 "/workspace/cand_all.txt /workspace/cand_fresh2.txt "
                 "2>/dev/null | head -1"],
                capture_output=True, text=True).stdout.strip()
            if not d:
                print(f"  !! no databag for {tag}")
                continue
            print(f"  building {tag} from {start}")
            cache_detections(d, start, a.n, a.stride, a.weights, imgszs, cache)

    rows = {}
    for tag, start, cache, audit in specs:
        if not os.path.exists(cache):
            print(f"  !! no cache {cache}")
            continue
        blob = json.load(open(cache))
        shape = tuple(blob["shape"])
        base = [[{"box": d["box"], "conf": d["conf"],
                  "kp": np.array(d["kp"])} for d in f]
                for f in blob["dets"][str(imgszs[0])]]
        runs = load_runs(base, audit, tag)
        if not runs:
            print(f"  !! no audited runs rejoined for {tag}")
            continue
        bx = [[d["box"] for d in f] for f in base]
        arms = [("P0 baseline .60", policy_baseline(base, shape), bx)]
        for nc in CONFIRM_N:
            arms.append((f"P1 offline N={nc}",
                         policy_offline(base, shape, nc), bx))
        for nc in CONFIRM_N:
            arms.append((f"P2 tentative N={nc}",
                         policy_tentative(base, shape, nc), bx))
        for sz in imgszs[1:]:
            hi = [[{"box": d["box"], "conf": d["conf"],
                    "kp": np.array(d["kp"])} for d in f]
                  for f in blob["dets"][str(sz)]]
            # The runs come from the 512 pass, so a larger input is scored on
            # whether it admits THE SAME hands, not on whether it finds
            # different ones. Matching is by overlap, so a box that moved a
            # little with the resolution still counts.
            arms.append((f"P3 imgsz {sz}", policy_baseline(hi, shape),
                         [[d["box"] for d in f] for f in hi]))

        print(f"\n=== {tag} ===  {len(runs)} audited runs "
              f"({sum(1 for r in runs if r['label'].startswith('hand'))} real, "
              f"{sum(1 for r in runs if r['label'] == 'nothand')} not a hand)")
        print(f"  {'policy':<20} {'真手进':>8} {'误进':>6} {'延迟中位':>9} "
              f"{'延迟最大':>9} {'零延迟':>7} {'每帧准入':>9}")
        for name, adm, abox in arms:
            v = score(adm, abox, runs)
            rows.setdefault(name, []).append(v)
            print(f"  {name:<20} {v['real_admitted']:>3}/{v['real']:<4} "
                  f"{v['junk_admitted']:>2}/{v['junk']:<3} "
                  f"{v['lat_med']:>9.0f} {v['lat_max']:>9.0f} "
                  f"{v['lat_zero']:>7} {v['admitted_per_frame']:>9.2f}")
        print(f"  detector seconds: {blob.get('detect_seconds')}")

    if len(specs) > 1 and rows:
        print(f"\n=== 合并 ===")
        print(f"  {'policy':<20} {'真手进':>8} {'误进':>6} {'延迟中位':>9} "
              f"{'零延迟':>7}")
        for name, v in rows.items():
            ra = sum(x["real_admitted"] for x in v)
            r = sum(x["real"] for x in v)
            ja = sum(x["junk_admitted"] for x in v)
            j = sum(x["junk"] for x in v)
            lz = sum(x["lat_zero"] for x in v)
            lm = [x["lat_med"] for x in v if x["lat_med"] == x["lat_med"]]
            print(f"  {name:<20} {ra:>3}/{r:<4} {ja:>2}/{j:<3} "
                  f"{(sorted(lm)[len(lm) // 2] if lm else float('nan')):>9.0f} "
                  f"{lz:>7}")
        print("\n  延迟是「这只手第一次被覆盖」减「它第一次出现」，单位帧。P1 "
              "离线回溯的设计目标就是它恒为 0；\n  P2 在线确认必然 >= N-1，"
              "而那几帧正是隐私管线漏出去的帧。")


if __name__ == "__main__":
    main()
