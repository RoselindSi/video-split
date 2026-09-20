"""The frames where the wearer's hand was there and the detector was not.

WHY THESE FRAMES AND NOT A RANDOM SAMPLE. The detector already finds the
wearer's hands on 97.5% of the images an independent reference says they are
in, so a random batch of frames would mostly teach it what it knows. What
costs the pipeline is the 2.5% it never proposes and the frames it proposes
too weakly to keep: each one breaks a track, and a broken track is what the
reacquire rule then turns into a covered frame on the wearer's own hand. One
400-frame clip had 63 within-track gaps.

WHAT IS MINED. A track the pipeline called the wearer's, a frame inside that
track's span with no detection in it, and the position the hand must have been
in -- interpolated from the frames on either side. Then the frame is detected
again at a floor far below the shipped one, which splits the gap into the two
failures that need different work:

    proposed    a box is there at 0.05-0.25, under the floor that ships.
                An annotator confirms it; the fix may be a threshold, not
                training data at all.
    missing     nothing at any score. The annotator draws it, and this is the
                training example that cannot be had any other way.

Every mined frame is written as an image with the interpolated position and
whatever the detector did propose drawn on it, plus a row in an index, so an
annotation pass has the picture and the reason it was chosen.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def own_tracks(dump_rows, rec, labels=None):
    """-> {tid: {frame: box}} for tracks the pipeline called the wearer's.

    The human or zone label is used when there is one; otherwise the pipeline's
    own verdict, taken over the track rather than per frame, because a track
    that is mostly self is a self hand however it was scored on one frame."""
    by_tid = collections.defaultdict(dict)
    verdict = collections.defaultdict(lambda: [0, 0])
    for r in dump_rows:
        if r["rec"] != rec:
            continue
        tid = str(r["tid"])
        by_tid[tid][int(r["frame"])] = [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
        verdict[tid][r["final_owner_post_cap"] == "1"] += 1
    out = {}
    for tid, frames in by_tid.items():
        g = labels.get((rec, tid)) if labels else None
        if g == "other":
            continue
        if g != "owner" and verdict[tid][True] <= verdict[tid][False]:
            continue
        out[tid] = frames
    return out


def gaps_of(frames, max_gap=5):
    """-> [(tid frame, interpolated box)] for every frame inside the track's
    span where there is no detection. The box is the straight line between the
    two frames that bracket it, which is where a hand moving at all is."""
    fs = sorted(frames)
    out = []
    for a, b in zip(fs, fs[1:]):
        n = b - a - 1
        if not 1 <= n <= max_gap:
            continue
        pa, pb = frames[a], frames[b]
        for k in range(1, n + 1):
            t = k / (n + 1.0)
            out.append((a + k, [pa[i] + t * (pb[i] - pa[i]) for i in range(4)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databag", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--weights", default="/shared/models/HaWoR/weights/external/detector.pt")
    ap.add_argument("--out", default="/workspace/gapmine")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    from src.rig.calibration import RigCalibration
    from src.rig.geometry import VirtualWideCamera
    from src.rig.hand_detect import detect
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader, Prefetch

    dump = list(csv.DictReader(open(a.dump, encoding="utf-8")))
    labels = None
    if a.labels and os.path.exists(a.labels):
        from src.semhand.distil_eval import read_zone_gold
        labels = read_zone_gold(a.labels)
    tracks = own_tracks(dump, a.rec, labels)
    # WHO ELSE IS STANDING THERE. A gap in one track's frames is not a gap in
    # the picture when another track has taken the hand over: the tracker
    # renames a hand more often than it loses one, and 16 of the first 175
    # mined frames were that rename rather than a miss. Two things follow, and
    # both need the other tracks' boxes at that frame.
    others = collections.defaultdict(list)
    for r in dump:
        if r["rec"] == a.rec:
            others[int(r["frame"])].append(
                (str(r["tid"]), [float(r[c]) for c in ("x0", "y0", "x1", "y1")]))
    want = collections.defaultdict(list)          # frame -> [(tid, box)]
    handoff = 0
    for tid, frames in tracks.items():
        for f, box in gaps_of(frames):
            if any(t != tid and iou(box, b) >= 0.3 for t, b in others.get(f, ())):
                handoff += 1                      # the hand is there, renamed
                continue
            want[f].append((tid, box))
    print(f"{a.rec}: 自己的手轨迹 {len(tracks)} 条，断档帧 {len(want)} 帧、"
          f"{sum(len(v) for v in want.values())} 只手"
          + (f"（另有 {handoff} 帧是同一只手换了 track id，不算断档）" if handoff else ""))
    if not want:
        return

    os.makedirs(os.path.join(a.out, a.rec), exist_ok=True)
    rig = RigCalibration(os.path.join(a.databag, "calibration.yaml"))
    vcam = VirtualWideCamera.from_rig(rig)
    vids = {k: os.path.join(a.databag, f"{k}.mp4") for k in ("cam12", "cam34", "cam56")}
    model = YOLO(a.weights)
    rd = Prefetch(ClipReader(rig, vids, a.start), skip=0)
    rows, mc = [], {}
    # STOP WHEN THE LAST WANTED FRAME IS DONE, AND WRITE AS YOU GO. The first
    # version read to the end of the clip whatever it needed and wrote its
    # index once at the end; two runs then sat on a hung SAN read for 22 hours
    # with every image already on disk and nothing to show for it.
    last_wanted = max(want)
    idx = os.path.join(a.out, f"index_{a.rec}.csv")
    fh = open(idx, "w", newline="")
    writer = None
    for k in range(a.n):
        src = rd.next()
        if not src:
            break
        f = a.start + k
        if f not in want:
            continue
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        props = detect(model, rgb, min_conf=a.floor)
        vis = rgb.copy()
        for d in props:
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(vis, f"{float(d['conf']):.2f}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        # A detection another track already owns is not evidence about this
        # one. Without this, a colleague's hand standing next to the gap reads
        # as "the detector proposed it at 0.79" and the frame is filed under a
        # threshold problem it does not have.
        taken = [b for t, b in others.get(f, ())
                 if not any(t == tid for tid, _ in want[f])]
        for tid, box in want[f]:
            best, bc = 0.0, 0.0
            for d in props:
                db = [float(x) for x in d["box"]]
                if any(iou(db, tb) >= 0.5 for tb in taken):
                    continue
                v = iou(box, db)
                if v > best:
                    best, bc = v, float(d["conf"])
            kind = "proposed" if best >= 0.3 else "missing"
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 220, 60), 2)
            cv2.putText(vis, f"tid {tid} {kind}", (x0, min(rgb.shape[0] - 6, y1 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2, cv2.LINE_AA)
            rows.append({"rec": a.rec, "frame": f, "tid": tid, "kind": kind,
                         "x0": int(box[0]), "y0": int(box[1]), "x1": int(box[2]),
                         "y1": int(box[3]), "best_iou": round(best, 3),
                         "conf": round(bc, 4), "n_props": len(props),
                         "image": f"{a.rec}/{a.rec}_f{f:06d}.jpg"})
        cv2.imwrite(os.path.join(a.out, a.rec, f"{a.rec}_f{f:06d}.jpg"),
                    cv2.resize(vis, (1200, int(1200 * rgb.shape[0] / rgb.shape[1]))),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
        for r in rows[-len(want[f]):]:
            writer.writerow(r)
        fh.flush()
        if f >= last_wanted:
            break
    fh.close()
    rd.close()
    c = collections.Counter(r["kind"] for r in rows)
    print(f"  -> {idx}   提了但分数不够 {c['proposed']}，完全没提出 {c['missing']}")
    if c["proposed"]:
        cf = sorted(r["conf"] for r in rows if r["kind"] == "proposed")
        print(f"     被挡掉的那些分数: 中位 {cf[len(cf)//2]:.2f} 最小 {cf[0]:.2f} 最大 {cf[-1]:.2f}")


if __name__ == "__main__":
    main()
