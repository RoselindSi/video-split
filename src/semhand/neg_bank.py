"""The third class as clean frames and boxes, not as baked views.

WHY THE HARVEST'S JPGS CANNOT BE THE TRAINING DATA. The harvester drew the
green box on the native 1920x1520 frame and then scaled to 1280, and it cut
the crop out of the drawn copy. The student's transform does neither: it
resizes first and draws the box at 1280x704, so a width-4 line stays four
pixels instead of becoming 2.7, and it cuts the crop from the UNDRAWN image,
so the crop has no rectangle in it at all. `student.py` says the order matters
and it is right -- a green line of a different thickness and a rectangle that
should not be there are both in the part of the picture the model uses to
find the thing it is judging.

Nothing is lost by re-doing it: the harvest's index carries the frame and the
box, so the labels stand and only the pixels are re-made. What this file
writes is the CLEAN native frame and the box, in the layout `crossv1.read_rows`
already reads, and the views are rendered at training time by `qwen.views` --
the one copy of that transform.

THE SHORTCUT THIS OPENS, STATED SO IT GETS TESTED. Every class-2 example here
comes from a new pass over new recordings, while classes 0 and 1 come from the
existing pool. If the two renders differ in any way a network can see -- JPEG
quality, decoder, colour handling -- the model can learn "this looks like the
new pass" instead of "this is not a hand". The quality is matched to
`distil_prep` at 92 and the decoder is the same OpenCV, and that is not proof.
The acceptance criteria carry the control that tests it: the class-2 rate on
NEW-pool boxes the probe called hands must stay near the probe's own 7%, not
climb toward the pool's provenance.

ONE FRAME, MANY BOXES. Frames are deduplicated before writing, because three
boxes in one frame are three rows and one picture, and writing it three times
would triple eleven gigabytes for nothing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

PAIR = {"cam1": "cam12", "cam2": "cam12", "cam3": "cam34",
        "cam4": "cam34", "cam5": "cam56", "cam6": "cam56"}
LEFT = {"cam1", "cam3", "cam5"}
PKG_COLS = ("stem", "pkg", "tag", "databag", "frame", "image", "x0", "y0",
            "x1", "y1", "W", "H", "y", "status")
CROSS_COLS = ("id", "kind", "rec", "hand", "ctx", "geom", "p_v1_jpeg",
              "p_dump", "y", "tag", "status")
Y_NOTHAND = 2
JPEG_Q = 92          # matched to distil_prep, so the two pools look alike


def load(views_dir, scores_glob):
    """-> [row] joining the harvest index to the probe's verdict."""
    import glob
    idx = {r["stem"]: r
           for r in csv.DictReader(open(os.path.join(views_dir, "index.csv")))}
    out = []
    seen = set()
    for p in sorted(glob.glob(scores_glob)):
        for line in open(p):
            d = json.loads(line)
            if d["stem"] in seen or d["stem"] not in idx:
                continue
            seen.add(d["stem"])
            r = idx[d["stem"]]
            out.append({"stem": d["stem"], "rec": r["rec"], "cam": r["cam"],
                        "frame": int(r["frame"]), "p": float(d["p"]),
                        "box": [int(r[c]) for c in ("x0", "y0", "x1", "y1")],
                        "w_px": int(r["w_px"]), "conf": float(r["conf"])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", action="append", required=True,
                    help="a harvest views dir; give its scores with --scores in order")
    ap.add_argument("--scores", action="append", required=True)
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--pos_thr", type=float, default=0.90,
                    help="boxes above this are written too, as the control pool")
    # THE CONTROL DOES NOT NEED TO BE THE WHOLE POOL. Its job is to show that
    # a class-2 rate measured on new-pass images stays near the probe's own,
    # and a few thousand boxes answer that as well as forty-six thousand
    # while writing a third of the frames.
    ap.add_argument("--max_pos", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--min_px", type=int, default=150)
    ap.add_argument("--out", required=True)
    # SPLIT BY RECORDING, because that is where the cost is. Each recording
    # costs one seek into a 28,000-frame file and then a short walk, so the
    # work divides cleanly and four workers are four times faster. Each shard
    # writes its own index; merging them is a concatenation.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rows = []
    for v, s in zip(a.views, a.scores):
        got = load(v, s)
        print("%-24s %d 个框" % (os.path.basename(v), len(got)))
        rows += got
    neg = [r for r in rows if r["p"] <= a.thr and r["w_px"] >= a.min_px]
    pos = [r for r in rows if r["p"] >= a.pos_thr and r["w_px"] >= a.min_px]
    if a.max_pos and len(pos) > a.max_pos:
        import random
        # Drawn per recording so the control spans the same scenes as the
        # negatives rather than whichever recordings are longest.
        by_rec = collections.defaultdict(list)
        for r in pos:
            by_rec[r["rec"]].append(r)
        rnd = random.Random(a.seed)
        per = max(1, a.max_pos // max(1, len(by_rec)))
        pos = []
        for rec in sorted(by_rec):
            rnd.shuffle(by_rec[rec])
            pos += by_rec[rec][:per]
        rnd.shuffle(pos)
        pos = pos[:a.max_pos]
    print("\n负例 %d（y=%d）  对照正例 %d  来自 %d 条录像"
          % (len(neg), Y_NOTHAND, len(pos), len({r["rec"] for r in neg + pos})))
    frames = {(r["rec"], r["cam"], r["frame"]) for r in neg + pos}
    print("要写的干净帧 %d 张（每帧平均 %.1f 个框）"
          % (len(frames), (len(neg) + len(pos)) / max(1, len(frames))))
    if a.dry:
        return

    import cv2
    import glob as _glob
    roots = _glob.glob("/shared/datasets/.incoming/video_1520p_raw/*/raw_databag")

    def bagpath(rec):
        for r in roots:
            for p in _glob.glob(os.path.join(r, "*", rec)):
                if os.path.isdir(p):
                    return p
        return None

    os.makedirs(os.path.join(a.out, "pkg"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "cross"), exist_ok=True)
    by_vid = collections.defaultdict(list)
    for rec, cam, f in sorted(frames):
        p = bagpath(rec)
        if p is None:
            continue
        by_vid[(p, cam, rec)].append(f)
    if a.nshard > 1:
        keys = sorted(by_vid)
        mine = {k for i, k in enumerate(keys) if i % a.nshard == a.shard}
        by_vid = {k: v for k, v in by_vid.items() if k in mine}
        print("分片 %d/%d：%d 条录像" % (a.shard, a.nshard, len(by_vid)))

    written = {}
    for (p, cam, rec), fs in sorted(by_vid.items()):
        v = os.path.join(p, PAIR[cam] + ".mp4")
        if not os.path.exists(v):
            continue
        d = os.path.join(a.out, "frames", rec, cam)
        os.makedirs(d, exist_ok=True)
        # RESUMABLE, because a four-hour job that has to start over on any
        # interruption is a four-hour job nobody can interrupt. A frame
        # already on disk is the same frame.
        todo = [f for f in sorted(fs)
                if not os.path.exists(os.path.join(d, "%06d.jpg" % f))]
        for f in sorted(fs):
            q = os.path.join(d, "%06d.jpg" % f)
            if os.path.exists(q):
                import cv2 as _c
                im = _c.imread(q)
                if im is not None:
                    written[(rec, cam, f)] = (q, im.shape[1], im.shape[0])
        if not todo:
            print("  %-28s %s  已存在，跳过" % (rec, cam), flush=True)
            continue
        cap = cv2.VideoCapture(v)
        at = -1
        for f in todo:
            # The harvest read consecutive frames from one anchor, so these
            # are clustered; walking forward is cheap and seeking is not.
            if 0 <= f - at <= 240:
                fr = None
                while at < f:
                    ok, fr = cap.read()
                    at += 1
                    if not ok:
                        break
                if fr is None:
                    continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, fr = cap.read()
                at = f
                if not ok:
                    continue
            W = fr.shape[1] // 2
            half = fr[:, :W] if cam in LEFT else fr[:, W:]
            img = os.path.join(d, "%06d.jpg" % f)
            cv2.imwrite(img, half, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            written[(rec, cam, f)] = (img, half.shape[1], half.shape[0])
        cap.release()
        print("  %-28s %s  %d/%d 帧" % (rec, cam, len(fs), len(fs)), flush=True)

    pkg, cross = [], []
    for y, group in ((Y_NOTHAND, neg), (1, pos)):
        for r in group:
            got = written.get((r["rec"], r["cam"], r["frame"]))
            if got is None:
                continue
            img, W, H = got
            tag = "nothand" if y == Y_NOTHAND else "hand_unowned"
            pkg.append({"stem": r["stem"], "pkg": "neg_harvest", "tag": tag,
                        "databag": r["rec"], "frame": r["frame"], "image": img,
                        "x0": r["box"][0], "y0": r["box"][1],
                        "x1": r["box"][2], "y1": r["box"][3],
                        "W": W, "H": H, "y": y, "status": "ok"})
            cross.append({"id": r["stem"], "kind": "VQ", "rec": r["rec"],
                          "hand": "", "ctx": "",
                          "geom": json.dumps([0.0] * 14),
                          "p_v1_jpeg": "", "p_dump": round(r["conf"], 4),
                          "y": y, "tag": tag, "status": "ok"})
    suffix = "" if a.nshard == 1 else "_%d" % a.shard
    with open(os.path.join(a.out, "pkg", "index_nothand%s.csv" % suffix), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PKG_COLS)
        w.writeheader()
        w.writerows(pkg)
    with open(os.path.join(a.out, "cross",
                           "index_bank_nothand%s.csv" % suffix), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CROSS_COLS)
        w.writeheader()
        w.writerows(cross)
    n2 = sum(1 for r in pkg if r["y"] == Y_NOTHAND)
    print("\n-> %s   y=%d 的 %d 行，y=1(未标归属，仅作对照) 的 %d 行，"
          "干净帧 %d 张" % (a.out, Y_NOTHAND, n2, len(pkg) - n2, len(written)))
    print("   视图由 qwen.views 在训练时渲染，这里不烘焙任何视图。")


if __name__ == "__main__":
    main()
