"""The tracks that started out `other` and were later promoted to the wearer's, put to a person.

WHY THIS IS A DIFFERENT QUESTION FROM THE FIRST BATCH. `birth_probe` asks
about verdicts made on one frame with no history. These tracks had history and
used it: the system called the hand somebody else's, then the belief climbed
past `hi` (0.70) and it changed its mind. Six such tracks carry 370 frames
delivered as the wearer's -- more than the nine tracks of the first batch
carry between them.

THOSE 370 ARE NOT 370 ERRORS, and nor were the first batch's 156. They are
frames the system called `owner`; how many are wrong is what this sheet is
for. A track promoted from `other` to `owner` may be a correction, and if it
is, the promotion mechanism is doing its job.

THREE INDEPENDENT QUESTIONS, AND THE THIRD IS THE ONE WITHOUT AN INSTRUMENT:

  ownership before      whose hand is it in the `other` stretch
  ownership after       whose hand is it in the `owner` stretch
  same physical hand    is it the SAME hand in both

The third does not follow from the first two. A track that runs from one
colleague's left hand onto another colleague's right hand is labelled `other`
at both ends and has still been swapped by the tracker -- and a swap is the
tracker's defect, not the classifier's, repaired somewhere else entirely. So
it is asked separately and in its own words.

FRAMES ARE ANCHORED ON THE TRANSITION, NOT SPREAD OVER THE TRACK. Five fixed
slots around the first `other` -> `owner` crossing:

  B1  early in the `other` stretch        establishes the original hand
  B2  the last observation before it      identity immediately before
  A1  the first frame called `owner`      the crossing itself
  A2  middle of the `owner` stretch       whether it is still that hand
  A3  end of the `owner` stretch          where it ended up

A short track that has no distinct frame for a slot leaves that slot empty
rather than repeating a neighbour, and the sheet records how many it really
had.

AND A CLIP, BECAUSE FIVE STILLS CANNOT SHOW A SWAP. Roughly a second either
side of the crossing at native rate, slowed for viewing, with the box drawn on
every frame the track was present and nothing drawn where it was not -- the
dropouts are where an association error happens, and a still sampled at a
fixed position will sit either side of one without showing it.

THE SHEET CARRIES NO SCORES. Not the belief, not the birth value, not what it
was promoted on, not the verdict, not how often the cap demoted it. Those live
in the key file and are joined back after the answers are in.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

from src.rig.demo_video import _to_h264
from src.semhand.birth_probe import candidates

CLIP_PRE, CLIP_POST = 30, 30      # frames each side of the crossing, at 30 fps
CLIP_FPS = 12                     # written slower than real time, to watch
CLIP_W = 960
SLOTS = ("B1", "B2", "A1", "A2", "A3")


def slots_for(fr):
    """-> [(slot, row)] anchored on the first `other` -> `owner` crossing."""
    own = [int(r["own"]) for r in fr]
    t = next((i for i, o in enumerate(own) if o), None)
    if t is None or t == 0:
        return []
    after = [i for i in range(t, len(fr)) if own[i]]
    pick = {
        "B1": max(0, t // 3),
        "B2": t - 1,
        "A1": t,
        "A2": after[len(after) // 2],
        "A3": after[-1],
    }
    out, used = [], set()
    for s in SLOTS:
        i = pick[s]
        # NO PADDING. A track whose `other` stretch is two frames long has one
        # frame for B1 and B2, and saying so is information; repeating it to
        # fill five slots is not.
        if i in used:
            continue
        used.add(i)
        out.append((s, fr[i]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+",
                    default=["/workspace/ten_base", "/workspace/cam3_ship2"])
    ap.add_argument("--jobs", nargs="+",
                    default=["/workspace/cam3_jobs10.txt", "/workspace/cam3_jobs.txt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", required=True)
    a = ap.parse_args()
    import cv2
    import numpy as np
    from src.rig.seam_fix import RawCameraReader
    from src.semhand import qwen

    jobs = {}
    for jp in a.jobs:
        for line in open(jp):
            f = line.strip().split("|")
            if len(f) >= 4 and not line.startswith("#"):
                jobs[f[0]] = (f[1], int(f[2]), int(f[3]))

    picked = []
    for arm in a.arms:
        for rec, tid, fr, why in candidates(arm):
            if "promoted" not in why:
                continue
            sl = slots_for(fr)
            if sl:
                picked.append((rec, tid, fr, sl))
    print("promoted 轨迹 %d 条" % len(picked))

    os.makedirs(a.frames, exist_ok=True)
    for sub in ("context", "clips"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    # Every frame either sheet needs, per recording, so each video is walked
    # once: the five stills and the whole clip window.
    want = collections.defaultdict(set)
    boxes = collections.defaultdict(dict)      # rec -> frame -> box, per track
    for rec, tid, fr, sl in picked:
        s, e = jobs[rec][1], jobs[rec][1] + jobs[rec][2]
        cross = int(next(r for k, r in sl if k == "A1")["frame"])
        for f in range(max(s, cross - CLIP_PRE), min(e, cross + CLIP_POST + 1)):
            want[rec].add(f)
        for _, r in sl:
            want[rec].add(int(r["frame"]))
        for r in fr:
            boxes[(rec, tid)][int(r["frame"])] = \
                [int(r[c]) for c in ("x0", "y0", "x1", "y1")]

    for rec in sorted(want):
        need = sorted(want[rec])
        rd = RawCameraReader({k: os.path.join(jobs[rec][0], f"{k}.mp4")
                              for k in ("cam12", "cam34", "cam56")}, "cam3", need[0])
        cur, img = need[0] - 1, None
        for f in need:
            while cur < f:
                img = rd.next()
                cur += 1
                if img is None:
                    break
            if img is None:
                break
            cv2.imwrite(os.path.join(a.frames, "%s_f%06d.jpg" % (rec, f)),
                        img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        rd.close()
        print("  %-20s %d 帧" % (rec, len(need)), flush=True)

    trk, frm, key = [], [], []
    for rec, tid, fr, sl in sorted(picked, key=lambda c: (c[0], int(c[1]))):
        own = [int(r["own"]) for r in fr]
        t = next(i for i, o in enumerate(own) if o)
        cross = int(fr[t]["frame"])
        for s, r in sl:
            f = int(r["frame"])
            p = os.path.join(a.frames, "%s_f%06d.jpg" % (rec, f))
            if not os.path.exists(p):
                continue
            box = [int(r[c]) for c in ("x0", "y0", "x1", "y1")]
            stem = "%s_t%s_%s_f%06d" % (rec, tid, s, f)
            view, _ = qwen.views({"image": p, "box": box})
            view.save(os.path.join(a.out, "context", stem + ".jpg"), quality=86)
            frm.append({"stem": stem, "rec": rec, "tid": tid, "slot": s,
                        "phase": "before" if s.startswith("B") else "after",
                        "frame": f, "w": box[2] - box[0], "label": ""})
            key.append({"stem": stem, "rec": rec, "tid": tid, "slot": s,
                        "frame": f, "p": r["p"], "own": r["own"],
                        "conf": r["conf"], "w": box[2] - box[0]})

        clip = os.path.join(a.out, "clips", "%s_t%s.mp4" % (rec, tid))
        s0, e0 = jobs[rec][1], jobs[rec][1] + jobs[rec][2]
        rng = range(max(s0, cross - CLIP_PRE), min(e0, cross + CLIP_POST + 1))
        wr, n_drawn = None, 0
        for f in rng:
            p = os.path.join(a.frames, "%s_f%06d.jpg" % (rec, f))
            if not os.path.exists(p):
                continue
            im = cv2.imread(p)
            sc = CLIP_W / float(im.shape[1])
            im = cv2.resize(im, (CLIP_W, int(im.shape[0] * sc)))
            b = boxes[(rec, tid)].get(f)
            if b is not None:
                cv2.rectangle(im, (int(b[0] * sc), int(b[1] * sc)),
                              (int(b[2] * sc), int(b[3] * sc)), (0, 230, 0), 3)
                n_drawn += 1
            if wr is None:
                wr = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"),
                                     CLIP_FPS, (im.shape[1], im.shape[0]))
            wr.write(im)
        if wr is not None:
            wr.release()
            # OpenCV's `mp4v` is MPEG-4 Part 2, which no browser will demux --
            # the first cut of this package shipped six clips that opened as a
            # dead black frame, and the sheet came back unanswered. Five stills
            # cannot show a swap, so a clip that will not play is a missing
            # instrument, not a cosmetic defect.
            if not _to_h264(clip):
                print("  警告：%s 仍是 MPEG-4 Part 2，浏览器放不了"
                      % os.path.basename(clip))

        trk.append({"rec": rec, "tid": tid,
                    "n_before": t, "n_after": len(fr) - t,
                    "n_slots": len(sl), "clip": os.path.basename(clip),
                    "clip_frames": len(rng), "box_drawn": n_drawn,
                    "ownership_before": "", "ownership_after": "",
                    "same_physical_hand": "", "evidence": ""})
        key.append({"stem": "TRACK_%s_t%s" % (rec, tid), "rec": rec, "tid": tid,
                    "slot": "", "frame": cross, "p": round(float(fr[0]["p"]), 4),
                    "own": "", "conf": "",
                    "w": max(float(r["p"]) for r in fr)})

    for name, rows in (("tracks.csv", trk), ("frames.csv", frm)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(a.out.rstrip("/") + "_key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    with open(os.path.join(a.out, "MAPPING.txt"), "w") as fh:
        fh.write(
            "每条轨迹两份材料：clips/ 里的小视频（首次转变前后各约 1 秒，放慢播放），\n"
            "和 context/ 里的五张固定帧 B1 B2 A1 A2 A3（B 在前、A 在后，按文件名排序即是时序）。\n"
            "绿框画在该轨迹出现的每一帧上；没有框的帧是这条轨迹当时不存在。\n\n"
            "frames.csv 每帧一行，填 label：\n"
            "  owner 佩戴者自己的 / other 别人的 / nothand 不是手 / uncertain 看不清\n\n"
            "tracks.csv 每条轨迹一行，填四列：\n"
            "  ownership_before   B 段那只手是谁的\n"
            "  ownership_after    A 段那只手是谁的\n"
            "  same_physical_hand 前后是不是同一只物理的手  yes / no / uncertain\n"
            "     （前后都填 other 也不等于同一只手——可能是从一个人的手串到另一个人的手）\n"
            "  evidence           凭什么判的：视觉连续性 / 左右手 / 手套袖子前臂 / 遮挡重现\n\n"
            "模型的分数、判定和挑中理由都不在这个包里。\n")
    print("-> %s  %d 轨迹 / %d 框 / %d 段视频"
          % (a.out, len(trk), len(frm), sum(1 for r in trk if r["box_drawn"])))


if __name__ == "__main__":
    main()
