"""Render a clip boxing only the hands Qwen3.8 calls the wearer's.

Clean frames, one green box per wearer hand, nothing on anyone else's. Raw
per-frame answers: no smoothing, no carrying a verdict across frames, so what
flickers in the video is what the model did. If a frame's answer names more
than two hands or fails to parse, the frame says so in its corner instead of
quietly dropping to the first two or to nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess


def v1_answers(dump, items):
    """The deployed V1 verdict (`final_owner_post_cap`) in the answer format.

    Same clip, same detections, same boxes as the Qwen video -- the manifest's
    boxes were taken from this dump -- so the two videos differ only in whose
    hands each system says are the wearer's."""
    import csv
    rec = {it["rec"] for it in items}
    owner = {(r["rec"], int(r["frame"]), str(r["tid"])): r["final_owner_post_cap"] == "1"
             for r in csv.DictReader(open(dump, encoding="utf-8")) if r["rec"] in rec}
    out = {}
    for it in items:
        wearer = [b["n"] for b in it["boxes"]
                  if owner.get((it["rec"], int(it["frame"]), str(b["tid"])))]
        out[it["id"]] = {"id": it["id"], "ok": True, "parsed": {"wearer": wearer}}
    return out


def perhand_answers(pattern, items):
    """One P(self) per hand from `semhand.qwen`, asked about that hand alone.
    A hand with no score is reported in the corner, never silently dropped."""
    import glob
    p = {}
    for f in glob.glob(pattern):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                p[d["id"]] = d["p"]
    out = {}
    for it in items:
        ids = [f"{it['rec']}|{it['frame']}|{b['tid']}" for b in it["boxes"]]
        if any(i not in p for i in ids):
            out[it["id"]] = {"id": it["id"], "ok": False}
            continue
        out[it["id"]] = {"id": it["id"], "ok": True,
                         "parsed": {"wearer": [b["n"] for b, i in zip(it["boxes"], ids) if p[i] >= 0.5]}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--answers", help="Qwen answers")
    ap.add_argument("--v1_dump", help="use V1's deployed verdicts from this dump instead")
    ap.add_argument("--perhand", help="semhand per-hand scores (<arm>_*.jsonl glob): wearer = P >= 0.5")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    import cv2

    items = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.v1_dump:
        answers, title = v1_answers(a.v1_dump, items), "V1 deployed"
    elif a.perhand:
        answers, title = perhand_answers(a.perhand, items), "Qwen3.8 per-hand"
    else:
        answers, title = {}, "Qwen3.8-27B"
        for line in open(a.answers):
            if line.strip():
                d = json.loads(line)
                answers[d["id"]] = d
    items.sort(key=lambda it: it["frame"])
    tmp = a.out + ".raw.mp4"
    writer = None
    n_bad = n_many = 0
    title = a.title or title
    for it in items:
        img = cv2.imread(it["clean"])
        H, W = img.shape[:2]
        ans = answers.get(it["id"])
        note = ""
        if ans is None:
            note, wearer = "no answer", set()
        elif not ans["ok"]:
            note, wearer = "unparsed", set()
            n_bad += 1
        else:
            wearer = set(ans["parsed"]["wearer"])
            if len(wearer) > 2:
                note = f"{len(wearer)} named"
                n_many += 1
        for b in it["boxes"]:
            if b["n"] in wearer:
                cv2.rectangle(img, (int(b["x0"]), int(b["y0"])), (int(b["x1"]), int(b["y1"])),
                              (60, 230, 60), 5)
        if note:
            cv2.putText(img, note, (20, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{title}  frame {it['frame']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out = cv2.resize(img, (a.width, int(a.width * H / W)), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", a.out],
                   check=True)
    os.remove(tmp)
    print(f"{len(items)} 帧 -> {a.out}（无法解析 {n_bad} 帧，点名超过两只 {n_many} 帧）")


if __name__ == "__main__":
    main()
