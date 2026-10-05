"""Score hand-ness for a finished run, in the form `post_pass` can consume.

THE MISSING LINK. `post_pass` runs three track-level decisions and the first
is hand-ness -- a box on a machine part is not evidence about ownership, so
letting it vote first is the one order that is not circular. But that stage
only fires when `--scores` is given, and nothing produced those scores for an
arbitrary arm: `handness.py` takes its stems from an existing review package,
not from a run. Without this, `post_pass` treats every box as a hand, and on
one recording of the frozen set that meant 25 tracks a person had called
non-hands were consolidated into "the wearer's own".

WHY THE TEACHER AND NOT THE DISTILLED STUDENT. On 100 human-confirmed
non-hands from that set the teacher put 1 above the deployment threshold and
the student put 31. The student was trained to imitate this teacher and lost
almost all of its discrimination, so on the population that matters it is not
a cheaper version of the teacher -- it is a different, much worse answer. The
volume here is small enough that the teacher is affordable: hand-ness is only
needed for boxes the run already called the wearer's, which is a few hundred
per recording, about four seconds of a four-hundred-second render.

THE RANKING HAS TO BE REBUILT EXACTLY. `post_pass` looks scores up by
`<rec>_f<frame>_b<rank>`, where rank comes from the own boxes of that frame,
largest area first, keeping only boxes that do not overlap an already-kept one
by more than 0.10 IoU. A different order silently attaches a score to the
wrong box, and nothing downstream can detect that -- so the selection here is
a copy of `post_pass.hand_scores`, not a reimplementation of its intent.

Crops follow `handness.py`: the green box is drawn before cropping, the crop
is 1.6x the longer side centred on the box, and both images are written at the
sizes the teacher was measured with.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

FRAME_W = 1280
CROP = 256
DISJOINT_IOU = 0.10


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def all_boxes(rows):
    """每个拿到 tid 的框都打一次，用于跟踪前过滤。

    `selected_boxes` 只取 own=1 的框，因为 post_pass 的查表就是按那套排名
    建的。但跟踪前的过滤发生在归属之前 —— 那时 own 还不存在 —— 所以这里
    改按框本身索引，且必须覆盖全部框：只打 own=1 的那批，等于让检测器先
    替语义层筛一遍，而它的置信度对手/非手的平衡准确率只有 53.3%。
    """
    out = []
    for row in rows:
        if row.get("tid") in (None, ""):
            continue
        try:
            box = [int(float(row[c])) for c in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            continue
        out.append((int(row["frame"]), None, box))
    return out


def selected_boxes(rows):
    """-> [(frame, rank, (x0,y0,x1,y1))]，和 post_pass.hand_scores 同一套顺序。"""
    by_frame = collections.defaultdict(list)
    for row in rows:
        if str(row.get("own")) != "1":
            continue
        try:
            box = [float(row[c]) for c in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            continue
        by_frame[int(row["frame"])].append(box)
    out = []
    for frame, boxes in sorted(by_frame.items()):
        boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        kept = []
        for box in boxes:
            if all(iou(box, k) < DISJOINT_IOU for k in kept):
                kept.append(box)
                out.append((frame, len(kept) - 1, box))
    return out


def read_jobs(path):
    jobs = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 4:
            jobs[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
    return jobs


def render(arm, jobs, views, recs=None, every_box=False):
    import cv2
    from src.rig.seam_fix import RawCameraReader

    os.makedirs(views, exist_ok=True)
    items = []
    for rec, (bag, _start, _n) in sorted(jobs.items()):
        if recs and rec not in recs:
            continue
        path = os.path.join(arm, "%s.csv" % rec)
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        picks = all_boxes(rows) if every_box else selected_boxes(rows)
        if not picks:
            continue
        videos = {k: os.path.join(bag, "%s.mp4" % k)
                  for k in ("cam12", "cam34", "cam56")}
        wanted = sorted({f for f, _r, _b in picks})
        reader = RawCameraReader(videos, "cam3", wanted[0])
        current, cache, image = wanted[0] - 1, {}, None
        for frame in wanted:
            while current < frame:
                image = reader.next()
                current += 1
                if image is None:
                    break
            if image is None:
                break
            cache[frame] = image.copy()
        reader.close()
        made = 0
        for frame, rank, box in picks:
            image = cache.get(frame)
            if image is None:
                continue
            stem = ("%s_f%06d_x%d_y%d" % (rec, frame, int(box[0]), int(box[1]))
                    if rank is None else "%s_f%06d_b%d" % (rec, frame, rank))
            full_path = os.path.join(views, stem + "_full.jpg")
            crop_path = os.path.join(views, stem + "_crop.jpg")
            if not (os.path.exists(full_path) and os.path.exists(crop_path)):
                height, width = image.shape[:2]
                vis = image.copy()
                x0, y0, x1, y1 = [int(v) for v in box]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 230, 0), 4)
                scale = FRAME_W / float(width)
                cv2.imwrite(full_path,
                            cv2.resize(vis, (FRAME_W, int(height * scale))),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                side = int(max(x1 - x0, y1 - y0) * 1.6)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                a0, b0 = max(0, cx - side // 2), max(0, cy - side // 2)
                a1, b1 = min(width, cx + side // 2), min(height, cy + side // 2)
                cv2.imwrite(crop_path,
                            cv2.resize(vis[b0:b1, a0:a1], (CROP, CROP)),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            items.append({"stem": stem, "full": full_path,
                          "crop": crop_path, "frame": frame,
                          "box": "%d,%d,%d,%d" % tuple(int(v) for v in box),
                          "bag": os.path.basename(bag), "rec": rec})
            made += 1
        print("  %-22s %d 个框" % (rec, made), flush=True)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True, help="post_pass --scores 吃的 jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:18997")
    ap.add_argument("--model", default="qwen-bf16")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--rec", action="append")
    ap.add_argument("--all-boxes", dest="all_boxes", action="store_true",
                    help="给每个拿到 tid 的框打分并按框索引，用于跟踪前过滤；"
                         "默认只打 own=1 的框并按 post_pass 的排名索引")
    a = ap.parse_args()

    from src.semhand.handness import QUESTION, PREFIX
    from src.semhand.vllm_probe import data_url, two_way, run_phase
    import json as _json
    import time
    import urllib.request

    jobs = read_jobs(a.jobs)
    items = render(a.arm, jobs, a.views, set(a.rec) if a.rec else None,
                   every_box=a.all_boxes)
    # neg_bank 要这份索引，而 rec 必须写 databag 的目录名：它按 rec 去
    # /shared/... 下找同名目录，写 R26_xxx 会一条都匹配不上而且不报错。
    if items and items[0].get("box"):
        index = os.path.join(a.views, "index.csv")
        with open(index, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "stem", "rec", "cam", "frame", "track_id", "conf",
                "w_px", "h_px", "x0", "y0", "x1", "y1"])
            writer.writeheader()
            for item in items:
                x0, y0, x1, y1 = [int(v) for v in item["box"].split(",")]
                writer.writerow({
                    "stem": item["stem"], "rec": item["bag"], "cam": "cam3",
                    "frame": item["frame"], "track_id": item.get("tid", ""),
                    "conf": item.get("conf", ""),
                    "w_px": x1 - x0, "h_px": y1 - y0,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        print("索引 -> %s（%d 行）" % (index, len(items)))

    print("共 %d 个框要打分" % len(items))
    if not items:
        raise SystemExit("没有 own=1 的框")

    def ask(item):
        body = {
            "model": a.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_url(item["full"])}},
                    {"type": "image_url",
                     "image_url": {"url": data_url(item["crop"])}},
                    {"type": "text", "text": QUESTION},
                ]},
                {"role": "assistant", "content": PREFIX},
            ],
            "max_tokens": 1, "temperature": 0.0,
            "logprobs": True, "top_logprobs": 64,
            "add_generation_prompt": False, "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            a.url.rstrip("/") + "/v1/chat/completions",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return _json.loads(response.read())

    start = time.time()
    responses = run_phase(items, ask, a.conc)
    written = missing = low = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for item, response in zip(items, responses):
            p, _seen = two_way(response)
            if p is None:
                missing += 1
                continue
            record = {"stem": item["stem"], "p": p}
            if a.all_boxes:
                record.update(frame=item["frame"], box=item["box"])
            fh.write(json.dumps(record) + "\n")
            written += 1
            low += int(p < 0.10)
    elapsed = time.time() - start
    print("%.1fs（%.0f 框/分钟），写出 %d 条 -> %s"
          % (elapsed, 60 * len(items) / elapsed, written, a.out))
    print("  其中 P(hand) < 0.10 的 %d 个（%.1f%%）—— 这些就是 post_pass "
          "第一阶段会摘掉的非手" % (low, 100 * low / max(1, written)))
    if missing:
        print("  true/false 没同时进 top-k 的 %d 个" % missing)


if __name__ == "__main__":
    main()
