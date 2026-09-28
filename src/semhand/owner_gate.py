"""Qwen admission gate for an additional camera-wearer hand.

The frame model and ``OwnHold`` nominate owner hands.  ``max_owner=2`` is only
an upper bound: it cannot know that a frame currently contains one wearer hand
and one colleague's hand.  This module audits every lower-ranked owner track
that appears beside a stronger owner candidate.

Two environments are used, as in ``semhand.handness``:

* ``--prep`` runs in the rig environment and writes five representative
  temporal-context-plus-crop views for each contested track.  The context is
  a chronological four-panel view of the preceding two seconds, with the same
  tracker id boxed wherever it is observed.
* the scoring pass runs in the Qwen environment.  It first writes a neutral
  description of the boxed hand (Q1), then scores ``P(wearer)`` from the next
  token logits and aggregates the sampled frames into ``owner/other/unsure``.

Only ``owner`` is admitted by the renderer.  ``other`` and ``unsure`` fail
closed and remain mosaicked.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics


MANIFEST = "manifest.csv"

CONTEXT_DESCRIBE = (
    "The first image is a chronological four-panel sequence from a camera worn "
    "on the head of a factory worker. The panels are labelled by time relative "
    "to the current frame. The same tracked hand is outlined in green wherever "
    "it is visible; a panel without a green box means that tracker id was not "
    "observed then. The second image is a crop of that hand in the current "
    "frame. Describe the hand's motion, where its arm enters from, and what it "
    "is doing. Do not say or guess whose hand it is.\nReply with JSON only:\n"
    '{"hand_side": "left" | "right" | "unclear", '
    '"motion_path": "<where the hand came from and moved, a short phrase>", '
    '"interaction": "<what the hand is doing, a short phrase>", '
    '"object": "<object in contact, or none>", '
    '"occlusion": "<what hides part of the hand, or none>", '
    '"ambiguous_regions": "<anything that makes the track hard to follow, or none>"}')

CONTEXT_INTRO = (
    "These images come from a camera worn on the head of a factory worker "
    "(the camera wearer). The first image is a chronological four-panel "
    "sequence covering the two seconds before the current frame. The green "
    "boxes follow one tracker id through time; the second image is its current "
    "hand crop. Use the arm's entry direction and motion through time, not only "
    "the current hand appearance. Other people's hands may also be visible. A "
    "person has at most two hands.")

CONTEXT_QUESTION = (
    "Is the green-boxed hand trajectory one of the camera wearer's own hands? "
    'Reply with JSON only: {"wearer": true} or {"wearer": false}')


def read_jobs(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 4:
            out[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
    return out


def contested_tracks(rows):
    """Return lower-ranked owner rows grouped by track id."""
    by_frame = collections.defaultdict(list)
    for row in rows:
        if str(row.get("own")) == "1" and str(row.get("not_hand", "0")) != "1":
            by_frame[int(row["frame"])].append(row)

    tracks = collections.defaultdict(list)
    for frame_rows in by_frame.values():
        owners = sorted(frame_rows, key=lambda r: -float(r["p"]))
        for rank, row in enumerate(owners[1:], start=2):
            item = dict(row)
            item["owner_rank"] = rank
            tracks[str(row["tid"])].append(item)
    return tracks


def all_tracks(rows, min_frames=3):
    """Every track the detector kept, grouped by id, whatever it called them.

    WHY THIS EXISTS BESIDE `contested_tracks`. The production gate audits only
    tracks the pipeline already nominated as the wearer's, which is the right
    thing for admission and the wrong thing for building a teacher corpus: of
    51 tracks it returned across ten recordings, 45 came back `owner` and 3
    `other`, and a head cannot be distilled from three examples of the class it
    exists to find. The same ten recordings hold 260 tracks once the filter is
    dropped, 195 of them called `other` by the detector -- the material was
    never missing, it was never submitted.

    The detector's own verdict rides along as `det_own` so the teacher's answer
    can be read against it. It is a prior, not a label: the detector is what
    the teacher is being asked to correct.
    """
    tracks = collections.defaultdict(list)
    for row in rows:
        tid = row.get("tid")
        if tid in (None, "") or str(row.get("not_hand", "0")) == "1":
            continue
        tracks[str(tid)].append(dict(row))
    out = {}
    for tid, track_rows in tracks.items():
        if len(track_rows) < min_frames:
            continue
        owner = sum(1 for r in track_rows if str(r.get("own")) == "1")
        verdict = "owner" if owner * 2 >= len(track_rows) else "other"
        for r in track_rows:
            r["det_own"] = verdict
            r["owner_rank"] = 1
        out[tid] = track_rows
    return out


def representative_rows(rows, n=5):
    """Evenly sample up to ``n`` observed frames, including both ends."""
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    if len(rows) <= n:
        return rows
    if n <= 1:
        return [rows[len(rows) // 2]]
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[i] for i in dict.fromkeys(idx)]


def context_frame_numbers(frame, source_fps=30.0, seconds=2.0):
    """Four chronological observations ending at ``frame``.

    The spacing is deliberately denser near the decision frame: entry
    direction is usually visible over seconds, while a hand crossing another
    hand can change appearance in a fraction of a second.
    """
    offsets = (seconds, seconds / 2.0, seconds / 4.0, 0.0)
    return [max(0, int(frame) - round(float(source_fps) * offset))
            for offset in offsets]


def _storyboard(images, boxes, labels, size=(1280, 960)):
    """Build a 2x2, aspect-preserving temporal storyboard."""
    import cv2
    import numpy as np

    width, height = size
    tile_w, tile_h = width // 2, height // 2
    board = np.zeros((height, width, 3), dtype=np.uint8)
    for index, (image, box, label) in enumerate(zip(images, boxes, labels)):
        ih, iw = image.shape[:2]
        scale = min(tile_w / float(iw), tile_h / float(ih))
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        ox, oy = (tile_w - rw) // 2, (tile_h - rh) // 2
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tile[oy:oy + rh, ox:ox + rw] = cv2.resize(image, (rw, rh))
        if box is not None:
            x0, y0, x1, y1 = box
            p0 = (ox + round(x0 * scale), oy + round(y0 * scale))
            p1 = (ox + round(x1 * scale), oy + round(y1 * scale))
            cv2.rectangle(tile, p0, p1, (0, 230, 0), 4)
        cv2.rectangle(tile, (8, 8), (180, 45), (20, 20, 20), -1)
        cv2.putText(tile, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        row, col = divmod(index, 2)
        board[row * tile_h:(row + 1) * tile_h,
              col * tile_w:(col + 1) * tile_w] = tile
    return board


def aggregate(records, admit=0.60, reject=0.40, min_positive_frac=0.60,
              min_views=3, base_rescue=0.95, base_strong=0.90,
              min_base_strong_frac=0.80, track_maps=None):
    """Fuse context judgments with stable full-track classifier evidence.

    Context consensus remains an independent admission path. Stable evidence
    over the full base track can rescue insufficient or mistaken context;
    a single confident frame is never enough. Both evidence states and the
    final reason are emitted for audit.
    """
    grouped = collections.defaultdict(list)
    track_maps = track_maps or {}
    for record in records:
        rec, raw_tid = record["rec"], str(record["tid"])
        canonical_tid = str(track_maps.get(rec, {}).get(raw_tid, raw_tid))
        grouped[(rec, canonical_tid)].append(record)

    out = []
    for (rec, tid), rr in sorted(grouped.items()):
        ps = [float(r["p"]) for r in rr]
        median = statistics.median(ps)
        positive_frac = sum(p >= admit for p in ps) / len(ps)
        track_medians = [float(r["base_track_p_median"])
                         for r in rr if r.get("base_track_p_median") not in
                         (None, "")]
        track_strong_fracs = [float(r["base_track_strong_frac"])
                              for r in rr
                              if r.get("base_track_strong_frac") not in
                              (None, "")]
        base_ps = [float(r["base_p"]) for r in rr
                   if r.get("base_p") not in (None, "")]
        base_median = (statistics.median(track_medians) if track_medians
                       else statistics.median(base_ps) if base_ps
                       else float("nan"))
        base_strong_frac = (
            statistics.median(track_strong_fracs) if track_strong_fracs
            else (sum(p >= base_strong for p in base_ps) / len(base_ps)
                  if base_ps else float("nan")))
        context_sufficient = len(ps) >= min_views
        context_owner = (context_sufficient and median >= admit
                         and positive_frac >= min_positive_frac)
        context_other = context_sufficient and median <= reject
        context_state = ("insufficient" if not context_sufficient
                         else "owner" if context_owner
                         else "other" if context_other
                         else "conflict")
        base_consensus = (
            base_median >= base_rescue
            and base_strong_frac >= min_base_strong_frac)
        base_state = ("owner" if base_consensus else
                      "missing" if not math.isfinite(base_median)
                      else "not_owner")
        if context_owner:
            verdict = "owner"
            reason = "context_consensus"
        elif base_consensus:
            verdict = "owner"
            if not context_sufficient:
                reason = "too_few_base_track_consensus"
            elif context_other:
                reason = "base_track_overrides_context_reject"
            else:
                reason = "base_track_consensus"
        elif not context_sufficient:
            verdict = "unsure"
            reason = "too_few_context_views"
        elif context_other:
            verdict = "other"
            reason = "context_reject"
        else:
            verdict = "unsure"
            reason = "conflicting_evidence"
        out.append({
            "rec": rec,
            "tid": tid,
            "verdict": verdict,
            "reason": reason,
            "context_state": context_state,
            "base_state": base_state,
            "p_median": round(median, 6),
            "p_min": round(min(ps), 6),
            "p_max": round(max(ps), 6),
            "positive_frac": round(positive_frac, 6),
            "base_p_median": round(base_median, 6),
            "base_strong_frac": round(base_strong_frac, 6),
            "n": len(ps),
        })
    return out


def write_gate_files(records, out_dir, track_maps=None, **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    by_rec = collections.defaultdict(list)
    track_maps = track_maps or {}
    for row in aggregate(records, track_maps=track_maps, **kwargs):
        rec, canonical_tid = row["rec"], row["tid"]
        members = sorted(
            (raw_tid for raw_tid, canonical in track_maps.get(rec, {}).items()
             if str(canonical) == canonical_tid),
            key=lambda tid: (not str(tid).isdigit(),
                             int(tid) if str(tid).isdigit() else str(tid)))
        for raw_tid in members or [canonical_tid]:
            by_rec[rec].append({**row, "tid": raw_tid,
                                "canonical_tid": canonical_tid})
    fields = ["rec", "tid", "canonical_tid", "verdict", "reason",
              "context_state", "base_state",
              "p_median", "p_min",
              "p_max", "positive_frac", "base_p_median",
              "base_strong_frac", "n"]
    for rec, rows in by_rec.items():
        path = os.path.join(out_dir, f"{rec}.owner_gate.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {rec}: {len(rows)} 条语义轨迹 -> {path}")


def prep(args):
    import cv2
    from src.rig.seam_fix import RawCameraReader

    jobs = read_jobs(args.jobs)
    wanted_recs = set(args.rec or jobs)
    os.makedirs(args.views, exist_ok=True)
    manifest = []

    for rec in sorted(wanted_recs):
        if rec not in jobs:
            raise KeyError(f"{rec}: not present in {args.jobs}")
        path = os.path.join(args.arm, f"{rec}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        tracks = (all_tracks(rows, args.min_track_frames)
                  if getattr(args, "all_tracks", False)
                  else contested_tracks(rows))
        all_track_rows = collections.defaultdict(list)
        for row in rows:
            if (str(row.get("not_hand", "0")) != "1"
                    and row.get("tid") not in (None, "")):
                all_track_rows[str(row["tid"])].append(row)
        by_tid_frame = {}
        for row in rows:
            key = (str(row.get("tid")), int(row["frame"]))
            if key not in by_tid_frame or float(row.get("conf") or 0) > float(
                    by_tid_frame[key].get("conf") or 0):
                by_tid_frame[key] = row
        selected = []
        for tid, track_rows in sorted(tracks.items(), key=lambda x: int(x[0])):
            selected.extend((tid, row) for row in representative_rows(
                track_rows, args.frames_per_track))
        if not selected:
            print(f"  {rec}: 没有可送的轨迹"
                  if getattr(args, "all_tracks", False)
                  else f"  {rec}: 没有第二只 owner 候选")
            continue

        bag, _start, _n = jobs[rec]
        videos = {name: os.path.join(bag, f"{name}.mp4")
                  for name in ("cam12", "cam34", "cam56")}
        frames = sorted({context_frame
                         for _tid, row in selected
                         for context_frame in context_frame_numbers(
                             int(row["frame"]), args.source_fps,
                             args.context_seconds)})
        reader = RawCameraReader(videos, "cam3", frames[0])
        current = frames[0] - 1
        cache = {}
        for frame in frames:
            image = None
            while current < frame:
                image = reader.next()
                current += 1
                if image is None:
                    break
            if image is None:
                break
            cache[frame] = image.copy()
        reader.close()

        for tid, row in selected:
            frame = int(row["frame"])
            image = cache.get(frame)
            if image is None:
                continue
            height, width = image.shape[:2]
            box = [int(float(row[key])) for key in ("x0", "y0", "x1", "y1")]
            x0, y0, x1, y1 = box
            stem = f"{rec}__tid{tid}__f{frame}"
            full_path = os.path.join(args.views, stem + "_full.jpg")
            context_path = os.path.join(args.views, stem + "_context.jpg")
            crop_path = os.path.join(args.views, stem + "_crop.jpg")

            full = image.copy()
            cv2.rectangle(full, (x0, y0), (x1, y1), (0, 230, 0), 4)
            scale = 1280 / float(width)
            cv2.imwrite(full_path,
                        cv2.resize(full, (1280, int(height * scale))),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            context_frames = context_frame_numbers(
                frame, args.source_fps, args.context_seconds)
            context_images, context_boxes = [], []
            for context_frame in context_frames:
                context_image = cache.get(context_frame)
                if context_image is None:
                    break
                context_images.append(context_image)
                context_row = by_tid_frame.get((str(tid), context_frame))
                context_boxes.append(
                    [int(float(context_row[key]))
                     for key in ("x0", "y0", "x1", "y1")]
                    if context_row is not None else None)
            if len(context_images) != 4:
                continue
            labels = [f"-{args.context_seconds:.1f}s",
                      f"-{args.context_seconds / 2.0:.1f}s",
                      f"-{args.context_seconds / 4.0:.1f}s", "current"]
            board = _storyboard(context_images, context_boxes, labels)
            cv2.imwrite(context_path, board,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            side = max(1, int(max(x1 - x0, y1 - y0) * 1.6))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            left, top = max(0, cx - side // 2), max(0, cy - side // 2)
            right, bottom = min(width, cx + side // 2), min(height, cy + side // 2)
            crop = image[top:bottom, left:right]
            if not crop.size:
                continue
            cv2.imwrite(crop_path, cv2.resize(crop, (256, 256)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            track_ps = [float(r["p"]) for r in all_track_rows[str(tid)]]
            manifest.append({
                "id": stem,
                "rec": rec,
                "tid": tid,
                "frame": frame,
                "base_p": row["p"],
                "owner_rank": row["owner_rank"],
                "full": full_path,
                "context": context_path,
                "crop": crop_path,
                "source_fps": args.source_fps,
                "context_seconds": args.context_seconds,
                "base_track_p_median": statistics.median(track_ps),
                "base_track_strong_frac": (
                    sum(p >= args.base_strong for p in track_ps)
                    / len(track_ps)),
            })
        print(f"  {rec}: {len(tracks)} 条候选轨迹，{len(selected)} 个视图")

    path = os.path.join(args.views, MANIFEST)
    fields = ["id", "rec", "tid", "frame", "base_p", "owner_rank",
              "full", "context", "crop", "source_fps", "context_seconds",
              "base_track_p_median", "base_track_strong_frac"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"-> {path} ({len(manifest)} 个视图)")


def score(args):
    from PIL import Image
    from src.semhand.qwen import ANSWER_PREFIX, Qwen, parse_desc, sem_text

    manifest_path = os.path.join(args.views, MANIFEST)
    items = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record
    # A prep pass can add track-level evidence without paying for Qwen again.
    # Merge those manifest fields into resumed semantic records in memory.
    for item in items:
        if item["id"] in done:
            done[item["id"]] = {**item, **done[item["id"]]}
    todo = [item for item in items if item["id"] not in done]
    if args.nshard > 1:
        # 分片按 id 的稳定哈希，不按顺序：顺序分片会让一条轨迹的五个视图落在
        # 同一个 worker 上，一个 worker 挂掉就整条轨迹没有分数，而聚合需要
        # min_views 个视图才出裁决。
        import hashlib
        todo = [i for i in todo
                if int(hashlib.sha1(i["id"].encode()).hexdigest(), 16)
                % args.nshard == args.shard]
        print(f"分片 {args.shard}/{args.nshard}")
    print(f"待判 {len(items)} 个视图；已有 {len(done)}，本次 {len(todo)}")
    if args.nshard > 1 and not todo and len(done) < len(items):
        # 空的分片几乎总是参数没传进来，而不是真的没活干。让它响，
        # 否则下游会拿到一个「跑过但没产出」的文件，和「跑完确实没发现」
        # 长得一模一样。
        raise SystemExit(f"分片 {args.shard}/{args.nshard} 没有分到任何视图，"
                         f"但 {len(items) - len(done)} 个还没判——检查参数")

    track_maps = {}
    if args.track_maps:
        from src.rig.track_fusion import load_track_map
        for path in glob.glob(os.path.join(args.track_maps,
                                           "*.track_map.csv")):
            rec = os.path.basename(path).split(".track_map.csv")[0]
            track_maps[rec] = load_track_map(path)

    if not todo:
        write_gate_files(
            done.values(), args.gate_dir, admit=args.admit,
            reject=args.reject,
            min_positive_frac=args.min_positive_frac,
            min_views=args.min_views, base_rescue=args.base_rescue,
            base_strong=args.base_strong,
            min_base_strong_frac=args.min_base_strong_frac,
            track_maps=track_maps)
        return

    qwen = Qwen(args.model, tiny=args.tiny, answer_prefix=ANSWER_PREFIX,
                question=CONTEXT_QUESTION)
    with open(args.out, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            pairs = [(Image.open(item.get("context") or item["full"]).convert("RGB"),
                      Image.open(item["crop"]).convert("RGB"))
                     for item in chunk]
            describe_convs = [[{"role": "user", "content": [
                {"type": "image", "image": full},
                {"type": "image", "image": crop},
                {"type": "text", "text": CONTEXT_DESCRIBE},
            ]}] for full, crop in pairs]
            descriptions = qwen.describe(describe_convs)

            for item, (full, crop), raw in zip(chunk, pairs, descriptions):
                parsed = parse_desc(raw)
                if parsed is None:
                    description = sem_text({"raw": raw, "parsed": parsed})
                else:
                    description = (
                        f"{parsed.get('hand_side', 'unclear')} hand; "
                        f"motion path: {parsed.get('motion_path', '')}; "
                        f"interaction: {parsed.get('interaction', '')}; "
                        f"object: {parsed.get('object', '')}; "
                        f"occlusion: {parsed.get('occlusion', '')}; "
                        "ambiguous: "
                        f"{parsed.get('ambiguous_regions', '')}")
                conv = [{"role": "user", "content": [
                    {"type": "text", "text": CONTEXT_INTRO},
                    {"type": "image", "image": full},
                    {"type": "image", "image": crop},
                    {"type": "text", "text": f"Description: {description}"},
                    {"type": "text", "text": CONTEXT_QUESTION},
                ]}]
                result = qwen.score(conv)
                record = {
                    **item,
                    "frame": int(item["frame"]),
                    "base_p": float(item["base_p"]),
                    "owner_rank": int(item["owner_rank"]),
                    "description": description,
                    "description_ok": parsed is not None,
                    **result,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                done[item["id"]] = record
                print(f"  {item['id']}: P(wearer)={result['p']:.3f}",
                      flush=True)

    write_gate_files(
        done.values(), args.gate_dir, admit=args.admit, reject=args.reject,
        min_positive_frac=args.min_positive_frac, min_views=args.min_views,
        base_rescue=args.base_rescue, base_strong=args.base_strong,
        min_base_strong_frac=args.min_base_strong_frac,
        track_maps=track_maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="/workspace/ten_ship_capfix",
                        help="directory containing final per-hand CSV files")
    parser.add_argument("--jobs", default="/workspace/cam3_jobs10.txt")
    parser.add_argument("--views", default="/workspace/owner_gate_views")
    parser.add_argument("--out", default="/workspace/owner_gate_scores.jsonl")
    parser.add_argument("--gate_dir", default="/workspace/owner_gates")
    parser.add_argument("--track_maps", default="",
                        help="directory of raw-to-canonical track maps")
    parser.add_argument("--model", default="/shared/datasets/public_model/Qwen3.8-27B")
    parser.add_argument("--rec", action="append")
    parser.add_argument("--frames_per_track", type=int, default=5)
    parser.add_argument("--source_fps", type=float, default=30.0)
    parser.add_argument("--context_seconds", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--admit", type=float, default=0.60)
    parser.add_argument("--reject", type=float, default=0.40)
    parser.add_argument("--min_positive_frac", type=float, default=0.60)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--base_rescue", type=float, default=0.95)
    parser.add_argument("--base_strong", type=float, default=0.90)
    parser.add_argument("--min_base_strong_frac", type=float, default=0.80)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--prep", action="store_true")
    # 建教师语料用：把检测器留下的每条轨迹都送进去，而不是只送有争议的 owner
    parser.add_argument("--all-tracks", dest="all_tracks", action="store_true",
                        help="送全部轨迹（建语料用），不只是有争议的 owner 候选")
    parser.add_argument("--min-track-frames", dest="min_track_frames",
                        type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.reject <= args.admit <= 1:
        parser.error("require 0 <= reject <= admit <= 1")
    if not (0 <= args.base_strong <= args.base_rescue <= 1
            and 0 <= args.min_base_strong_frac <= 1):
        parser.error("invalid base-track consensus thresholds")
    if (args.frames_per_track < 1 or args.min_views < 1
            or args.source_fps <= 0 or args.context_seconds <= 0):
        parser.error("frame counts must be positive")
    (prep if args.prep else score)(args)


if __name__ == "__main__":
    main()
