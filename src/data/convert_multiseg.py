"""Convert the hand-task dataset's segmentation_reference/*.segments.json into
the multi-segment training format consumed by the GRPO trainer (approach B).

Each output row = ONE video with the full ordered list of GT segments:

    {
      "video": "raw_videos/recording_0001.mp4",
      "duration": 25.7,
      "solution": [["Adjust and fold first outer box flap", 2.6, 15.3], ...],
      "reasons":  ["Initial occlusion clears ...", ...],   # for SFT think block
      "recording_id": "recording_0001"
    }

Usage (on the server, where the dataset + decord live):
    python -m src.data.convert_multiseg \
        --dataset_dir /workspace/datasets/task_segmentation_annotation_dataset_v1 \
        --out dataset/handtask/train_multiseg.json
"""

import argparse
import glob
import json
import os


def _video_duration(path, fallback):
    """Real duration via decord if available, else the fallback (max GT end)."""
    try:
        from decord import VideoReader  # lazy; only available on server

        vr = VideoReader(path)
        return len(vr) / vr.get_avg_fps()
    except Exception:
        return fallback


def build_rows(dataset_dir, read_duration=True):
    rows = []
    pattern = os.path.join(dataset_dir, "segmentation_reference", "*.segments.json")
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        rid = d["recording_id"]
        rel_video = os.path.join("raw_videos", rid + ".mp4")
        abs_video = os.path.join(dataset_dir, rel_video)

        segs, reasons = [], []
        for s in d["segments"]:
            segs.append(
                [
                    s["task_label"],
                    float(s["recording_relative_start_s"]),
                    float(s["recording_relative_end_s"]),
                ]
            )
            br = s.get("boundary_reason", {}) or {}
            reason = " ".join(x for x in [br.get("start"), br.get("end")] if x)
            reasons.append(reason or s.get("brief_description", ""))

        if not segs:
            continue
        max_end = max(e for _, _, e in segs)
        duration = _video_duration(abs_video, max_end) if read_duration else max_end

        rows.append(
            {
                "video": abs_video,  # absolute: Time-R1 loader does os.path.isfile()
                "duration": round(duration, 3),
                "solution": segs,
                "reasons": reasons,
                "recording_id": rid,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default="/workspace/datasets/task_segmentation_annotation_dataset_v1",
    )
    ap.add_argument("--out", default="dataset/handtask/train_multiseg.json")
    ap.add_argument(
        "--no-read-duration",
        action="store_true",
        help="skip decord; use max GT end as duration",
    )
    args = ap.parse_args()

    rows = build_rows(args.dataset_dir, read_duration=not args.no_read_duration)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=2)

    n_seg = sum(len(r["solution"]) for r in rows)
    print(f"Wrote {len(rows)} videos / {n_seg} segments -> {args.out}")
    if rows:
        print("example[0]:", json.dumps(rows[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
