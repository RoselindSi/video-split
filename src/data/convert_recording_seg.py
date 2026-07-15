"""Convert human_ego_recording_segmentation_10fps into our training format.

Reads each recording's segments.json (schema human_ego_recording_segments_v1) and
emits rows compatible with the boundary-head pipeline:

    {"video": ".../mid.mp4", "duration": <max end_s>,
     "solution": [[label_en, start_s, end_s], ...], "recording_id": "..."}

part_01 = dev/train snapshot (recordings 1-220); part_02 (test, recordings
221-471) is released later. Until then we hold out a slice of part_01 for val.
The read-only dataset mount is never written to; output json goes under /workspace.

Usage (server):
    python -m src.data.convert_recording_seg \
        --root /shared/datasets/human_ego_recording_segmentation_10fps_r01_part_01 \
        --out_dir /workspace/tr1/data_recseg --n_val 30
"""
import argparse, glob, json, os, random


def load_rows(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "recordings", "*", "segments.json"))):
        d = json.load(open(f))
        segs = d.get("segments", [])
        if not segs:
            continue
        rec_dir = os.path.dirname(f)
        video = os.path.join(rec_dir, "mid.mp4")
        sol = [[s["label_en"], float(s["start_s"]), float(s["end_s"])] for s in segs]
        rows.append({
            "video": video,
            "duration": max(s["end_s"] for s in segs),
            "solution": sol,
            "recording_id": d.get("recording_id", os.path.basename(rec_dir)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = load_rows(a.root)
    random.Random(a.seed).shuffle(rows)
    val, train = rows[:a.n_val], rows[a.n_val:]

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(train, open(os.path.join(a.out_dir, "recseg_train.json"), "w"),
              ensure_ascii=False)
    json.dump(val, open(os.path.join(a.out_dir, "recseg_val.json"), "w"),
              ensure_ascii=False)

    def stat(rs):
        segc = [len(r["solution"]) for r in rs]
        durs = [r["duration"] for r in rs]
        return (f"{len(rs)} recs, segs/rec min {min(segc)} mean {sum(segc)/len(segc):.0f} "
                f"max {max(segc)}, dur min {min(durs):.0f}s max {max(durs):.0f}s")
    print("train:", stat(train))
    print("val  :", stat(val))
    print("wrote ->", a.out_dir)


if __name__ == "__main__":
    main()
