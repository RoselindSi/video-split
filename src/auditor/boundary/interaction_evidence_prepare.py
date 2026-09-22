"""Prepare anonymous development clips and recover measured feature caches."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.auditor.boundary.final_timeline_audit import atomic_json, file_sha256
from src.auditor.boundary.interaction_evidence_pilot import read_csv, unique_index, assert_development


def selected_events(packet):
    packet = Path(packet)
    review = unique_index([{**r, "event_id": r["review_id"]} for r in read_csv(packet / "review.csv")])
    keys = read_csv(packet / "private_key.csv")
    rows = [r for r in keys if r["review_id"] in review]
    if len(rows) != 150 or len({r["event_id"] for r in rows}) != 150:
        raise ValueError("Expected exactly 150 unique selected events")
    return rows


def video_index(paths):
    result = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()):
            p = Path(row["video"])
            if not p.is_file() and str(p).startswith("/shared/datasets/"):
                moved = Path("/shared/datasets/datasets") / p.relative_to("/shared/datasets")
                if moved.is_file():
                    p = moved
            rid = row["recording_id"]
            if rid in result and result[rid] != p:
                raise ValueError(f"Conflicting video paths for {rid}")
            result[rid] = p
    return result


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)]))
    s = data["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]),
            "duration": float(data["format"]["duration"])}


def clip_bounds(t, duration, context=15.0):
    if not all(math.isfinite(v) for v in (t, duration, context)) or not 0 <= t <= duration or context <= 0:
        raise ValueError("Invalid candidate time, duration or context")
    lo, hi = max(0.0, t - context), min(duration, t + context)
    return lo, hi, t - lo


def export_clips(rows, videos, out, packet, contract, workers=3):
    out.mkdir(parents=True, exist_ok=False)
    clips = out / "clips"
    clips.mkdir()
    metadata = {rid: probe(videos[rid]) for rid in {r["recording_id"] for r in rows}}
    records = []
    for r in rows:
        info = metadata[r["recording_id"]]
        if info["width"] != 1280 or info["height"] != 480:
            raise ValueError(f"Unverified packed-stereo shape: {r['recording_id']} {info}")
        lo, hi, offset = clip_bounds(float(r["candidate_time_s"]), info["duration"])
        records.append({"review_id": r["review_id"], "video": str(clips / (r["review_id"] + ".mp4")),
                        "candidate_offset_s": offset, "duration_s": hi - lo,
                        "context_before_s": offset, "context_after_s": hi - lo - offset,
                        "source": str(videos[r["recording_id"]]), "start_s": lo})
    public = [{k: v for k, v in r.items() if k not in ("source", "start_s")} for r in records]
    manifest = {"schema_version": "interaction_review_clips_v2", "n_events": len(rows),
                "packet_sha256": file_sha256(Path(packet) / "manifest.json"),
                "contract_sha256": file_sha256(contract), "eye": "left", "clips": public}

    def encode(r):
        target = Path(r["video"])
        temp = target.with_suffix(".part.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(r["start_s"]),
                        "-i", r["source"], "-t", str(r["duration_s"]), "-vf", "crop=640:480:0:0",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                        "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)],
                       check=True, timeout=180)
        checked = probe(temp)
        if checked["width"] != 640 or checked["height"] != 480 or abs(checked["duration"] - r["duration_s"]) > 0.3:
            raise ValueError(f"Clip verification failed: {r['review_id']}")
        temp.replace(target)
        return r["review_id"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(encode, r) for r in records]), 1):
            print(f"clips {i}/{len(records)} {future.result()}", flush=True)
    atomic_json(out / "clips_manifest.json", manifest)
    atomic_json(out / "private_clip_sources.json", {"sources": records})
    print(f"clips_ready {out / 'clips_manifest.json'}", flush=True)


def write_table(path, rows):
    names = sorted({k for r in rows for k in r})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def recover_hands(rows, raw_path, csv_path, out):
    import torch
    import numpy as np
    from src.boundary.hand_traj_features import all_features

    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    old = unique_index(read_csv(csv_path))
    expected = {"eye": "left", "window": [2.0, 2.0], "max_hands": 2,
                "upscale": "auto", "margin": 0.3, "max_interp_gap_frames": 2,
                "min_confidence": 0.3, "running_mode": "video"}
    recovered, absent, measured, discrepancies = [], [], [], []
    for r in rows:
        eid = r["event_id"]
        if eid not in raw:
            absent.append(r)
            continue
        entry = raw[eid]
        if entry["recording_id"] != r["recording_id"] or entry["config"] != expected or entry["fps"] != 10:
            raise ValueError(f"Raw cache identity/configuration mismatch: {eid}")
        if abs(entry["candidate_time"] - float(r["candidate_time_s"])) > 0.001:
            raise ValueError(f"Raw cache timestamp mismatch: {eid}")
        if entry["n_frames"] != len(entry["frames"]) or entry["n_frames"] != 41:
            raise ValueError(f"Incomplete raw frame window: {eid}")
        if not all(f.get("decode_success") and f.get("detector_success") for f in entry["frames"]):
            absent.append(r)
            continue
        feature = all_features(entry)
        if eid in old:
            for name, value in feature.items():
                previous = old[eid].get(name, "")
                previous = float(previous) if previous else float("nan")
                if (np.isfinite(previous) != np.isfinite(value) or
                    (np.isfinite(value) and not math.isclose(previous, value, rel_tol=1e-5, abs_tol=1e-6))):
                    discrepancies.append({"event_id": eid, "feature": name})
        if eid not in old:
            old[eid] = {"event_id": eid, "recording_id": r["recording_id"],
                        **{k: f"{v:.6f}" if np.isfinite(v) else "" for k, v in feature.items()}}
            recovered.append(eid)
        measured.append(eid)
    out.mkdir(parents=True, exist_ok=False)
    if discrepancies:
        atomic_json(out / "configuration_mismatch.json", {"discrepancies": discrepancies})
        raise ValueError("Recovered computation disagrees with existing feature rows; no merge published")
    write_table(out / "hand_trajectory_features.csv", list(old.values()))
    if absent:
        write_table(out / "needs_hand_extraction.csv", absent)
    atomic_json(out / "hand_recovery.json", {"selected": len(rows), "raw_verified": len(measured),
                "recovered_csv_rows": recovered, "needs_extraction": absent,
                "raw_sha256": file_sha256(raw_path), "old_csv_sha256": file_sha256(csv_path),
                "features_code_sha256": file_sha256(Path(__file__).parents[2] / "boundary/hand_traj_features.py"),
                "method": "safe_weights_only_raw_cache_recovery_no_label_filter", "configuration": expected})
    print(f"hand recovery {len(recovered)} recovered, {len(absent)} need extraction", flush=True)


def recover_visual(rows, csv_path, caches, out):
    import torch
    import numpy as np

    # Keep this recovery route independent of the legacy cache loader, which
    # predates torch's safe ``weights_only`` loading mode.  These descriptors
    # are pure NumPy transforms of already safely deserialised tensors.
    n_blocks = 5

    def change_trajectory(feats, times, candidate_time, win=2.0):
        mask = (times >= candidate_time - win) & (times <= candidate_time + win)
        if int(mask.sum()) < 6:
            return None, None
        selected = feats[mask].astype(np.float64)
        selected_times = times[mask]
        normalized = selected / np.maximum(
            np.linalg.norm(selected, axis=1, keepdims=True), 1e-9)
        change = 1.0 - (normalized[1:] * normalized[:-1]).sum(1)
        midpoint = (selected_times[1:] + selected_times[:-1]) / 2.0
        return midpoint - candidate_time, change

    def concentration_features(offsets, change):
        total = float(change.sum())
        if total <= 0:
            return None
        feature = {}
        for width in (0.25, 0.5, 1.0):
            fraction = float(change[np.abs(offsets) <= width].sum()) / total
            uniform = min(1.0, width / 2.0)
            feature[f"conc_{width}s"] = fraction
            feature[f"conc_{width}s_over_uniform"] = fraction / uniform
        peak = float(change.max())
        median = float(np.median(change))
        feature["peak_over_median"] = peak / median if median > 0 else np.nan
        feature["peak_width_s"] = (
            float((change > peak / 2).sum()) * float(np.median(np.diff(offsets)))
            if len(offsets) > 1 else np.nan)
        feature["peak_offset_s"] = float(offsets[int(np.argmax(change))])
        left, right = change[offsets < 0].sum(), change[offsets >= 0].sum()
        feature["asymmetry"] = float((right - left) / total)
        feature["side_change_left"] = float(left / max(1, (offsets < 0).sum()))
        feature["side_change_right"] = float(right / max(1, (offsets >= 0).sum()))
        feature["side_change_imbalance"] = abs(
            feature["side_change_left"] - feature["side_change_right"])
        return feature

    old = unique_index(read_csv(csv_path))
    pending = {r["event_id"]: r for r in rows if r["event_id"] not in old}
    provenance, diagnostics = [], []
    for cache in caches:
        if not pending:
            break
        records = torch.load(cache, weights_only=True, map_location="cpu", mmap=True)
        for record in records:
            matches = [r for r in pending.values() if r["recording_id"] == record.get("recording_id")]
            for r in matches:
                times = np.asarray(record["times"])
                feats = np.asarray(record["feats"])
                dt, d = change_trajectory(feats, times, float(r["candidate_time_s"]))
                if dt is None:
                    diagnostics.append({"event_id": r["event_id"], "cache": cache, "reason": "too_few_frames"})
                    continue
                feature = concentration_features(dt, d)
                if feature is None:
                    diagnostics.append({"event_id": r["event_id"], "cache": cache, "reason": "no_positive_change"})
                    continue
                old[r["event_id"]] = {"event_id": r["event_id"], "recording_id": r["recording_id"],
                                      **{k: f"{v:.6f}" if np.isfinite(v) else "" for k, v in feature.items()}}
                provenance.append({"event_id": r["event_id"], "cache": cache, "cache_bytes": Path(cache).stat().st_size,
                                   "window_samples": len(d) + 1, "block": "full", "n_blocks_definition": n_blocks})
                pending.pop(r["event_id"])
        del records
    out.mkdir(parents=True, exist_ok=False)
    write_table(out / "concentration_features_full.csv", list(old.values()))
    atomic_json(out / "visual_recovery.json", {"recovered": provenance, "remaining": list(pending),
                "diagnostics": diagnostics, "old_csv_sha256": file_sha256(csv_path)})
    print(f"visual recovery {len(provenance)} recovered, {len(pending)} missing", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["clips", "hands", "visual"])
    p.add_argument("--packet", required=True)
    p.add_argument("--final_manifest", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--data", action="append", default=[])
    p.add_argument("--raw_hands")
    p.add_argument("--hand_csv")
    p.add_argument("--visual_csv")
    p.add_argument("--visual_cache", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    packet = json.loads((Path(a.packet) / "manifest.json").read_text())
    if file_sha256(a.final_manifest) != packet["sources"]["final_test_manifest"]["sha256"]:
        raise ValueError("Final-test manifest hash mismatch")
    rows = selected_events(a.packet)
    forbidden = {r["recording_id"] for r in json.loads(Path(a.final_manifest).read_text())["selected"]}
    assert_development(rows, forbidden)
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if a.stage == "clips":
        export_clips(rows, video_index(a.data), out, a.packet, a.contract)
    elif a.stage == "hands":
        recover_hands(rows, a.raw_hands, a.hand_csv, out)
    else:
        recover_visual(rows, a.visual_csv, a.visual_cache, out)


if __name__ == "__main__":
    main()
