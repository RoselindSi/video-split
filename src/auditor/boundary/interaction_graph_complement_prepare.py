"""Select and render a reproducible hand-mosaic complement batch.

The selector runs where the small mosaic annotation files are available.  The
renderer is intentionally standalone so the same file can be copied to the
data machine and run without installing this repository.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


SELECTION_SCHEMA = "interaction_graph_mosaic_complement_selection_v1"
RENDER_SCHEMA = "interaction_graph_mosaic_complement_render_v1"
MANIFEST_SCHEMA = "interaction_graph_timeline_manifest_v1"
STRATA = ("low", "medium", "high")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def keyframe_stratum(count: int) -> str:
    if count <= 2:
        return "low"
    if count <= 8:
        return "medium"
    return "high"


def annotation_keyframe_count(path: str | Path) -> int:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    eyes = value.get("eyes") or {}
    if not isinstance(eyes, dict):
        raise ValueError(f"mosaic annotation has invalid eyes: {path}")
    return sum(
        len(eye.get("keyframes") or [])
        for eye in eyes.values()
        if isinstance(eye, dict)
    )


def _recording_date(databag_id: str) -> str:
    match = re.search(r"(?:^|_)(\d{4})(?:_|$)", databag_id)
    if not match:
        raise ValueError(f"cannot read MMDD date from databag id: {databag_id}")
    return match.group(1)


def _source_group(source_video: str) -> str:
    source = Path(source_video)
    if len(source.parents) < 2:
        raise ValueError(f"source path has no databag parent: {source_video}")
    return source.parent.parent.name


def _stable_rank(seed: str, row: dict) -> str:
    identity = f"{seed}:{row['databag_id']}:{row['recording_id']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _targets(total: int) -> dict[str, int]:
    if total < len(STRATA):
        raise ValueError("sample size must be at least three")
    quotient, remainder = divmod(total, len(STRATA))
    return {
        name: quotient + (1 if index < remainder else 0)
        for index, name in enumerate(STRATA)
    }


def _excluded_databags(paths: list[str | Path]) -> tuple[set[str], list[dict]]:
    excluded: set[str] = set()
    provenance = []
    for name in paths:
        path = Path(name).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("clips") or value.get("videos") or []
        before = len(excluded)
        for row in rows:
            databag_id = row.get("databag_id")
            if not databag_id and isinstance(row.get("provenance"), dict):
                databag_id = row["provenance"].get("databag_id")
            if databag_id:
                excluded.add(str(databag_id))
        provenance.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "n_new_exclusions": len(excluded) - before,
        })
    return excluded, provenance


def select_complement(catalog: str | Path, out: str | Path, sample_size: int = 30,
                      context_s: float = 120.0,
                      seed: str = "mosaic-complement-v1",
                      exclude_selections: list[str | Path] | None = None,
                      video_prefix: str = "mosaic_graph") -> dict:
    catalog_path = Path(catalog).resolve()
    value = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = value.get("databags") or []
    if not rows:
        raise ValueError("catalog has no databags")
    if not math.isfinite(context_s) or context_s <= 0:
        raise ValueError("context must be positive")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", video_prefix):
        raise ValueError("video_prefix contains unsupported characters")
    excluded, exclusion_provenance = _excluded_databags(exclude_selections or [])

    enriched = []
    for row in rows:
        if str(row.get("databag_id") or "") in excluded:
            continue
        annotation = Path(row["mosaic_annotation"])
        count = annotation_keyframe_count(annotation)
        fps = float(row["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid fps for {row['databag_id']}")
        enriched.append({
            **row,
            "source_group": _source_group(str(row["source_video"])),
            "recording_date": _recording_date(str(row["databag_id"])),
            "keyframe_count": count,
            "keyframe_stratum": keyframe_stratum(count),
            "mosaic_annotation_sha256": file_sha256(annotation),
            "anchor_s": round(int(row["clip_start_frame"]) / fps + 15.0, 6),
            "requested_context_s": float(context_s),
        })

    targets = _targets(sample_size)
    available = Counter(row["keyframe_stratum"] for row in enriched)
    for name, target in targets.items():
        if available[name] < target:
            raise ValueError(
                f"stratum {name} has {available[name]} rows but needs {target}"
            )

    # Select the rare stratum first so source-group uniqueness is not consumed
    # by the plentiful low-motion examples.
    chosen = []
    used_groups: set[str] = set()
    date_counts: Counter[str] = Counter()
    for name in ("high", "medium", "low"):
        pool = [row for row in enriched if row["keyframe_stratum"] == name]
        for _ in range(targets[name]):
            eligible = [row for row in pool if row["source_group"] not in used_groups]
            if not eligible:
                eligible = pool
            pick = min(
                eligible,
                key=lambda row: (
                    date_counts[row["recording_date"]] >= 2,
                    date_counts[row["recording_date"]],
                    _stable_rank(seed, row),
                ),
            )
            pool.remove(pick)
            chosen.append(pick)
            used_groups.add(pick["source_group"])
            date_counts[pick["recording_date"]] += 1

    chosen.sort(key=lambda row: (row["recording_date"], row["recording_id"]))
    clips = []
    for index, row in enumerate(chosen, 1):
        clean = dict(row)
        clean.pop("source_exists_here", None)
        clean["video_id"] = f"{video_prefix}_{index:04d}"
        clips.append(clean)

    result = {
        "schema_version": SELECTION_SCHEMA,
        "selection_seed": seed,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": file_sha256(catalog_path),
        "sample_size": len(clips),
        "requested_context_s": float(context_s),
        "excluded_databag_count": len(excluded),
        "exclusion_sources": exclusion_provenance,
        "video_prefix": video_prefix,
        "sampling_policy": {
            "targets_by_keyframe_stratum": targets,
            "strata": {"low": "0-2", "medium": "3-8", "high": "9+"},
            "prefer_unique_source_group": True,
            "prefer_at_most_two_per_date": True,
            "anchor": "mosaic 30-second clip midpoint",
        },
        "realized": {
            "by_keyframe_stratum": dict(sorted(Counter(
                row["keyframe_stratum"] for row in clips).items())),
            "by_recording_date": dict(sorted(Counter(
                row["recording_date"] for row in clips).items())),
            "unique_source_groups": len({row["source_group"] for row in clips}),
        },
        "clips": clips,
    }
    atomic_json(out, result)
    return result


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> dict:
    output = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", str(path),
    ], text=True, timeout=120)
    value = json.loads(output)
    stream = (value.get("streams") or [{}])[0]
    duration = float((value.get("format") or {}).get("duration") or 0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"cannot probe video: {path}")
    return {"duration_s": duration, "width": width, "height": height}


def clip_bounds(anchor_s: float, source_duration_s: float,
                requested_context_s: float) -> tuple[float, float]:
    duration = min(float(requested_context_s), float(source_duration_s))
    start = max(0.0, float(anchor_s) - duration / 2.0)
    start = min(start, max(0.0, float(source_duration_s) - duration))
    return round(start, 6), round(duration, 6)


def _render_one(row: dict, out_dir: Path, ffmpeg: str, ffprobe: str,
                width: int, crf: int) -> dict:
    source = Path(row["source_video"])
    if not source.is_file():
        raise FileNotFoundError(f"source video is missing: {source}")
    source_info = probe_video(source, ffprobe)
    start_s, duration_s = clip_bounds(
        row["anchor_s"], source_info["duration_s"], row["requested_context_s"])
    target = out_dir / "videos" / f"{row['video_id']}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.part.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_s), "-i", str(source), "-t", str(duration_s),
        "-map", "0:v:0", "-an",
        "-vf", f"crop=iw/2:ih:0:0,scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(target)
    rendered = probe_video(target, ffprobe)
    return {
        "video_id": row["video_id"],
        "output_video": str(target.resolve()),
        "output_sha256": file_sha256(target),
        "output_duration_s": round(rendered["duration_s"], 3),
        "output_width": rendered["width"],
        "output_height": rendered["height"],
        "source_video": str(source),
        "source_duration_s": round(source_info["duration_s"], 3),
        "source_width": source_info["width"],
        "source_height": source_info["height"],
        "source_start_s": start_s,
        "requested_duration_s": duration_s,
        "anchor_s": row["anchor_s"],
        "databag_id": row["databag_id"],
        "recording_id": row["recording_id"],
        "keyframe_count": row["keyframe_count"],
        "keyframe_stratum": row["keyframe_stratum"],
    }


def render_selection(selection: str | Path, out_dir: str | Path,
                     workers: int = 2, ffmpeg: str = "ffmpeg",
                     ffprobe: str = "ffprobe", width: int = 640,
                     crf: int = 28) -> dict:
    selection_path = Path(selection).resolve()
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError(f"unexpected selection schema: {value.get('schema_version')}")
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = value["clips"]
    rendered = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_render_one, row, root, ffmpeg, ffprobe, width, crf): row
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            rendered.append(item)
            print(f"rendered {len(rendered)}/{len(rows)} {item['video_id']}", flush=True)
    rendered.sort(key=lambda row: row["video_id"])
    result = {
        "schema_version": RENDER_SCHEMA,
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "view": "left eye from side-by-side cam34",
        "output_width": width,
        "crf": crf,
        "n_videos": len(rendered),
        "videos": rendered,
    }
    atomic_json(root / "render_manifest.json", result)
    return result


def build_manifest(selection: str | Path, render_manifest: str | Path,
                   video_dir: str | Path, out: str | Path,
                   source_pool: str = "hand_mosaic_complement_v1") -> dict:
    selection_path = Path(selection).resolve()
    render_path = Path(render_manifest).resolve()
    root = Path(video_dir).resolve()
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    rendered = json.loads(render_path.read_text(encoding="utf-8"))
    if selected.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unexpected selection schema")
    if rendered.get("schema_version") != RENDER_SCHEMA:
        raise ValueError("unexpected render schema")
    render_by_id = {row["video_id"]: row for row in rendered["videos"]}
    videos = []
    for row in selected["clips"]:
        video = root / f"{row['video_id']}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"rendered clip is missing: {video}")
        render_row = render_by_id[row["video_id"]]
        videos.append({
            "video_id": row["video_id"],
            "video": str(video),
            "duration_s": float(render_row["output_duration_s"]),
            "source_pool": source_pool,
            "source_ref": row["databag_id"],
            "video_sha256": file_sha256(video),
            "provenance": {
                "recording_id": row["recording_id"],
                "databag_id": row["databag_id"],
                "source_group": row["source_group"],
                "source_video": row["source_video"],
                "source_start_s": render_row["source_start_s"],
                "anchor_s": row["anchor_s"],
                "clip_start_frame": row["clip_start_frame"],
                "fps": row["fps"],
                "keyframe_count": row["keyframe_count"],
                "keyframe_stratum": row["keyframe_stratum"],
                "mosaic_annotation": row["mosaic_annotation"],
                "mosaic_annotation_sha256": row["mosaic_annotation_sha256"],
            },
        })
    result = {
        "schema_version": MANIFEST_SCHEMA,
        "n_videos": len(videos),
        "source_pool": source_pool,
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "render_manifest": str(render_path),
        "render_manifest_sha256": file_sha256(render_path),
        "video_root": str(root),
        "videos": videos,
    }
    atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--catalog", required=True)
    select.add_argument("--out", required=True)
    select.add_argument("--sample-size", type=int, default=30)
    select.add_argument("--context-s", type=float, default=120.0)
    select.add_argument("--seed", default="mosaic-complement-v1")
    select.add_argument("--exclude-selection", action="append", default=[])
    select.add_argument("--video-prefix", default="mosaic_graph")

    render = subparsers.add_parser("render")
    render.add_argument("--selection", required=True)
    render.add_argument("--out-dir", required=True)
    render.add_argument("--workers", type=int, default=2)
    render.add_argument("--ffmpeg", default="ffmpeg")
    render.add_argument("--ffprobe", default="ffprobe")
    render.add_argument("--width", type=int, default=640)
    render.add_argument("--crf", type=int, default=28)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--selection", required=True)
    manifest.add_argument("--render-manifest", required=True)
    manifest.add_argument("--video-dir", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--source-pool", default="hand_mosaic_complement_v1")

    args = parser.parse_args()
    if args.command == "select":
        result = select_complement(
            args.catalog, args.out, args.sample_size, args.context_s, args.seed,
            args.exclude_selection, args.video_prefix)
        print(f"selection_ready {args.out} ({result['sample_size']} clips)")
    elif args.command == "render":
        result = render_selection(
            args.selection, args.out_dir, args.workers, args.ffmpeg,
            args.ffprobe, args.width, args.crf)
        print(f"render_ready {args.out_dir} ({result['n_videos']} videos)")
    else:
        result = build_manifest(
            args.selection, args.render_manifest, args.video_dir, args.out,
            args.source_pool)
        print(f"manifest_ready {args.out} ({result['n_videos']} videos)")


if __name__ == "__main__":
    main()
