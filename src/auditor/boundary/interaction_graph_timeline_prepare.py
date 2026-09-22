"""Prepare video manifests and hand-mosaic databag source catalogs.

Examples:

    python -m src.auditor.boundary.interaction_graph_timeline_prepare videos \
      --video-dir dist/interaction_graph_reviewer_2_offline_v2/videos \
      --source-pool repeated_actions \
      --out results/auditor/interaction_graph_timeline_demo/manifest.json

    python -m src.auditor.boundary.interaction_graph_timeline_prepare databags \
      --annotations-root ../Long_Video_Splitting/annotations \
      --out results/auditor/interaction_graph_timeline_databags/catalog.json

    python -m src.auditor.boundary.interaction_graph_timeline_prepare catalog-videos \
      --catalog results/auditor/interaction_graph_timeline_databags/catalog.json \
      --source-pool diverse_hand_mosaic_databags \
      --out results/auditor/interaction_graph_timeline_databags/manifest.json

The catalog command does not require the raw ``/shared`` mount.  It records the
source paths retained by the hand-mosaic annotation files so a manifest can be
prepared later on the data machine, where those videos are available.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from src.auditor.boundary.final_timeline_audit import atomic_json, file_sha256
from src.auditor.boundary.interaction_graph_timeline import MANIFEST_SCHEMA


CATALOG_SCHEMA = "interaction_graph_databag_catalog_v1"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def probe_duration(path: str | Path, ffprobe: str = "ffprobe") -> float:
    output = subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True, timeout=60).strip()
    duration = round(float(output), 3)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"video has no positive duration: {path}")
    return duration


def _video_id(path: Path, root: Path, used: set[str]) -> str:
    relative = path.relative_to(root).with_suffix("")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(relative.parts)).strip("_.-")
    base = (base or "video")[:120]
    candidate = base
    number = 2
    while candidate in used:
        suffix = f"_{number}"
        candidate = base[:128 - len(suffix)] + suffix
        number += 1
    used.add(candidate)
    return candidate


def _unique_id(value: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    base = (base or "video")[:120]
    candidate = base
    number = 2
    while candidate in used:
        suffix = f"_{number}"
        candidate = base[:128 - len(suffix)] + suffix
        number += 1
    used.add(candidate)
    return candidate


def prepare_video_manifest(video_dir: str | Path, out: str | Path,
                           source_pool: str, ffprobe: str = "ffprobe",
                           include_hashes: bool = False) -> dict:
    root = Path(video_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"video directory is missing: {root}")
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise ValueError(f"no videos found under: {root}")

    used: set[str] = set()
    rows = []
    for index, path in enumerate(paths, 1):
        row = {
            "video_id": _video_id(path, root, used),
            "video": str(path.resolve()),
            "duration_s": probe_duration(path, ffprobe),
            "source_pool": source_pool,
            "source_ref": str(path.relative_to(root)),
        }
        if include_hashes:
            row["video_sha256"] = file_sha256(path)
        rows.append(row)
        print(f"probed {index}/{len(paths)} {row['video_id']}", flush=True)

    value = {
        "schema_version": MANIFEST_SCHEMA,
        "n_videos": len(rows),
        "source_pool": source_pool,
        "video_root": str(root),
        "videos": rows,
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, value)
    return value


def _preferred_annotation(files: list[Path]) -> tuple[Path, str]:
    finals = sorted(path for path in files if "__final_" in path.name)
    if finals:
        return finals[-1], "final"
    drafts = sorted(path for path in files if path.name.endswith("__draft.json"))
    if drafts:
        return drafts[-1], "draft"
    raise ValueError("annotation directory has no final or draft JSON")


def build_databag_catalog(annotations_root: str | Path, out: str | Path) -> dict:
    root = Path(annotations_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"annotations directory is missing: {root}")

    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(directory.glob("*.json"))
        if not files:
            continue
        annotation, status = _preferred_annotation(files)
        value = json.loads(annotation.read_text(encoding="utf-8"))
        source = Path(str(value.get("source") or ""))
        recording = str(value.get("recording") or directory.name)
        if not source.name:
            raise ValueError(f"annotation has no source: {annotation}")
        rows.append({
            "recording_id": recording,
            "databag_id": directory.name,
            "source_video": str(source),
            "source_exists_here": source.is_file(),
            "clip_start_frame": int(value.get("clip_start_frame") or 0),
            "fps": float(value.get("fps") or 0),
            "mosaic_annotation": str(annotation),
            "mosaic_annotation_status": status,
        })

    if not rows:
        raise ValueError(f"no databag annotations found under: {root}")
    sources = {row["source_video"] for row in rows}
    if len(sources) != len(rows):
        raise ValueError("multiple annotation directories point to the same source video")
    value = {
        "schema_version": CATALOG_SCHEMA,
        "annotations_root": str(root),
        "n_databags": len(rows),
        "n_sources_available_here": sum(row["source_exists_here"] for row in rows),
        "databags": rows,
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, value)
    return value


def _mapped_source(source: str, path_maps: list[tuple[str, str]]) -> Path:
    candidates = [Path(source)]
    for old, new in path_maps:
        if source == old or source.startswith(old.rstrip("/") + "/"):
            suffix = source[len(old.rstrip("/")):].lstrip("/")
            candidates.insert(0, Path(new) / suffix)
    prefix = "/shared/datasets/"
    if source.startswith(prefix):
        candidates.append(Path("/shared/datasets/datasets") / source[len(prefix):])
    return next((path for path in candidates if path.is_file()), candidates[0])


def manifest_from_databag_catalog(catalog: str | Path, out: str | Path,
                                  source_pool: str, ffprobe: str = "ffprobe",
                                  include_hashes: bool = False, limit: int = 0,
                                  path_maps: list[tuple[str, str]] | None = None) -> dict:
    catalog_path = Path(catalog).resolve()
    value = json.loads(catalog_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError(f"unexpected catalog schema: {value.get('schema_version')!r}")
    if limit < 0:
        raise ValueError("limit must be zero or positive")

    used: set[str] = set()
    rows = []
    missing = []
    for item in value.get("databags", []):
        source = _mapped_source(str(item["source_video"]), path_maps or [])
        if not source.is_file():
            missing.append(str(item["source_video"]))
            continue
        row = {
            "video_id": _unique_id(str(item["recording_id"]), used),
            "video": str(source.resolve()),
            "duration_s": probe_duration(source, ffprobe),
            "source_pool": source_pool,
            "source_ref": str(item["databag_id"]),
            "mosaic_annotation": str(item["mosaic_annotation"]),
        }
        if include_hashes:
            row["video_sha256"] = file_sha256(source)
        rows.append(row)
        print(f"probed {len(rows)} {row['video_id']}", flush=True)
        if limit and len(rows) >= limit:
            break
    if not rows:
        example = missing[0] if missing else "(catalog is empty)"
        raise FileNotFoundError(f"no catalog source videos are available; first missing: {example}")

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "n_videos": len(rows),
        "source_pool": source_pool,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": file_sha256(catalog_path),
        "n_missing_sources_skipped": len(missing),
        "videos": rows,
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    videos = subparsers.add_parser("videos", help="build a playable video manifest")
    videos.add_argument("--video-dir", required=True)
    videos.add_argument("--source-pool", required=True)
    videos.add_argument("--out", required=True)
    videos.add_argument("--ffprobe", default="ffprobe")
    videos.add_argument("--include-hashes", action="store_true")

    databags = subparsers.add_parser(
        "databags", help="catalog source videos referenced by mosaic annotations")
    databags.add_argument("--annotations-root", required=True)
    databags.add_argument("--out", required=True)

    catalog_videos = subparsers.add_parser(
        "catalog-videos", help="build a manifest from catalog sources available here")
    catalog_videos.add_argument("--catalog", required=True)
    catalog_videos.add_argument("--source-pool", required=True)
    catalog_videos.add_argument("--out", required=True)
    catalog_videos.add_argument("--ffprobe", default="ffprobe")
    catalog_videos.add_argument("--include-hashes", action="store_true")
    catalog_videos.add_argument("--limit", type=int, default=0)
    catalog_videos.add_argument(
        "--path-map", action="append", default=[], metavar="OLD=NEW",
        help="rewrite a catalog path prefix when the data mount moved")

    args = parser.parse_args()
    if args.command == "videos":
        result = prepare_video_manifest(args.video_dir, args.out, args.source_pool,
                                        args.ffprobe, args.include_hashes)
        print(f"manifest_ready {args.out} ({result['n_videos']} videos)")
    elif args.command == "databags":
        result = build_databag_catalog(args.annotations_root, args.out)
        print(f"catalog_ready {args.out} ({result['n_databags']} databags, "
              f"{result['n_sources_available_here']} sources available here)")
    else:
        path_maps = []
        for item in args.path_map:
            if "=" not in item:
                parser.error("--path-map must be OLD=NEW")
            old, new = item.split("=", 1)
            if not old or not new:
                parser.error("--path-map must be OLD=NEW")
            path_maps.append((old.rstrip("/"), new.rstrip("/")))
        result = manifest_from_databag_catalog(
            args.catalog, args.out, args.source_pool, args.ffprobe,
            args.include_hashes, args.limit, path_maps)
        print(f"manifest_ready {args.out} ({result['n_videos']} videos, "
              f"{result['n_missing_sources_skipped']} missing sources skipped)")


if __name__ == "__main__":
    main()
