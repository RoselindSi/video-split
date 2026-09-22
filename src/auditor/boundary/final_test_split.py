"""Freeze an untouched, natural recording-level final-test split.

Selection is deliberately label-blind. The sampler may inspect recording IDs,
directory names, ``mid.mp4`` existence/size, and optional ffprobe metadata. It
must never open ``segments.json`` or use an action name, boundary count, model
score, or candidate count. Those are all ways of selecting an easier test set
while still calling it random.

The rank is SHA256(salt, recording_id), so the same eligible population always
produces the same split regardless of filesystem order. Every exclusion source
is hashed into the manifest, and every selected recording carries its rank.

Typical use:

    python -m src.auditor.boundary.final_test_split \
        --exclude-source data/gold \
        --exclusions-out data/splits/final_test_v1_excluded_recordings.txt \
        --only-exclusions

    python final_test_split.py \
        --corpus-root /path/to/part_01/recordings \
        --corpus-root /path/to/part_02/recordings \
        --exclude-id-file final_test_v1_excluded_recordings.txt \
        --n-recordings 48 --ffprobe --out final_test_v1.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


RID_RE = re.compile(r"recording_[0-9]{6}")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
DEFAULT_SALT = "taxonomy-free-episode-boundary-final-v1-2026-09-12"
SCHEMA = "episode_boundary_final_test_split_v1"


def normalize_recording_id(value: object) -> str | None:
    text = str(value or "").strip()
    match = re.search(r"(?:recording[_-]?)?([0-9]+)$", text)
    if not match:
        return None
    return f"recording_{int(match.group(1)):06d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"exclusion source does not exist: {source}")
    return sorted(set(files), key=lambda p: str(p))


def _structured_recording_ids(path: Path) -> set[str]:
    """Read explicit recording_id columns/fields, including numeric IDs."""
    out: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if key and key.lstrip("\ufeff").strip() == "recording_id":
                        rid = normalize_recording_id(value)
                        if rid:
                            out.add(rid)
    elif path.suffix.lower() == ".json":
        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "recording_id":
                        rid = normalize_recording_id(item)
                        if rid:
                            out.add(rid)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        with path.open(encoding="utf-8-sig") as handle:
            walk(json.load(handle))
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict) and "recording_id" in value:
                    rid = normalize_recording_id(value["recording_id"])
                    if rid:
                        out.add(rid)
    return out


def collect_exclusions(sources: list[str], id_files: list[str]) -> tuple[set[str], dict]:
    excluded: set[str] = set()
    file_rows = []
    for path in _source_files(sources):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        raw = path.read_bytes()
        excluded.update(RID_RE.findall(raw.decode("utf-8", errors="ignore")))
        excluded.update(_structured_recording_ids(path))
        file_rows.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})

    explicit_rows = []
    for name in id_files:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"exclusion id file does not exist: {name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            rid = normalize_recording_id(line)
            if rid:
                excluded.add(rid)
        explicit_rows.append({"path": str(path), "sha256": _sha256(path)})

    aggregate = hashlib.sha256()
    for row in file_rows + explicit_rows:
        aggregate.update(f"{row['path']}\0{row['sha256']}\n".encode())
    provenance = {
        "scanned_sources": list(sources),
        "source_files": file_rows,
        "explicit_id_files": explicit_rows,
        "aggregate_sha256": aggregate.hexdigest(),
        "n_excluded_recordings": len(excluded),
    }
    return excluded, provenance


def _duration(path: Path) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=False, capture_output=True, text=True)
    try:
        return float(proc.stdout.strip()) if proc.returncode == 0 else None
    except ValueError:
        return None


def inventory(corpus_roots: list[str], use_ffprobe: bool) -> list[dict]:
    rows: dict[str, dict] = {}
    for root_name in corpus_roots:
        root = Path(root_name)
        if not root.is_dir():
            raise FileNotFoundError(f"corpus root does not exist: {root}")
        for directory in sorted(root.glob("recording_*")):
            if not directory.is_dir() or not RID_RE.fullmatch(directory.name):
                continue
            video = directory / "mid.mp4"
            if not video.is_file() or video.stat().st_size <= 0:
                continue
            if directory.name in rows:
                raise ValueError(f"duplicate recording across corpus roots: {directory.name}")
            row = {
                "recording_id": directory.name,
                "video": str(video),
                "video_bytes": video.stat().st_size,
            }
            if use_ffprobe:
                row["duration_s"] = _duration(video)
            rows[directory.name] = row
    return [rows[key] for key in sorted(rows)]


def selection_rank(salt: str, recording_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{recording_id}".encode()).hexdigest()


def build_manifest(corpus: list[dict], excluded: set[str], provenance: dict,
                   n_recordings: int, salt: str, min_duration_s: float) -> dict:
    eligible = []
    technical_rejects = []
    for row in corpus:
        rid = row["recording_id"]
        if rid in excluded:
            continue
        duration = row.get("duration_s")
        if "duration_s" in row and duration is None:
            technical_rejects.append({"recording_id": rid,
                                      "reason": "ffprobe_unreadable"})
            continue
        if duration is not None and duration < min_duration_s:
            technical_rejects.append({"recording_id": rid,
                                      "reason": "duration_below_minimum",
                                      "duration_s": duration})
            continue
        ranked = dict(row)
        ranked["selection_rank"] = selection_rank(salt, rid)
        eligible.append(ranked)
    eligible.sort(key=lambda row: row["selection_rank"])
    if len(eligible) < n_recordings:
        raise ValueError(f"requested {n_recordings} recordings but only "
                         f"{len(eligible)} are eligible")
    selected = eligible[:n_recordings]
    return {
        "schema_version": SCHEMA,
        "frozen": True,
        "selection_salt": salt,
        "selection_algorithm": "ascending sha256(salt + NUL + recording_id)",
        "label_blind_guarantee": (
            "Selection did not open segments.json and did not use action labels, "
            "boundaries, candidates, model outputs, or task categories."),
        "n_corpus_recordings": len(corpus),
        "n_excluded_recordings_present_in_corpus": len(
            {row["recording_id"] for row in corpus} & excluded),
        "n_eligible_recordings": len(eligible),
        "n_selected_recordings": len(selected),
        "minimum_duration_s": min_duration_s,
        "exclusion_provenance": provenance,
        "technical_rejects": technical_rejects,
        "selected": selected,
    }


def _write_json(path: str, value: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-root", action="append", default=[])
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--exclude-id-file", action="append", default=[])
    parser.add_argument("--exclusions-out")
    parser.add_argument("--only-exclusions", action="store_true")
    parser.add_argument("--n-recordings", type=int, default=48)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--min-duration-s", type=float, default=30.0)
    parser.add_argument("--ffprobe", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    excluded, provenance = collect_exclusions(args.exclude_source,
                                               args.exclude_id_file)
    print(f"excluded {len(excluded)} recording IDs")
    if args.exclusions_out:
        path = Path(args.exclusions_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.only_exclusions:
        return
    if not args.corpus_root or not args.out:
        parser.error("selection requires --corpus-root and --out")

    corpus = inventory(args.corpus_root, args.ffprobe)
    manifest = build_manifest(corpus, excluded, provenance, args.n_recordings,
                              args.salt, args.min_duration_s)
    _write_json(args.out, manifest)
    print(f"corpus {manifest['n_corpus_recordings']}, eligible "
          f"{manifest['n_eligible_recordings']}, selected "
          f"{manifest['n_selected_recordings']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
