"""Prepare and score the independent 30-case graph-contract review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.auditor.boundary.final_timeline_audit import atomic_json, file_sha256


SEED = "interaction-graph-contract-v1-review30"
FIXED_CASES = {
    # Continuous-contact boundaries.
    "review_0093": "continuous_contact_boundary",
    "review_0015": "continuous_contact_boundary",
    "review_0344": "continuous_contact_boundary",
    "review_0209": "continuous_contact_boundary",
    # Releases that do not establish a boundary.
    "review_0163": "release_without_boundary",
    "review_0148": "release_without_boundary",
    "review_0225": "release_without_boundary",
    "review_0115": "release_without_boundary",
    # Same-type renewed episodes, including no-release examples.
    "review_0230": "self_loop",
    "review_0274": "self_loop",
    "review_0258": "self_loop",
    "review_0193": "self_loop",
    # Cross-type transitions with varied objects and transition shapes.
    "review_0355": "cross_node",
    "review_0305": "cross_node",
    "review_0089": "cross_node",
    "review_0124": "cross_node",
    # Within-episode motion and pauses.
    "review_0087": "continue_internal",
    "review_0007": "continue_internal",
    "review_0138": "continue_internal",
    "review_0378": "continue_internal",
    # Visibility failures.
    "review_0002": "uncertain",
    "review_0391": "uncertain",
    # Proposal-time offsets, kept distinct from the true transition.
    "review_0229": "proposal_offset",
    "review_0314": "proposal_offset",
    # Closed-loop and gradual-transition calibration.
    "review_0240": "closed_loop",
    "review_0046": "gradual_boundary",
}

REVIEW_FIELDS = [
    "review_id", "relation", "type_equivalence", "prior_episode_status",
    "next_episode_status", "return_to_prior_type", "instance_link",
    "observed_type_path", "boundary_start_s", "boundary_end_s",
    "transition_shape", "continuous_contact", "release_observed",
    "overlap_present", "visibility", "observable_evidence", "reviewer_id",
    "human_confirmed",
]

ALLOWED = {
    "relation": {"CONTINUE", "SAME_ACTION_NEW_INSTANCE", "NEW_ACTION",
                 "BOUNDARY_TYPE_UNRESOLVED", "UNOBSERVABLE", "INITIAL",
                 "TERMINAL", "END_ONLY"},
    "type_equivalence": {"same", "different", "unknown", "not_applicable"},
    "prior_episode_status": {"ongoing", "completed", "terminated", "suspended",
                             "coactive", "unknown", "not_applicable"},
    "next_episode_status": {"new", "resumed", "ongoing", "unknown", "not_applicable"},
    "return_to_prior_type": {"yes", "no", "unknown", "not_applicable"},
    "instance_link": {"same_instance", "new_instance", "resumed_instance",
                      "unknown", "not_applicable"},
    "transition_shape": {"sharp", "gradual", "overlap", "no_transition", "uncertain"},
    "continuous_contact": {"yes", "no", "uncertain"},
    "release_observed": {"yes", "no", "uncertain"},
    "overlap_present": {"yes", "no", "uncertain"},
    "visibility": {"visible", "partial", "not_visible"},
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path = Path(path)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_label(value):
    value = re.sub(r"\([^)]*\)", "", value.lower())
    value = re.sub(r"\b(first|second|third|fourth|one|two|three|another|new)\b", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", value))


def stable_rank(value):
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def recording_files(recording_id, roots):
    found = []
    for root in map(Path, roots):
        directory = root / recording_id
        if (directory / "mid.mp4").is_file() and (directory / "segments.json").is_file():
            found.append((directory / "mid.mp4", directory / "segments.json"))
    if len(found) != 1:
        raise ValueError(f"Expected one source for {recording_id}, found {len(found)}")
    return found[0]


def probe_duration(video):
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video)], text=True)
    duration = float(raw.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration for {video}")
    return duration


def contextual_bounds(candidate, segments, duration, radius=35.0, limit=150.0):
    if not 0 <= candidate <= duration:
        raise ValueError("Candidate outside recording")
    nearby = [s for s in segments
              if float(s["start_s"]) <= candidate + radius
              and float(s["end_s"]) >= candidate - radius]
    lo = min([candidate - radius, *[float(s["start_s"]) - 3.0 for s in nearby]])
    hi = max([candidate + radius, *[float(s["end_s"]) + 3.0 for s in nearby]])
    lo, hi = max(0.0, lo), min(duration, hi)
    if hi - lo > limit:
        lo, hi = max(0.0, candidate - limit / 2), min(duration, candidate + limit / 2)
    return lo, hi


def structural_candidates(recording_ids, roots, excluded_recordings):
    aba, overlap = [], []
    for recording_id in recording_ids:
        if recording_id in excluded_recordings:
            continue
        video, segments_path = recording_files(recording_id, roots)
        segments = sorted(json.loads(segments_path.read_text())["segments"],
                          key=lambda x: (float(x["start_s"]), float(x["end_s"])))
        for first, middle, last in zip(segments, segments[1:], segments[2:]):
            a, b, c = map(lambda x: normalized_label(x["label_en"]),
                          (first, middle, last))
            lo, hi = float(first["start_s"]), float(last["end_s"])
            ordered = (float(first["end_s"]) <= float(middle["start_s"]) + 0.5
                       and float(middle["end_s"]) <= float(last["start_s"]) + 0.5)
            if a and a == c and a != b and ordered and hi - lo <= 150:
                aba.append({"kind": "aba_candidate", "recording_id": recording_id,
                            "candidate_time_s": float(last["start_s"]), "clip_start_s": max(0.0, lo - 5),
                            "clip_end_s": hi + 5, "source_video": str(video),
                            "selection_labels": [first["label_en"], middle["label_en"], last["label_en"]]})
        for index, first in enumerate(segments):
            for second in segments[index + 1:]:
                if float(second["start_s"]) >= float(first["end_s"]):
                    break
                intersection = min(float(first["end_s"]), float(second["end_s"])) - max(
                    float(first["start_s"]), float(second["start_s"]))
                a, b = normalized_label(first["label_en"]), normalized_label(second["label_en"])
                if intersection >= 2.0 and a and b and a != b and a not in b and b not in a:
                    lo = max(0.0, min(float(first["start_s"]), float(second["start_s"])) - 10)
                    hi = max(float(first["end_s"]), float(second["end_s"])) + 10
                    if hi - lo <= 150:
                        overlap.append({"kind": "overlap_candidate", "recording_id": recording_id,
                                        "candidate_time_s": max(float(first["start_s"]), float(second["start_s"])),
                                        "clip_start_s": lo, "clip_end_s": hi, "source_video": str(video),
                                        "selection_labels": [first["label_en"], second["label_en"]]})
    def pick(rows, n, used):
        result = []
        for row in sorted(rows, key=lambda r: stable_rank(
                f"{r['recording_id']}|{r['candidate_time_s']}|{r['kind']}")):
            if row["recording_id"] not in used:
                result.append(row)
                used.add(row["recording_id"])
            if len(result) == n:
                return result
        raise ValueError(f"Only found {len(result)}/{n} independent {rows[0]['kind'] if rows else 'structural'} cases")
    used = set(excluded_recordings)
    return pick(aba, 2, used) + pick(overlap, 2, used)


def prepare(annotations_path, key_path, roots, contract, final_manifest_path, out, workers=3):
    annotation_rows = read_csv(annotations_path)
    if len(annotation_rows) != 150 or len({r["review_id"] for r in annotation_rows}) != 150:
        raise ValueError("Expected 150 unique completed development annotations")
    if any(r.get("status") != "final" for r in annotation_rows):
        raise ValueError("All 150 development annotations must be final")
    annotations = {r["review_id"]: r for r in annotation_rows}
    keys = {r["review_id"]: r for r in read_csv(key_path)}
    packet_manifest = json.loads((Path(key_path).parent / "manifest.json").read_text())
    expected_hash = packet_manifest["sources"]["final_test_manifest"]["sha256"]
    if file_sha256(final_manifest_path) != expected_hash:
        raise ValueError("Final-test manifest hash mismatch")
    forbidden = {r["recording_id"] for r in json.loads(
        Path(final_manifest_path).read_text())["selected"]}
    selected_ids = {keys[rid]["recording_id"] for rid in annotations}
    if selected_ids & forbidden:
        raise ValueError(f"Final-test overlap forbidden: {sorted(selected_ids & forbidden)}")
    if set(FIXED_CASES) - annotations.keys() or set(FIXED_CASES) - keys.keys():
        raise ValueError("Fixed review cases are missing from inputs")
    if any(annotations[r]["status"] != "final" for r in FIXED_CASES):
        raise ValueError("Every fixed case must be final")

    cases = []
    for review_id, kind in FIXED_CASES.items():
        key, old = keys[review_id], annotations[review_id]
        video, segments_path = recording_files(key["recording_id"], roots)
        segments = json.loads(segments_path.read_text())["segments"]
        duration = probe_duration(video)
        candidate = float(key["candidate_time_s"])
        lo, hi = contextual_bounds(candidate, segments, duration)
        cases.append({"kind": kind, "source_review_id": review_id,
                      "recording_id": key["recording_id"], "candidate_time_s": candidate,
                      "clip_start_s": lo, "clip_end_s": hi, "source_video": str(video),
                      "old_development_annotation": {k: old[k] for k in (
                          "relation", "transition_shape", "continuous_contact",
                          "release_observed", "visibility", "why_one_line")}})

    development_recordings = sorted({r["recording_id"] for r in keys.values()
                                     if r["recording_id"] not in forbidden})
    cases.extend(structural_candidates(development_recordings, roots,
                                       {r["recording_id"] for r in cases}))
    if len(cases) != 30:
        raise AssertionError("Review packet must contain exactly 30 cases")
    for row in cases:
        duration = probe_duration(row["source_video"])
        row["clip_start_s"] = max(0.0, row["clip_start_s"])
        row["clip_end_s"] = min(duration, row["clip_end_s"])
        if not row["clip_start_s"] <= row["candidate_time_s"] <= row["clip_end_s"]:
            raise ValueError(f"Candidate lies outside clip for {row['recording_id']}")
        if row["clip_end_s"] - row["clip_start_s"] < 10:
            raise ValueError(f"Insufficient context for {row['recording_id']}")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    clips = out / "clips"
    clips.mkdir()
    ordered = sorted(cases, key=lambda r: stable_rank(
        f"alias|{r['recording_id']}|{r['candidate_time_s']}|{r['kind']}"))
    for index, row in enumerate(ordered, 1):
        row["review_id"] = f"graph_review_{index:04d}"
        row["clip_duration_s"] = row["clip_end_s"] - row["clip_start_s"]
        row["candidate_offset_s"] = row["candidate_time_s"] - row["clip_start_s"]
        row["output_video"] = str(clips / f"{row['review_id']}.mp4")

    def encode(row):
        target = Path(row["output_video"])
        temporary = target.with_suffix(".part.mp4")
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(row["clip_start_s"]),
            "-i", row["source_video"], "-t", str(row["clip_duration_s"]),
            "-vf", "crop=640:480:0:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "24", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temporary)], check=True, timeout=600)
        actual = probe_duration(temporary)
        if abs(actual - row["clip_duration_s"]) > 0.5:
            raise ValueError(f"Clip duration mismatch for {row['review_id']}")
        temporary.replace(target)
        return row["review_id"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(encode, row) for row in ordered]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"clips {index}/30 {future.result()}", flush=True)

    public_cases = [{"review_id": r["review_id"],
                     "video": str(Path(r["output_video"]).resolve()),
                     "video_sha256": file_sha256(r["output_video"]),
                     "duration_s": r["clip_duration_s"],
                     "candidate_offset_s": r["candidate_offset_s"]} for r in ordered]
    public = {"schema_version": "interaction_graph_contract_review_v1", "n_cases": 30,
              "contract_sha256": file_sha256(contract), "cases": public_cases}
    atomic_json(out / "review_manifest.json", public)
    atomic_json(out / "private_selection_key.json", {
        "schema_version": "interaction_graph_contract_review_selection_v1",
        "selection_seed": SEED, "annotations_sha256": file_sha256(annotations_path),
        "private_key_sha256": file_sha256(key_path),
        "final_test_manifest_sha256": file_sha256(final_manifest_path), "cases": ordered})
    blank = [{field: (r["review_id"] if field == "review_id" else "")
              for field in REVIEW_FIELDS} for r in public_cases]
    write_csv(out / "reviewer_1.csv", blank, REVIEW_FIELDS)
    write_csv(out / "reviewer_2.csv", blank, REVIEW_FIELDS)
    print(out / "review_manifest.json", flush=True)


def validate_review(rows, expected_ids, manifest_hash, path):
    if len(rows) != 30 or {r["review_id"] for r in rows} != expected_ids:
        raise ValueError(f"{path}: expected the same 30 unique review IDs")
    for row in rows:
        if row.get("status") != "final" or row.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{path}: not a final review from this manifest")
        if not row.get("reviewer_id", "").strip():
            raise ValueError(f"{path}: missing reviewer ID for {row['review_id']}")
        for field, allowed in ALLOWED.items():
            if row.get(field) not in allowed:
                raise ValueError(f"{path}: invalid {field} for {row['review_id']}")
        if not row.get("observable_evidence", "").strip():
            raise ValueError(f"{path}: missing evidence for {row['review_id']}")
        if row.get("human_confirmed", "").lower() not in {"true", "yes", "1"}:
            raise ValueError(f"{path}: unconfirmed {row['review_id']}")
        for field in ("boundary_start_s", "boundary_end_s"):
            if row.get(field):
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}: invalid {field} for {row['review_id']}")
        start, end = row.get("boundary_start_s"), row.get("boundary_end_s")
        if start and end and float(start) > float(end):
            raise ValueError(f"{path}: reversed boundary interval for {row['review_id']}")
        if row["relation"] in {"SAME_ACTION_NEW_INSTANCE", "NEW_ACTION",
                               "BOUNDARY_TYPE_UNRESOLVED", "END_ONLY"} and not start:
            raise ValueError(f"{path}: boundary time required for {row['review_id']}")
        if row["return_to_prior_type"] == "yes" and not row.get("observed_type_path", "").strip():
            raise ValueError(f"{path}: return path required for {row['review_id']}")


def score(manifest_path, first_path, second_path, out):
    manifest = json.loads(Path(manifest_path).read_text())
    expected = {r["review_id"] for r in manifest["cases"]}
    if manifest.get("n_cases") != 30 or len(expected) != 30:
        raise ValueError("Not a complete 30-case manifest")
    manifest_hash = file_sha256(manifest_path)
    first, second = read_csv(first_path), read_csv(second_path)
    validate_review(first, expected, manifest_hash, first_path)
    validate_review(second, expected, manifest_hash, second_path)
    reviewer_1 = {r["reviewer_id"] for r in first}
    reviewer_2 = {r["reviewer_id"] for r in second}
    if len(reviewer_1) != 1 or len(reviewer_2) != 1 or reviewer_1 == reviewer_2:
        raise ValueError("Two complete reviews from distinct reviewer IDs are required")
    a, b = ({r["review_id"]: r for r in rows} for rows in (first, second))
    primary = [rid for rid in sorted(expected) if a[rid]["relation"] == b[rid]["relation"]]
    fields = ["relation", "type_equivalence", "prior_episode_status",
              "return_to_prior_type", "instance_link", "overlap_present"]
    agreement = {field: sum(a[rid][field] == b[rid][field] for rid in expected) / 30
                 for field in fields}
    timing = []
    for rid in sorted(expected):
        if a[rid]["relation"] not in {"SAME_ACTION_NEW_INSTANCE", "NEW_ACTION",
                                      "BOUNDARY_TYPE_UNRESOLVED", "END_ONLY"}:
            continue
        if b[rid]["relation"] not in {"SAME_ACTION_NEW_INSTANCE", "NEW_ACTION",
                                      "BOUNDARY_TYPE_UNRESOLVED", "END_ONLY"}:
            continue
        a_start = float(a[rid]["boundary_start_s"])
        a_end = float(a[rid]["boundary_end_s"] or a_start)
        b_start = float(b[rid]["boundary_start_s"])
        b_end = float(b[rid]["boundary_end_s"] or b_start)
        gap = max(0.0, max(a_start, b_start) - min(a_end, b_end))
        timing.append({"review_id": rid, "interval_gap_s": round(gap, 3),
                       "within_1s": gap <= 1.0})
    disagreements = [{"review_id": rid,
                      "reviewer_1": {f: a[rid][f] for f in fields},
                      "reviewer_2": {f: b[rid][f] for f in fields}}
                     for rid in sorted(expected) if any(a[rid][f] != b[rid][f] for f in fields)]
    result = {"schema_version": "interaction_graph_contract_agreement_v1",
              "manifest_sha256": manifest_hash,
              "reviewer_1_sha256": file_sha256(first_path),
              "reviewer_2_sha256": file_sha256(second_path),
              "n_cases": 30, "primary_exact": len(primary),
              "primary_exact_rate": len(primary) / 30,
              "passes_primary_gate": len(primary) >= 27,
              "field_exact_agreement": agreement, "timing_comparisons": timing,
              "timing_within_1s": sum(r["within_1s"] for r in timing),
              "disagreements": disagreements,
              "ratified": False,
              "ratification_note": "Set true only after every disagreement is human-adjudicated against the contract."}
    atomic_json(out, result)
    print(json.dumps({k: result[k] for k in ("primary_exact", "primary_exact_rate",
                                             "passes_primary_gate", "field_exact_agreement")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--annotations", required=True)
    prepare_parser.add_argument("--private-key", required=True)
    prepare_parser.add_argument("--data-root", action="append", required=True)
    prepare_parser.add_argument("--contract", required=True)
    prepare_parser.add_argument("--final-manifest", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--workers", type=int, default=3)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--reviewer-1", required=True)
    score_parser.add_argument("--reviewer-2", required=True)
    score_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.annotations, args.private_key, args.data_root, args.contract,
                args.final_manifest, args.out, args.workers)
    else:
        score(args.manifest, args.reviewer_1, args.reviewer_2, Path(args.out))


if __name__ == "__main__":
    main()
