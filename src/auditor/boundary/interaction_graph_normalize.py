"""Create a normalized, non-destructive export from final timeline labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


NORMALIZED_SCHEMA = "interaction_graph_normalized_v1"
DEFAULT_SPECIAL_LABELS = {
    "idle", "未开始", "无内容", "看不到动作", "看不到内容", "特殊事件",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _latest_finals(annotation_dir: str | Path) -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in Path(annotation_dir).glob("*.final.*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "final":
            continue
        video_id = str(value.get("video_id") or "")
        saved_at = str(value.get("saved_at") or path.name)
        if video_id and (video_id not in latest or saved_at > latest[video_id][0]):
            latest[video_id] = (saved_at, path.resolve())
    return {video_id: item[1] for video_id, item in latest.items()}


def _compress(sequence: list[str]) -> list[str]:
    result = []
    for event_id in sequence:
        if not result or result[-1] != event_id:
            result.append(event_id)
    return result


def normalize_annotation(annotation: dict,
                         special_labels: set[str] | None = None) -> dict:
    labels = {label.strip().casefold() for label in
              (special_labels or DEFAULT_SPECIAL_LABELS)}
    descriptions = {
        row["event_id"]: str(row.get("description") or "").strip()
        for row in annotation.get("events", [])
    }
    roles = {
        event_id: ("special" if description.casefold() in labels else "action")
        for event_id, description in descriptions.items()
    }
    graph = annotation.get("graph") or {}
    nodes = []
    for row in graph.get("nodes", []):
        item = dict(row)
        item["node_role"] = roles.get(row["event_id"], "action")
        nodes.append(item)

    occurrences = []
    for row in graph.get("edge_occurrences", []):
        item = dict(row)
        item["source_role"] = roles.get(row["source"], "action")
        item["target_role"] = roles.get(row["target"], "action")
        occurrences.append(item)
    action_occurrences = [
        row for row in occurrences
        if row["source_role"] == "action" and row["target_role"] == "action"
    ]
    action_ids = {event_id for event_id, role in roles.items() if role == "action"}
    action_sequence = [
        event_id for event_id in graph.get("event_sequence", [])
        if event_id in action_ids
    ]
    compressed = _compress(action_sequence)
    return_cycle = len(compressed) != len(set(compressed))
    self_loop = any(
        row["source"] == row["target"] and
        row["kind"] in {"self_transition", "internal_repeat"}
        for row in action_occurrences
    )
    kind_counts = Counter(row["kind"] for row in action_occurrences)

    return {
        "video_id": annotation["video_id"],
        "source_annotation_status": annotation.get("status"),
        "events": [
            {**row, "node_role": roles.get(row["event_id"], "action")}
            for row in annotation.get("events", [])
        ],
        "segments": [
            {**row, "node_role": roles.get(row["event_id"], "action")}
            for row in annotation.get("segments", [])
        ],
        "full_graph": {**graph, "nodes": nodes, "edge_occurrences": occurrences},
        "action_graph": {
            "nodes": [row for row in nodes if row["node_role"] == "action"],
            "edge_occurrences": action_occurrences,
            "event_sequence": action_sequence,
            "compressed_event_sequence": compressed,
            "edge_kind_counts": dict(sorted(kind_counts.items())),
            "self_loop_present": self_loop,
            "return_cycle_present": return_cycle,
        },
        "special_event_ids": sorted(
            event_id for event_id, role in roles.items() if role == "special"),
    }


def normalize_batch(manifest: str | Path, annotation_dir: str | Path,
                    out: str | Path,
                    special_labels: set[str] | None = None) -> dict:
    manifest_path = Path(manifest).resolve()
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    finals = _latest_finals(annotation_dir)
    expected = [row["video_id"] for row in manifest_value.get("videos", [])]
    missing = [video_id for video_id in expected if video_id not in finals]
    if missing:
        raise ValueError(f"missing final annotations: {', '.join(missing)}")

    videos = []
    for video_id in expected:
        path = finals[video_id]
        annotation = json.loads(path.read_text(encoding="utf-8"))
        item = normalize_annotation(annotation, special_labels)
        item["source_annotation"] = str(path)
        item["source_annotation_sha256"] = _sha256(path)
        videos.append(item)
    labels = sorted(special_labels or DEFAULT_SPECIAL_LABELS)
    result = {
        "schema_version": NORMALIZED_SCHEMA,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "n_videos": len(videos),
        "normalization_policy": {
            "special_labels": labels,
            "special_nodes_excluded_from_action_graph": True,
            "special_gaps_are_not_bridged_with_synthetic_edges": True,
            "self_transition_and_internal_repeat_kept_separate": True,
            "event_ids_are_local_to_each_video": True,
            "source_annotations_modified": False,
        },
        "summary": {
            "videos_with_action_self_loop": sum(
                row["action_graph"]["self_loop_present"] for row in videos),
            "videos_with_action_return_cycle": sum(
                row["action_graph"]["return_cycle_present"] for row in videos),
            "special_nodes": sum(len(row["special_event_ids"]) for row in videos),
        },
        "videos": videos,
    }
    _atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--special-label", action="append", default=[])
    args = parser.parse_args()
    labels = set(args.special_label) if args.special_label else None
    result = normalize_batch(args.manifest, args.annotation_dir, args.out, labels)
    print(f"normalized_ready {args.out} ({result['n_videos']} videos)")


if __name__ == "__main__":
    main()
