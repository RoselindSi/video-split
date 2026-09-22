"""Score two independent whole-timeline interaction-graph annotations.

Event letters are recording-local and arbitrary.  Agreement therefore compares
boundaries, same-vs-different transition relations, and the induced temporal
partition; it never compares raw event letters or free-text descriptions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from src.auditor.boundary.final_timeline_audit import file_sha256


PRIMARY_TOLERANCE_S = 1.0
TOLERANCES_S = (0.25, 0.5, 1.0)


def latest_finals(directory: str | Path) -> dict[str, dict]:
    root = Path(directory)
    out = {}
    for path in sorted(root.glob("*.final.*.json")):
        video_id = path.name.split(".final.", 1)[0]
        out[video_id] = json.loads(path.read_text(encoding="utf-8"))
    return out


def load_annotations(path: str | Path) -> tuple[dict[str, dict], dict]:
    source = Path(path)
    if source.is_dir():
        rows = latest_finals(source)
        return rows, {"kind": "final_directory", "path": str(source.resolve())}
    value = json.loads(source.read_text(encoding="utf-8"))
    videos = value.get("videos") or []
    rows = {
        str(row["video_id"]): row
        for row in videos
        if row.get("status") in (None, "final")
    }
    return rows, {
        "kind": "export_json",
        "path": str(source.resolve()),
        "sha256": file_sha256(source),
        "manifest_sha256": value.get("manifest_sha256"),
        "annotator": value.get("annotator"),
    }


def transition_boundaries(annotation: dict) -> list[dict]:
    segments = annotation["segments"]
    return [{
        "time_s": float(right["start_s"]),
        "relation": (
            "SAME_ACTION_NEW_INSTANCE"
            if left["event_id"] == right["event_id"]
            else "NEW_ACTION"
        ),
        "left_event_id": left["event_id"],
        "right_event_id": right["event_id"],
    } for left, right in zip(segments, segments[1:])]


def match_boundaries(left: list[dict], right: list[dict],
                     tolerance_s: float) -> list[tuple[int, int]]:
    """Order-preserving maximum-cardinality, minimum-distance matching."""
    n, m = len(left), len(right)
    score = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        back[i][0] = "left"
    for j in range(1, m + 1):
        back[0][j] = "right"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [(score[i - 1][j], "left"),
                          (score[i][j - 1], "right")]
            delta = abs(left[i - 1]["time_s"] - right[j - 1]["time_s"])
            if delta <= tolerance_s:
                prior = score[i - 1][j - 1]
                candidates.append(((prior[0] + 1, prior[1] - delta), "match"))
            best, action = max(
                candidates,
                key=lambda item: (item[0][0], item[0][1], item[1] == "match"),
            )
            score[i][j], back[i][j] = best, action
    pairs = []
    i, j = n, m
    while i or j:
        action = back[i][j]
        if action == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "left":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _f1(matches: int, n_left: int, n_right: int) -> dict:
    precision = matches / n_right if n_right else (1.0 if not n_left else 0.0)
    recall = matches / n_left if n_left else (1.0 if not n_right else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"matches": matches, "n_reviewer_1": n_left,
            "n_reviewer_2": n_right, "precision": precision,
            "recall": recall, "f1": f1}


def restricted_growth(sequence: list[str]) -> list[int]:
    mapping = {}
    out = []
    for value in sequence:
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return out


def _segment_at(annotation: dict, time_s: float) -> dict:
    for segment in annotation["segments"]:
        if segment["start_s"] <= time_s < segment["end_s"]:
            return segment
    return annotation["segments"][-1]


def _comb2(value: int) -> float:
    return value * (value - 1) / 2


def adjusted_rand(left: list[str], right: list[str]) -> float:
    if len(left) != len(right):
        raise ValueError("partition vectors have different lengths")
    if len(left) < 2:
        return 1.0
    table = Counter(zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    sum_cells = sum(_comb2(v) for v in table.values())
    sum_left = sum(_comb2(v) for v in left_counts.values())
    sum_right = sum(_comb2(v) for v in right_counts.values())
    total = _comb2(len(left))
    expected = sum_left * sum_right / total if total else 0.0
    maximum = (sum_left + sum_right) / 2
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if restricted_growth(left) == restricted_growth(right) else 0.0
    return (sum_cells - expected) / denominator


def cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    p_left = sum(left) / len(left)
    p_right = sum(right) / len(right)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def temporal_vectors(left: dict, right: dict, duration_s: float,
                     step_s: float = 0.5) -> tuple[list[str], list[str],
                                                      list[bool], list[bool]]:
    count = max(1, math.ceil(duration_s / step_s))
    left_events, right_events, left_repeat, right_repeat = [], [], [], []
    for index in range(count):
        time_s = min(duration_s - 1e-6, (index + 0.5) * step_s)
        a, b = _segment_at(left, time_s), _segment_at(right, time_s)
        left_events.append(str(a["event_id"]))
        right_events.append(str(b["event_id"]))
        left_repeat.append(bool(a.get("repeats_within_segment")))
        right_repeat.append(bool(b.get("repeats_within_segment")))
    return left_events, right_events, left_repeat, right_repeat


def compare_video(video_id: str, duration_s: float, left: dict, right: dict) -> dict:
    a, b = transition_boundaries(left), transition_boundaries(right)
    tolerance_rows = {}
    for tolerance in TOLERANCES_S:
        pairs = match_boundaries(a, b, tolerance)
        row = _f1(len(pairs), len(a), len(b))
        row["mean_abs_error_s"] = (
            sum(abs(a[i]["time_s"] - b[j]["time_s"]) for i, j in pairs) / len(pairs)
            if pairs else None
        )
        tolerance_rows[str(tolerance)] = row
    primary_pairs = match_boundaries(a, b, PRIMARY_TOLERANCE_S)
    confusion = Counter((a[i]["relation"], b[j]["relation"])
                        for i, j in primary_pairs)
    relation_correct = sum(x == y for x, y in confusion.elements())
    left_events, right_events, left_repeat, right_repeat = temporal_vectors(
        left, right, duration_s
    )
    path_left = restricted_growth([s["event_id"] for s in left["segments"]])
    path_right = restricted_growth([s["event_id"] for s in right["segments"]])
    repeat_agreement = sum(x == y for x, y in zip(left_repeat, right_repeat)) / len(left_repeat)
    return {
        "video_id": video_id,
        "duration_s": duration_s,
        "n_segments_reviewer_1": len(left["segments"]),
        "n_segments_reviewer_2": len(right["segments"]),
        "boundary": tolerance_rows,
        "primary_matched_boundary_relation": {
            "n": len(primary_pairs),
            "n_agree": relation_correct,
            "agreement": relation_correct / len(primary_pairs) if primary_pairs else None,
            "confusion": {f"{x}__vs__{y}": count
                          for (x, y), count in sorted(confusion.items())},
        },
        "normalized_event_path_reviewer_1": path_left,
        "normalized_event_path_reviewer_2": path_right,
        "exact_normalized_path": path_left == path_right,
        "temporal_type_partition_ari": adjusted_rand(left_events, right_events),
        "internal_repeat_time_agreement": repeat_agreement,
        "internal_repeat_vectors": [left_repeat, right_repeat],
    }


def score(manifest_path: str | Path, reviewer_1: str | Path,
          reviewer_2: str | Path) -> dict:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {str(row["video_id"]): row for row in manifest["videos"]}
    left, left_source = load_annotations(reviewer_1)
    right, right_source = load_annotations(reviewer_2)
    missing_left = sorted(set(expected) - set(left))
    missing_right = sorted(set(expected) - set(right))
    extra_left = sorted(set(left) - set(expected))
    extra_right = sorted(set(right) - set(expected))
    complete_ids = sorted(set(expected) & set(left) & set(right))
    per_video = [compare_video(
        video_id, float(expected[video_id]["duration_s"]),
        left[video_id], right[video_id],
    ) for video_id in complete_ids]

    boundary = {}
    for tolerance in TOLERANCES_S:
        key = str(tolerance)
        matches = sum(row["boundary"][key]["matches"] for row in per_video)
        n_left = sum(row["boundary"][key]["n_reviewer_1"] for row in per_video)
        n_right = sum(row["boundary"][key]["n_reviewer_2"] for row in per_video)
        boundary[key] = _f1(matches, n_left, n_right)
    confusions = Counter()
    relation_n = relation_agree = 0
    repeat_left: list[bool] = []
    repeat_right: list[bool] = []
    for row in per_video:
        relation = row["primary_matched_boundary_relation"]
        relation_n += relation["n"]
        relation_agree += relation["n_agree"]
        confusions.update(relation["confusion"])
        repeat_vectors = row.pop("internal_repeat_vectors")
        repeat_left.extend(repeat_vectors[0])
        repeat_right.extend(repeat_vectors[1])
    repeat_agreement = (
        sum(a == b for a, b in zip(repeat_left, repeat_right)) / len(repeat_left)
        if repeat_left else None
    )
    median_ari = statistics.median(
        row["temporal_type_partition_ari"] for row in per_video
    ) if per_video else None
    relation_rate = relation_agree / relation_n if relation_n else None
    primary_f1 = boundary[str(PRIMARY_TOLERANCE_S)]["f1"]
    complete = not (missing_left or missing_right or extra_left or extra_right)
    gate = {
        "all_manifest_videos_present_and_no_extras": complete,
        "boundary_f1_at_1s_gte_0_80": primary_f1 >= 0.80,
        "matched_relation_agreement_gte_0_80": (
            relation_rate is not None and relation_rate >= 0.80
        ),
        "median_temporal_partition_ari_gte_0_80": (
            median_ari is not None and median_ari >= 0.80
        ),
        "internal_repeat_time_agreement_gte_0_80": (
            repeat_agreement is not None and repeat_agreement >= 0.80
        ),
    }
    return {
        "schema_version": "interaction_graph_timeline_agreement_v1",
        "manifest": str(manifest_file.resolve()),
        "manifest_sha256": file_sha256(manifest_file),
        "reviewer_1_source": left_source,
        "reviewer_2_source": right_source,
        "n_expected_videos": len(expected),
        "n_compared_videos": len(per_video),
        "missing_reviewer_1": missing_left,
        "missing_reviewer_2": missing_right,
        "extra_reviewer_1": extra_left,
        "extra_reviewer_2": extra_right,
        "metrics": {
            "boundary_micro_by_tolerance_s": boundary,
            "matched_relation_at_1s": {
                "n": relation_n,
                "n_agree": relation_agree,
                "agreement": relation_rate,
                "confusion": dict(sorted(confusions.items())),
            },
            "median_temporal_type_partition_ari": median_ari,
            "exact_normalized_path_videos": sum(
                row["exact_normalized_path"] for row in per_video
            ),
            "internal_repeat_time_agreement": repeat_agreement,
            "internal_repeat_time_kappa": cohen_kappa(repeat_left, repeat_right),
        },
        "prespecified_gate": {
            "thresholds_were_fixed_before_reviewer_2_results": True,
            "checks": gate,
            "passes_all": all(gate.values()),
        },
        "interpretation_note": (
            "Agreement measures reproducibility of the annotation contract, not "
            "which reviewer is correct. Disagreements require blinded adjudication."
        ),
        "per_video": per_video,
    }


def write_outputs(result: dict, out_dir: str | Path) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["video_id", "duration_s", "n_segments_reviewer_1",
              "n_segments_reviewer_2", "boundary_f1_0.25s",
              "boundary_f1_0.5s", "boundary_f1_1.0s",
              "relation_agreement_1.0s", "temporal_type_partition_ari",
              "exact_normalized_path", "internal_repeat_time_agreement"]
    with (root / "per_video.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["per_video"]:
            writer.writerow({
                "video_id": row["video_id"], "duration_s": row["duration_s"],
                "n_segments_reviewer_1": row["n_segments_reviewer_1"],
                "n_segments_reviewer_2": row["n_segments_reviewer_2"],
                "boundary_f1_0.25s": row["boundary"]["0.25"]["f1"],
                "boundary_f1_0.5s": row["boundary"]["0.5"]["f1"],
                "boundary_f1_1.0s": row["boundary"]["1.0"]["f1"],
                "relation_agreement_1.0s": row["primary_matched_boundary_relation"]["agreement"],
                "temporal_type_partition_ari": row["temporal_type_partition_ari"],
                "exact_normalized_path": row["exact_normalized_path"],
                "internal_repeat_time_agreement": row["internal_repeat_time_agreement"],
            })
    metrics = result["metrics"]
    report = f"""# Whole-timeline double-annotation agreement

- Videos compared: {result['n_compared_videos']} / {result['n_expected_videos']}
- Boundary F1 at 0.25 s: {metrics['boundary_micro_by_tolerance_s']['0.25']['f1']:.3f}
- Boundary F1 at 0.5 s: {metrics['boundary_micro_by_tolerance_s']['0.5']['f1']:.3f}
- Boundary F1 at 1.0 s: {metrics['boundary_micro_by_tolerance_s']['1.0']['f1']:.3f}
- Same-new-instance vs new-action agreement at matched 1.0 s boundaries: {metrics['matched_relation_at_1s']['agreement']}
- Median temporal event-partition ARI: {metrics['median_temporal_type_partition_ari']}
- Internal-repeat time agreement: {metrics['internal_repeat_time_agreement']}
- Prespecified gate passes: {result['prespecified_gate']['passes_all']}

Raw event letters and free-text descriptions are intentionally not compared.
Disagreements must be adjudicated without treating either reviewer as ground truth.
"""
    (root / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reviewer-1", required=True)
    parser.add_argument("--reviewer-2", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = score(args.manifest, args.reviewer_1, args.reviewer_2)
    write_outputs(result, args.out_dir)
    print(json.dumps({
        "n_compared_videos": result["n_compared_videos"],
        "metrics": result["metrics"],
        "prespecified_gate": result["prespecified_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
