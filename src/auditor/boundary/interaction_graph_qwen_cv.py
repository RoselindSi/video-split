"""Five-fold blind Qwen evaluation for whole-timeline interaction graphs.

The human annotation for a held-out video is never placed in that video's
prompt.  Each fold uses only annotations from the other four folds as compact
structural demonstrations.  Event letters are recording-local; evaluation is
delegated to ``interaction_graph_timeline_agreement`` so labels are compared as
temporal partitions instead of by their arbitrary A/B/C names.

Prepare and smoke-test locally::

    python -m src.auditor.boundary.interaction_graph_qwen_cv prepare ...
    python -m src.auditor.boundary.interaction_graph_qwen_cv run \
        --backend mock ...

Run the real model on the GPU host::

    python -m src.auditor.boundary.interaction_graph_qwen_cv run \
        --backend qwen --model-id /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.auditor.boundary.interaction_graph_timeline import validate_submission
from src.auditor.boundary.interaction_graph_timeline_agreement import (
    TOLERANCES_S,
    score,
    write_outputs,
)
from src.auditor.boundary.final_timeline_audit import file_sha256


SCHEMA = "interaction_graph_qwen_5fold_v1"
DEFAULT_SEED = "interaction-graph-qwen-5fold-v1-2026-09-22"
SYSTEM_PROMPT = (
    "你是独立的视频动作图标注者。严格遵守用户给出的标注规则，"
    "完整观看视频，只输出合法JSON。"
)
RULES = """你是独立的视频动作图标注者。目标视频的人工答案对你完全隐藏。

把整条视频从 0 秒到结尾切成连续、无空白、无重叠的时间段，并建立当前视频内可复用的事件节点 A、B、C……

硬规则：
1. 同一次动作继续进行时不要切段。来回擦、反复拧、调整手势、短暂停顿都仍可属于同一次动作。
2. 同一种动作已经结束、随后重新开始时要切段，并让新段复用原事件 ID。这形成 A->A，表示同类新实例。
3. 目标或任务改变时要切段。若回到此前事件就复用旧 ID，例如 A->B->A；否则创建新 ID。
4. repeats_within_segment 只表示同一次事件内部有肉眼可见的重复，不代替同类新实例之间的切点。
5. 事件 ID 只在当前视频内有效。描述使用简短、可观察的中文动作，不猜测看不到的意图。
6. 只有确实无法给出可观察动作描述的过渡或遮挡段才写“特殊事件”，不要把普通等待、走路或拿取笼统地写成特殊事件。
7. 边界保留一位小数。第一段从 0.0 开始，最后一段必须到视频时长。

只输出一个 JSON 对象，不要使用 Markdown。格式：
{"events":[{"event_id":"A","description":"动作描述"}],"segments":[{"start_s":0.0,"end_s":12.3,"event_id":"A","repeats_within_segment":false,"note":""}]}"""


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def assign_folds(manifest: dict, n_folds: int = 5,
                 seed: str = DEFAULT_SEED) -> dict:
    """Return a deterministic fold manifest stratified by mosaic complexity."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    strata: defaultdict[str, list[str]] = defaultdict(list)
    for row in manifest["videos"]:
        stratum = str((row.get("provenance") or {}).get(
            "keyframe_stratum") or "unknown")
        strata[stratum].append(str(row["video_id"]))
    folds = [{"fold": index, "test_video_ids": []}
             for index in range(n_folds)]
    for stratum, video_ids in sorted(strata.items()):
        ordered = sorted(video_ids, key=lambda x: _stable_key(
            seed, f"{stratum}|{x}"))
        for index, video_id in enumerate(ordered):
            folds[index % n_folds]["test_video_ids"].append(video_id)
    all_ids = sorted(row["video_id"] for row in manifest["videos"])
    for fold in folds:
        fold["test_video_ids"].sort()
        fold["train_video_ids"] = sorted(
            set(all_ids) - set(fold["test_video_ids"])
        )
        fold["test_strata"] = dict(Counter(
            str((row.get("provenance") or {}).get("keyframe_stratum") or
                "unknown")
            for row in manifest["videos"]
            if row["video_id"] in fold["test_video_ids"]
        ))
    return {
        "schema_version": SCHEMA,
        "seed": seed,
        "n_folds": n_folds,
        "n_videos": len(all_ids),
        "folds": folds,
    }


def annotation_tags(annotation: dict) -> set[str]:
    segments = annotation["segments"]
    sequence = [row["event_id"] for row in segments]
    tags = set()
    if len(segments) == 1:
        tags.add("single_segment")
    if any(row.get("repeats_within_segment") for row in segments):
        tags.add("internal_repeat")
    if any(left == right for left, right in zip(sequence, sequence[1:])):
        tags.add("same_type_new_instance")
    if any(sequence[i] == sequence[j] for i in range(len(sequence))
           for j in range(i + 2, len(sequence))):
        tags.add("return_to_prior_type")
    if len(set(sequence)) > 1:
        tags.add("type_switch")
    if any(event["description"] == "特殊事件"
           for event in annotation["events"]):
        tags.add("special_event")
    if len(segments) >= 5:
        tags.add("many_segments")
    return tags


def select_demos(train_ids: list[str], gold: dict[str, dict], seed: str,
                 maximum: int = 8) -> list[str]:
    """Greedily cover structural patterns without consulting held-out labels."""
    candidates = set(train_ids)
    chosen: list[str] = []
    covered: set[str] = set()
    while candidates and len(chosen) < maximum:
        ranked = sorted(
            candidates,
            key=lambda video_id: (
                -len(annotation_tags(gold[video_id]) - covered),
                _stable_key(seed, video_id),
            ),
        )
        selected = ranked[0]
        chosen.append(selected)
        covered |= annotation_tags(gold[selected])
        candidates.remove(selected)
    return chosen


def compact_demo(annotation: dict) -> dict:
    return {
        "events": annotation["events"],
        "segments": annotation["segments"],
    }


def build_prompt(duration_s: float, demos: list[dict],
                 draft: dict | str | None = None,
                 draft_error: str | None = None) -> str:
    examples = "\n\n".join(
        f"示例 {index + 1}：\n" + json.dumps(
            compact_demo(item), ensure_ascii=False, separators=(",", ":")
        )
        for index, item in enumerate(demos)
    )
    target = (
        f"\n\n现在独立标注目标视频。视频时长为 {duration_s:.3f} 秒。"
        "不要假设目标视频和示例有相同事件。"
    )
    if draft is not None:
        if isinstance(draft, str):
            target += (
                "\n第一遍输出没有通过格式校验：" + str(draft_error or "未知错误") +
                "。重新观看视频并重新输出完整JSON。不要逐帧创建节点，不要为同义动作反复创建"
                "新事件ID；events和segments各不得超过60项。第一遍原始输出如下：\n" + draft
            )
        else:
            target += (
                "\n这是你第一遍给出的草稿。重新观看视频后，检查漏切、过切、事件ID复用和段内重复，"
                "输出修正后的完整JSON：\n" +
                json.dumps(draft, ensure_ascii=False, separators=(",", ":"))
            )
    return RULES + ("\n\n以下示例全部来自其他视频：\n" + examples
                    if examples else "") + target


def extract_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen reply contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen reply is not an object")
    return value


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def canonicalize_model_output(value: dict, duration_s: float,
                              log: list[dict]) -> dict:
    """Apply schema-only repairs that do not change the predicted graph."""
    value = dict(value)
    events = value.get("events")
    segments = value.get("segments")
    if not isinstance(events, list) or not isinstance(segments, list):
        return value
    events = [dict(item) if isinstance(item, dict) else item for item in events]
    segments = [dict(item) if isinstance(item, dict) else item
                for item in segments]

    used = {
        str(item.get("event_id") or "").strip().upper()
        for item in segments if isinstance(item, dict)
    }
    dropped = [
        str(item.get("event_id") or "").strip().upper()
        for item in events
        if isinstance(item, dict)
        and str(item.get("event_id") or "").strip().upper() not in used
    ]
    if dropped:
        events = [
            item for item in events
            if not isinstance(item, dict)
            or str(item.get("event_id") or "").strip().upper() in used
        ]
        log.append({"kind": "drop_unused_events", "event_ids": dropped})

    sortable = all(
        isinstance(item, dict) and _number(item.get("start_s")) is not None
        and _number(item.get("end_s")) is not None
        for item in segments
    )
    if sortable:
        segments.sort(key=lambda item: (
            _number(item["start_s"]), _number(item["end_s"])
        ))
        first_start = _number(segments[0]["start_s"])
        if first_start is not None and 0 < abs(first_start) <= 0.11:
            log.append({"kind": "snap_start", "from_s": first_start,
                        "to_s": 0.0})
            segments[0]["start_s"] = 0.0
        last_end = _number(segments[-1]["end_s"])
        if last_end is not None and 0 < abs(last_end - duration_s) <= 0.11:
            log.append({"kind": "snap_end", "from_s": last_end,
                        "to_s": duration_s})
            segments[-1]["end_s"] = duration_s
        for index in range(1, len(segments)):
            left, right = segments[index - 1], segments[index]
            left_end = _number(left["end_s"])
            right_start = _number(right["start_s"])
            if left_end is None or right_start is None:
                continue
            delta = right_start - left_end
            if 0 < abs(delta) <= 0.11:
                boundary = round((left_end + right_start) / 2, 3)
                left_start = _number(left["start_s"])
                right_end = _number(right["end_s"])
                if left_start < boundary < right_end:
                    log.append({
                        "kind": "snap_adjacent_boundary",
                        "segments": [index, index + 1],
                        "from_s": [left_end, right_start],
                        "to_s": boundary,
                    })
                    left["end_s"] = boundary
                    right["start_s"] = boundary
    value["events"] = events
    value["segments"] = segments
    return value


def normalize_prediction(raw: str, video_id: str, duration_s: float,
                         normalization_log: list[dict] | None = None) -> dict:
    value = extract_object(raw)
    fixes = normalization_log if normalization_log is not None else []
    value = canonicalize_model_output(value, duration_s, fixes)
    annotation = {
        "schema_version": "interaction_graph_timeline_v1",
        "video_id": video_id,
        "events": value.get("events"),
        "segments": value.get("segments"),
        "confirmed_full_video": True,
    }
    normalized = validate_submission(annotation, video_id, duration_s, final=True)
    normalized["status"] = "final"
    return normalized


def _video_path(row: dict, video_root: str | Path | None) -> str:
    if video_root:
        path = Path(video_root) / Path(str(row["video"])).name
    else:
        path = Path(str(row["video"]))
    if not path.is_file():
        raise FileNotFoundError(f"video is missing: {path}")
    return str(path.resolve())


def mock_prediction(video_id: str, duration_s: float) -> dict:
    return {
        "schema_version": "interaction_graph_timeline_v1",
        "video_id": video_id,
        "events": [{"event_id": "A", "description": "mock only"}],
        "segments": [{
            "start_s": 0.0,
            "end_s": duration_s,
            "event_id": "A",
            "repeats_within_segment": False,
            "note": "",
        }],
        "confirmed_full_video": True,
        "status": "final",
    }


def fold_lookup(folds: dict) -> dict[str, dict]:
    out = {}
    for fold in folds["folds"]:
        for video_id in fold["test_video_ids"]:
            if video_id in out:
                raise ValueError(f"video appears in multiple folds: {video_id}")
            out[video_id] = fold
    return out


def prepare(manifest_path: str | Path, gold_path: str | Path,
            out_dir: str | Path, seed: str, max_demos: int) -> dict:
    manifest, export = read_json(manifest_path), read_json(gold_path)
    gold = {row["video_id"]: row for row in export["videos"]}
    expected = {row["video_id"] for row in manifest["videos"]}
    if set(gold) != expected:
        raise ValueError("gold and manifest video IDs differ")
    folds = assign_folds(manifest, seed=seed)
    for fold in folds["folds"]:
        fold["demo_video_ids"] = select_demos(
            fold["train_video_ids"], gold, f"{seed}|fold{fold['fold']}",
            maximum=max_demos,
        )
        if set(fold["demo_video_ids"]) & set(fold["test_video_ids"]):
            raise AssertionError("held-out annotation leaked into demonstrations")
    folds.update({
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "gold": str(Path(gold_path).resolve()),
        "gold_sha256": file_sha256(gold_path),
        "max_demos_per_fold": max_demos,
    })
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "folds.json"
    path.write_text(json.dumps(folds, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return folds


def run(args) -> dict:
    manifest = read_json(args.manifest)
    export = read_json(args.gold)
    gold = {row["video_id"]: row for row in export["videos"]}
    folds = read_json(args.folds)
    by_fold = fold_lookup(folds)
    rows = {row["video_id"]: row for row in manifest["videos"]}
    root = Path(args.out_dir)
    raw_root, prompt_root = root / "raw", root / "prompts"
    raw_root.mkdir(parents=True, exist_ok=True)
    prompt_root.mkdir(parents=True, exist_ok=True)

    backend = None
    if args.backend == "qwen":
        from src.auditor.vision_backends import build_backend
        backend = build_backend(
            "qwen", args.model_id, max_pixels=args.max_pixels,
            attn=args.attn,
        )

    predictions = []
    for video_id in sorted(rows):
        fold = by_fold[video_id]
        if args.fold is not None and fold["fold"] != args.fold:
            continue
        raw_path = raw_root / f"{video_id}.json"
        saved = None
        if raw_path.exists() and not args.overwrite:
            saved = read_json(raw_path)
            if saved.get("prediction"):
                predictions.append(saved["prediction"])
                continue
        duration = float(rows[video_id]["duration_s"])
        demos = [gold[item] for item in fold["demo_video_ids"]]
        prompt1 = build_prompt(duration, demos)
        prompt_path = prompt_root / f"{video_id}.txt"
        prompt_path.write_text(prompt1, encoding="utf-8")
        record = {
            "schema_version": SCHEMA,
            "video_id": video_id,
            "fold": fold["fold"],
            "test_video_ids": fold["test_video_ids"],
            "demo_video_ids": fold["demo_video_ids"],
            "target_gold_was_in_prompt": video_id in fold["demo_video_ids"],
            "backend": args.backend,
            "model_id": args.model_id if args.backend == "qwen" else None,
            "fps": args.fps,
            "passes": args.passes,
        }
        try:
            if args.backend == "mock":
                prediction = mock_prediction(video_id, duration)
                record["raw_passes"] = []
            else:
                video = _video_path(rows[video_id], args.video_root)
                reusable = (saved or {}).get("raw_passes") or []
                if reusable:
                    raw1 = reusable[0]
                    record["resumed_from_saved_raw"] = True
                    record["previous_error"] = saved.get("error")
                else:
                    raw1 = backend.generate(
                        SYSTEM_PROMPT, prompt1, video=video, fps=args.fps,
                        temperature=0.0, max_new_tokens=args.max_new_tokens,
                    )
                record["raw_passes"] = [raw1]
                normalizations1: list[dict] = []
                record["normalizations_by_pass"] = [normalizations1]
                prediction = None
                pass1_error = None
                try:
                    prediction = normalize_prediction(
                        raw1, video_id, duration, normalizations1
                    )
                except Exception as error:
                    pass1_error = f"{type(error).__name__}: {error}"
                    record["pass_1_error"] = pass1_error
                if args.passes == 2:
                    prompt2 = build_prompt(
                        duration, demos,
                        prediction if prediction is not None else raw1,
                        pass1_error,
                    )
                    (prompt_root / f"{video_id}.pass2.txt").write_text(
                        prompt2, encoding="utf-8"
                    )
                    raw2 = backend.generate(
                        SYSTEM_PROMPT, prompt2, video=video, fps=args.fps,
                        temperature=0.0, max_new_tokens=args.max_new_tokens,
                    )
                    record["raw_passes"].append(raw2)
                    normalizations2: list[dict] = []
                    record["normalizations_by_pass"].append(normalizations2)
                    try:
                        prediction = normalize_prediction(
                            raw2, video_id, duration, normalizations2
                        )
                    except Exception as error:
                        record["pass_2_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                        if prediction is not None:
                            record["used_pass"] = 1
                    else:
                        record["used_pass"] = 2
                elif prediction is not None:
                    record["used_pass"] = 1
                if prediction is None:
                    raise ValueError(
                        "both model passes failed validation: "
                        f"pass1={record.get('pass_1_error')}; "
                        f"pass2={record.get('pass_2_error')}"
                    )
            record["prediction"] = prediction
            record["status"] = "ok"
            predictions.append(prediction)
        except Exception as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
        raw_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"{video_id} fold={fold['fold']} {record['status']}", flush=True)
        _write_export(root / "predictions.json", predictions, folds, args)
    return _write_export(root / "predictions.json", predictions, folds, args)


def _write_export(path: Path, predictions: list[dict], folds: dict, args) -> dict:
    value = {
        "schema_version": "interaction_graph_timeline_offline_export_v1",
        "annotator": f"qwen_5fold_{args.backend}",
        "batch_id": "interaction_graph_blind30_v2",
        "fold_schema": folds["schema_version"],
        "fold_seed": folds["seed"],
        "n_videos": len(predictions),
        "videos": sorted(predictions, key=lambda row: row["video_id"]),
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return value


def _micro_boundary(rows: list[dict], tolerance: str) -> dict:
    matched = sum(row["boundary"][tolerance]["matches"] for row in rows)
    gold = sum(row["boundary"][tolerance]["n_reviewer_1"] for row in rows)
    pred = sum(row["boundary"][tolerance]["n_reviewer_2"] for row in rows)
    precision = matched / pred if pred else (1.0 if not gold else 0.0)
    recall = matched / gold if gold else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"matches": matched, "n_gold": gold, "n_pred": pred,
            "precision": precision, "recall": recall, "f1": f1}


def add_fold_metrics(result: dict, folds: dict) -> dict:
    lookup = {video_id: fold["fold"] for fold in folds["folds"]
              for video_id in fold["test_video_ids"]}
    grouped: defaultdict[int, list[dict]] = defaultdict(list)
    for row in result["per_video"]:
        row["fold"] = lookup[row["video_id"]]
        grouped[row["fold"]].append(row)
    by_fold = []
    for fold_spec in folds["folds"]:
        fold = fold_spec["fold"]
        rows = grouped[fold]
        relation_n = sum(row["primary_matched_boundary_relation"]["n"]
                         for row in rows)
        relation_ok = sum(row["primary_matched_boundary_relation"]["n_agree"]
                          for row in rows)
        by_fold.append({
            "fold": fold,
            "n_videos": len(rows),
            "n_expected_videos": len(fold_spec["test_video_ids"]),
            "valid_output_coverage": (
                len(rows) / len(fold_spec["test_video_ids"])
            ),
            "boundary_micro_by_tolerance_s": {
                str(t): _micro_boundary(rows, str(t)) for t in TOLERANCES_S
            },
            "matched_relation_at_1s": {
                "n": relation_n,
                "n_agree": relation_ok,
                "agreement": relation_ok / relation_n if relation_n else None,
            },
            "median_temporal_type_partition_ari": statistics.median(
                row["temporal_type_partition_ari"] for row in rows
            ),
            "mean_internal_repeat_time_agreement": statistics.mean(
                row["internal_repeat_time_agreement"] for row in rows
            ),
        })
    result["five_fold"] = {
        "schema_version": folds["schema_version"],
        "seed": folds["seed"],
        "by_fold": by_fold,
    }
    return result


def add_failure_aware_metrics(result: dict, gold_path: str | Path) -> dict:
    """Count missing model outputs as producing no boundaries."""
    export = read_json(gold_path)
    gold = {row["video_id"]: row for row in export["videos"]}
    missing = result["missing_reviewer_2"]
    missing_boundaries = sum(
        max(0, len(gold[video_id]["segments"]) - 1) for video_id in missing
    )
    metrics = result["metrics"]
    metrics["valid_output_coverage"] = (
        result["n_compared_videos"] / result["n_expected_videos"]
        if result["n_expected_videos"] else None
    )
    adjusted = {}
    for tolerance, row in metrics["boundary_micro_by_tolerance_s"].items():
        matches = row["matches"]
        n_gold = row["n_reviewer_1"] + missing_boundaries
        n_pred = row["n_reviewer_2"]
        precision = matches / n_pred if n_pred else (1.0 if not n_gold else 0.0)
        recall = matches / n_gold if n_gold else (1.0 if not n_pred else 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        adjusted[tolerance] = {
            "matches": matches,
            "n_gold": n_gold,
            "n_pred": n_pred,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    metrics["boundary_micro_failure_aware_by_tolerance_s"] = adjusted
    result["model_output_failures"] = {
        "n": len(missing),
        "video_ids": missing,
        "gold_boundaries_in_failed_videos": missing_boundaries,
    }
    return result


def score_run(manifest: str | Path, gold: str | Path, predictions: str | Path,
              folds_path: str | Path, out_dir: str | Path) -> dict:
    result = score(manifest, gold, predictions)
    result["interpretation_note"] = (
        "This is model-vs-human out-of-fold agreement. It is not human "
        "inter-annotator reliability and descriptions are retained for audit "
        "but not automatically scored."
    )
    add_failure_aware_metrics(result, gold)
    add_fold_metrics(result, read_json(folds_path))
    write_outputs(result, out_dir)
    path = Path(out_dir) / "agreement.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--gold", required=True)
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--seed", default=DEFAULT_SEED)
    prep.add_argument("--max-demos", type=int, default=8)

    runp = sub.add_parser("run")
    runp.add_argument("--manifest", required=True)
    runp.add_argument("--gold", required=True)
    runp.add_argument("--folds", required=True)
    runp.add_argument("--out-dir", required=True)
    runp.add_argument("--backend", choices=("mock", "qwen"), required=True)
    runp.add_argument("--model-id")
    runp.add_argument("--video-root")
    runp.add_argument("--fold", type=int)
    runp.add_argument("--fps", type=float, default=1.0)
    runp.add_argument("--passes", type=int, choices=(1, 2), default=2)
    runp.add_argument("--max-pixels", type=int, default=200704)
    runp.add_argument("--max-new-tokens", type=int, default=2048)
    runp.add_argument("--attn")
    runp.add_argument("--overwrite", action="store_true")

    scoring = sub.add_parser("score")
    scoring.add_argument("--manifest", required=True)
    scoring.add_argument("--gold", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--folds", required=True)
    scoring.add_argument("--out-dir", required=True)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        value = prepare(args.manifest, args.gold, args.out_dir,
                        args.seed, args.max_demos)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command == "run":
        if args.backend == "qwen" and not args.model_id:
            raise SystemExit("--model-id is required for --backend qwen")
        run(args)
    else:
        result = score_run(args.manifest, args.gold, args.predictions,
                           args.folds, args.out_dir)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
