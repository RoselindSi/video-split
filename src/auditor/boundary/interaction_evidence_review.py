"""Prepare a label-blind development target-contract audit, not new gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.auditor.boundary.interaction_evidence_pilot import (
    assert_development, dump_csv, dump_json, explicit_targets, forbidden_ids,
    numeric, read_csv, recording_id, sha256, unique_index,
)


def candidate_time(event_id):
    match = re.search(r"_t(\d+(?:\.\d+)?)$", event_id)
    if not match:
        raise ValueError(f"Candidate lacks a timestamp: {event_id}")
    return float(match[1])


def expand_selection(ordered, anchors, n_events, seed, min_separation_s, max_per_recording):
    if not len(anchors) <= n_events <= len(ordered):
        raise ValueError("Requested size must retain all legacy events and fit the candidate pool")
    if not math.isfinite(min_separation_s) or min_separation_s < 0 or max_per_recording < 2:
        raise ValueError("Invalid separation or recording cap")
    selected = set(anchors)
    groups, times = defaultdict(list), {}
    for eid in ordered:
        groups[recording_id(eid)].append(eid)
        times[eid] = candidate_time(eid)
    rank = sorted(groups, key=lambda rid: hashlib.sha256(
        f"{seed}:recording-pair-expansion:{rid}".encode()).hexdigest())
    chosen = {rid: [e for e in ids if e in selected] for rid, ids in groups.items()}
    for ids in chosen.values():
        if len(ids) > max_per_recording:
            raise ValueError("Legacy events already exceed the recording cap")
        tt = sorted(times[e] for e in ids)
        if any(b - a < min_separation_s for a, b in zip(tt, tt[1:])):
            raise ValueError("Legacy events violate minimum time separation")

    def available(rid):
        if len(chosen[rid]) >= max_per_recording:
            return []
        return [e for e in groups[rid] if e not in selected and
                all(abs(times[e] - times[k]) >= min_separation_s for k in chosen[rid])]

    def add(eid):
        selected.add(eid)
        chosen[recording_id(eid)].append(eid)

    # Pair legacy singletons first, then add pairs from new recordings.
    for rid in rank:
        if len(selected) == n_events:
            break
        candidates = available(rid)
        if len(chosen[rid]) == 1 and candidates:
            add(candidates[0])
    for rid in rank:
        if n_events - len(selected) < 2:
            break
        if chosen[rid]:
            continue
        candidates = available(rid)
        pair = next(((a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                     if abs(times[a] - times[b]) >= min_separation_s), None)
        if pair:
            for eid in pair:
                add(eid)
    while len(selected) < n_events:
        rid = next((r for r in sorted(rank, key=lambda r: len(chosen[r])) if available(r)), None)
        if rid is None:
            raise ValueError("Cannot reach requested size under recording/separation constraints")
        add(available(rid)[0])
    return selected


def prepare_packet(migrated, legacy_targets, forbidden, seed, n_events=None,
                   min_separation_s=2.0, max_per_recording=3):
    assert_development(migrated, forbidden)
    assert_development(legacy_targets, forbidden)
    candidates, legacy = unique_index(migrated), unique_index(legacy_targets)
    if set(legacy) - set(candidates):
        raise ValueError("Legacy audit has events outside the candidate pool")
    ordered = sorted(candidates, key=lambda eid: hashlib.sha256(
        f"{seed}:target-contract-v2:{eid}".encode()).hexdigest())
    selected = (set(legacy) if n_events is None else expand_selection(
        ordered, legacy, n_events, seed, min_separation_s, max_per_recording))
    primary, reserve, private_key = [], [], []
    for i, eid in enumerate(ordered):
        time_s = candidate_time(eid)
        alias = f"review_{i + 1:04d}"
        row = {"review_id": alias, "relation": "", "boundary_scope": "",
               "visibility": "", "context_sufficient": "", "transition_shape": "",
               "continuous_contact": "", "release_observed": "", "why_one_line": ""}
        (primary if eid in selected else reserve).append(row)
        # IDs such as 'gt_boundary' and 'false_near_edge' leak the old answer.
        private_key.append({"review_id": alias, "event_id": eid,
                            "recording_id": recording_id(eid),
                            "candidate_time_s": time_s,
                            "cohort": ("legacy_definition_recheck" if eid in legacy else
                                       "expansion" if eid in selected else "reserve")})
    return primary, reserve, private_key


def coverage_audit(selected_key, visual, hand, cfg):
    vi, hi = unique_index(visual), unique_index(hand)
    counts, missing = Counter(), []
    for row in selected_key:
        eid = row["event_id"]
        has_v = any(math.isfinite(x) for x in numeric(vi.get(eid, {}), cfg["feature_groups"]["visual"]))
        has_h = any(math.isfinite(x) for x in numeric(hi.get(eid, {}), cfg["feature_groups"]["hand"]))
        counts["both_signals" if has_v and has_h else "visual_only" if has_v else
               "hand_only" if has_h else "neither_signal"] += 1
        if not has_v or not has_h:
            missing.append({"review_id": row["review_id"], "event_id": eid,
                            "recording_id": row["recording_id"],
                            "candidate_time_s": row["candidate_time_s"],
                            "visual_input": ("available" if has_v else "row_absent" if eid not in vi else "signal_missing"),
                            "hand_input": ("available" if has_h else "row_absent" if eid not in hi else "signal_missing")})
    return {"status": "coverage_only_not_used_for_selection", "events": len(selected_key),
            "signal_coverage": dict(counts), "accepted_v2_labels": 0, "missing_inputs": missing}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/auditor/interaction_evidence_v1.json")
    parser.add_argument("--final_test_manifest", default="data/splits/final_test_v1.json")
    parser.add_argument("--relations", default="data/gold/relation_audit_annotator1_filled.csv")
    parser.add_argument("--migrated", default="data/gold/pair_schema_v2_migrated.csv")
    parser.add_argument("--contract", default="docs/interaction_evidence_v2_target_contract.md")
    parser.add_argument("--pilot_report", required=True)
    parser.add_argument("--n_events", type=int)
    parser.add_argument("--min_separation_s", type=float, default=2.0)
    parser.add_argument("--max_per_recording", type=int, default=3)
    parser.add_argument("--sampling_protocol")
    parser.add_argument("--visual_csv")
    parser.add_argument("--hand_csv")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)
    if bool(args.visual_csv) != bool(args.hand_csv):
        parser.error("Supply both feature CSVs for a post-selection coverage audit")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {out}")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    forbidden = forbidden_ids(args.final_test_manifest, cfg["final_test_manifest_sha256"])
    migrated, raw = read_csv(args.migrated), read_csv(args.relations)
    targets = explicit_targets(raw)
    primary, reserve, key = prepare_packet(migrated, targets, forbidden, cfg["seed"],
                                            args.n_events, args.min_separation_s, args.max_per_recording)
    pilot = json.loads(Path(args.pilot_report).read_text(encoding="utf-8"))
    for field in ("relations", "migrated", "final_test_manifest"):
        if pilot["sources"][field]["sha256"] != sha256(getattr(args, field)):
            raise ValueError(f"Pilot source changed: {field}")
    selected_key = [r for r in key if r["cohort"] != "reserve"]
    counts = Counter(r["recording_id"] for r in selected_key)
    file_args = ("config", "final_test_manifest", "relations", "migrated", "contract",
                 "pilot_report", "sampling_protocol", "visual_csv", "hand_csv")
    report = {"status": "prepared_not_annotated", "contract_version": "v2_draft",
              "scope": "development_target_definition_audit_not_confirmatory",
              "primary_events": len(primary), "reserve_events": len(reserve),
              "legacy_events_retained": len(targets), "new_events": len(primary) - len(targets),
              "recordings": len(counts), "recordings_with_multiple_events": sum(n >= 2 for n in counts.values()),
              "events_per_recording_histogram": dict(sorted(Counter(counts.values()).items())),
              "selection": {"method": "recording_pair_expansion_v1" if args.n_events is not None else "legacy_only",
                            "requested_events": args.n_events, "seed": cfg["seed"],
                            "min_separation_s": args.min_separation_s,
                            "max_per_recording": args.max_per_recording,
                            "uses_label_values_scores_or_feature_coverage": False},
              "protected_recordings": len(forbidden), "clips_prepared": False,
              "legacy_labels_accepted_as_v2_gold": False,
              "reserve_is_not_a_request_to_relabel_every_event": True,
              "pilot_model_or_threshold_changed": False,
              "sources": {field: {"path": str(Path(getattr(args, field)).resolve()),
                                   "sha256": sha256(getattr(args, field))}
                          for field in file_args if getattr(args, field)},
              "code_sha256": sha256(__file__)}
    missing = []
    if args.visual_csv:
        visual, hand = read_csv(args.visual_csv), read_csv(args.hand_csv)
        for table in (visual, hand):
            assert_development(table, forbidden)
        report["feature_coverage"] = coverage_audit(selected_key, visual, hand, cfg)
        missing = report["feature_coverage"].pop("missing_inputs")
        report["feature_coverage"]["missing_input_manifest"] = "private_missing_features.csv" if missing else None
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("review.csv", primary), ("reserve.csv", reserve), ("private_key.csv", key)):
        if rows:
            dump_csv(out / name, rows)
    if missing:
        dump_csv(out / "private_missing_features.csv", missing)
    dump_json(out / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "code_sha256")}, indent=2))
    return report


if __name__ == "__main__":
    main()
