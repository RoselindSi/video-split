"""Development-only target audit and evidence pilot; no final-test scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

RELATIONS = {"same_instance": 0, "same_action_new_instance": 1, "new_action": 1}
EXCLUDED = {"cannot_determine", "initial_action_start", "terminal_action_end"}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def recording_id(event_id):
    match = re.match(r"^(recording_\d{6})(?:_|$)", event_id)
    if not match:
        raise ValueError(f"Invalid event/recording ID: {event_id!r}")
    return match[1]


def forbidden_ids(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError("Final-test manifest hash mismatch; refusing pilot")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = {r["recording_id"] for r in data["selected"]}
    if not selected or len(selected) != data["n_selected_recordings"]:
        raise ValueError("Invalid final-test selection")
    return selected


def assert_development(rows, forbidden):
    found = set()
    for row in rows:
        rid = recording_id(row.get("event_id") or row["recording_id"])
        if row.get("recording_id") not in (None, "", rid):
            raise ValueError(f"Recording ID disagrees with event ID: {row}")
        if rid in forbidden:
            found.add(rid)
    if found:
        raise ValueError(f"Final-test overlap forbidden: {sorted(found)}")


def unique_index(rows):
    out = {}
    for row in rows:
        eid = row["event_id"]
        if eid in out:
            raise ValueError(f"Duplicate event ID: {eid}")
        out[eid] = row
    return out


def explicit_targets(rows):
    out = []
    for row in rows:
        columns = [k for k in row if k and k.startswith("your_call")]
        if len(columns) != 1:
            raise ValueError("Expected one explicit your_call relation column")
        relation = row[columns[0]].strip()
        if relation not in set(RELATIONS) | EXCLUDED:
            raise ValueError(f"Unknown explicit relation {relation!r}")
        eid = row["event_id"]
        match = re.search(r"_t(\d+(?:\.\d+)?)$", eid)
        if not match:
            raise ValueError(f"Event lacks candidate time: {eid}")
        out.append({"event_id": eid, "recording_id": recording_id(eid),
                    "candidate_time": float(match[1]), "relation": relation,
                    "target": RELATIONS.get(relation),
                    "target_source": "explicit_relation_audit"})
    unique_index(out)
    return out


def support(rows, cfg):
    counts = {str(y): sum(r["target"] == y for r in rows) for y in (0, 1)}
    groups = {str(y): len({r["recording_id"] for r in rows if r["target"] == y})
              for y in (0, 1)}
    return {"events_per_class": counts, "recordings_per_class": groups,
            "sufficient": min(counts.values()) >= cfg["minimum_class_events"]
            and min(groups.values()) >= cfg["minimum_class_recordings"]}


def target_audit(migrated, targets, cfg):
    unique_index(migrated)
    known = unique_index(targets)
    queue = []
    for row in sorted(migrated, key=lambda r: hashlib.sha256(
            f"{cfg['seed']}:{r['event_id']}".encode()).hexdigest()):
        eid = row["event_id"]
        if eid in known:
            continue
        match = re.search(r"_t(\d+(?:\.\d+)?)$", eid)
        queue.append({"event_id": eid, "recording_id": recording_id(eid),
                      "candidate_time_s": float(match[1]) if match else "",
                      "instance_relation": "", "transition_shape": "",
                      "visibility": "", "contact_release_evidence": "",
                      "continuous_contact_transition": "",
                      "negative_subtype": "", "annotator_id": "",
                      "why_one_line": ""})
    report = {
        "scope": "development_only_not_a_new_confirmatory_gold",
        "migrated_rows": len(migrated),
        "legacy_relation_counts": dict(Counter(r["instance_relation"] for r in migrated)),
        "legacy_source_counts": dict(Counter(r["relation_source"] for r in migrated)),
        "explicit_rows": len(targets),
        "explicit_relation_counts": dict(Counter(r["relation"] for r in targets)),
        "binary_support": support(targets, cfg),
        "reannotation_queue_rows": len(queue),
        "unmeasured_strata": ["continuous_contact_transition", "release_without_boundary",
                               "human_observability_at_candidate"],
        "blocked_cues": cfg["unavailable_cues"],
    }
    return report, queue


def historical_pilot_audit(rows):
    keys = ("frames", "hand_visible_frac", "object_visible_frac",
            "n_release", "n_target_switch")
    distributions = {k: dict(Counter(r.get(k, "") for r in rows)) for k in keys}
    signature = bool(rows) and all(
        int(float(r.get("frames", "0"))) == 41
        and float(r.get("hand_visible_frac", "0")) == 1
        and abs(float(r.get("object_visible_frac", "0")) - 35 / 41) < 1e-9
        and float(r.get("n_release", "0")) == 1
        and float(r.get("n_target_switch", "0")) == 1 for r in rows)
    return {"n": len(rows), "distributions": distributions,
            "matches_scripted_synthetic_summary": signature,
            "status": "quarantined_unverified_provenance", "usable_as_features": False}


def numeric(row, names):
    values = []
    for name in names:
        try:
            value = float(row[name])
        except (KeyError, ValueError, TypeError):
            value = float("nan")
        values.append(value if math.isfinite(value) else float("nan"))
    return values


def build_pools(targets, visual_rows, hand_rows, cfg):
    visual, hand = unique_index(visual_rows), unique_index(hand_rows)
    names = cfg["feature_groups"]
    included, excluded = [], []
    for row in targets:
        eid = row["event_id"]
        v = numeric(visual.get(eid, {}), names["visual"])
        h = numeric(hand.get(eid, {}), names["hand"])
        q = numeric(hand.get(eid, {}), names["quality"])
        reasons = []
        if not any(math.isfinite(x) for x in v):
            reasons.append("no_visual_signal")
        if not any(math.isfinite(x) for x in h):
            reasons.append("no_hand_signal")
        if reasons:
            excluded.append({**row, "reason": "+".join(reasons)})
        else:
            included.append({**row, "visual": v, "hand": h, "quality": q})
    return included, excluded


def fold_predict(rows, cfg):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labeled = [r for r in rows if r["target"] is not None]
    if len({r["recording_id"] for r in labeled}) < 2:
        return [], [], "insufficient_recording_groups"
    y = np.asarray([r["target"] for r in labeled])
    groups = np.asarray([r["recording_id"] for r in labeled])
    v = np.asarray([r["visual"] for r in rows], float)
    h = np.asarray([r["hand"] for r in rows], float)
    q = np.asarray([r["quality"] for r in rows], float)
    # Identical quality/missingness controls in all arms expose shortcut gains.
    quality = np.column_stack([q, ~np.isfinite(np.column_stack([v, h, q]))])
    matrices = {"visual_change_shape": np.column_stack([v, quality]),
                "hand_kinematics": np.column_stack([h, quality]),
                "visual_plus_hand": np.column_stack([v, h, quality]),
                "missingness_only": quality}
    index = {r["event_id"]: i for i, r in enumerate(rows)}
    positions = np.asarray([index[r["event_id"]] for r in labeled])
    preds = {arm: np.full(len(rows), np.nan) for arm in matrices}
    folds = []
    splitter = GroupKFold(n_splits=min(cfg["folds"], len(set(groups))))
    for k, (tr, te) in enumerate(splitter.split(positions, y, groups)):
        training, testing = set(groups[tr]), set(groups[te])
        assert not training & testing
        apply = np.asarray([i for i, r in enumerate(rows) if r["recording_id"] in testing])
        fold = {"fold": k, "train_recordings": sorted(training),
                "test_recordings": sorted(testing), "train_events": len(tr),
                "test_events": len(te), "status": "ok"}
        if len(set(y[tr])) != 2:
            fold["status"] = "withheld_single_class_training"
            folds.append(fold)
            continue
        for arm, matrix in matrices.items():
            # Drop all-missing training columns; never inspect test statistics.
            available = np.isfinite(matrix[positions[tr]]).any(axis=0)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=cfg["logistic_C"],
                                                     class_weight="balanced",
                                                     max_iter=2000, random_state=cfg["seed"]))
            model.fit(matrix[positions[tr]][:, available], y[tr])
            preds[arm][apply] = model.predict_proba(matrix[apply][:, available])[:, 1]
        folds.append(fold)
    out = [{k: v for k, v in row.items() if k not in ("visual", "hand", "quality")}
           for row in rows]
    for i, row in enumerate(out):
        row["scores"] = {arm: float(p[i]) if np.isfinite(p[i]) else None
                         for arm, p in preds.items()}
    return out, folds, "exploratory_oof"


def metrics(rows, arm, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    valid = [r for r in rows if r["target"] is not None and r["scores"][arm] is not None]
    y, p = [r["target"] for r in valid], [r["scores"][arm] for r in valid]
    counts = Counter(y)
    out = {"n": len(valid), "recordings": len({r["recording_id"] for r in valid}),
           "positives": counts[1], "negatives": counts[0], "auroc": None,
           "average_precision": None, "candidate_tpr": None, "candidate_fpr": None}
    if len(counts) == 2:
        out.update(auroc=float(roc_auc_score(y, p)),
                   average_precision=float(average_precision_score(y, p)))
    for label, field in ((0, "candidate_fpr"), (1, "candidate_tpr")):
        if counts[label]:
            out[field] = sum(s >= threshold for a, s in zip(y, p) if a == label) / counts[label]
    by = {}
    for r in valid:
        by.setdefault(r["recording_id"], []).append(r)
    accuracy, pairs = [], []
    for rr in by.values():
        accuracy.append(sum((r["scores"][arm] >= threshold) == r["target"] for r in rr) / len(rr))
        pos = [r["scores"][arm] for r in rr if r["target"] == 1]
        neg = [r["scores"][arm] for r in rr if r["target"] == 0]
        if pos and neg:
            pairs.append(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))
    out["recording_macro_accuracy"] = sum(accuracy) / len(accuracy) if accuracy else None
    out["within_recording_pair_accuracy"] = sum(pairs) / len(pairs) if pairs else None
    out["pair_eligible_recordings"] = len(pairs)
    return out


def paired_bootstrap(rows, a, b, cfg):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    by = {}
    for row in rows:
        if row["target"] is not None and all(row["scores"][x] is not None for x in (a, b)):
            by.setdefault(row["recording_id"], []).append(row)
    keys = sorted(by)
    if not keys:
        return {"ci95": None, "valid_replicates": 0}
    rng, diffs = np.random.default_rng(cfg["seed"]), []
    for _ in range(cfg["bootstrap_replicates"]):
        sample = [r for k in rng.choice(keys, size=len(keys), replace=True) for r in by[k]]
        y = [r["target"] for r in sample]
        if len(set(y)) < 2:
            continue
        diffs.append(float(roc_auc_score(y, [r["scores"][a] for r in sample]) -
                           roc_auc_score(y, [r["scores"][b] for r in sample])))
    return {"ci95": np.percentile(diffs, [2.5, 97.5]).tolist() if diffs else None,
            "valid_replicates": len(diffs), "conditional_on_fixed_oof_predictions": True}


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def dump_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/auditor/interaction_evidence_v1.json")
    ap.add_argument("--final_test_manifest", default="data/splits/final_test_v1.json")
    ap.add_argument("--relations", default="data/gold/relation_audit_annotator1_filled.csv")
    ap.add_argument("--migrated", default="data/gold/pair_schema_v2_migrated.csv")
    ap.add_argument("--negative_subtypes", default="data/gold/same_action_subtype_v1.csv")
    ap.add_argument("--visual_csv")
    ap.add_argument("--hand_csv")
    ap.add_argument("--historical_hoi_csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    if bool(args.visual_csv) != bool(args.hand_csv):
        ap.error("Supply both feature CSVs for a common-pool comparison")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    forbidden = forbidden_ids(args.final_test_manifest, cfg["final_test_manifest_sha256"])
    migrated, raw = read_csv(args.migrated), read_csv(args.relations)
    for table in (migrated, raw):
        assert_development(table, forbidden)
    targets = explicit_targets(raw)
    audit, queue = target_audit(migrated, targets, cfg)
    sources = {k: {"path": str(Path(v).resolve()), "sha256": sha256(v)}
               for k, v in vars(args).items() if v and k != "out_dir"}
    audit["protected_final_recordings"] = len(forbidden)
    result = {"stage": "E0_complete_E1_waiting_for_features", "config": cfg,
              "sources": sources, "code_sha256": sha256(__file__),
              "python": sys.version, "target_audit": audit}
    if args.historical_hoi_csv:
        historical = read_csv(args.historical_hoi_csv)
        assert_development(historical, forbidden)
        result["historical_hand_object_cache"] = historical_pilot_audit(historical)
    if args.visual_csv:
        visual, hand = read_csv(args.visual_csv), read_csv(args.hand_csv)
        for table in (visual, hand):
            assert_development(table, forbidden)
        rows, excluded = build_pools(targets, visual, hand, cfg)
        predictions, folds, status = fold_predict(rows, cfg)
        import sklearn
        import numpy as np
        scored = [r for r in predictions if all(s is not None for s in r["scores"].values())]
        scored_support = support(scored, cfg)
        result.update(stage=("E1_exploratory_pilot_complete" if scored else "E1_evaluation_withheld"),
                      evaluation_status=status,
                      evidence_status=("exploratory_only" if scored_support["sufficient"]
                                       else "insufficient_scored_class_support"),
                      sklearn=sklearn.__version__, numpy=np.__version__,
                      common_pool_support=support(rows, cfg), scored_support=scored_support,
                      excluded_features=excluded,
                      predictions=predictions, folds=folds)
        subtypes = read_csv(args.negative_subtypes)
        assert_development(subtypes, forbidden)
        subtype_by = unique_index(subtypes)
        for row in predictions:
            row["negative_subtype"] = (subtype_by.get(row["event_id"], {}).get("subtype", "unmeasured")
                                        if row["relation"] == "same_instance" else None)
        arms = ("visual_change_shape", "hand_kinematics", "visual_plus_hand", "missingness_only")
        result["arms"] = {arm: metrics(predictions, arm, cfg["threshold"]) for arm in arms}
        result["by_relation"] = {
            rel: {arm: metrics([r for r in predictions if r["relation"] == rel], arm, cfg["threshold"])
                  for arm in arms} for rel in sorted(set(RELATIONS) | EXCLUDED)}
        result["by_negative_subtype"] = {
            name: {arm: metrics([r for r in predictions if r["negative_subtype"] == name], arm, cfg["threshold"])
                   for arm in arms} for name in sorted({r["negative_subtype"] for r in predictions
                                                      if r["negative_subtype"] is not None})}
        result["paired_auroc_difference"] = {
            f"visual_plus_hand minus {arm}": paired_bootstrap(predictions, "visual_plus_hand", arm, cfg)
            for arm in ("visual_change_shape", "missingness_only")}
        result["promotion_allowed"] = False
    out.mkdir(parents=True, exist_ok=False)
    dump_json(out / "report.json", result)
    dump_csv(out / "explicit_targets.csv", targets)
    if queue:
        dump_csv(out / "reannotation_queue.csv", queue)
    print(json.dumps({"stage": result["stage"], "target_audit": audit,
                      "arms": result.get("arms"), "out_dir": str(out.resolve())}, indent=2))
    return result


if __name__ == "__main__":
    main()
