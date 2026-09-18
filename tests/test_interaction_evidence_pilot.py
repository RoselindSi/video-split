import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.auditor.boundary.interaction_evidence_pilot import (
    RELATIONS, assert_development, build_pools, explicit_targets, fold_predict,
    forbidden_ids, historical_pilot_audit, main, metrics, numeric,
    paired_bootstrap, sha256, target_audit, unique_index,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "configs/auditor/interaction_evidence_v1.json").read_text())
HAS_ML = importlib.util.find_spec("sklearn") is not None


def target(n=1, rel="new_action"):
    return {"event_id": f"recording_{n:06d}_candidate_t12.0",
            "recording_id": f"recording_{n:06d}", "candidate_time": 12.0,
            "relation": rel, "target": RELATIONS.get(rel),
            "target_source": "explicit_relation_audit"}


class TargetAuditTest(unittest.TestCase):
    def test_explicit_relation_not_release_or_old_morphology(self):
        rows = []
        for n, rel in enumerate(["new_action", "same_action_new_instance", "same_instance",
                                 "cannot_determine", "terminal_action_end"], start=1):
            rows.append({"event_id": target(n)["event_id"], "your_call(label)": rel,
                         "release": "yes", "temporal_pair_subtype": "NO_TRANSITION"})
        actual = explicit_targets(rows)
        self.assertEqual([r["target"] for r in actual], [1, 1, 0, None, None])

    def test_unknown_or_duplicate_labels_fail(self):
        row = {"event_id": target()["event_id"], "your_call": "UNKNOWN"}
        with self.assertRaises(ValueError):
            explicit_targets([row])
        row["your_call"] = "same_instance"
        with self.assertRaises(ValueError):
            explicit_targets([row, row])

    def test_final_test_rows_and_disagreeing_recording_fail(self):
        with self.assertRaisesRegex(ValueError, "Final-test overlap"):
            assert_development([target()], {"recording_000001"})
        bad = {**target(), "recording_id": "recording_000002"}
        with self.assertRaisesRegex(ValueError, "disagrees"):
            assert_development([bad], set())

    def test_manifest_integrity_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "split.json"
            p.write_text(json.dumps({"selected": [{"recording_id": "recording_000999"}],
                                     "n_selected_recordings": 1}))
            self.assertEqual(forbidden_ids(p, sha256(p)), {"recording_000999"})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                forbidden_ids(p, "0" * 64)

    def test_migration_never_fills_relation_target_or_blind_queue(self):
        migrated = [{"event_id": target(1)["event_id"], "instance_relation": "same_instance",
                     "relation_source": "prose", "evidence": "release", "score": 0.9},
                    {"event_id": target(2)["event_id"], "instance_relation": "new_action",
                     "relation_source": "explicit"}]
        report, queue = target_audit(migrated, [target(2)], CONFIG)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["instance_relation"], "")
        self.assertNotIn("evidence", queue[0])
        self.assertNotIn("score", queue[0])
        self.assertEqual(report["explicit_rows"], 1)
        self.assertFalse(report["binary_support"]["sufficient"])
        with self.assertRaisesRegex(ValueError, "Duplicate event"):
            target_audit(migrated + migrated, [target(2)], CONFIG)

    def test_historical_synthetic_signature_quarantined(self):
        r = {"frames": "41", "hand_visible_frac": "1", "object_visible_frac": str(35 / 41),
             "n_release": "1", "n_target_switch": "1"}
        report = historical_pilot_audit([r] * 36)
        self.assertTrue(report["matches_scripted_synthetic_summary"])
        self.assertFalse(report["usable_as_features"])
        self.assertFalse(historical_pilot_audit([])["usable_as_features"])

    def test_missing_is_not_zero_and_stored_targets_never_features(self):
        import math
        vals = numeric({"a": "", "b": "inf", "c": "0", "y": "1"}, ["a", "b", "c", "d"])
        self.assertTrue(all(math.isnan(vals[k]) for k in (0, 1, 3)))
        self.assertEqual(vals[2], 0)
        r = target()
        vr = {"event_id": r["event_id"], "y": 1, "score": 1}
        pool, excluded = build_pools([r], [vr], [vr], CONFIG)
        self.assertEqual(pool, [])
        self.assertIn("no_visual_signal", excluded[0]["reason"])

    def test_common_pool_and_unobservable_rows_preserved(self):
        rows = [target(), target(2, "cannot_determine"), target(3)]
        visual = [{"event_id": r["event_id"], "conc_0.5s": "0.4"} for r in rows]
        hand = [{"event_id": r["event_id"], "pre_motion_speed": "0"} for r in rows[:2]]
        pool, excluded = build_pools(rows, visual, hand, CONFIG)
        self.assertEqual(len(pool), 2)
        self.assertIsNone(pool[1]["target"])
        self.assertEqual(excluded[0]["event_id"], rows[2]["event_id"])

    def test_end_to_end_audit_without_ml_or_remote_data(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            split = p / "split.json"
            split.write_text(json.dumps({"selected": [{"recording_id": "recording_000999"}],
                                        "n_selected_recordings": 1}))
            cfg = copy.deepcopy(CONFIG)
            cfg["final_test_manifest_sha256"] = sha256(split)
            (p / "config.json").write_text(json.dumps(cfg))
            (p / "relations.csv").write_text("event_id,your_call\n" + target()["event_id"] + ",new_action\n")
            (p / "migrated.csv").write_text("event_id,instance_relation,relation_source\n" +
                                             target()["event_id"] + ",UNKNOWN,\n")
            args = ["--config", str(p / "config.json"), "--final_test_manifest", str(split),
                    "--relations", str(p / "relations.csv"), "--migrated", str(p / "migrated.csv"),
                    "--out_dir", str(p / "out")]
            result = main(args)
            self.assertEqual(result["target_audit"]["explicit_rows"], 1)
            self.assertTrue((p / "out/report.json").is_file())
            with self.assertRaises(FileExistsError):
                main(args)


@unittest.skipUnless(HAS_ML, "optional scikit-learn dependency")
class EvaluationTest(unittest.TestCase):
    def test_csv_to_evaluation_report_underpowered_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            split = p / "split.json"
            split.write_text(json.dumps({"selected": [{"recording_id": "recording_000999"}],
                                        "n_selected_recordings": 1}))
            cfg = {**CONFIG, "final_test_manifest_sha256": sha256(split),
                   "bootstrap_replicates": 10}
            (p / "config.json").write_text(json.dumps(cfg))
            tables = {"relations": [], "migrated": [], "visual": [], "hand": [], "subtypes": []}
            for n in range(1, 5):
                for y in (0, 1):
                    eid = f"recording_{n:06d}_candidate_t{12 + y}.0"
                    tables["relations"].append({"event_id": eid, "your_call": "new_action" if y else "same_instance"})
                    tables["migrated"].append({"event_id": eid, "instance_relation": "UNKNOWN", "relation_source": ""})
                    tables["visual"].append({"event_id": eid, "conc_0.5s": y})
                    tables["hand"].append({"event_id": eid, "pre_motion_speed": y})
                    tables["subtypes"].append({"event_id": eid, "subtype": "regrasp_reposition"})
            for name, rows in tables.items():
                with (p / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            args = ["--config", str(p / "config.json"), "--final_test_manifest", str(split),
                    "--relations", str(p / "relations.csv"), "--migrated", str(p / "migrated.csv"),
                    "--negative_subtypes", str(p / "subtypes.csv"), "--visual_csv", str(p / "visual.csv"),
                    "--hand_csv", str(p / "hand.csv"), "--out_dir", str(p / "out")]
            result = main(args)
            stored = json.loads((p / "out/report.json").read_text())
            self.assertEqual(stored, result)
            self.assertEqual(result["evidence_status"], "insufficient_scored_class_support")
            self.assertFalse(result["promotion_allowed"])
            self.assertEqual(result["scored_support"]["events_per_class"], {"0": 4, "1": 4})
            self.assertEqual(result["arms"]["visual_change_shape"]["auroc"], 1.0)

    def rows(self):
        rows = []
        for n in range(1, 13):
            for y in (0, 1):
                r = target(n, "new_action" if y else "same_instance")
                r["event_id"] += f"_{y}"
                r.update(visual=[float(y), n / 20], hand=[float(y), float("nan")], quality=[1.0])
                rows.append(r)
        return rows

    def test_recording_disjoint_folds_same_pool_and_useful_baseline(self):
        predictions, folds, status = fold_predict(self.rows(), CONFIG)
        self.assertEqual(status, "exploratory_oof")
        self.assertEqual(len(predictions), 24)
        for fold in folds:
            self.assertFalse(set(fold["train_recordings"]) & set(fold["test_recordings"]))
        self.assertTrue(all(all(s is not None for s in r["scores"].values()) for r in predictions))
        a = metrics(predictions, "visual_change_shape", 0.5)
        self.assertGreater(a["auroc"], 0.99)
        self.assertEqual(a["within_recording_pair_accuracy"], 1)
        self.assertEqual(metrics(predictions, "missingness_only", 0.5)["auroc"], 0.5)

    def test_test_fold_values_cannot_change_its_training_predictions(self):
        rows = self.rows()
        p, folds, _ = fold_predict(rows, CONFIG)
        altered = copy.deepcopy(rows)
        test_groups = set(folds[0]["test_recordings"])
        for r in altered:
            if r["recording_id"] in test_groups:
                r["target"] = 1 - r["target"]
        q, _, _ = fold_predict(altered, CONFIG)
        for a, b in zip(p, q):
            if a["recording_id"] in test_groups:
                self.assertEqual(a["scores"], b["scores"])

    def test_missing_and_single_class_metrics_are_undefined(self):
        r = target()
        r["scores"] = {"x": 0.8}
        self.assertIsNone(metrics([r], "x", 0.5)["auroc"])
        self.assertIsNone(metrics([r], "x", 0.5)["candidate_fpr"])
        self.assertIsNone(metrics([r], "x", 0.5)["within_recording_pair_accuracy"])

    def test_paired_bootstrap_same_arm_is_zero(self):
        predictions, _, _ = fold_predict(self.rows(), CONFIG)
        result = paired_bootstrap(predictions, "visual_change_shape", "visual_change_shape",
                                  {**CONFIG, "bootstrap_replicates": 30})
        self.assertEqual(result["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
