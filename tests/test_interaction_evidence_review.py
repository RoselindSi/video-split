import copy
import unittest
from collections import Counter

from src.auditor.boundary.interaction_evidence_review import coverage_audit, prepare_packet


class ReviewPacketTest(unittest.TestCase):
    def inputs(self):
        events = ["recording_000001_gt_boundary_t12.0",
                  "recording_000002_false_near_edge_t14.5",
                  "recording_000003_raw_change_peak_t20.0"]
        rows = [{"event_id": e, "instance_relation": "new_action", "score": "0.99"}
                for e in events]
        targets = [{"event_id": e, "relation": "same_instance"} for e in events[:2]]
        return rows, targets

    def test_blind_aliases_hide_labels_scores_and_candidate_origin(self):
        rows, targets = self.inputs()
        primary, reserve, key = prepare_packet(rows, targets, set(), 1)
        self.assertEqual((len(primary), len(reserve), len(key)), (2, 1, 3))
        for row in primary + reserve:
            self.assertRegex(row["review_id"], r"^review_\d{4}$")
            self.assertTrue(all(v == "" for k, v in row.items() if k != "review_id"))
        self.assertEqual({r["event_id"] for r in key}, {r["event_id"] for r in rows})
        self.assertEqual({r["review_id"] for r in key},
                         {r["review_id"] for r in primary + reserve})

    def test_labels_scores_and_input_order_do_not_affect_packet(self):
        rows, targets = self.inputs()
        expected = prepare_packet(rows, targets, set(), 2)
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update(score="0.0", instance_relation="same_instance")
        for row in targets:
            row["relation"] = "new_action"
        self.assertEqual(prepare_packet(list(reversed(changed)), targets, set(), 2), expected)
        self.assertEqual(prepare_packet(list(reversed(changed)), targets, set(), 2, n_events=3),
                         prepare_packet(rows, targets, set(), 2, n_events=3))

    def test_expand_to_150_keeps_54_aliases_and_caps_recordings(self):
        rows = [{"event_id": f"recording_{n:06d}_candidate_t{t}.0"}
                for n in range(1, 81) for t in (10, 30, 50)]
        targets = [{"event_id": r["event_id"]} for r in rows[:54]]
        old, _, old_key = prepare_packet(rows, targets, set(), 7)
        primary, reserve, key = prepare_packet(rows, targets, set(), 7, n_events=150)
        self.assertEqual((len(primary), len(reserve)), (150, 90))
        self.assertTrue({r["review_id"] for r in old} <= {r["review_id"] for r in primary})
        self.assertEqual({r["event_id"]: r["review_id"] for r in old_key},
                         {r["event_id"]: r["review_id"] for r in key})
        counts = Counter(r["recording_id"] for r in key if r["cohort"] != "reserve")
        self.assertLessEqual(max(counts.values()), 3)
        self.assertGreaterEqual(sum(n >= 2 for n in counts.values()), 50)
        self.assertEqual(sum(r["cohort"] == "expansion" for r in key), 96)

    def test_pair_legacy_singleton_and_exclude_near_duplicate(self):
        ids = ["recording_000001_anchor_t10.0", "recording_000001_other_t10.5",
               "recording_000001_other_t25.0", "recording_000002_other_t4.0",
               "recording_000002_other_t15.0"]
        rows = [{"event_id": eid} for eid in ids]
        _, _, key = prepare_packet(rows, rows[:1], set(), 3, n_events=4)
        selected = {r["event_id"] for r in key if r["cohort"] != "reserve"}
        self.assertEqual(selected, {ids[0], ids[2], ids[3], ids[4]})
        with self.assertRaisesRegex(ValueError, "Cannot reach"):
            prepare_packet(rows, rows[:1], set(), 3, n_events=5)

    def test_invalid_size_and_retained_constraints_fail(self):
        rows, targets = self.inputs()
        for n in (1, 4):
            with self.assertRaisesRegex(ValueError, "Requested size"):
                prepare_packet(rows, targets, set(), 1, n_events=n)
        close = [{"event_id": f"recording_000001_candidate_t{t}"} for t in (10.0, 10.5)]
        with self.assertRaisesRegex(ValueError, "Legacy events violate"):
            prepare_packet(close, close, set(), 1, n_events=2)
        crowded = [{"event_id": f"recording_000001_candidate_t{t}"} for t in (0.0, 10.0, 20.0, 30.0)]
        with self.assertRaisesRegex(ValueError, "exceed the recording cap"):
            prepare_packet(crowded, crowded, set(), 1, n_events=4)

    def test_post_selection_coverage_never_claims_new_gold(self):
        rows, targets = self.inputs()
        _, _, key = prepare_packet(rows, targets, set(), 2, n_events=3)
        cfg = {"feature_groups": {"visual": ["v"], "hand": ["h"]}}
        coverage = coverage_audit(key, [{"event_id": rows[0]["event_id"], "v": "0"}],
                                  [{"event_id": rows[0]["event_id"], "h": "1"}], cfg)
        self.assertEqual(coverage["signal_coverage"], {"both_signals": 1, "neither_signal": 2})
        self.assertEqual(coverage["accepted_v2_labels"], 0)
        self.assertEqual(len(coverage["missing_inputs"]), 2)
        self.assertTrue(all(r["hand_input"] == "row_absent" for r in coverage["missing_inputs"]))

    def test_final_test_duplicates_missing_time_and_cohort_mismatch_fail(self):
        rows, targets = self.inputs()
        with self.assertRaisesRegex(ValueError, "Final-test overlap"):
            prepare_packet(rows, targets, {"recording_000001"}, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate event"):
            prepare_packet(rows + rows, targets, set(), 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_packet(rows[1:], targets, set(), 1)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            prepare_packet([{"event_id": "recording_000001_missing"}], [], set(), 1)


if __name__ == "__main__":
    unittest.main()
