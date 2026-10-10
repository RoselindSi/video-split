"""Properties the fragment join has to hold, because it runs before the verdicts.

These are the three that make it safe to put in front of `post_pass` rather
than behind it: it can only merge, running it twice is running it once, and the
cases it is not sure about it leaves alone. The time-overlap veto is tested
separately because the audit that motivated the rule only ever looked at
positive gaps -- a rule that is correct only because of how its input happened
to be filtered is not a rule.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rig.track_join import (apply_to, fragments,
                                join_to_fixed_point, WINDOW)


def row(frame, tid, x0=100, y0=100, w=50, side="right", not_hand="0"):
    return {"frame": str(frame), "tid": tid, "side": side,
            "not_hand": not_hand, "own": "1",
            "x0": str(x0), "y0": str(y0),
            "x1": str(x0 + w), "y1": str(y0 + w)}


def seq(tid, a, b, x0=100, **kw):
    return [row(f, tid, x0=x0, **kw) for f in range(a, b + 1)]


class Join(unittest.TestCase):
    def pipeline(self, rows):
        frag = fragments(rows)
        mapping, _rounds = join_to_fixed_point(frag)
        new_rows, _ = apply_to(rows, mapping)
        return frag, mapping, new_rows

    def test_single_candidate_is_joined(self):
        rows = seq("1", 0, 20) + seq("2", 30, 50)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["2"], "1")

    def test_two_candidates_left_alone(self):
        # 两条都在窗口内、都够近、且彼此时间重叠（并不起来）-> 歧义不消解，
        # 不动点迭代也救不了，规则必须保持不动
        rows = seq("1", 0, 20) + seq("2", 25, 40) + seq("3", 26, 41, x0=110)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["2"], "2")
        self.assertEqual(mapping["3"], "3")

    def test_far_candidate_not_joined(self):
        rows = seq("1", 0, 20) + seq("2", 30, 50, x0=900)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["2"], "2")

    def test_beyond_window_not_joined(self):
        rows = seq("1", 0, 20) + seq("2", 20 + WINDOW + 5, 60)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["2"], "2")

    def test_time_overlap_is_vetoed(self):
        # 同一位置但时间重叠：一只手不可能同时在两处
        rows = seq("1", 0, 40) + seq("2", 20, 60)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["2"], "2")

    def test_transitive_chain_becomes_one(self):
        rows = seq("1", 0, 10) + seq("2", 20, 30) + seq("3", 40, 50)
        frag, mapping, _r = self.pipeline(rows)
        self.assertEqual(len({mapping[t] for t in frag}), 1)

    def test_merge_only_never_increases(self):
        rows = seq("1", 0, 10) + seq("2", 20, 30) + seq("3", 40, 50)
        frag, mapping, _r = self.pipeline(rows)
        self.assertLessEqual(len({mapping[t] for t in frag}), len(frag))

    def test_idempotent(self):
        rows = seq("1", 0, 10) + seq("2", 20, 30) + seq("3", 40, 50)
        _f, _m, once = self.pipeline(rows)
        _f2, _m2, twice = self.pipeline(once)
        self.assertEqual([r["tid"] for r in once], [r["tid"] for r in twice])

    def test_side_recomputed_over_merged_track(self):
        # 碎片内部 side 有噪声；合并后按整轨多数票重算
        rows = (seq("1", 0, 10, side="right")
                + seq("2", 20, 30, side="left")[:2]
                + seq("2", 32, 40, side="right"))
        _f, mapping, new_rows = self.pipeline(rows)
        self.assertEqual(mapping["2"], "1")
        self.assertEqual({r["side"] for r in new_rows}, {"right"})

    def test_non_hand_rows_ignored(self):
        # not_hand 的框不参与接续判断
        rows = seq("1", 0, 20) + seq("2", 25, 35, not_hand="1") + seq("3", 30, 50)
        _f, mapping, _r = self.pipeline(rows)
        self.assertEqual(mapping["3"], "1")

    def test_other_columns_untouched(self):
        rows = seq("1", 0, 10) + seq("2", 20, 30)
        _f, _m, new_rows = self.pipeline(rows)
        self.assertEqual([r["own"] for r in new_rows], ["1"] * len(rows))
        self.assertEqual([r["frame"] for r in new_rows],
                         [r["frame"] for r in rows])


if __name__ == "__main__":
    unittest.main(verbosity=2)
