"""Local CPU unit tests for the multi-segment rewards. No GPU, no model.

Run:  python -m pytest tests/test_seg_rewards.py -v
  or: python tests/test_seg_rewards.py   (falls back to plain asserts)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rewards.seg_rewards import (  # noqa: E402
    parse_segments,
    greedy_match,
    format_seg_reward,
    iou_seg_reward,
    seq_reward,
    name_seg_reward,
)
from prompt.seg_prompt import render_target  # noqa: E402


GT = [
    ["Adjust and fold first outer box flap", 2.6, 15.3],
    ["Align inner box flaps", 15.3, 19.8],
    ["Fold second outer box flap into position", 19.8, 25.7],
]


def _completion(segments, think="reasoning here"):
    return "<think>\n" + think + "\n</think>\n" + render_target(segments).split("</think>\n")[-1]


def test_parse():
    comp = _completion([("a", 1.0, 2.0), ("b", 2.0, 3.5)])
    segs = parse_segments(comp)
    assert segs == [("a", 1.0, 2.0), ("b", 2.0, 3.5)], segs


def test_format():
    good = _completion([("a", 1.0, 2.0), ("b", 2.0, 3.0)])
    bad_struct = "<segments><seg><name>a</name><span>1 to 2</span></seg></segments>"
    non_chrono = _completion([("b", 5.0, 6.0), ("a", 1.0, 2.0)])
    inverted = _completion([("a", 3.0, 1.0)])
    r = format_seg_reward([good, bad_struct, non_chrono, inverted])
    assert r[0] == 1.0, r
    assert r[1] == 0.0, r      # missing <think>
    assert r[2] == 0.5, r      # parses but not chronological
    assert r[3] == 0.5, r      # inverted span


def test_greedy_match():
    preds = [("x", 2.5, 15.0), ("y", 15.0, 20.0)]
    pairs = greedy_match(preds, GT)
    matched_gt = sorted(gj for _, gj, _ in pairs)
    assert matched_gt == [0, 1], pairs   # two best matches, each used once


def test_iou_perfect_vs_wrong():
    perfect = _completion([(n, s, e) for n, s, e in GT])
    wrong = _completion([("z", 40.0, 60.0)])
    rp = iou_seg_reward([perfect], [GT], durations=[25.7])[0]
    rw = iou_seg_reward([wrong], [GT], durations=[25.7])[0]
    assert rp > 0.95, rp
    assert rw == 0.0, rw


def test_iou_missed_segment_penalized():
    # only 1 of 3 GT predicted perfectly -> reward ~= 1/3
    partial = _completion([(GT[0][0], GT[0][1], GT[0][2])])
    r = iou_seg_reward([partial], [GT], durations=[25.7])[0]
    assert 0.30 < r < 0.36, r


def test_seq_coverage_and_count():
    perfect = _completion([(n, s, e) for n, s, e in GT])
    r_perfect = seq_reward([perfect], [GT])[0]
    assert r_perfect > 0.95, r_perfect

    # one blanket segment covering everything: coverage high, count wrong
    blanket = _completion([("all", 2.6, 25.7)])
    r_blanket = seq_reward([blanket], [GT])[0]
    assert r_blanket < r_perfect, (r_blanket, r_perfect)


def test_seq_overlap_penalized():
    overlapping = _completion([("a", 2.6, 20.0), ("b", 10.0, 25.7)])
    r = seq_reward([overlapping], [GT])[0]
    non_overlap = seq_reward([_completion([(n, s, e) for n, s, e in GT])], [GT])[0]
    assert r < non_overlap, (r, non_overlap)


def test_solution_as_json_string():
    # rewards must accept solution stored as a JSON string (Arrow-safe path)
    import json
    perfect = _completion([(n, s, e) for n, s, e in GT])
    sol_str = json.dumps(GT)
    r = iou_seg_reward([perfect], [sol_str], durations=[25.7])[0]
    assert r > 0.95, r
    rs = seq_reward([perfect], [sol_str])[0]
    assert rs > 0.95, rs


def test_name_with_stub_sim():
    # stub similarity: exact string match = 1.0, else 0.0
    def stub(a, bs):
        return [1.0 if a == b else 0.0 for b in bs]

    good = _completion([(n, s, e) for n, s, e in GT])
    r = name_seg_reward([good], [GT], sim_fn=stub)[0]
    assert r > 0.95, r

    generic = _completion([("box flap", 2.6, 15.3)])  # matches nothing exactly
    rg = name_seg_reward([generic], [GT], sim_fn=stub)[0]
    assert rg == 0.0, rg


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
