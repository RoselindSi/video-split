"""Controlled mechanism ablation for the temporal hand-filter changes.

The fixtures isolate three failures with known ground truth:

1. a moving foreign hand is missed for two frames;
2. a self hand leaves and a different hand appears nearby;
3. a real low-score continuation arrives beside a low-score false candidate.

This is a causal regression experiment, not a deployment estimate. Run the
same cumulative variants on held-out recordings before choosing thresholds.

Run:
    python3 -m src.rig.hand_filter_ablation
"""
from __future__ import annotations

import json

import numpy as np

from src.rig.hand_detect import OwnHold
from src.rig.hand_track import Tracker, box_iou


SHAPE = (900, 1600, 3)
T_NEW = 0.60
T_CONTINUE = 0.25


VARIANTS = (
    {"name": "baseline", "motion": False, "safe": False, "byte": False},
    {"name": "+motion", "motion": True, "safe": False, "byte": False},
    {"name": "+safe association", "motion": True, "safe": True,
     "byte": False},
    {"name": "+two thresholds", "motion": True, "safe": True,
     "byte": True},
)


def _det(cx, *, conf=0.9, edge="left", owner=False, owner_p=0.1,
         size=40):
    cy = 400
    box = np.array([cx - size, cy - size, cx + size, cy + size], int)
    return {"box": box, "kp": np.array([[cx, cy]] * 21, float),
            "conf": float(conf), "edge": edge, "side": "right",
            "owner": bool(owner), "owner_p": float(owner_p),
            "rule_owner": bool(owner), "exit": None}


def _tracker(variant):
    return Tracker(
        predict_motion=variant["motion"],
        rich_association=variant["safe"],
        max_assoc_cost=None if not variant["safe"] else 1.60,
        unmatched_cost=None if not variant["safe"] else 1.35)


def _update(tracker, detections, byte):
    if byte:
        return tracker.update(detections, SHAPE, new_track_conf=T_NEW,
                              continue_conf=T_CONTINUE)
    return tracker.update([d for d in detections if d["conf"] >= T_NEW],
                          SHAPE)


def _motion_iou(variant):
    tracker = _tracker(variant)
    _update(tracker, [_det(100)], variant["byte"])
    _update(tracker, [_det(120)], variant["byte"])
    scores = []
    for truth_x in (140, 160):
        _update(tracker, [], variant["byte"])
        predicted = tracker.coasting(max_prediction_age=2)[0][1]["box"]
        scores.append(box_iou(predicted, _det(truth_x)["box"]))
    return float(np.mean(scores))


def _self_identity_leak(variant):
    tracker = _tracker(variant)
    hold = OwnHold(rule_w=0.0, self_reconfirm_frames=2)
    first = _det(400, edge="bottom", owner=True, owner_p=0.95)
    ids = _update(tracker, [first], variant["byte"])
    hold.update([first], [(True, 0.95)], ids=ids)

    _update(tracker, [], variant["byte"])
    hold.update([], [], ids=[])

    newcomer = _det(405, edge="left", owner=False, owner_p=0.40)
    ids = _update(tracker, [newcomer], variant["byte"])
    flags = hold.update(
        [newcomer], [(False, 0.40)], ids=ids,
        reacquired=tracker.reacquired if variant["safe"] else ())
    return int(flags[0][0])       # 1 means the foreign hand leaked as self


def _low_conf_policy(variant):
    tracker = _tracker(variant)
    first = _update(tracker, [_det(300, conf=0.90)], variant["byte"])[0]
    before = tracker.n_new
    candidates = [_det(310, conf=0.30), _det(1200, conf=0.30)]
    ids = _update(tracker, candidates, variant["byte"])
    recovered = int(bool(ids) and ids[0] == first)
    low_score_births = tracker.n_new - before
    return recovered, int(low_score_births)


def run():
    rows = []
    for variant in VARIANTS:
        recovered, births = _low_conf_policy(variant)
        rows.append({
            "variant": variant["name"],
            "missing_box_iou": round(_motion_iou(variant), 3),
            "foreign_as_self": _self_identity_leak(variant),
            "low_conf_recovered": recovered,
            "low_conf_new_tracks": births,
        })
    return rows


def _self_test(rows):
    by_name = {row["variant"]: row for row in rows}
    assert by_name["+motion"]["missing_box_iou"] \
        > by_name["baseline"]["missing_box_iou"]
    assert by_name["baseline"]["foreign_as_self"] == 1
    assert by_name["+safe association"]["foreign_as_self"] == 0
    assert by_name["+two thresholds"]["low_conf_recovered"] == 1
    assert by_name["+two thresholds"]["low_conf_new_tracks"] == 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = run()
    _self_test(rows)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"  {'variant':<20} {'miss IoU':>9} {'other->self':>12} "
          f"{'low recovered':>14} {'low births':>11}")
    for row in rows:
        print(f"  {row['variant']:<20} {row['missing_box_iou']:>9.3f} "
              f"{row['foreign_as_self']:>12d} "
              f"{row['low_conf_recovered']:>14d} "
              f"{row['low_conf_new_tracks']:>11d}")


if __name__ == "__main__":
    main()
