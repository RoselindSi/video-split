"""The three track-level decisions, in the one order that is not circular.

WHAT THIS IS. Three questions that can only be answered once a track has
ended -- is this thing a hand, whose is it, which hand is it -- run as a
single pass over a finished run and written back as decisions a re-render
consumes. Separately they were three scripts and three numbers; together they
are what actually changes the delivered video.

THE ORDER IS FORCED, NOT CHOSEN.

    1. hand-ness   A box on a machine part is not evidence about ownership
                   and not evidence about handedness. Letting it vote first
                   would put bench clutter into both later tallies, and 7.7%
                   of large boxes are exactly that.
    2. ownership   Which hand it is only matters for hands that are the
                   wearer's, and a track the vote moves to `other` must not
                   then contribute a side label to anything.
    3. handedness  Last, over what survives.

Run the other way round, each stage would be deciding on a population the
next stage is about to change.

EVERY STAGE VOTES PER CANONICAL TRACK, INCLUDING HAND-NESS. Raw tracker ids
are joined only by the separately audited offline mapping. A track is one physical
thing, so it is one hand or it is not; a per-frame verdict that flickers is
the same noise that made the handedness vote necessary. The floor makes this
matter in practice -- boxes under 150 px are never asked, so a track can have
three scored frames out of forty, and those three should decide the track
rather than three frames of it.

AND THE TWO ASYMMETRIES SURVIVE THE MERGE, which is the thing to check when
reading this file. A veto still never uncovers: `own=0` is untouched at every
stage. And ownership still needs a strict majority to call a track the
wearer's, while anything short of one stays covered. Handedness alone is
symmetric, because being wrong about it costs a label rather than a person.

A TIE DOES NOT VETO. Removing a box from the deliverable needs evidence, and
half the frames saying "not a hand" is not a majority saying so.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

MIN_PX = 150
SIDES = ("left", "right")


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def score_key(rec, frame, rank):
    return "%s_f%06d_b%d" % (rec, frame, rank)


def hand_scores(rows, rec, scores, disjoint_iou=0.10):
    """-> {row_id: p}. The probe ran on the disjoint-kept own boxes per frame,
    ranked largest first, so the rank has to be rebuilt the same way."""
    out = {}
    by_f = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if str(r.get("own")) == "1":
            by_f[int(r["frame"])].append(i)
    for f, idxs in by_f.items():
        idxs.sort(key=lambda i: -(float(rows[i]["x1"]) - float(rows[i]["x0"]))
                  * (float(rows[i]["y1"]) - float(rows[i]["y0"])))
        picked = []
        for i in idxs:
            b = [float(rows[i][c]) for c in ("x0", "y0", "x1", "y1")]
            if all(iou(b, k) < disjoint_iou for k in picked):
                picked.append(b)
                p = scores.get(score_key(rec, f, len(picked) - 1))
                if p is not None:
                    out[i] = p
    return out


def decide(rows, rec, scores, thr=0.10, min_px=MIN_PX, min_frac=0.5,
           track_map=None):
    """-> (decisions, stats). decisions: {(frame, tid): (own, side, is_hand)}"""
    p_by_row = hand_scores(rows, rec, scores)
    track_map = track_map or {}

    def canonical(tid):
        return str(track_map.get(tid, tid))

    # ---- 1. hand-ness, per track, over the boxes that were asked
    tally_h = collections.defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        raw_t = str(r.get("tid") or "")
        if not raw_t or str(r.get("own")) != "1":
            continue
        t = canonical(raw_t)
        w = float(r["x1"]) - float(r["x0"])
        p = p_by_row.get(i)
        if p is None or w < min_px:
            continue
        tally_h[t][0 if p > thr else 1] += 1
    not_hand = {t for t, (h, n) in tally_h.items() if n > h}

    # ---- 2. ownership, per track, over boxes that are not vetoed
    tally_o = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        raw_t = str(r.get("tid") or "")
        if not raw_t:
            continue
        t = canonical(raw_t)
        if t in not_hand:
            continue
        tally_o[t][0 if str(r.get("own")) == "1" else 1] += 1
    own_track = {t: 1 if o > (o + n) * min_frac else 0
                 for t, (o, n) in tally_o.items()}

    # ---- 3. handedness, per track, over own tracks only
    tally_s = collections.defaultdict(collections.Counter)
    for r in rows:
        raw_t = str(r.get("tid") or "")
        if not raw_t:
            continue
        t = canonical(raw_t)
        if own_track.get(t) != 1:
            continue
        s = (r.get("side") or "").strip()
        if s in SIDES:
            tally_s[t][s] += 1
    side_track = {}
    for t, c in tally_s.items():
        side_track[t] = ("left" if c["left"] > c["right"]
                         else ("right" if c["right"] > c["left"] else ""))

    dec, st = {}, collections.Counter()
    for r in rows:
        raw_t = str(r.get("tid") or "")
        t = canonical(raw_t) if raw_t else ""
        f = int(r["frame"])
        was_own = str(r.get("own")) == "1"
        if t in not_hand:
            own, is_hand = 0, 0
            st["vetoed_boxes"] += 1
        else:
            own = own_track.get(t, 1 if was_own else 0)
            is_hand = 1
        side = side_track.get(t, "") if own == 1 else ""
        dec[(f, raw_t)] = (own, side, is_hand)
        if was_own and own == 0 and is_hand:
            st["demoted_to_other"] += 1
        if not was_own and own == 1:
            st["promoted_to_own"] += 1
        if own == 1 and side and (r.get("side") or "") in SIDES \
                and side != r["side"]:
            st["side_changed"] += 1
        # THE INVARIANT, checked on every row rather than argued for once.
        if not was_own and own == 1 and t in not_hand:
            raise RuntimeError("a vetoed track was promoted at frame %d" % f)
    raw_tracks = set(str(r.get("tid") or "") for r in rows) - {""}
    st["tracks"] = len(raw_tracks)
    st["canonical_tracks"] = len({canonical(t) for t in raw_tracks})
    st["not_hand_tracks"] = len(not_hand)
    st["own_tracks"] = sum(1 for v in own_track.values() if v == 1)
    return dec, st


def materialize_rows(rows, decisions, max_owner=2):
    """Apply track decisions and the final owner cap without replaying video.

    The owner-context gate consumes only box metadata. Re-decoding, masking and
    encoding a complete ``clean`` video to obtain that table changes no value
    it reads. This function reproduces the decision and cap portion of the
    cached replay, leaving pixel-derived audit columns explicitly zero.
    """
    out = []
    by_frame = collections.defaultdict(list)
    for row in rows:
        item = dict(row)
        frame = int(item["frame"])
        tid = str(item.get("tid") or "")
        own, side, is_hand = decisions[(frame, tid)]
        item["own"] = str(int(own))
        # Replay only overwrites detector handedness when the track vote has
        # a non-empty result; foreign/tied tracks keep their detector label.
        item["side"] = side or item.get("side", "")
        item["not_hand"] = str(int(not is_hand))
        for field in ("covered", "face_on_hand", "biggest_face",
                      "face_px_frac", "oth_px", "veto_px"):
            if field in item:
                item[field] = "0"
        by_frame[frame].append(len(out))
        out.append(item)

    if max_owner is not None:
        for indexes in by_frame.values():
            owners = [i for i in indexes
                      if out[i]["own"] == "1"
                      and out[i].get("not_hand", "0") != "1"]
            owners.sort(key=lambda i: -float(out[i].get("p") or 0.0))
            for index in owners[int(max_owner):]:
                out[index]["own"] = "0"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--scores", action="append", default=[],
                    help="hand-ness jsonl for this arm; omit to skip stage 1")
    ap.add_argument("--jobs", default="/workspace/cam3_jobs.txt")
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--min_px", type=int, default=MIN_PX)
    ap.add_argument("--min_frac", type=float, default=0.5)
    ap.add_argument("--track_maps", default="",
                    help="directory of <rec>.track_map.csv canonical-id maps")
    ap.add_argument("--out", default="",
                    help="where to write <rec>.decisions.csv; default beside the run")
    ap.add_argument("--materialized_arm", default="",
                    help="also write decision-applied <rec>.csv tables here")
    ap.add_argument("--max_owner", type=int, default=2,
                    help="per-frame owner cap used by materialized tables")
    a = ap.parse_args()

    recs = []
    for line in open(a.jobs):
        f = line.strip().split("|")
        if len(f) >= 4 and not line.startswith("#"):
            recs.append(f[0])

    scores = {}
    for p in a.scores:
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            scores[d["stem"]] = float(d["p"])
    print("hand-ness 打分 %d 个" % len(scores))

    out_dir = a.out or a.arm
    os.makedirs(out_dir, exist_ok=True)
    if a.materialized_arm:
        os.makedirs(a.materialized_arm, exist_ok=True)
    total = collections.Counter()
    for rec in recs:
        p = os.path.join(a.arm, rec + ".csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows or "tid" not in rows[0]:
            print("  !! %s 没有 tid 列，跳过" % rec)
            continue
        track_map = {}
        if a.track_maps:
            from src.rig.track_fusion import load_track_map
            track_map = load_track_map(
                os.path.join(a.track_maps, rec + ".track_map.csv"))
        dec, st = decide(rows, rec, scores, a.thr, a.min_px, a.min_frac,
                         track_map)
        with open(os.path.join(out_dir, rec + ".decisions.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "tid", "own", "side", "is_hand"])
            for (f, t), (own, side, is_hand) in sorted(dec.items()):
                w.writerow([f, t, own, side, is_hand])
        if a.materialized_arm:
            clean = materialize_rows(rows, dec, a.max_owner)
            clean_path = os.path.join(a.materialized_arm, rec + ".csv")
            with open(clean_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(clean)
        for k, v in st.items():
            total[k] += v
        print("  %-16s 轨迹 %3d -> canonical %3d（非手 %2d，自己的 %2d）  降为别人 %3d  升为自己 %3d  改左右 %3d"
              % (rec, st["tracks"], st["canonical_tracks"],
                 st["not_hand_tracks"], st["own_tracks"],
                 st["demoted_to_other"], st["promoted_to_own"],
                 st["side_changed"]))
    print("\n合计：轨迹 %d -> canonical %d，其中判为非手 %d、判为自己的手 %d"
          % (total["tracks"], total["canonical_tracks"],
             total["not_hand_tracks"], total["own_tracks"]))
    print("      框级改动：非手 %d，降为别人 %d，升为自己 %d，改左右 %d"
          % (total["vetoed_boxes"], total["demoted_to_other"],
             total["promoted_to_own"], total["side_changed"]))
    print("-> %s/<rec>.decisions.csv" % out_dir)


if __name__ == "__main__":
    main()
