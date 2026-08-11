"""One question, five answers: what is the relation between the two sides?

`instance_relation` is UNKNOWN on 223 of 415 events and cannot be derived,
because the old seven subtypes never asked it. Worse, the values that ARE
known are not a sample of anything: 156 of them are a mechanical `same_instance`
read off `same_action_internal_motion`, and `new_action` appears ZERO times --
not because the data has none, but because no pass has ever asked. The one
distinction the repeated-instance cases turn on, new_action against
same_action_new_instance, has no gold at all.

SO THIS SHEET ASKS ONE THING. The annotator does not judge sharp against
gradual; `transition_shape` is already 85.5% covered and re-asking it would
add cognitive load for information already held. Five options, one line of
reasoning, nothing else.

THE RULE EXISTS NOW, which is what makes this different from the 48-event
sheet. That one asked "is there a boundary here" while the taxonomy had no
answer for repeated instances, so two annotators would have disagreed about an
undefined question and their agreement would have measured the rule's absence.
configs/auditor/instance_relation_policy_v1.yaml now says what a
same_action_new_instance requires -- a completed instance, then a meaningful
disengagement, then the same action again -- and the condition is on the
DISENGAGEMENT rather than the repetition. With that written down, an agreement
number finally measures people rather than the schema.

TWO SHEETS. The second annotator is what makes this measurable at all; without
it the result is one person's opinion about their own rule. Different shuffles,
because a shared order puts fatigue and drift on the same events for both and
inflates exactly the number the exercise is for.

STRATIFIED, AND THE STORED `POINT` CELLS COME FIRST. Of the 189 unknown
relations, 138 carry a stored POINT. What fraction of the current POINT class
is really new_action, how much is same_action_new_instance, and how much is
same_instance is the number that decides whether the boundary target means
what it says -- and it is worth more right now than another AUROC.

Usage:
    python -m src.auditor.boundary.relation_audit_sheet \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --labels data/gold/boundary_v1_labels_ontology_v2.json \
        --data /workspace/tr1/data_recseg/recseg_batch3_10fps.json \
        --n 54 --out_dir /workspace/tr1/results/auditor/relation_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

RELATIONS = ["new_action", "same_action_new_instance", "same_instance",
             "terminal_action_end", "cannot_determine"]

GUIDELINES = """# One question per clip: what is the relation between the two sides?

The candidate is the exact middle of the clip and is not marked. You are NOT
being asked whether the change is sharp or gradual -- that is already recorded.

    new_action
        The goal changed. A different task-level action begins.
        Rinsing the strainer, then drying your hands.

    same_action_new_instance
        The SAME action starts again, as a new instance. All three must hold:
          1. the previous instance completed
          2. a meaningful disengagement -- release, idle, hands leaving the
             workspace, a reset
          3. the same action starts again after it
        Cup carried left; hands release and leave; hands return; cup carried
        right. The test is the DISENGAGEMENT, not the repetition.

    same_instance
        One continuous instance. Whatever changes is internal to it.
        Holding the cup throughout while the direction reverses. One more
        cycle of a continuous back-and-forth. A pause with the grip kept.

    terminal_action_end
        The action ends and nothing follows in the clip -- the recording
        stops, or the hands go idle and stay idle.
        We have not decided what the dataset does with these. Say so and move
        on; the answer is being collected, not used yet.

    cannot_determine
        The relevant hands or objects are off-frame or occluded.

## The distinction that matters most
`same_action_new_instance` against `same_instance` is what this whole sheet is
for. Both involve the same action happening more than once. The difference is
whether the person LET GO and re-engaged, or kept going.

    let go, hands left, came back      -> same_action_new_instance
    kept hold, motion reversed         -> same_instance

## Confidence
    1_guess   2_lean   3_sure

## Do not
Do not look at the stored label or any model output. Do not go back to revise
earlier rows once you notice a pattern -- two people answering independently is
the point, and a sheet smoothed by hand measures nothing.
"""


def src_of(eid):
    if "_batch3_gt_boundary_" in eid:
        return "batch3 gt_boundary"
    if "_batch3_raw_change_peak_" in eid:
        return "batch3 raw_change_peak"
    if "_batch3_" in eid:
        return "batch3 other"
    return "dev"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrated", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        mig = {r["event_id"]: r for r in csv.DictReader(f)}
    lab = {e["event_id"]: e
           for e in json.load(open(a.labels, encoding="utf-8"))["events"]}
    unk = [e for e, r in mig.items() if r["instance_relation"] == "UNKNOWN"
           and e in lab]
    print(f"{len(mig)} migrated events, {len(unk)} with an UNKNOWN relation "
          f"and a label row")

    def cell(e):
        m = lab[e].get("morphology") or "MASKED"
        s = mig[e]["transition_shape"]
        if m == "POINT_TRANSITION":
            return f"POINT x {src_of(e)}"
        if s in ("gap", "gradual", "overlap"):
            return f"shape {s}"
        return f"{m} x {src_of(e)}"

    cells = defaultdict(list)
    for e in unk:
        cells[cell(e)].append(e)
    # the stored-POINT cells are the ones the headline number comes from, so
    # they are filled first and the rest share what is left
    order = sorted(cells, key=lambda k: (not k.startswith("POINT"), k))
    print(f"\n  {'cell':<34} {'available':>10}")
    for k in order:
        print(f"  {k:<34} {len(cells[k]):>10}")

    rng = random.Random(a.seed)
    per = max(1, a.n // max(len(order), 1))
    picked = []
    for k in order:
        g = sorted(cells[k])
        rng.shuffle(g)
        picked += g[:per]
    if len(picked) < a.n:
        rest = [e for e in unk if e not in set(picked)]
        # top up from the POINT cells first, for the same reason
        rest.sort(key=lambda e: (not cell(e).startswith("POINT"), e))
        picked += rest[:a.n - len(picked)]
    picked = picked[:a.n]
    print(f"\n  drew {len(picked)} events over "
          f"{len({lab[e]['recording_id'] for e in picked})} recordings")
    print(f"  by cell: {dict(Counter(cell(e) for e in picked))}")
    n_point = sum(1 for e in picked
                  if (lab[e].get('morphology') or '') == 'POINT_TRANSITION')
    print(f"  {n_point} of them carry a stored POINT, which is the cell the "
          f"headline number comes from")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "GUIDELINES.md"), "w",
              encoding="utf-8") as f:
        f.write(GUIDELINES)

    hdr = [f"your_call({'|'.join(RELATIONS)})",
           "confidence(1_guess|2_lean|3_sure)", "why_one_line", "event_id",
           "clip"]
    for tag, seed in (("annotator1", a.seed + 1), ("annotator2", a.seed + 2)):
        rows = list(picked)
        random.Random(seed).shuffle(rows)
        p = os.path.join(a.out_dir, f"relation_audit_{tag}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for e in rows:
                w.writerow(["", "", "", e, f"{e}.mp4"])
        print(f"  wrote {os.path.basename(p)}")

    with open(os.path.join(a.out_dir, "relation_audit_key.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "recording_id", "candidate_time",
                    "stored_morphology", "transition_shape", "source", "cell"])
        for e in picked:
            w.writerow([e, lab[e]["recording_id"], lab[e]["candidate_time"],
                        lab[e].get("morphology"), mig[e]["transition_shape"],
                        src_of(e), cell(e)])

    video = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video[r["recording_id"]] = r.get("video")
    sh = os.path.join(a.out_dir, "make_clips.sh")
    miss = []
    with open(sh, "w", encoding="utf-8") as f:
        f.write('#!/bin/sh\nD="$(dirname "$0")"\nmkdir -p "$D/clips"\n')
        for e in picked:
            v = video.get(lab[e]["recording_id"])
            if not v:
                miss.append(e)
                continue
            t0 = max(0.0, float(lab[e]["candidate_time"]) - a.half_s)
            f.write(f'ffmpeg -nostdin -loglevel error -y -ss {t0:.2f} -i "{v}" '
                    f'-t {2 * a.half_s:.2f} -c:v libx264 -crf 23 -an '
                    f'"$D/clips/{e}.mp4" || echo "FAILED {e}"\n')
        f.write(f'echo "clips: $(ls "$D/clips" | wc -l) of {len(picked)}"\n')
    os.chmod(sh, 0o755)
    if miss:
        print(f"  !! {len(miss)} events have no video path; add the --data "
              f"json holding their recordings ({sorted({src_of(e) for e in miss})})")

    print(f"\n{'=' * 78}\nWHAT THIS DECIDES, written before the sheets come "
          f"back\n{'=' * 78}")
    print(f"  1  Of the stored POINT events, the split across new_action / "
          f"same_action_new_instance / same_instance.\n     A large "
          f"same_instance share means the current POINT class does not mean "
          f"what the target says it means,\n     and no amount of "
          f"representation work fixes a target that mixes them.")
    print(f"  2  Whether `new_action` even appears. It is at zero across every "
          f"pass so far, and if it stays rare here the\n     distinction the "
          f"schema was split for is rarer than the repeated-instance case that "
          f"prompted it.")
    print(f"  3  Human agreement on a DEFINED question. The 48-event sheet "
          f"could not measure it because the rule did not\n     exist; it does "
          f"now, so agreement here is about people rather than about the "
          f"schema. Score it with\n     src.auditor.boundary.human_ceiling_auroc "
          f"once both sheets are back.")


if __name__ == "__main__":
    main()
