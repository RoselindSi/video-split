"""Blind DOUBLE annotation of never-re-checked batch3 events. Two sheets, one key.

The 36-event REVIEW-band audit showed model-human discrimination (0.844, 0.851)
sitting inside human-human (0.861, 0.864) while both humans agree with the
stored label at only 0.732 and 0.745. That makes annotation consistency a
first-order bottleneck. It does NOT show that representation is no longer one:
human-human is itself only 0.86, the re-checked half of batch3 reaches only
0.79-0.81, and individual events still look like genuine model failures
(000192 reads P1 0.96 against a temporal 0.001). Those two claims need
different evidence and this sheet is aimed at the first.

WHY THESE EVENTS. 115 batch3 events with a decisive POINT or NO_TRANSITION
label were never re-checked from video; their subtypes descend from
machine-made calls. They are also the population where every scorer is
weakest. If a blind double pass reproduces the 36-event pattern --
human-human roughly equal to model-human, both far above human-stored -- then
annotation dominance holds on the population the deployment number is actually
computed on, not only on the hardest slice. If instead human-human is clearly
above model-human, that is the representation gap, measured, with a size.

THE VOCABULARY IS THE 36-EVENT VOCABULARY, deliberately. sharp / same /
cannot with the same three confidence levels, so `human_ceiling_auroc` reads
these sheets unchanged and the two results sit on one scale. A better-designed
vocabulary would not be comparable to the number it is meant to extend.

TWO SHEETS, TWO ORDERS. Each annotator gets the same events shuffled with a
different seed. A shared order makes fatigue, drift and any run of similar
clips land on the same events for both, which would inflate their agreement --
the one quantity the whole exercise depends on.

STRATIFIED, because the two batch3 generators are not interchangeable:
raw_change_peak and gt_boundary differ in what they select for, and the
current label is either POINT or NO. Equal cells keep any cell from being
carried by three events, and the cell counts are printed so an imbalance in
the pool is visible rather than absorbed.

NOTHING ABOUT THE MODEL OR THE STORED LABEL IS IN EITHER SHEET.

Usage:
    python -m src.auditor.boundary.batch3_double_audit_sheet \
        --labels data/gold/boundary_v1_labels_ontology_v2.json \
        --rechecked data/gold/claude_labelled_events.txt \
        --rechecked data/gold/batch3_relabel_claude_90.csv \
        --data /workspace/tr1/data_recseg/recseg_batch3_10fps.json \
        --n 48 --out_dir /workspace/tr1/results/auditor/batch3_double_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

POINT, NONE = "POINT_TRANSITION", "NO_TRANSITION"
SOURCES = ("raw_change_peak", "gt_boundary")

GUIDELINES = """# Is there a boundary here?

One question per clip. The candidate is the exact middle of the clip and is
not marked.

    sharp    the interaction changes at a compact moment you could point at
    same     one ongoing action; whatever changes is internal to it
    cannot   you cannot tell from this clip

Then a confidence: 1_guess, 2_lean, 3_sure.

`cannot` is a real answer, not a failure. It is scored between the two votes,
so using it costs nothing and guessing does.

## The granularity that matters
Judge at the level of a task-level action, the way the dataset is segmented.
Unrolling wrap, laying it over a bowl, tearing it and pressing it down is ONE
wrapping action even though the hands do several different things inside it.
A new action means the goal changed, not that the grip did.

## Do not
Do not look up what the label says, do not look at any model output, and do
not go back to revise earlier rows once you notice a pattern. Two people
answering independently is the entire point; a sheet you smoothed by hand
measures nothing.
"""


def load_rechecked(paths):
    out = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e = line.split(",")[0].strip()
                if e and e != "event_id":
                    out.add(e)
    return out


def source_of(eid):
    for s in SOURCES:
        if f"_{s}_t" in eid:
            return s
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rechecked", action="append", default=[])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    lab = json.load(open(a.labels, encoding="utf-8"))["events"]
    done = load_rechecked(a.rechecked)
    pool = [e for e in lab
            if "_batch3_" in e["event_id"]
            and e.get("morphology") in (POINT, NONE)
            and e["event_id"] not in done]
    print(f"{len(lab)} labelled events; {len(done)} batch3 events already "
          f"re-checked")
    print(f"  pool: {len(pool)} never-re-checked batch3 events with a decisive "
          f"POINT/NO label, over "
          f"{len({e['recording_id'] for e in pool})} recordings")

    cells = defaultdict(list)
    for e in pool:
        cells[(source_of(e["event_id"]), e["morphology"])].append(e)
    print(f"\n  {'cell':<40} {'available':>10}")
    for k in sorted(cells):
        print(f"  {k[0]} x {k[1]:<24} {len(cells[k]):>10}")

    rng = random.Random(a.seed)
    per = max(1, a.n // max(len(cells), 1))
    picked = []
    for k in sorted(cells):
        g = sorted(cells[k], key=lambda e: e["event_id"])
        rng.shuffle(g)
        picked += g[:per]
    # top up from the largest remaining cells if a cell was short, so the
    # target n is met without silently reweighting toward one generator
    if len(picked) < a.n:
        rest = [e for e in pool if e not in picked]
        rng.shuffle(rest)
        picked += rest[:a.n - len(picked)]
    print(f"\n  drew {len(picked)} events over "
          f"{len({e['recording_id'] for e in picked})} recordings")
    print(f"  by cell: "
          f"{dict(Counter((source_of(e['event_id']), e['morphology']) for e in picked))}")
    short = [k for k in sorted(cells) if len(cells[k]) < per]
    if short:
        print(f"  !! cells short of {per}: {short}. The top-up came from the "
              f"largest remaining cells, so the design is not exactly\n     "
              f"balanced and the printed counts above are the real ones.")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "GUIDELINES.md"), "w",
              encoding="utf-8") as f:
        f.write(GUIDELINES)

    hdr = ["your_call(sharp|same|cannot)",
           "confidence(1_guess|2_lean|3_sure)", "event_id", "clip", "notes"]
    for tag, seed in (("annotator1", a.seed + 1), ("annotator2", a.seed + 2)):
        rows = list(picked)
        random.Random(seed).shuffle(rows)
        path = os.path.join(a.out_dir, f"batch3_double_audit_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for e in rows:
                w.writerow(["", "", e["event_id"],
                            f"{e['event_id']}.mp4", ""])
        print(f"  wrote {os.path.basename(path)} ({len(rows)} rows, own "
              f"shuffle)")

    key = os.path.join(a.out_dir, "batch3_double_audit_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "recording_id", "candidate_time",
                    "stored_morphology", "source"])
        for e in picked:
            w.writerow([e["event_id"], e["recording_id"], e["candidate_time"],
                        e["morphology"], source_of(e["event_id"])])

    video = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video[r["recording_id"]] = r.get("video")
    sh = os.path.join(a.out_dir, "make_clips.sh")
    miss = []
    with open(sh, "w", encoding="utf-8") as f:
        f.write('#!/bin/sh\nD="$(dirname "$0")"\nmkdir -p "$D/clips"\n')
        for e in picked:
            v = video.get(e["recording_id"])
            if not v:
                miss.append(e["event_id"])
                continue
            t0 = max(0.0, float(e["candidate_time"]) - a.half_s)
            f.write(f'ffmpeg -nostdin -loglevel error -y -ss {t0:.2f} '
                    f'-i "{v}" -t {2 * a.half_s:.2f} -c:v libx264 -crf 23 '
                    f'-an "$D/clips/{e["event_id"]}.mp4" '
                    f'|| echo "FAILED {e["event_id"]}"\n')
        f.write(f'echo "clips: $(ls "$D/clips" | wc -l) of {len(picked)}"\n')
    os.chmod(sh, 0o755)
    if miss:
        print(f"  !! {len(miss)} events have no video path; add the --data "
              f"json holding their recordings")

    print(f"\n  the key is a SEPARATE file, so either sheet can be handed over "
          f"whole. Score them with:")
    print(f"    python -m src.auditor.boundary.human_ceiling_auroc \\")
    print(f"      --a {a.out_dir}/batch3_double_audit_annotator1_filled.csv \\")
    print(f"      --b {a.out_dir}/batch3_double_audit_annotator2_filled.csv \\")
    print(f"      --labels {a.labels} --predictions <oof json>")
    print(f"\n  human-human close to model-human, both well above "
          f"human-stored, repeats the 36-event finding on the population the\n"
          f"  deployment number is computed on. human-human clearly above "
          f"model-human is the representation gap, with a size.")


if __name__ == "__main__":
    main()
