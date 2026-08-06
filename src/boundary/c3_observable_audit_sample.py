"""Sample REVIEW-band events for the four-way observability audit.

The question this audit exists to answer is what a human needs in order to
decide an event that both P1 and hand trajectories fail on. Everything built
so far measured whether a given representation separates the classes; none of
it asked what information the decision actually requires. Four answers, fixed
before any event is looked at:

  1 hand-kinematically observable   the hand motion alone settles it, so the
                                    tracking or the feature vocabulary is what
                                    fell short
  2 object-relative                 needs which object the hand is near, and
                                    whether it made or broke contact
  3 semantic / long-context         needs the purpose of the action or more
                                    time than the window shows
  4 not visually resolvable         occlusion, off-frame, or a labelling
                                    convention with no visual correlate

The sample is STRATIFIED, not random, because a 36-event random draw from 169
would land mostly in the majority cell and answer nothing about the corners.
Four binary axes, all fixed in advance:

  the label            sharp vs same-action, so a bias toward one error
                       direction shows up as a difference between them
  P1 confidence        near its threshold vs far from it -- an event P1 is
                       confidently wrong about is a different failure from one
                       it has no opinion on
  detection coverage   above vs below 0.5, since 83 of 412 events fall below
                       and their features are largely imputed
  one hand vs two      the two-hand features were the only ones to clear a
                       baseline anywhere, so both sides must be represented

A PER-RECORDING CAP applies on top. Without it a single recording with many
REVIEW events can supply a quarter of the sample, and since camera behaviour
and task are shared within a recording, the audit would then be describing
that recording rather than the REVIEW band.

The output is deliberately blind: the sheet carries no model score, no
predicted class and no taxonomy subtype. The reviewer sees the clip, the
contact sheet and the segment labels. Anything else anchors the answer, and
the whole value of the audit is that it is independent of what the model
thinks.

This does NOT resample or relabel anything. The taxonomy label stays as it is;
the audit adds an orthogonal annotation about what evidence the decision
needs.

Usage:
    python -m src.boundary.c3_observable_audit_sample \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --features /workspace/tr1/results/hal/c3/hand_trajectory_features.csv \
        --blind_csv data/gold/audit_188_context.jsonl \
        --blind_csv /workspace/tr1/results/hal/batch3/batch3_blind_review.csv \
        --n 36 --max_per_recording 2 \
        --out_dir /workspace/tr1/results/hal/c3/observable_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

SCORE_KEYS = ("score", "p1_score", "primary_score", "fused_score", "oof_score")
ANSWERS = ("1_hand_kinematic", "2_object_relative", "3_semantic_context",
           "4_not_resolvable")


# utf-8-sig, not utf-8: a spreadsheet writes a BOM and the FIRST column
# name comes back as "\ufeffyour_call(...)", so a prefix match on it
# fails and the file looks like it is missing the column it plainly has.
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--features", required=True,
                    help="hand_trajectory_features.csv from the probe")
    ap.add_argument("--blind_csv", action="append", default=[],
                    help="repeatable; supplies segment-label context for rendering")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--max_per_recording", type=int, default=2)
    ap.add_argument("--coverage_split", type=float, default=0.5)
    ap.add_argument("--two_hand_split", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    dec = {r["event_id"]: r for r in read_csv(a.decisions)}
    feat = {r["event_id"]: r for r in read_csv(a.features)}
    score_key = next((k for k in SCORE_KEYS
                      if any(k in r for r in list(dec.values())[:1])), None)
    print(f"{len(dec)} decisions, {len(feat)} feature rows; "
          f"confidence column: {score_key or 'NONE -- that axis collapses'}")

    rows = [r for eid, r in feat.items()
            if dec.get(eid, {}).get("decision") == "REVIEW"]
    print(f"REVIEW band: {len(rows)} events, "
          f"{len({r['recording_id'] for r in rows})} recordings")
    if len(rows) < a.n:
        raise SystemExit(f"only {len(rows)} REVIEW events, cannot draw {a.n}")

    # confidence axis: distance from the median score, split at its own median
    # so both halves are populated regardless of the score's distribution
    if score_key:
        sc = np.array([fnum(dec[r["event_id"]], score_key) for r in rows])
        d = np.abs(sc - np.nanmedian(sc))
        far_cut = np.nanmedian(d)
    else:
        d = np.zeros(len(rows))
        far_cut = np.inf

    def cell(i, r):
        cov = fnum(r, "hand_detect_coverage")
        nh = fnum(r, "mean_n_hands")
        return (
            "sharp" if r.get("y") == "1" else "same",
            "far" if d[i] > far_cut else "near",
            "cov_hi" if not np.isfinite(cov) or cov >= a.coverage_split else "cov_lo",
            "two" if np.isfinite(nh) and nh >= a.two_hand_split else "one",
        )

    cells = defaultdict(list)
    for i, r in enumerate(rows):
        cells[cell(i, r)].append(r)
    print(f"{len(cells)} non-empty strata of 16; sizes "
          f"{sorted((len(v) for v in cells.values()), reverse=True)}")

    # proportional allocation with a floor of 1 per non-empty stratum, so the
    # corners are represented without letting them dominate
    rng = np.random.RandomState(a.seed)
    keys = sorted(cells)
    quota = {k: 1 for k in keys}
    left = a.n - len(keys)
    if left < 0:
        raise SystemExit(f"{len(keys)} strata but n={a.n}; raise --n")
    tot = sum(len(cells[k]) for k in keys)
    for k in keys:
        quota[k] += int(round(left * len(cells[k]) / tot))

    picked, per_rec = [], Counter()
    for k in keys:
        pool = list(cells[k])
        rng.shuffle(pool)
        take = 0
        for r in pool:
            if take >= quota[k] or len(picked) >= a.n:
                break
            if per_rec[r["recording_id"]] >= a.max_per_recording:
                continue
            picked.append((k, r))
            per_rec[r["recording_id"]] += 1
            take += 1
    # top up if the per-recording cap starved some strata
    if len(picked) < a.n:
        chosen = {r["event_id"] for _, r in picked}
        pool = [r for r in rows if r["event_id"] not in chosen]
        rng.shuffle(pool)
        for r in pool:
            if len(picked) >= a.n:
                break
            if per_rec[r["recording_id"]] >= a.max_per_recording:
                continue
            picked.append(("topup", r))
            per_rec[r["recording_id"]] += 1

    print(f"drew {len(picked)} events from {len(per_rec)} recordings "
          f"(max {max(per_rec.values())} per recording)")
    print(f"  by stratum: {dict(Counter(k for k, _ in picked))}")

    # The two halves of the development set store context in different
    # formats -- the 145 in audit_188_context.jsonl, batch3 in a blind review
    # CSV -- and a sample stratified across both draws from both. Accepting
    # only one format would silently leave a third of the sheet with no
    # segment labels, which is the one piece of legitimate context the blind
    # reviewer is given.
    ctx = {}
    for p in a.blind_csv:
        recs = ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                if p.endswith(".jsonl") else read_csv(p))
        for r in recs:
            ctx.setdefault(r["event_id"], r)
    missing = [r["event_id"] for _, r in picked if r["event_id"] not in ctx]
    if missing:
        print(f"  !! {len(missing)} of {len(picked)} sampled events have no "
              f"segment-label context in --blind_csv; they will render with "
              f"blank labels. Pass the context file that covers them rather "
              f"than accepting a sheet where a third of the rows are missing "
              f"the only legitimate context the reviewer gets.")

    os.makedirs(a.out_dir, exist_ok=True)
    man = os.path.join(a.out_dir, "audit_manifest.jsonl")
    with open(man, "w", encoding="utf-8") as f:
        for _, r in picked:
            eid = r["event_id"]
            t = float(eid.rsplit("_t", 1)[1])
            f.write(json.dumps({"event_id": eid,
                                "recording_id": r["recording_id"],
                                "t": t}, ensure_ascii=False) + "\n")

    blind = os.path.join(a.out_dir, "audit_blind_context.csv")
    cols = ["event_id", "prev_segment_label", "next_segment_label",
            "containing_segment_label"]
    with open(blind, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for _, r in picked:
            c = ctx.get(r["event_id"], {})
            w.writerow({k: c.get(k, "") for k in cols})

    sheet = os.path.join(a.out_dir, "audit_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # your_call and confidence come FIRST and are answered before the
        # observability question, for two reasons. They measure whether the
        # REVIEW band is resolvable by a human at all -- an upper bound on
        # anything that could be built for it, which nothing so far has
        # established -- and answering "what evidence would settle this"
        # before attempting the decision invites a plausible-sounding
        # explanation for a judgement never actually made.
        w.writerow(["event_id", "your_call(sharp|same|cannot)",
                    "confidence(1_guess|2_lean|3_sure)",
                    "answer(" + "|".join(ANSWERS) + ")",
                    "what_evidence_would_settle_it", "notes"])
        for _, r in picked:
            w.writerow([r["event_id"], "", "", "", "", ""])

    # The key is written SEPARATELY so the sheet a reviewer opens carries no
    # stratum, no label and no coverage -- knowing an event was drawn from the
    # low-coverage cell is enough to suggest answer 4 before the clip is
    # played. It is joined back only at analysis time.
    key = os.path.join(a.out_dir, "audit_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "recording_id", "stratum", "y",
                    "hand_detect_coverage", "mean_n_hands"])
        for k, r in picked:
            w.writerow([r["event_id"], r["recording_id"], "|".join(k)
                        if isinstance(k, tuple) else k, r.get("y", ""),
                        r.get("hand_detect_coverage", ""),
                        r.get("mean_n_hands", "")])

    print(f"\nwrote {man}\n      {blind}\n      {sheet}\n      {key}")
    print("\nThe sheet carries no score, no subtype and no stratum -- the key "
          "is a separate file, joined back only when the answers come in.")


if __name__ == "__main__":
    main()
