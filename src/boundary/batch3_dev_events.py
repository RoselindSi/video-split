"""Turn completed batch3 labels into development-set-compatible events, now
that batch3 has been promoted from "one-shot confirmatory test" to
"development data" (2026-08-02 decision -- its 240 labels are fully known,
so it can no longer serve as a blind test for anything built AFTER seeing
them; batch4 is the reserved test set for C1-C5 instead).

batch3's blind review already used the SAME 7-value subtype vocabulary as
pair_taxonomy.py's SUBTYPES (this was a deliberate choice made while
building the blind-review sheet), so folding it in needs no new taxonomy --
just two adapters:

  write_pair_labels()   batch3_blind_review_*.csv's `temporal_truth` column
                        -> a pair_taxonomy.load_pair_labels()-compatible CSV
                        (temporal_pair_subtype + pair_supervision, via the
                        existing SUBTYPE_TO_SUPERVISION mapping).

  build_events()        batch3_manifest.jsonl (event_id/recording_id/t) +
                        a feature cache -> the same event dict shape
                        state_adapter.build_events() produces for the
                        original 145-pair pipeline (event_id, recording_id,
                        t, y placeholder, rec), so pair_taxonomy.
                        apply_to_events() and everything downstream of it
                        (build_matrices, p1_fold_eval, continuity_features)
                        treats batch3 events identically to the original
                        ones -- no separate code path needed once these two
                        adapters have run.

The ORIGINAL 145-pair diagnostic (C1's subtype cross-tab, the frozen P1
comparison) must keep using pair_labels_v1.csv alone -- merging batch3 in
there would conflate "why did C1 fail on the original clean set" with "does
more/different data help", which is a different, later question. This
module is for whenever C2/C3 development explicitly wants the larger
merged pool, not a drop-in replacement for pair_labels_v1.csv.

Usage (standalone conversion):
    python -m src.boundary.batch3_dev_events \
        --blind_review ~/Downloads/batch3_blind_review_complete_240.csv \
        --out data/gold/batch3_pair_labels_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import json

from src.boundary.pair_taxonomy import SUBTYPES, SUBTYPE_TO_SUPERVISION


def write_pair_labels(blind_review_csv: str, out_csv: str):
    """batch3's temporal_truth -> a load_pair_labels()-compatible CSV.
    Rows with an empty temporal_truth (not yet labelled) are skipped, not
    written as blank -- load_pair_labels would otherwise treat a truly
    unlabelled row as a hard error if it ever got exercised."""
    rows_out, skipped = [], 0
    with open(blind_review_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sub = (r.get("temporal_truth") or "").strip()
            if not sub:
                skipped += 1
                continue
            if sub not in SUBTYPES:
                raise ValueError(f"{r['event_id']}: unknown temporal_truth {sub!r} "
                                 f"(allowed: {SUBTYPES})")
            rows_out.append({"event_id": r["event_id"],
                             "temporal_pair_subtype": sub,
                             "pair_supervision": SUBTYPE_TO_SUPERVISION[sub]})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "temporal_pair_subtype",
                                          "pair_supervision"])
        w.writeheader()
        w.writerows(rows_out)
    return len(rows_out), skipped


def build_events(manifest_path: str, by_rid: dict):
    """Mirrors state_adapter.build_events()'s output shape, sourced from
    batch3_manifest.jsonl instead of audit_188_gold+context. `y` is a
    placeholder (0) -- pair_taxonomy.apply_to_events() overwrites it from
    CLEAN_BINARY[pair_supervision], exactly as for the original 145 events,
    so callers should always run apply_to_events() before trusting `y`."""
    events, n_missing_rec = [], 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            rec = by_rid.get(m["recording_id"])
            if rec is None:
                n_missing_rec += 1
                continue
            events.append({"event_id": m["event_id"], "recording_id": m["recording_id"],
                           "t": float(m["t"]), "y": 0, "rec": rec})
    if n_missing_rec:
        print(f"  !! {n_missing_rec} batch3 manifest events skipped: recording "
              f"not in the supplied feature cache(s)")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind_review", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n, skipped = write_pair_labels(a.blind_review, a.out)
    print(f"wrote {n} labelled events -> {a.out}"
          + (f" ({skipped} unlabelled rows skipped)" if skipped else ""))
    from collections import Counter
    dist = Counter()
    with open(a.out, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dist[r["pair_supervision"]] += 1
    print("pair_supervision distribution:", dict(dist))


if __name__ == "__main__":
    main()
