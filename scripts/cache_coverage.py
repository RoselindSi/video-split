"""Which caches cover which labelled recordings. Run before training, not after.

The loader drops an event whose recording is missing from either cache, and it
counts them -- but a run that silently trains on 145 of 415 events because the
local cache only ever covered the dev recordings still produces a plausible
number. That failure has already cost this project one round: a --data flag
that was not append-mode kept only the last file and squeezed the window to
four frames.

So this scans the .pt caches under the given roots, reports how many of the
labelled recordings each one holds, and then reports the coverage of the UNION
per label source -- because a cache that covers every clean-145 recording and
no batch3 recording looks fine on the total and removes a whole label source.

Usage:
    python scripts/cache_coverage.py --labels data/gold/boundary_v1_labels.json \
        --root /workspace/tr1/data_recseg --root /workspace/tr1/data_recseg_part2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import torch


def cache_recordings(path):
    try:
        blob = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:60]}"
    if not isinstance(blob, list):
        return None, f"not a list of records ({type(blob).__name__})"
    out, has_feats = set(), 0
    for rec in blob:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("recording_id") or rec.get("video")
        if rid:
            out.add(rid)
        has_feats += int("feats" in rec and "times" in rec)
    return out, (None if has_feats else "no record carries feats/times")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--pattern", default="*.pt")
    a = ap.parse_args()

    lab = json.load(open(a.labels, encoding="utf-8"))["events"]
    want = {e["recording_id"] for e in lab}
    # batch3 events are the ones with no audit record; they are also the ones a
    # dev-only local cache would silently remove
    batch3 = {e["recording_id"] for e in lab if not e.get("audited")}
    clean = want - batch3
    print(f"{len(lab)} labelled events over {len(want)} recordings "
          f"({len(clean)} audited, {len(batch3)} batch3-only)")

    paths = []
    for r in a.root:
        paths += sorted(glob.glob(os.path.join(r, a.pattern)))
    if not paths:
        raise SystemExit(f"no {a.pattern} under {a.root}")

    print(f"\n  {'cache':<52} {'recs':>6} {'audited':>8} {'batch3':>7} "
          f"{'frames?':>8}")
    found = {}
    for p in paths:
        rec, err = cache_recordings(p)
        if rec is None:
            print(f"  {os.path.basename(p)[:52]:<52} {'--':>6} {'--':>8} "
                  f"{'--':>7}   {err}")
            continue
        found[p] = rec
        print(f"  {os.path.basename(p)[:52]:<52} {len(rec):>6} "
              f"{len(rec & clean):>8} {len(rec & batch3):>7} "
              f"{'no' if err else 'yes':>8}")

    print(f"\n{'=' * 78}\nUNION OF EVERY READABLE CACHE\n{'=' * 78}")
    union = set().union(*found.values()) if found else set()
    miss = want - union
    print(f"  covers {len(want & union)}/{len(want)} recordings; "
          f"{len(miss)} missing")
    if miss:
        print(f"  missing audited: {len(miss & clean)}   "
              f"missing batch3: {len(miss & batch3)}")
        print(f"  e.g. {sorted(miss)[:4]}")
        n_ev = sum(1 for e in lab if e["recording_id"] in miss)
        print(f"  those recordings carry {n_ev} of the {len(lab)} labelled "
              f"events, which would be dropped without appearing in any "
              f"metric.")
        by = Counter(e.get("morphology") or "MASKED" for e in lab
                     if e["recording_id"] in miss)
        print(f"  and they are not spread evenly across the target: {dict(by)}")
    else:
        print("  every labelled recording is covered.")

    print(f"\n  Pass the caches that matter with repeated --feat_cache / "
          f"--local_cache flags. Both are append-mode; a single flag\n  given "
          f"twice keeps both files, but a comma-separated string does not.")


if __name__ == "__main__":
    main()
