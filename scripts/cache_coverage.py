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


def frame_spacing(blob):
    """Median seconds between cached frames. Caches at different rates can be
    mixed, and the loader resamples both onto the same grid, but a 10 fps
    source will land a real frame near every grid point while a 2 fps source
    will not -- so window coverage differs BY SOURCE, and a coverage gap that
    tracks the label source is a confound rather than a signal."""
    import numpy as np
    for rec in blob:
        if isinstance(rec, dict) and "times" in rec:
            t = rec["times"]
            t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            if len(t) > 1:
                return float(np.median(np.diff(np.sort(t))))
    return float("nan")


def cache_recordings(path):
    try:
        blob = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:60]}"
    if not isinstance(blob, list):
        return None, f"not a list of records ({type(blob).__name__})"
    out, has_feats = set(), 0
    globals()["_spacing"] = frame_spacing(blob)
    for rec in blob:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("recording_id") or rec.get("video")
        if rid:
            out.add(rid)
        has_feats += int("feats" in rec and "times" in rec)
    return out, (None if has_feats else "no record carries feats/times")


def report_chosen(a, lab, want, clean, batch3):
    """The two sets exactly as train.py will receive them.

    An event survives only if BOTH streams hold its recording, so the
    intersection is what gets trained on -- reporting each stream's coverage
    separately hides the case where each is nearly complete and their overlap
    is not."""
    import numpy as np
    for tag, paths in (("--feat_cache", a.feat_cache),
                       ("--local_cache", a.local_cache)):
        print(f"\n{tag}: {len(paths)} file(s)")
        for p in paths:
            rec, err = cache_recordings(p)
            if rec is None:
                print(f"    !! {os.path.basename(p)}: {err}")
                continue
            print(f"    {os.path.basename(p)[:46]:<47} {len(rec):>5} recs   "
                  f"spacing {globals().get('_spacing', float('nan')):.2f}s")
    def union(paths):
        u = set()
        sp = []
        for p in paths:
            rec, err = cache_recordings(p)
            if rec:
                u |= rec
                sp.append(globals().get("_spacing", float("nan")))
        return u, sp
    ug, spg = union(a.feat_cache)
    ul, spl = union(a.local_cache)
    both = ug & ul
    print(f"\n{'=' * 78}\nWHAT TRAIN WILL ACTUALLY SEE\n{'=' * 78}")
    print(f"  global union {len(ug & want)}/{len(want)}   "
          f"local union {len(ul & want)}/{len(want)}   "
          f"BOTH {len(both & want)}/{len(want)}")
    lost = want - both
    n_ev = sum(1 for e in lab if e["recording_id"] in lost)
    print(f"  {len(lost)} recordings carry no usable pair, dropping {n_ev} of "
          f"{len(lab)} events")
    if lost:
        by = Counter(e.get("morphology") or "MASKED" for e in lab
                     if e["recording_id"] in lost)
        print(f"  the dropped events by class: {dict(by)}")
        print(f"  audited {len(lost & clean)}, batch3 {len(lost & batch3)}")
    sp = [x for x in spg + spl if x == x]
    if sp and (max(sp) / max(min(sp), 1e-9)) > 1.5:
        print(f"\n  !! frame spacing differs across the caches "
              f"({min(sp):.2f}s to {max(sp):.2f}s). The loader resamples onto "
              f"one grid, so this\n     does not break anything, but the denser "
              f"source will hit more grid points and its window coverage will "
              f"read\n     higher. If that tracks the label source it is a "
              f"confound, not a signal -- check the per-source coverage the "
              f"trainer prints.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--pattern", default="*.pt")
    ap.add_argument("--feat_cache", action="append", default=[],
                    help="check the EXACT set you will pass to train")
    ap.add_argument("--local_cache", action="append", default=[])
    a = ap.parse_args()

    lab = json.load(open(a.labels, encoding="utf-8"))["events"]
    want = {e["recording_id"] for e in lab}
    # batch3 events are the ones with no audit record; they are also the ones a
    # dev-only local cache would silently remove
    batch3 = {e["recording_id"] for e in lab if not e.get("audited")}
    clean = want - batch3
    print(f"{len(lab)} labelled events over {len(want)} recordings "
          f"({len(clean)} audited, {len(batch3)} batch3-only)")

    if a.feat_cache or a.local_cache:
        report_chosen(a, lab, want, clean, batch3)
        return
    paths = []
    for r in a.root:
        paths += sorted(glob.glob(os.path.join(r, a.pattern)))
    if not paths:
        raise SystemExit(f"no {a.pattern} under {a.root}")

    print(f"\n  {'cache':<52} {'recs':>6} {'audited':>8} {'batch3':>7} "
          f"{'frames?':>8} {'spacing':>9}")
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
              f"{'no' if err else 'yes':>8} {globals().get('_spacing', float('nan')):>8.2f}s")

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
