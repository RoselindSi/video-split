"""Build ONE recseg json holding exactly the recordings the continuity
experiment needs, so the 10 fps extraction covers everything in a single run.

Why this exists: extract_features_recseg.py's --data is NOT action="append"
(argparse silently keeps only the last one), and extracting every recording
at 10 fps costs 5x the 2 fps run for recordings C1 never touches. C1 needs:

  * every recording carrying a clean-145 eval event  -- otherwise those
    events score NaN and drop out of the comparison (the first run lost
    52 events exactly this way);
  * N recordings for self-supervised training that are in NEITHER the
    development set NOR batch3.

batch3 recordings are deliberately absent: they are not evaluated here and
must stay untouched for the one-shot confirmation later.

Usage:
    python -m src.boundary.make_cont_subset \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --data /workspace/tr1/data_recseg_part2/recseg_train.json \
        --data /workspace/tr1/data_recseg_part2/recseg_val.json \
        --exclude_manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --n_train 100 \
        --out /workspace/tr1/data_recseg/recseg_cont_subset.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.auditor import gold_schema as S
from src.boundary import pair_taxonomy as T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--exclude_manifest", action="append", default=[])
    ap.add_argument("--have_cache", action="append", default=[],
                    help="existing feature cache(s); recordings already present "
                         "are dropped from the subset so extraction only covers "
                         "what is genuinely missing")
    ap.add_argument("--min_frames", type=int, default=0,
                    help="a recording in --have_cache with fewer than this many "
                         "frames counts as NOT cached. Blur/black filtering is "
                         "unconditional in extract_features_recseg.py and can "
                         "reduce a recording to 0-30 frames, which is present but "
                         "unusable; re-extract those with --th_blur 0 --th_black 0.")
    ap.add_argument("--force_reextract_eval", action="store_true",
                    help="ALWAYS include every eval-needed recording in the "
                         "output, regardless of --have_cache/--min_frames. "
                         "Total frame count is a weak proxy: a recording can hold "
                         "thousands of frames yet still have candidate-sized gaps "
                         "concentrated exactly where they matter (e.g. blur "
                         "removes frames preferentially during fast hand motion, "
                         "which is disproportionately likely right at a real "
                         "action boundary -- the least acceptable place to lose "
                         "coverage). The eval set is only ~46 recordings, so "
                         "forcing a clean --th_blur 0 --th_black 0 re-extract of "
                         "all of them is cheap and removes this failure mode "
                         "entirely instead of chasing per-recording thresholds.")
    ap.add_argument("--include_recordings_from", action="append", default=[],
                    help="additional manifest(s) (e.g. batch3_manifest.jsonl) whose "
                         "recordings must ALSO be in the output, on top of the "
                         "clean-145 eval-needed set -- for extracting a diagnostic "
                         "cache over a promoted batch's recordings without writing "
                         "a separate script. Subject to the same --have_cache/"
                         "--min_frames/--force_reextract_eval logic as eval_recs.")
    ap.add_argument("--n_train", type=int, default=100,
                    help="TARGET total training recordings. Recordings already "
                         "present in --have_cache count toward it, so re-running "
                         "after a partial extraction tops up rather than doubling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, seen = [], set()
    for p in a.data:
        for r in json.load(open(os.path.expanduser(p), encoding="utf-8")):
            rid = r.get("recording_id") or r.get("video")
            if rid not in seen:
                seen.add(rid)
                rows.append(r)
    by_rid = {(r.get("recording_id") or r.get("video")): r for r in rows}
    print(f"pooled {len(rows)} unique recordings from {len(a.data)} data files")

    gold = S.load_gold(a.gold)
    # Only recordings that still carry an event after the clean-pair filter
    # actually need a cache; soft/excluded rows never reach the comparison.
    keep_ids = None
    if os.path.exists(a.pair_labels):
        labels = T.load_pair_labels(a.pair_labels)
        keep_ids = {eid for eid, v in labels.items()
                    if v.get("pair_supervision") in ("strong_separate", "strong_align")}
    eval_recs = {g.get("recording_id") for g in gold
                 if g.get("recording_id")
                 and (keep_ids is None or g.get("event_id") in keep_ids)}
    all_dev = {g.get("recording_id") for g in gold if g.get("recording_id")}
    print(f"eval recordings needed (clean-145 events): {len(eval_recs)} "
          f"(of {len(all_dev)} development recordings)")

    extra_recs = set()
    for mp in a.include_recordings_from:
        with open(os.path.expanduser(mp), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    extra_recs.add(json.loads(line)["recording_id"])
    if extra_recs:
        print(f"extra recordings requested via --include_recordings_from: "
              f"{len(extra_recs)} (from {len(a.include_recordings_from)} manifest(s))")
    eval_recs = eval_recs | extra_recs

    b3 = set()
    for mp in a.exclude_manifest:
        with open(os.path.expanduser(mp), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    b3.add(json.loads(line)["recording_id"])
    print(f"batch3 recordings to exclude from training: {len(b3)}")

    have, too_sparse = set(), {}
    for hp in a.have_cache:
        import torch
        for rec in torch.load(os.path.expanduser(hp), weights_only=False):
            rid = rec.get("recording_id") or rec.get("video")
            n_fr = len(rec["times"])
            if a.min_frames and n_fr < a.min_frames:
                too_sparse[rid] = n_fr
            else:
                have.add(rid)
    if a.have_cache:
        print(f"already cached and usable: {len(have)} recordings "
              f"({len(eval_recs & have)}/{len(eval_recs)} of the eval-needed ones)")
        if too_sparse:
            ev = {r: n for r, n in too_sparse.items() if r in eval_recs}
            print(f"  {len(too_sparse)} cached recordings hold <{a.min_frames} frames "
                  f"and will be RE-extracted ({len(ev)} of them carry eval events)")
            for rid, n in sorted(too_sparse.items(), key=lambda kv: kv[1])[:10]:
                print(f"     {rid}  frames={n}"
                      + ("  [eval]" if rid in eval_recs else ""))

    missing = sorted(eval_recs - set(by_rid))
    if missing:
        print(f"!! {len(missing)} eval recordings are not in any --data file: "
              f"{' '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")

    already_train = have - all_dev - b3
    need = max(0, a.n_train - len(already_train))
    pool = sorted(set(by_rid) - all_dev - b3 - have)
    rng = np.random.RandomState(a.seed)
    train_ids = sorted(rng.choice(pool, min(need, len(pool)),
                                  replace=False).tolist()) if need and pool else []
    print(f"training recordings: {len(already_train)} already cached, "
          f"target {a.n_train} -> need {need}, sampled {len(train_ids)} "
          f"from a pool of {len(pool)}")

    eval_out = (eval_recs & set(by_rid)) if a.force_reextract_eval else                ((eval_recs & set(by_rid)) - have)
    if a.force_reextract_eval:
        print(f"--force_reextract_eval: re-extracting all {len(eval_out)} "
              f"eval-needed recordings regardless of cache status")
    selected = sorted(eval_out | set(train_ids))
    out_rows = [by_rid[r] for r in selected]
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False)
    print(f"\nwrote {len(out_rows)} recordings -> {a.out}")
    print(f"  eval recordings in this output: {len(eval_out)} "
          f"({'forced, ignoring cache status' if a.force_reextract_eval else 'missing from cache'})"
          f" + new training {len(train_ids)}")
    if a.have_cache:
        print(f"  (pass BOTH the old and new caches as --cont_feat_cache when "
              f"running predictive_continuity)")
    print(f"  batch3 recordings included: "
          f"{len(set(selected) & b3)} (must be 0)")
    assert not (set(selected) & b3), "batch3 recording leaked into the subset"
    if a.min_frames:
        print(f"  re-extract these with --th_blur 0 --th_black 0, and pass the new "
              f"cache LAST to --cont_feat_cache so it overrides the sparse copies")


if __name__ == "__main__":
    main()
