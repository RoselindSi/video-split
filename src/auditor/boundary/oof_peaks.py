"""Recording-held-out peaks for the timing-audit recordings. Nothing else changes.

WHAT THIS IS FOR AND WHAT IT MUST NOT TOUCH. The event-matched comparison ran
on peaks from a head fitted on the very recordings it was then inferred on.
Everything else about that experiment is frozen -- the 25-event identity join,
tol 0.5s, both distance definitions, delta hit rate as the pre-registered
primary statistic, and the permutation procedure. The only thing being replaced
is where the peaks come from.

So this file produces peaks and stops. It does not choose a threshold, a sigma
or a min_gap: those come from the frozen manifests, and tuning any of them on
these 25 events would turn a pre-registered test into a search on the sample
it is meant to judge.

THE FOLD RULE. Each audit recording must get its peaks from a head that never
saw that recording's annotations. Folds are over RECORDINGS, assigned by a
seeded shuffle and nothing else -- not by length, not by score, not by how
many audit events a recording carries, because any of those correlate with
what is being measured.

    fold k   train on   every training recording EXCEPT this fold's audit ones
             infer on   this fold's audit recordings

Non-audit training recordings stay in every fold's training set. They are not
being predicted, and holding them out too would shrink the training set for no
benefit and make each fold's head weaker than the reference head in a way that
has nothing to do with the question.

DISK IS THE REASON THIS IS INCREMENTAL. A fold needs its own filtered copy of
the training features, and K copies of a multi-region feature dump is a lot.
--write_fold writes one fold at a time and prints its command; the caller runs
it, then writes the next. --plan shows the sizes first so that decision is
made with numbers rather than after filling a disk.

WHAT THE RESULT WILL MEAN, stated now:

    human > stored, permutation p < .05, bootstrap excluding zero
        the timing claim holds on held-out predictions and stored timing is
        established as a first-order problem
    human > stored, p > .05
        directional but not established. The answer then is a larger
        peak-blind timing audit, NOT another statistic
    the advantage disappears
        the training-recording result was in-sample detector behaviour and
        the mechanism hypothesis is downgraded

Usage:
    python -m src.auditor.boundary.oof_peaks --plan \
        --features /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --gold_json data/gold/alignment_timing_gold_45.json --n_folds 5
    python -m src.auditor.boundary.oof_peaks --write_fold 0 ... --out_dir ...
    python -m src.auditor.boundary.oof_peaks --merge --out_dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import torch


def fold_of(recordings, n_folds, seed):
    r = sorted(recordings)
    random.Random(seed).shuffle(r)
    return {rid: i % n_folds for i, rid in enumerate(r)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    help="the TRAIN feature .pt the reference head was fitted "
                         "on")
    ap.add_argument("--gold_json",
                    default="data/gold/alignment_timing_gold_45.json")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest",
                    help="b2_logits.pt.manifest.json, for the frozen head "
                         "config. Required by --write_fold")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--write_fold", type=int)
    ap.add_argument("--run", action="store_true",
                    help="execute the reconstructed command instead of "
                         "printing it, and delete the fold's train copy on "
                         "success. Opt-in: the default stays print-only "
                         "because the command retrains a head")
    ap.add_argument("--run_all", action="store_true",
                    help="write, run and clean up every fold in turn, "
                         "skipping folds whose logits already exist. 5 x 5.5 "
                         "GB of writes and five training runs -- resumable "
                         "because a job that long will be interrupted")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate the per-fold logits into one file")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    audit = sorted({e["recording_id"] for e in
                    json.load(open(a.gold_json, encoding="utf-8"))["events"]})
    assign = fold_of(audit, a.n_folds, a.seed)
    json.dump(assign, open(os.path.join(a.out_dir, "fold_assignment.json"),
                           "w"), indent=2)

    if a.merge:
        parts, missing = [], []
        for k in range(a.n_folds):
            p = os.path.join(a.out_dir, f"fold{k}_logits.pt")
            if not os.path.exists(p):
                missing.append(k)
                continue
            parts += torch.load(p, weights_only=False)
        if missing:
            raise SystemExit(
                f"folds {missing} have no logits yet. Merging a subset would "
                f"produce peaks for\n  only some audit recordings and the "
                f"event-matched test would silently run on\n  fewer events "
                f"than it reports.")
        got = {v.get("recording_id") for v in parts}
        out = os.path.join(a.out_dir, "oof_logits.pt")
        torch.save(parts, out)
        # the merged file had no manifest, so boundary_error_audit reported it
        # could not tell which config produced it -- on the one file in this
        # chain whose provenance is the whole point
        from src.eval.run_manifest import write_manifest
        write_manifest(out,
                       input_paths=[os.path.join(a.out_dir,
                                                 f"fold{k}_logits.pt")
                                    for k in range(a.n_folds)],
                       extra={"note": "recording-held-out peaks: each "
                                      "recording's logits come from a fold "
                                      "whose head excluded it",
                              "n_folds": a.n_folds, "fold_seed": a.seed,
                              "fold_assignment": assign,
                              "audit_recordings": len(audit)})
        print(f"merged {len(parts)} records over {len(got)} recordings "
              f"-> {out}")
        miss = [r for r in audit if r not in got]
        if miss:
            print(f"  !! {len(miss)} audit recordings still absent: "
                  f"{miss[:6]}")
        print(f"\nnext, with the FROZEN decode flags from "
              f"predictions.jsonl.manifest.json:")
        print(f"  python -m src.boundary.boundary_error_audit --logits {out} "
              f"\\\n    --out_dir {a.out_dir}/audit --thr 0.45 --min_gap 1.0 "
              f"--tol 0.5 --exact_tol 0.25")
        print(f"  python -m src.auditor.boundary.event_matched_timing "
              f"\\\n    --gold_json {a.gold_json} --migrated "
              f"data/gold/pair_schema_v2_migrated.csv \\\n    --predictions "
              f"{a.out_dir}/audit/predictions.jsonl --recseg ... "
              f"\\\n    --peaks held_out --n_boot 2000 --n_perm 2000")
        return

    if not a.features:
        raise SystemExit("--features is required unless --merge")
    print(f"loading {a.features} ...")
    feats = torch.load(a.features, weights_only=False)
    have = {x.get("recording_id") for x in feats}
    print(f"  {len(feats)} records over {len(have)} recordings")
    absent = [r for r in audit if r not in have]
    if absent:
        print(f"  !! {len(absent)} of the {len(audit)} audit recordings are "
              f"NOT in this feature file: {absent[:6]}")
        print(f"     they cannot get out-of-fold peaks from it, and the "
              f"event-matched test would\n     quietly run on the rest. "
              f"Point --features at a file covering them.")

    # `not a.run_all` matters: run_all leaves --write_fold unset, so without
    # it the loop fell through to the plan branch and returned having done
    # nothing while printing something that looked like success
    if a.plan or (a.write_fold is None and not a.run_all):
        n_bytes = os.path.getsize(a.features)
        print(f"\nPLAN -- {a.n_folds} folds over {len(audit)} audit "
              f"recordings, seed {a.seed}:")
        for k in range(a.n_folds):
            hold = [r for r in audit if assign[r] == k]
            print(f"  fold {k}: infer on {len(hold):>3} audit recordings, "
                  f"train on {len(have) - len(hold):>3}")
        print(f"\n  {a.features} is {n_bytes / 1e9:.1f} GB. Each fold writes "
              f"a filtered train copy of\n  roughly that size plus a small "
              f"infer file, so write and run ONE fold at a time\n  and delete "
              f"it before the next unless {a.n_folds} x {n_bytes / 1e9:.1f} "
              f"GB is comfortable.")
        print(f"\n  python -m src.auditor.boundary.oof_peaks --write_fold 0 "
              f"--features {a.features} \\\n    --manifest <b2 manifest> "
              f"--out_dir {a.out_dir}")
        return

    if not a.manifest or not os.path.exists(a.manifest):
        raise SystemExit(
            "--manifest is required: the head config must be the frozen one, "
            "and reading it\n  from the manifest is the only way to be sure "
            "it is.")

    def build(k):
        hold = {r for r in audit if assign[r] == k}
        return hold, os.path.join(a.out_dir, f"fold{k}_train.pt"), \
            os.path.join(a.out_dir, f"fold{k}_infer.pt"), \
            os.path.join(a.out_dir, f"fold{k}_logits.pt")

    if a.run_all:
        need = os.path.getsize(a.features) * 1.1
        for k in range(a.n_folds):
            _h, tp_, _ip, lg = build(k)
            if os.path.exists(lg):
                print(f"fold {k}: logits already present, skipping")
                continue
            free = shutil.disk_usage(a.out_dir).free
            if free < need:
                raise SystemExit(
                    f"fold {k}: {free / 1e9:.1f} GB free, this fold needs "
                    f"about {need / 1e9:.1f} GB.\n  Stopping before the "
                    f"write rather than half way through it.")
            print(f"\n{'=' * 74}\nfold {k}\n{'=' * 74}", flush=True)
            rc = subprocess.run([sys.executable, "-m",
                                 "src.auditor.boundary.oof_peaks",
                                 "--write_fold", str(k), "--run",
                                 "--features", a.features,
                                 "--gold_json", a.gold_json,
                                 "--manifest", a.manifest,
                                 "--n_folds", str(a.n_folds),
                                 "--seed", str(a.seed),
                                 "--out_dir", a.out_dir])
            if rc.returncode != 0:
                raise SystemExit(
                    f"fold {k} failed (exit {rc.returncode}). Its files are "
                    f"left in place;\n  rerun --run_all to resume from here.")
        print(f"\nall {a.n_folds} folds done. Now:\n  python -m "
              f"src.auditor.boundary.oof_peaks --merge --n_folds "
              f"{a.n_folds} --out_dir {a.out_dir}")
        return

    k = a.write_fold
    if not (0 <= k < a.n_folds):
        raise SystemExit(f"--write_fold must be in [0, {a.n_folds})")
    hold, _t, _i, _l = build(k)
    tr = [x for x in feats if x.get("recording_id") not in hold]
    inf = [x for x in feats if x.get("recording_id") in hold]
    tp = os.path.join(a.out_dir, f"fold{k}_train.pt")
    ip = os.path.join(a.out_dir, f"fold{k}_infer.pt")
    torch.save(tr, tp)
    torch.save(inf, ip)
    print(f"fold {k}: {len(tr)} train records, {len(inf)} infer records "
          f"({len(hold)} recordings held out)")
    print(f"  {tp}\n  {ip}")

    m = json.load(open(a.manifest, encoding="utf-8"))
    argv, extra = list(m.get("argv") or []), m.get("extra") or {}
    drop = {"--save_logits", "--infer_extra", "--infer_extra_out", "--seeds",
            "--train", "--val"}
    keep, i = [], 1
    while i < len(argv):
        if argv[i] in drop:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        keep.append(argv[i])
        i += 1
    val = next((argv[i + 1] for i in range(len(argv))
                if argv[i] == "--val" and i + 1 < len(argv)), None)
    cmd = (["python", "-m", "src.boundary.train_head_multi",
            "--train", tp] + (["--val", val] if val else []) + keep
           + ["--seeds", str(extra.get("seed", 0)),
              "--infer_extra", ip,
              "--infer_extra_out",
              os.path.join(a.out_dir, f"fold{k}_logits.pt")])
    print(f"\n  config from the manifest: variant "
          f"{extra.get('variant')}, seed {extra.get('seed')}, sigma_s "
          f"{extra.get('sigma_s')}, pos_weight {extra.get('pos_weight')}")
    print(f"\nRUN:")
    parts, j = [], 3
    while j < len(cmd):
        if cmd[j].startswith("--"):
            vals = []
            n = j + 1
            while n < len(cmd) and not cmd[n].startswith("--"):
                vals.append(cmd[n])
                n = n + 1
            parts.append(" ".join([cmd[j]] + vals))
            j = n
        else:
            parts.append(cmd[j])
            j += 1
    if a.run:
        print("  " + " ".join(cmd[:3]) + " \\\n    "
              + " \\\n    ".join(parts), flush=True)
        rc = subprocess.run(cmd)
        if rc.returncode != 0:
            raise SystemExit(f"training failed (exit {rc.returncode}); "
                             f"{tp} left in place")
        os.remove(tp)
        print(f"fold {k} done, removed {tp}")
        return
    print("  " + " ".join(cmd[:3]) + " \\\n    " + " \\\n    ".join(parts))
    print(f"\nthen `rm {tp}` before writing the next fold, and when all "
          f"{a.n_folds} are done:\n  python -m src.auditor.boundary.oof_peaks "
          f"--merge --out_dir {a.out_dir} --n_folds {a.n_folds}")


if __name__ == "__main__":
    main()
