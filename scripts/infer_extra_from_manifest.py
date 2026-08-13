"""Rebuild a train_head_multi command from a logits manifest, plus --infer_extra.

The timing gold's 36 recordings are all in the training split and none of them
appear in predictions.jsonl, so peaks have to be dumped for them before the
null test can run at all. Doing that means re-running the head with the SAME
configuration that produced the reference logits -- a different variant, seed,
sigma or pos_weight would produce a different peak set, and the 6.21 and 0.54
references would stop being comparable to the new number.

write_manifest already records the exact argv beside every logits file. So the
command is reconstructed from it rather than retyped, and the parts that must
change are changed explicitly and printed:

    --seeds            forced to the single seed the manifest names, since
                       --infer_extra requires one
    --save_logits      dropped; this run is not re-dumping the val logits and
                       overwriting the reference file would destroy the thing
                       the new number is being compared against
    --infer_extra      added, pointing at the features to infer on
    --infer_extra_out  added

IT PRINTS, IT DOES NOT RUN. The reconstructed command retrains the head, and a
script that silently launches training because someone wanted a filename is
worse than one that hands over a line to read first.

WHAT THE RESULT WILL AND WILL NOT SUPPORT. Inferring on --train means the head
saw these recordings' stored annotations during fitting. The resulting ratio is
contaminated, and usefully asymmetric: the contamination makes peaks on seen
recordings sharper and more numerous, which can only push alignment UP. So a
ratio near 1 despite that is already evidence against the hypothesis, while a
high ratio settles nothing and needs a checkpoint that held these recordings
out. This is a cheap one-sided test and it should be quoted as one.

Usage:
    python scripts/infer_extra_from_manifest.py \
        --manifest /workspace/tr1/results/boundary/b2_logits.pt.manifest.json \
        --infer_extra /workspace/tr1/data_recseg/feat_train_full_noblur_multi.pt \
        --infer_extra_out /workspace/tr1/results/boundary/timing36_logits.pt
"""
from __future__ import annotations

import argparse
import json
import os

DROP_WITH_VALUE = {"--save_logits", "--infer_extra", "--infer_extra_out",
                   "--seeds"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--infer_extra", required=True)
    ap.add_argument("--infer_extra_out", required=True)
    ap.add_argument("--decode_manifest",
                    help="predictions.jsonl.manifest.json from the run whose "
                         "peaks the reference ratios were computed on. The "
                         "decode threshold and min_gap define the peak set "
                         "just as much as the head does, and I hard-coded "
                         "defaults for them in the first version of this "
                         "script -- exactly the guess it exists to prevent")
    a = ap.parse_args()

    if not os.path.exists(a.manifest):
        raise SystemExit(
            f"{a.manifest} not found. Every logits file written by "
            f"train_head_multi has a\n  `<path>.manifest.json` beside it; "
            f"point at that one. Without it the variant,\n  seed and sigma "
            f"would have to be guessed, and a guess makes the new number "
            f"incomparable\n  to the references it exists to be compared "
            f"against.")

    m = json.load(open(a.manifest, encoding="utf-8"))
    argv = list(m.get("argv") or [])
    extra = m.get("extra") or {}
    print(f"manifest: {a.manifest}")
    print(f"  git commit  {str(m.get('git_commit'))[:12]}"
          f"{'  (DIRTY working tree)' if m.get('git_dirty_uncommitted_changes') else ''}")
    print(f"  written     {m.get('timestamp')}")
    print(f"  variant     {extra.get('variant')}   seed {extra.get('seed')}")
    for k in ("sigma_s", "pos_weight", "delta_mode", "val_f5", "train_f5"):
        if k in extra:
            print(f"  {k:<11} {extra[k]}")
    if not argv:
        raise SystemExit("the manifest has no argv; nothing to reconstruct")

    # argv[0] is the module path as python -m rewrote it
    out, i = [], 1
    dropped = []
    while i < len(argv):
        tok = argv[i]
        if tok in DROP_WITH_VALUE:
            dropped.append(tok)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(tok)
        i += 1

    seed = extra.get("seed", 0)
    cmd = (["python", "-m", "src.boundary.train_head_multi"] + out
           + ["--seeds", str(seed),
              "--infer_extra", a.infer_extra,
              "--infer_extra_out", a.infer_extra_out])

    print(f"\n  dropped from the original: {', '.join(dropped) or 'nothing'}")
    print(f"  --save_logits is dropped on purpose: overwriting the reference "
          f"val logits would\n  destroy what the new number is compared "
          f"against.")
    if extra.get("variant") and "--variant" not in out:
        print(f"  !! the argv carries no --variant but the manifest records "
              f"`{extra['variant']}`;\n     it defaulted at the time. Adding "
              f"it explicitly.")
        cmd += ["--variant", str(extra["variant"])]

    print(f"\n{'=' * 74}\nRUN THIS (it RETRAINS the head, then infers):\n"
          f"{'=' * 74}")
    # group each --flag with its value, or the line is unreadable and gets
    # pasted wrong
    parts, j = [], 3
    head = " ".join(cmd[:3])
    while j < len(cmd):
        if cmd[j].startswith("--"):
            vals = []
            k = j + 1
            while k < len(cmd) and not cmd[k].startswith("--"):
                vals.append(cmd[k])
                k += 1
            parts.append(" ".join([cmd[j]] + vals))
            j = k
        else:
            parts.append(cmd[j])
            j += 1
    print("  " + head + " \\\n    " + " \\\n    ".join(parts))
    print(f"\nthen turn the logits into peaks:")
    dec = []
    if a.decode_manifest and os.path.exists(a.decode_manifest):
        dm = json.load(open(a.decode_manifest, encoding="utf-8"))
        dargv = list(dm.get("argv") or [])
        j = 1
        while j < len(dargv):
            if dargv[j] in ("--thr", "--min_gap", "--tol", "--exact_tol"):
                dec.append(dargv[j])
                j += 1
                while j < len(dargv) and not dargv[j].startswith("--"):
                    dec.append(dargv[j])
                    j += 1
            else:
                j += 1
        print(f"  decode flags recovered from "
              f"{os.path.basename(a.decode_manifest)}: "
              f"{' '.join(dec) or 'none recorded -> defaults were used'}")
    elif a.decode_manifest:
        print(f"  !! {a.decode_manifest} not found")
    else:
        print(f"  !! no --decode_manifest given, so the line below uses "
              f"DEFAULT thr / min_gap.\n     Those define the peak set as "
              f"much as the head does. If the reference run used\n     "
              f"anything else, the new ratio is not comparable to 6.21 or "
              f"0.54. Pass the\n     predictions.jsonl.manifest.json to "
              f"pin them.")
    print(f"  python -m src.boundary.boundary_error_audit \\\n"
          f"    --logits {a.infer_extra_out} \\\n"
          f"    --out_dir {os.path.dirname(a.infer_extra_out) or '.'}"
          f"/timing36_audit"
          + (" \\\n    " + " ".join(dec) if dec else ""))
    print(f"\nthen the test, with the split stated:")
    print(f"  python -m src.auditor.boundary.timing_null_test \\\n"
          f"    --gold_json data/gold/alignment_timing_gold_45.json \\\n"
          f"    --predictions {os.path.dirname(a.infer_extra_out) or '.'}"
          f"/timing36_audit/predictions.jsonl \\\n"
          f"    --recseg_train /workspace/tr1/data_recseg/recseg_train.json "
          f"\\\n    --recseg_val /workspace/tr1/data_recseg/recseg_val.json "
          f"--n_perm 2000")
    print(f"\nCODE DRIFT, which the flags do not cover. The manifest "
          f"records a DIRTY working\ntree, so the reference logits came from "
          f"an uncommitted state that cannot be\nrestored. Since that "
          f"commit, src/boundary/train_head_multi.py has changed once\n"
          f"(7c662d4, which ADDED --infer_extra and leaves the training path "
          f"alone) and\nboundary_error_audit.py six times. Matching flags "
          f"makes the runs as comparable as\nthey can be made; it does not "
          f"make them identical, and if the new number lands\nnear the "
          f"decision boundary that gap is the first thing to suspect.\n")

    print(f"The head will have been fitted on every recording it is then "
          f"inferred on. That\ncontamination can only raise the ratio, so a "
          f"ratio near 1 is already evidence\nagainst the hypothesis and a "
          f"high one settles nothing.")


if __name__ == "__main__":
    main()
