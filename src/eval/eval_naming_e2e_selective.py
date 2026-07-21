"""END-TO-END selective prediction -- N10 redone WITHOUT oracle-primary
leakage. Reads eval_naming_e2e_predicted_primary.py's predictions and scores
selective acceptance on the DEPLOYABLE decode (predicted primary + grouped-
OOF secondary tau), using only deployment-visible confidence signals, with
the ACCEPT threshold chosen on a calibration fold (not the same rows being
evaluated -- the flaw N10 was flagged for).

Confidence signal (deployment-visible): min secondary margin =
min over secondary candidates of |contrastive_score - tau| (the riskiest
accept/reject decision). NOTE: a primary-MCQ confidence margin would
strengthen this but the single-choice module used generate() and didn't save
a logit margin -- capturing that is a documented follow-up; this uses the
secondary-side signal only.

"Correct" for an accepted segment = ORDERED full-exact (primary role + full
secondary set both right) -- the real end-to-end target.

Reports coverage at >=90/95/98% accepted ordered-exact, computed
out-of-fold: for each fold, the accept threshold reaching the target
precision is chosen on the OTHER folds and applied to the held-out fold.

Usage (server, no GPU):
    python -m src.eval.eval_naming_e2e_selective \
        --predictions /workspace/tr1/results/naming/n11_e2e_predictions.jsonl \
        --out /workspace/tr1/results/naming/n11_e2e_selective.json \
        --n_folds 5
"""
import argparse, json, random

from src.eval.eval_naming_e2e_predicted_primary import grouped_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.predictions)
    records = [json.loads(l) for l in open(a.predictions)]
    oof_tau, fold_of = grouped_tau(records, a.n_folds, a.seed)

    items = []
    for r in records:
        key = (r["recording_id"], r["segment_idx"])
        tau = oof_tau[key]
        scores = r["e2e_secondary_scores"]
        gt_letters = set(r["gt_letters"])
        gt_secondary = gt_letters - {r["gt_primary_letter"]}
        primary = r["pred_primary_letter"]
        primary_correct = (primary == r["gt_primary_letter"])
        pred_secondary = {l for l, s in scores.items() if s > tau}
        ordered_exact = primary_correct and (pred_secondary == gt_secondary)
        confidence = min((abs(s - tau) for s in scores.values()), default=999)
        items.append({"key": f"{key[0]}_seg{key[1]}", "fold": fold_of[r["recording_id"]],
                      "ordered_exact": ordered_exact, "confidence": confidence,
                      "is_compound": bool(gt_secondary)})

    n = len(items)
    overall = sum(it["ordered_exact"] for it in items) / n
    print(f"\n=== END-TO-END selective (n={n}, deployable decode) ===")
    print(f"full-coverage ordered-exact (accept all): {overall:.1%}")

    def oof_coverage_at_precision(target):
        """for each fold, pick min confidence threshold on TRAIN folds that
        reaches target accepted-precision, apply to held-out; aggregate."""
        accepted = 0; correct = 0
        for fi in range(a.n_folds):
            train = [it for it in items if it["fold"] != fi]
            test = [it for it in items if it["fold"] == fi]
            # candidate thresholds = train confidences; pick lowest threshold
            # (max coverage) whose train accepted-precision >= target
            best_thr = None
            for thr in sorted({it["confidence"] for it in train}):
                acc = [it for it in train if it["confidence"] >= thr]
                if acc and sum(i["ordered_exact"] for i in acc) / len(acc) >= target:
                    best_thr = thr
                    break
            if best_thr is None:
                continue
            for it in test:
                if it["confidence"] >= best_thr:
                    accepted += 1; correct += int(it["ordered_exact"])
        return accepted, correct

    for target in (0.90, 0.95, 0.98):
        acc_n, corr = oof_coverage_at_precision(target)
        cov = acc_n / n
        prec = corr / max(acc_n, 1)
        print(f"  target >={target:.0%}: OOF coverage={cov:.1%} ({acc_n}/{n})  "
              f"realized accepted ordered-exact={prec:.1%}")
    print("read: OOF coverage is the honest 'how much can be auto-labeled' -- "
          "if it's near 0 even at 90%, the confidence signal can't isolate a "
          "reliable subset at end-to-end difficulty, and everything routes to "
          "review. Compare to N10's oracle-primary coverage to see how much of "
          "N10's apparent auto-labelability was primary-oracle-assisted.")

    json.dump({"n": n, "overall_ordered_exact": overall, "items": items},
              open(a.out, "w"), indent=2)
    write_manifest(a.out, input_paths=[a.predictions], extra={"overall_ordered_exact": overall})
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
