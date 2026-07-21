"""END-TO-END re-decode (offline, no GPU) -- fixes the threshold-objective
bug in eval_naming_e2e_predicted_primary.py's "deployable" condition.

That script fit the grouped-OOF secondary threshold by best-F1. F1 rewards
recall, so on the e2e score distribution it picked a permissive threshold
that over-selects secondaries (atomic false-secondary jumped to 1.14/seg) and
made ordered-exact WORSE (25.0%) than the conservative frozen tau (42.9%) --
because F1-optimal != exact-match-optimal. The raw scores are all saved in
predictions.jsonl, so this recomputes the three-threshold table + error
decomposition offline with the threshold chosen to directly maximize the
TARGET metric (ordered-exact on the training fold), and prints the F1-tau
result alongside for comparison.

Usage (server, no GPU):
    python -m src.eval.eval_naming_e2e_redecode \
        --predictions /workspace/tr1/results/naming/n11_e2e_predictions.jsonl \
        --frozen_tau 10.25 --n_folds 5
"""
import argparse, json, random
from collections import Counter

from src.eval.eval_naming_e2e_predicted_primary import decode_and_score


def grouped_tau_objective(records, objective, n_folds, seed, use_oracle_primary=False):
    """objective in {'f1','exact'}. 'exact' picks tau maximizing ordered-exact
    on the training fold (directly the reported target); 'f1' reproduces the
    original best-F1 rule. use_oracle_primary controls which primary/anchor
    the exact objective scores against."""
    rng = random.Random(seed)
    rids = sorted({r["recording_id"] for r in records})
    rng.shuffle(rids)
    folds = [rids[i::n_folds] for i in range(n_folds)]
    fold_of = {rid: fi for fi, fold in enumerate(folds) for rid in fold}

    def scores_of(r):
        return r["oracle_secondary_scores"] if use_oracle_primary else r["e2e_secondary_scores"]

    def primary_of(r):
        return r["gt_primary_letter"] if use_oracle_primary else r["pred_primary_letter"]

    def best_tau(train):
        cand_taus = sorted({s for r in train for s in scores_of(r).values()})
        cand_taus = [cand_taus[0] - 1] + cand_taus if cand_taus else [0.0]
        best_t, best_score = 0.0, -1
        for tau in cand_taus:
            if objective == "f1":
                tp = fp = fn = 0
                for r in train:
                    gt = set(r["gt_letters"])
                    for l, s in scores_of(r).items():
                        pred_pos = s > tau
                        tp += pred_pos and (l in gt); fp += pred_pos and (l not in gt)
                        fn += (not pred_pos) and (l in gt)
                p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                val = 2 * p * rc / max(p + rc, 1e-9)
            else:  # exact: maximize ordered-exact on train
                ok = 0
                for r in train:
                    gt = set(r["gt_letters"]); primary = primary_of(r)
                    primary_correct = (primary == r["gt_primary_letter"])
                    gt_secondary = gt - {r["gt_primary_letter"]}
                    pred_secondary = {l for l, s in scores_of(r).items() if s > tau}
                    ok += int(primary_correct and pred_secondary == gt_secondary)
                val = ok / max(len(train), 1)
            if val > best_score:
                best_score, best_t = val, tau
        return best_t

    tau_of = {}
    for fi in range(n_folds):
        train = [r for r in records if fold_of[r["recording_id"]] != fi]
        tau = best_tau(train)
        for r in records:
            if fold_of[r["recording_id"]] == fi:
                tau_of[(r["recording_id"], r["segment_idx"])] = tau
    return tau_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--frozen_tau", type=float, default=10.25)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.predictions)
    records = [json.loads(l) for l in open(a.predictions)]
    n = len(records)
    primary_acc = sum(r["primary_correct"] for r in records) / n
    print(f"\n==== END-TO-END re-decode (offline, n={n}, primary acc={primary_acc:.1%}) ====")

    frozen = lambda r: a.frozen_tau
    exact_tau = grouped_tau_objective(records, "exact", a.n_folds, a.seed)
    f1_tau = grouped_tau_objective(records, "f1", a.n_folds, a.seed)

    conditions = [
        ("oracle-primary + frozen tau (== N9 reference)", frozen, True),
        ("predicted-primary + frozen tau (anchor degradation)", frozen, False),
        ("predicted-primary + grouped-OOF tau [EXACT objective, deployable]",
         lambda r: exact_tau[(r["recording_id"], r["segment_idx"])], False),
        ("predicted-primary + grouped-OOF tau [F1 objective, original buggy]",
         lambda r: f1_tau[(r["recording_id"], r["segment_idx"])], False),
    ]
    deploy_summ = None
    for label, tau_fn, oracle in conditions:
        summ, rows = decode_and_score(records, tau_fn, oracle)
        if "EXACT objective" in label:
            deploy_summ = summ
        print(f"\n--- {label} ---")
        print(f"  ordered full-exact: {summ['ordered_exact']:.1%}   "
              f"unordered action-set exact: {summ['unordered_exact']:.1%}")
        print(f"  atomic ordered-exact: {summ['atomic_ordered']:.1%}   "
              f"compound ordered-exact: {summ['compound_ordered']:.1%}")
        pu = summ["sec_uncond_prf"]; pc = summ["sec_cond_prf"]
        print(f"  secondary P/R/F1 uncond: {pu[0]:.1%}/{pu[1]:.1%}/{pu[2]:.1%}   "
              f"cond-on-primary-correct: {pc[0]:.1%}/{pc[1]:.1%}/{pc[2]:.1%}")
        print(f"  atomic false-secondary per seg: {summ['atomic_false_sec_per_seg']:.2f}")
        if not oracle and summ["ordered_exact"] > primary_acc + 1e-9:
            print(f"  !! SANITY VIOLATION: ordered {summ['ordered_exact']:.1%} > primary {primary_acc:.1%}")

    print(f"\n=== error decomposition (EXACT-objective deployable, n={n}) ===")
    err = deploy_summ["error_decomp"]
    for cat in ("primary_correct_secondary_correct", "primary_correct_secondary_wrong",
                "primary_secondary_role_swap", "primary_wrong_action_set_recovered",
                "primary_wrong_secondary_also_wrong"):
        c = err.get(cat, 0)
        print(f"  {cat:38s} {c:4d}  {c/n:.1%}")
    print("\nread: with the exact-objective threshold, primary_correct_secondary_wrong "
          "should drop vs the F1-objective run (fewer atomic segments given a "
          "spurious secondary), isolating the genuine compound-secondary "
          "failures from threshold-induced over-selection.")


if __name__ == "__main__":
    main()
