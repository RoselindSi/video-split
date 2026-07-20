"""N7 follow-up (4) -- does the gate score carry independent signal beyond a
pure object/primary-verb PRIOR, under leave-one-out cross-validation? No
model calls -- reads eval_naming_n7c_gate.py's saved records.

Rather than a full one-hot logistic regression (17 objects + 30 verbs is
already more categorical levels than the n=84 sample, guaranteed to overfit
uselessly), each item's object/primary_verb prior is reduced to ONE number
each: the leave-one-out empirical P(compound | that object) and P(compound |
that primary_verb) (excluding the item itself, to avoid leakage; falls back
to the global rate if the item is the only example of its object/verb). Then
a tiny (3-4 parameter) logistic regression is fit under proper LOOCV:

  prior-only  : logit(compound) = a + b*prior_obj + c*prior_verb
  prior+gate  : logit(compound) = a + b*prior_obj + c*prior_verb + d*gate_score

For each fold, the regression is refit on the other 83 items (so the held-out
item's prediction never sees its own label, directly or via a prior computed
from it), and AUROC is computed over all 84 out-of-fold predictions.

Read: if prior+gate AUROC isn't meaningfully above prior-only AUROC, the gate
has no independent signal beyond "which object/verb this is" and should not
be pursued further as a feature, matched-pair confounds or not.

CAVEAT: n=84 is still small for LOOCV logistic regression; treat this as a
sanity check, not a precise estimate.

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n7e_prior_ablation --jsonl /tmp/n7c_gate_v2.jsonl
"""
import argparse, json
import numpy as np


def fit_logistic(X, y, l2=1.0, iters=500, lr=0.5):
    """X: [n,d] (no bias column -- added internally), y: [n] in {0,1}."""
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n  # don't regularize the bias
        w -= lr * grad
    return w


def predict(w, x_row):
    z = w[0] + x_row @ w[1:]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def loo_prior(recs, key, i):
    others = [r for j, r in enumerate(recs) if j != i and r[key] == recs[i][key]]
    if others:
        return sum(r["gt_has_secondary"] for r in others) / len(others)
    all_others = [r for j, r in enumerate(recs) if j != i]
    return sum(r["gt_has_secondary"] for r in all_others) / max(len(all_others), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="output of eval_naming_n7c_gate.py")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.jsonl)
    recs = [json.loads(l) for l in open(a.jsonl)]
    n = len(recs)
    y_all = np.array([int(r["gt_has_secondary"]) for r in recs])

    prior_obj = np.array([loo_prior(recs, "object", i) for i in range(n)])
    prior_verb = np.array([loo_prior(recs, "primary_verb", i) for i in range(n)])
    gate = np.array([r["gate_score"] for r in recs])
    # normalize gate score to a comparable scale to the [0,1] priors
    gate_z = (gate - gate.mean()) / (gate.std() + 1e-9)

    X_prior = np.stack([prior_obj, prior_verb], axis=1)
    X_full = np.stack([prior_obj, prior_verb, gate_z], axis=1)

    def loocv_auroc(X):
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            w = fit_logistic(X[mask], y_all[mask])
            preds[i] = predict(w, X[i])
        return auroc(preds.tolist(), y_all.tolist()), preds

    prior_auc, prior_preds = loocv_auroc(X_prior)
    full_auc, full_preds = loocv_auroc(X_full)

    print(f"\n=== N7e: prior-only vs prior+gate, LOOCV AUROC (n={n}) ===")
    print(f"prior-only (object+primary_verb LOO empirical rate):  AUROC={prior_auc:.3f}")
    print(f"prior + gate_score:                                    AUROC={full_auc:.3f}")
    print(f"delta: {full_auc - prior_auc:+.3f}")
    if full_auc - prior_auc > 0.05:
        print("gate adds a real (if modest, n=84 caveat) increment over prior -- "
              "worth keeping as a soft feature, NOT a hard gate.")
    else:
        print("gate adds little/nothing beyond object+verb prior at this sample "
              "size -- consistent with the within-group AUC finding "
              "(mean within-object AUC ~0.51). Do not invest further in a "
              "global has_secondary gate.")


if __name__ == "__main__":
    main()
