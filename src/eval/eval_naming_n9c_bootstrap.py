"""N9 final step (2) -- recording-grouped paired bootstrap CI on the
independent-vs-contrastive deltas, on the FULL 84-item benchmark (not N8's
n=25 compound-only bootstrap). Without this, the N8/N9 improvement is a
single point estimate on a small sample and can be dismissed as noise.

Thresholds are fixed ONCE (fit on the full 84-item set via the same
best-F1-threshold rule used elsewhere) rather than refit inside every
bootstrap resample -- refitting a grouped-CV threshold 2000+ times is
unnecessary cost here; this measures sampling variance of the METRIC under a
fixed decision rule, not threshold-selection variance (that was already
addressed by grouped CV in N9 itself). Documented, not hidden.

Reports 95% CI (percentile method, recording-level resampling) for:
AUROC delta, secondary F1 delta, empty-secondary accuracy delta, compound
Recall@1 delta, compound-only exact-accuracy delta (contrastive - independent).

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n9c_bootstrap \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --n_boot 2000
"""
import argparse, json, random
from collections import defaultdict


def best_f1_threshold(pairs):
    if not pairs:
        return 0.0
    best_tau, best_f1 = 0.0, -1
    for tau in sorted({s for s, _ in pairs}):
        tp = sum(1 for s, l in pairs if s > tau and l == 1)
        fp = sum(1 for s, l in pairs if s > tau and l == 0)
        fn = sum(1 for s, l in pairs if s <= tau and l == 1)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def compute_metrics(items, tau):
    """items: list of (gt_letters, primary_letter, scores). Fixed tau."""
    tp = fp = fn = 0; n_atomic = 0; empty_correct = 0
    n_compound = 0; recall1_hit = 0; compound_exact_total = compound_exact_correct = 0
    pooled_pos, pooled_neg = [], []
    for gt_letters, primary_letter, scores in items:
        secondary_gt = set(gt_letters) - {primary_letter}
        pred = {l for l, s in scores.items() if s > tau}
        tp += len(pred & secondary_gt); fp += len(pred - secondary_gt); fn += len(secondary_gt - pred)
        if not secondary_gt:
            n_atomic += 1; empty_correct += int(len(pred) == 0)
        else:
            n_compound += 1
            ranked = sorted(scores, key=lambda l: -scores[l])
            best_rank = min(ranked.index(l) + 1 for l in secondary_gt if l in ranked)
            recall1_hit += int(best_rank <= 1)
            compound_exact_total += 1
            full_pred = {primary_letter} | pred
            compound_exact_correct += int(full_pred == set(gt_letters))
        for l, s in scores.items():
            (pooled_pos if l in secondary_gt else pooled_neg).append(s)
    if n_atomic == 0 or n_compound == 0 or not pooled_pos or not pooled_neg:
        return None
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    wins = sum(1 for p in pooled_pos for n in pooled_neg if p > n)
    ties = sum(1 for p in pooled_pos for n in pooled_neg if p == n)
    auroc = (wins + 0.5 * ties) / max(len(pooled_pos) * len(pooled_neg), 1)
    return {"auroc": auroc, "f1": f1, "empty_acc": empty_correct / n_atomic,
            "recall1": recall1_hit / n_compound,
            "compound_exact": compound_exact_correct / max(compound_exact_total, 1)}


def percentile_ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals); n = len(s)
    return s[int(n * lo / 100)], s[min(int(n * hi / 100), n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    n7 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n9_jsonl))}
    keys = sorted(set(n7) & set(n9))
    print(f"n_items={len(keys)}")

    ind_pairs = []
    for k in keys:
        r = n7[k]
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        for l, s in r["scores"].items():
            ind_pairs.append((s, int(l in secondary_gt)))
    con_pairs = []
    for k in keys:
        r = n9[k]
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        for l, s in r["contrastive_scores"].items():
            con_pairs.append((s, int(l in secondary_gt)))
    ind_tau = best_f1_threshold(ind_pairs)
    con_tau = best_f1_threshold(con_pairs)
    print(f"fixed thresholds (fit once on full 84): independent tau={ind_tau:.2f}  contrastive tau={con_tau:.2f}")

    by_recording = defaultdict(list)
    for k in keys:
        rid = k[0]
        by_recording[rid].append({
            "ind": (n7[k]["gt_letters"], n7[k]["primary_letter"], n7[k]["scores"]),
            "con": (n9[k]["gt_letters"], n9[k]["primary_letter"], n9[k]["contrastive_scores"]),
        })
    recording_ids = list(by_recording)

    rng = random.Random(a.seed)
    deltas = {"auroc": [], "f1": [], "empty_acc": [], "recall1": [], "compound_exact": []}
    skipped = 0
    for _ in range(a.n_boot):
        sampled = [rng.choice(recording_ids) for _ in recording_ids]
        ind_items = [it["ind"] for rid in sampled for it in by_recording[rid]]
        con_items = [it["con"] for rid in sampled for it in by_recording[rid]]
        m_ind = compute_metrics(ind_items, ind_tau)
        m_con = compute_metrics(con_items, con_tau)
        if m_ind is None or m_con is None:
            skipped += 1; continue
        for kk in deltas:
            deltas[kk].append(m_con[kk] - m_ind[kk])

    point_ind = compute_metrics([it["ind"] for its in by_recording.values() for it in its], ind_tau)
    point_con = compute_metrics([it["con"] for its in by_recording.values() for it in its], con_tau)

    print(f"\n==== N9c paired bootstrap, contrastive - independent "
          f"(recording-level resampling, n_boot={a.n_boot}, usable={a.n_boot - skipped}) ====")
    for k, label in [("auroc", "AUROC"), ("f1", "secondary F1"), ("empty_acc", "empty-secondary accuracy"),
                     ("recall1", "compound Recall@1"), ("compound_exact", "compound-only exact accuracy")]:
        if not deltas[k]:
            continue
        lo, hi = percentile_ci(deltas[k])
        point_delta = point_con[k] - point_ind[k]
        sig = "SIGNIFICANT (95% CI excludes 0)" if (lo > 0 or hi < 0) else "not significant (95% CI includes 0)"
        print(f"{label:28s} delta={point_delta:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
