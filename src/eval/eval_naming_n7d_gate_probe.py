"""N7 follow-up (3) -- what is the has_secondary gate actually keying off?
No model calls -- reads eval_naming_n7c_gate.py's saved per-item records
(gate_score, object, primary_verb, duration, gt_has_secondary).

  - gate_score vs duration correlation (Pearson): if strongly positive,
    the gate may just be responding to "longer clip" as a proxy for
    "more likely to contain more than one action", not actual compound
    detection.
  - per-object and per-primary-verb mean gate_score by class + WITHIN-GROUP
    AUC: if overall AUROC (0.613) comes mostly from BETWEEN-group separation
    (some objects/verbs just have systematically higher scores, correlated
    with how often they happen to be compound in this data) rather than
    WITHIN-group discrimination, within-group AUC will collapse toward 0.5
    even though the pooled AUC looks like it has signal. That would mean the
    gate is largely reading an object/verb prior, not visual compoundness.

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n7d_gate_probe --jsonl /tmp/n7c_gate.jsonl
"""
import argparse, json, statistics
from collections import defaultdict


def pairwise_auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="output of eval_naming_n7c_gate.py")
    ap.add_argument("--min_group_n", type=int, default=4,
                     help="skip within-group AUC for groups smaller than this")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.jsonl)
    recs = [json.loads(l) for l in open(a.jsonl)]

    durations = [r["duration"] for r in recs if r.get("duration") is not None]
    scores_for_dur = [r["gate_score"] for r in recs if r.get("duration") is not None]
    if durations:
        r_val = pearson(durations, scores_for_dur)
        print(f"gate_score vs duration: Pearson r={r_val:.3f} (n={len(durations)})"
              if r_val is not None else "gate_score vs duration: undefined (no variance)")
        r_dur_label = pearson(durations, [int(x["gt_has_secondary"]) for x in recs if x.get("duration") is not None])
        print(f"duration vs gt_has_secondary: Pearson r={r_dur_label:.3f} "
              f"(if this is also strongly positive, longer clips really ARE more "
              f"often compound in this data -- duration correlation in gate_score "
              f"could be legitimate signal, not just a confound)")

    def within_group_report(key_fn, label):
        groups = defaultdict(lambda: {"pos": [], "neg": []})
        for r in recs:
            k = key_fn(r)
            (groups[k]["pos"] if r["gt_has_secondary"] else groups[k]["neg"]).append(r["gate_score"])
        print(f"\n=== per-{label} mean gate_score by class + within-group AUC ===")
        aucs = []
        for k in sorted(groups):
            pos, neg = groups[k]["pos"], groups[k]["neg"]
            n = len(pos) + len(neg)
            mp = statistics.mean(pos) if pos else float("nan")
            mn = statistics.mean(neg) if neg else float("nan")
            auc = pairwise_auc(pos, neg) if (len(pos) >= 2 and len(neg) >= 2 and n >= a.min_group_n) else None
            if auc is not None:
                aucs.append(auc)
            print(f"  {k:20s} n={n:3d} (pos={len(pos)},neg={len(neg)})  "
                  f"mean_score[pos]={mp:6.2f}  mean_score[neg]={mn:6.2f}  "
                  f"within-group AUC={f'{auc:.2f}' if auc is not None else 'n/a (too few of one class)'}")
        if aucs:
            print(f"  mean within-group AUC (groups with enough of both classes, "
                  f"n={len(aucs)}): {statistics.mean(aucs):.3f}")
            print("  read: if this is close to 0.5 while the pooled AUROC (0.613) "
                  "is clearly above it, the pooled number is mostly explained by "
                  f"BETWEEN-{label} differences (an object/verb prior correlated "
                  "with how often that group happens to be compound), not real "
                  "within-group visual compound detection.")

    within_group_report(lambda r: r["object"], "object")
    within_group_report(lambda r: r["primary_verb"], "primary_verb")


if __name__ == "__main__":
    main()
