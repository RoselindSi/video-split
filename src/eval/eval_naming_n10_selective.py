"""N10 -- selective prediction (auto-label / human-review / abstain) on the
frozen N9 contrastive pipeline. Offline, no model calls: reads N7 (candidate
identities) + N9 (contrastive scores) and reframes the evaluation from
full-coverage F1 into the production-relevant question:

    if we only AUTO-ACCEPT the most confident segments, how much of the data
    can we label automatically at 95% / 98% accepted-set precision, and how
    much has to go to human review?

Confidence signal (offline-available features only):
  min_margin : the segment's exact-set prediction is correct iff EVERY
               candidate is on the correct side of the decision threshold
               tau. The riskiest candidate is the one whose contrastive
               score is CLOSEST to tau (most likely to flip). So
               min_margin = min over all non-primary candidates of
               |score - tau| is a principled per-segment confidence: large =
               every accept/reject decision is far from the boundary. This is
               the primary confidence score used for the coverage curve.
  primary_margin (reported as an alternative signal): score(top1) -
               score(top2) among ALL candidates -- how decisively the primary
               was selected.

NOT included (require re-scoring, flagged for a later pass): window
agreement, frame-sampling agreement, supporting-window length. This is the
subset of the mentor's confidence-model features computable without new GPU
work; adding the rest is the natural follow-up once windowed atomic scores
exist.

"Precision" of the accepted set = accepted-set EXACT accuracy (fraction of
auto-accepted segments whose full predicted set {primary} u secondary equals
the GT set). Coverage = fraction of all segments auto-accepted.

Threshold tau is fit once on the full 84 (same best-F1 rule as N9c) -- same
same-set caveat; recalibrate on N3 once it exists. n=84 is small: the
coverage-at-precision points are indicative, not tight estimates.

Usage (server, no GPU):
    python -m src.eval.eval_naming_n10_selective \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --out /workspace/tr1/results/naming/n10_selective.json
"""
import argparse, json


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.n7_jsonl); print_manifest_if_exists(a.n9_jsonl)
    n7 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n9_jsonl))}
    keys = sorted(set(n7) & set(n9))

    pairs = []
    for k in keys:
        r = n9[k]
        secondary_gt = set(r["gt_letters"]) - {r["primary_letter"]}
        for l, s in r["contrastive_scores"].items():
            pairs.append((s, int(l in secondary_gt)))
    tau = best_f1_threshold(pairs)

    items = []
    for k in keys:
        r7, r9 = n7[k], n9[k]
        primary = r9["primary_letter"]
        secondary_gt = set(r9["gt_letters"]) - {primary}
        scores = r9["contrastive_scores"]  # non-primary candidates
        pred_secondary = {l for l, s in scores.items() if s > tau}
        exact = ({primary} | pred_secondary) == set(r9["gt_letters"])
        min_margin = min((abs(s - tau) for s in scores.values()), default=999)
        # primary margin: how decisively the primary beat the runner-up among
        # ALL candidates (primary score from N7's independent scores, since
        # N9 only stored non-primary contrastive scores)
        all_ind = r7["scores"]
        ranked = sorted(all_ind.values(), reverse=True)
        primary_margin = (ranked[0] - ranked[1]) if len(ranked) >= 2 else 999
        items.append({"key": f"{k[0]}_seg{k[1]}", "exact": exact,
                      "is_compound": bool(secondary_gt),
                      "min_margin": min_margin, "primary_margin": primary_margin})

    n = len(items)
    overall_exact = sum(it["exact"] for it in items) / n
    print(f"\n=== N10 selective prediction (n={n}, tau={tau:.2f}) ===")
    print(f"full-coverage exact accuracy (accept everything): {overall_exact:.1%}")

    def coverage_curve(signal):
        # sort most-confident first; sweep accept fraction
        ordered = sorted(items, key=lambda it: -it[signal])
        rows = []
        for cut in range(1, n + 1):
            accepted = ordered[:cut]
            acc = sum(it["exact"] for it in accepted) / cut
            rows.append((cut / n, acc, cut))
        return rows

    for signal in ("min_margin", "primary_margin"):
        rows = coverage_curve(signal)
        print(f"\nconfidence signal = {signal}:")
        print(f"  {'coverage':>9s} {'accepted-exact-acc':>19s} {'n_accepted':>11s}")
        # print a few operating points across the curve
        for target_cov in (0.1, 0.25, 0.5, 0.75, 1.0):
            # nearest cut at/above the target coverage
            row = min((r for r in rows if r[0] >= target_cov), key=lambda r: r[0])
            print(f"  {row[0]:9.1%} {row[1]:19.1%} {row[2]:11d}")
        for target_prec in (0.98, 0.95, 0.90):
            feasible = [r for r in rows if r[1] >= target_prec]
            if feasible:
                best = max(feasible, key=lambda r: r[0])  # max coverage meeting precision
                print(f"  coverage at >={target_prec:.0%} accepted-exact-acc: "
                      f"{best[0]:.1%} ({best[2]}/{n} segments)")
            else:
                print(f"  coverage at >={target_prec:.0%} accepted-exact-acc: "
                      f"0% (no operating point reaches this precision on n={n})")

    # separate the two by difficulty (atomic vs compound), since atomic is
    # where the pipeline is strong -- selective acceptance should lean on that
    atomic = [it for it in items if not it["is_compound"]]
    compound = [it for it in items if it["is_compound"]]
    print(f"\nby difficulty: atomic exact={sum(i['exact'] for i in atomic)}/{len(atomic)}"
          f"={sum(i['exact'] for i in atomic)/max(len(atomic),1):.1%}  "
          f"compound exact={sum(i['exact'] for i in compound)}/{len(compound)}"
          f"={sum(i['exact'] for i in compound)/max(len(compound),1):.1%}")
    print("read: coverage-at-95% mostly reflects how many high-confidence "
          "(usually atomic + clear-compound) segments exist; the abstain/review "
          "bucket is where the unresolved compound grounding concentrates. This "
          "reframes 'compound-only exact 20%' as 'route low-confidence segments "
          "to review' rather than a single global accuracy.")

    json.dump({"tau": tau, "n": n, "overall_exact": overall_exact,
               "items": items}, open(a.out, "w"), indent=2)
    write_manifest(a.out, input_paths=[a.n7_jsonl, a.n9_jsonl],
                   extra={"tau": tau, "overall_exact": overall_exact})
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
