"""N7 follow-up (5b) -- score the naturally-occurring matched atomic/compound
pairs from mine_matched_pairs.py with the SAME has_secondary gate mechanism
as N7c, but this time object+primary_verb are held FIXED within each pair --
so any remaining score gap can't be explained by "this object/verb just
tends to be compound" (the confound N7d's within-group AUC (~0.51 for
objects) flagged in the small n=84 slice).

Reports:
  - paired win rate: fraction of pairs where compound_score > atomic_score
    (>50%, ideally well above, if there's a real, object/verb-independent
    signal), with an exact binomial test against 50%.
  - mean/median paired score difference (compound - atomic).
  - pooled AUROC over all atomic vs all compound scores in this matched set
    -- a much larger, better-controlled estimate than N7d's tiny within-
    object groups (n=3-10) to compare against the earlier pooled 0.613 and
    within-group ~0.51.

Usage (server):
    python -m src.eval.eval_naming_n7f_matched_gate \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --pairs /tmp/matched_pairs.jsonl --out /tmp/n7f_matched_scored.jsonl
"""
import argparse, json, math, os

import torch
from decord import VideoReader

from src.eval.eval_naming_n7_scored import resolve_first_token_ids, YES_SURFACES, NO_SURFACES
from src.eval.eval_naming_n7c_gate import score_gate
from src.eval.eval_naming_n5_sampling import sample_uniform
from src.boundary.decode_sweep import pr_auc
from transformers import AutoModelForImageTextToText, AutoProcessor


def binom_test_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test p-value, no scipy dependency."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    return sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--pairs", required=True, help="output of mine_matched_pairs.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.pairs)
    pairs = [json.loads(l) for l in open(a.pairs)]

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    yes_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, YES_SURFACES))
    no_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, NO_SURFACES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    vr_cache = {}
    results = []

    def score_item(item, obj, verb):
        vp = item["video"]
        if vp not in vr_cache:
            vr_cache[vp] = VideoReader(vp, num_threads=1)
        vr = vr_cache[vp]
        vfps = vr.get_avg_fps()
        frames, _ = sample_uniform(vr, vfps, item["start"], item["end"], 16)
        return score_gate(proc, model, frames, obj, verb, yes_ids, no_ids)

    for p in pairs:
        a_score = score_item(p["atomic"], p["object"], p["primary_verb"])
        c_score = score_item(p["compound"], p["object"], p["primary_verb"])
        rec = {**p, "atomic_score": a_score, "compound_score": c_score,
               "compound_wins": c_score > a_score}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        results.append(rec)
        print(f"{p['pair_id']}: atomic={a_score:.2f} ('{p['atomic']['gt_name']}')  "
              f"compound={c_score:.2f} ('{p['compound']['gt_name']}')  "
              f"{'compound higher' if c_score > a_score else 'atomic higher or tied'}")

    n = len(results)
    wins = sum(r["compound_wins"] for r in results)
    diffs = [r["compound_score"] - r["atomic_score"] for r in results]
    print(f"\n==== N7f matched-pair gate comparison (n={n} pairs, "
          f"object+primary_verb held fixed within each pair) ====")
    print(f"compound > atomic: {wins}/{n} = {wins/max(n,1):.1%}  "
          f"(binomial two-sided p-value vs 50%: {binom_test_two_sided(wins, n):.4f})")
    print(f"mean paired diff (compound-atomic): {sum(diffs)/max(n,1):.2f}  "
          f"median: {sorted(diffs)[n//2] if n else float('nan'):.2f}")

    all_scores = [r["atomic_score"] for r in results] + [r["compound_score"] for r in results]
    all_labels = [0] * n + [1] * n
    pooled_auc = pr_auc(all_scores, all_labels)
    pos = [r["compound_score"] for r in results]; neg = [r["atomic_score"] for r in results]
    pairwise_wins = sum(1 for cs in pos for as_ in neg if cs > as_)
    pairwise_ties = sum(1 for cs in pos for as_ in neg if cs == as_)
    unpaired_auroc = (pairwise_wins + 0.5 * pairwise_ties) / max(len(pos) * len(neg), 1)
    print(f"pooled (unpaired) PR-AUC: {pooled_auc:.3f}   unpaired AUROC: {unpaired_auroc:.3f}")
    print(f"\ncompare to N7c pooled AUROC=0.613 and N7d mean within-object AUC~0.51: "
          f"this matched-pair estimate holds object+verb fixed for every single "
          f"comparison (not just within a group), so it's the most controlled "
          f"estimate so far of whether the gate has real signal beyond prior.")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=[a.pairs],
                   extra={"n_pairs": n, "win_rate": wins / max(n, 1),
                          "pooled_auc": pooled_auc, "unpaired_auroc": unpaired_auroc})


if __name__ == "__main__":
    main()
