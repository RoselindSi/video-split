"""N6.1 iter3 -- OFFLINE decoder comparison on the multi_select_ordered
rankings already saved by eval_naming_n6b_cardinality.py. No model calls.

The iter2 result showed the VLM count predictor is badly broken (5.2% exact,
MAE 2.69, 89.7% over-count, 33/58 items predicted the max count=6) and,
worse, the benchmark itself is 91.4% GT=2 -- so before concluding anything
about cardinality prediction as a general capability, we need the two
missing reference points: does a TRIVIAL constant-K=2 baseline already beat
predicted-K (meaning the count predictor added negative value), and does
ORACLE-K (told the true count) actually improve over untruncated (meaning
the candidate RANKING is good enough that count is the only bottleneck) or
not (meaning ranking itself is also broken, independent of count).

Four decoders on the SAME saved ranking:
  untruncated : the full multi_select_ordered set (no truncation)
  fixed_k2    : first min(2, len(ordered)) letters (constant K=2 baseline)
  oracle_k    : first len(gt_letters) letters (told the TRUE count -- upper
                bound on what a perfect count predictor could buy)
  predicted_k : first cardinality_pred letters (iter2's actual approach,
                recomputed here for a same-script, same-metric comparison)

Usage (server, no GPU needed):
    python -m src.eval.eval_naming_n6c_decode_compare --jsonl /tmp/n6b_cardinality_v2.jsonl
"""
import argparse, json

from src.eval.eval_naming_n6b_cardinality import score_set, aggregate_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                     help="output of eval_naming_n6b_cardinality.py")
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.jsonl)
    recs = [json.loads(l) for l in open(a.jsonl)]

    decoders = {
        "untruncated": lambda r: r["multi_select_ordered"],
        "fixed_k2": lambda r: r["multi_select_ordered"][:min(2, len(r["multi_select_ordered"]))],
        "oracle_k": lambda r: r["multi_select_ordered"][:len(r["gt_letters"])],
        "predicted_k": lambda r: (r["multi_select_ordered"][:max(1, min(r["cardinality_pred"], len(r["options"])))]
                                  if r.get("cardinality_parsed") else r["multi_select_ordered"]),
    }

    primary_correct = sum(r["single_choice_correct"] for r in recs)
    print(f"\n==== N6.1 iter3: offline decoder comparison (n={len(recs)}) ====")
    print(f"GT cardinality distribution: "
          f"{ {k: sum(len(r['gt_letters']) == k for r in recs) for k in sorted({len(r['gt_letters']) for r in recs})} }")
    for name, fn in decoders.items():
        rows = [score_set(fn(r), r["gt_letters"], r["primary_letter"]) for r in recs]
        aggregate_report(name, rows, primary_correct)

    print("\nread: fixed_k2 >= predicted_k -> the count predictor is actively "
          "hurting vs just always guessing 2 (matches the benchmark's own "
          "skew, not a real capability signal either way). oracle_k >> "
          "untruncated -> candidate ranking is good enough that count truly "
          "is the bottleneck, worth building a real count model on a "
          "cardinality-BALANCED benchmark. oracle_k close to untruncated -> "
          "ranking itself is also broken; fix candidate ranking (independent "
          "per-verb scoring) before investing in cardinality prediction at all.")


if __name__ == "__main__":
    main()
