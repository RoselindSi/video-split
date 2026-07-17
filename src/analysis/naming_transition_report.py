"""Analyze the transition/structured/reverse naming experiments correctly.

Two things the naive analysis got wrong (fixed here):
  1. reverse-pred vs ORIGINAL forward GT similarity is meaningless -- reversing
     the frames changes what action is actually shown (unpack -> repack), so a
     low similarity there does NOT mean the reversed prediction is wrong.
  2. duplicate rate alone is not a valid optimization target (many GT segments
     ARE legitimately the same coarse action repeated -- see gt_granularity.py).

What this script reports instead:
  A. transition vs structured vs (prior) local/16f baseline, DECOMPOSED into
     verb_match / obj_f1 / emb_sim (reusing the deterministic scorer), so we
     can see whether structured output fixes the dominant right-object/
     wrong-verb failure specifically, not just move mean_sim around.
  B. forward vs reverse ORDER-SENSITIVITY, done properly:
       - prediction change rate (any change at all)
       - on the REVERSIBLE-verb subset only: does the reverse prediction's
         verb match the INVERSE of the forward prediction's verb (not GT)?
       - object consistency: forward/reverse predictions should name the SAME
         physical object -- if the object itself changes, that's a grounding
         failure (hallucination), not a meaningful reversibility signal, and
         is reported SEPARATELY rather than folded into an order-sensitivity
         score.

Usage (server):
    python -m src.analysis.naming_transition_report \
        --forward /tmp/naming_transition.jsonl \
        --structured /tmp/naming_transition_struct.jsonl \
        --reverse /tmp/naming_transition_reverse.jsonl
"""
import argparse, json, statistics
from collections import defaultdict

try:                                            # server flat layout
    from eval_naming_decoupled import verb_match, obj_f1, primary_verb, content
except ImportError:
    try:
        from src.eval.eval_naming_decoupled import verb_match, obj_f1, primary_verb, content
    except ImportError:
        from eval_naming_decoupled import verb_match, obj_f1, primary_verb, content

try:
    from src.seg_rewards import _default_sim_fn
except ImportError:
    from src.rewards.seg_rewards import _default_sim_fn

# Reversible action pairs: verb -> set of acceptable INVERSE verbs. Independent
# of the synonym clusters in eval_naming_decoupled (those group near-synonyms,
# not opposites). Matching is substring-based against the predicted name.
INVERSE_PAIRS = [
    ({"remove", "take", "extract", "unpack", "unbox", "withdraw"},
     {"place", "insert", "replace", "put", "seat", "repack", "pack", "box", "return"}),
    ({"open", "unwrap", "uncover", "unzip"}, {"close", "wrap", "cover", "zip", "seal"}),
    ({"fold"}, {"unfold"}),
    ({"coil"}, {"uncoil"}),
    ({"extend"}, {"retract"}),
    ({"fill"}, {"empty", "drain", "pour"}),
    ({"tighten", "screw"}, {"loosen", "unscrew"}),
    ({"attach", "mount", "install"}, {"detach", "remove", "uninstall"}),
    ({"pick", "grab", "grasp", "lift"}, {"put", "place", "drop", "lower"}),
]


def verb_of(name):
    toks = name.lower().split()
    return toks[0] if toks else ""


def is_inverse(v_fwd, v_rev):
    for a, b in INVERSE_PAIRS:
        if (v_fwd in a and v_rev in b) or (v_fwd in b and v_rev in a):
            return True
    return False


def reversible_verb(v):
    return any(v in a or v in b for a, b in INVERSE_PAIRS)


def object_overlap(name_a, name_b):
    ca, cb = content(name_a), content(name_b)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / max(len(ca | cb), 1)


def section_a(paths):
    print("=== A. decomposed quality: verb_match / obj_f1 / emb_sim ===")
    sim = _default_sim_fn()
    print(f"{'set':16s} {'n':>4} {'verb_acc':>9} {'obj_f1':>7} {'emb_sim':>8}")
    for tag, path in paths:
        recs = [json.loads(l) for l in open(path)]
        vm = [verb_match(r["pred_name"], r["gt_name"]) for r in recs]
        of = [obj_f1(r["pred_name"], r["gt_name"]) for r in recs]
        es = [r["emb_sim"] for r in recs]
        print(f"{tag:16s} {len(recs):4d} {statistics.mean(vm):9.3f} "
              f"{statistics.mean(of):7.3f} {statistics.mean(es):8.3f}")
    print()


def section_b(fwd_path, rev_path):
    print("=== B. forward vs reverse order-sensitivity (correct methodology) ===")
    fwd = {(r["recording_id"], r["segment_idx"]): r for r in map(json.loads, open(fwd_path))}
    rev = {(r["recording_id"], r["segment_idx"]): r for r in map(json.loads, open(rev_path))}
    keys = sorted(set(fwd) & set(rev))
    sim = _default_sim_fn()

    changed = 0
    obj_overlaps = []
    hallucinated = []          # object changed entirely between fwd/rev
    reversible_cases = []      # (gt_verb reversible) -> did rev flip correctly?
    for k in keys:
        f, r = fwd[k]["pred_name"], rev[k]["pred_name"]
        d = 1.0 - sim(f, [r])[0]
        if d > 0.05:
            changed += 1
        ov = object_overlap(f, r)
        obj_overlaps.append(ov)
        if ov == 0.0:
            hallucinated.append((k, f, r))

        gt_v = verb_of(fwd[k]["gt_name"])
        if reversible_verb(gt_v) and ov > 0:      # only score direction on same-object cases
            v_f, v_r = verb_of(f), verb_of(r)
            reversible_cases.append(is_inverse(v_f, v_r))

    print(f"n paired segments: {len(keys)}")
    print(f"prediction changed (1-cos > 0.05): {changed}/{len(keys)} "
          f"({changed/len(keys):.1%})")
    print(f"mean object word overlap (fwd vs rev pred): "
          f"{statistics.mean(obj_overlaps):.2f}")
    print(f"hallucination rate (object overlap == 0, i.e. grounding broke on "
          f"reversal, NOT a valid order-sensitivity sample): "
          f"{len(hallucinated)}/{len(keys)} ({len(hallucinated)/len(keys):.1%})")
    if reversible_cases:
        print(f"on REVERSIBLE-verb + same-object subset (n={len(reversible_cases)}): "
              f"reverse correctly flipped to inverse verb "
              f"{sum(reversible_cases)}/{len(reversible_cases)} "
              f"({sum(reversible_cases)/len(reversible_cases):.1%})")
    else:
        print("no reversible-verb + same-object cases found in this sample")
    if hallucinated:
        print("\nsample hallucination cases (object identity broke on reversal):")
        for k, f, r in hallucinated[:5]:
            print(f"  {k[0]} seg{k[1]}: fwd='{f}' rev='{r}'")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", required=True)
    ap.add_argument("--structured", default=None)
    ap.add_argument("--reverse", required=True)
    a = ap.parse_args()

    paths = [("transition/free-text", a.forward)]
    if a.structured:
        paths.append(("transition/structured", a.structured))
    section_a(paths)
    section_b(a.forward, a.reverse)

    print("Read: (1) does structured output raise verb_acc specifically "
          "(vs just moving emb_sim)? (2) hallucination rate tells you how often "
          "reversal breaks grounding entirely -- those cases must be excluded "
          "before trusting any order-sensitivity number. (3) the reversible-"
          "subset flip-rate is the real 'does the model use motion direction' "
          "answer -- compare it to random-flip baseline (~50% for a 2-way "
          "inverse-or-not guess) to see if it's above chance.")


if __name__ == "__main__":
    main()
