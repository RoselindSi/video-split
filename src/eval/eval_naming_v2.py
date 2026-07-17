"""N2 -- naming evaluator built on the FROZEN ontology (build_ontology.py).
Replaces primary-verb-exact-match and raw-text similarity with:
  - verb-SET precision/recall/F1 (multi-verb aware; GT{rinse,seat} vs
    Pred{rinse} -> P=1.0 R=0.5, not a flat wrong)
  - exact verb-set accuracy
  - secondary-verb recall (denominator ONLY segments where GT has >=2 verbs)
  - canonical-object accuracy (via ontology phrase-first extraction, not raw
    content-word overlap)
  - inverse-direction error split into STRICT (unconditional, e.g.
    open<->close) vs CONTEXTUAL (only counts when the object matches a
    known object-conditioned pair, e.g. remove<->seat on sink strainer)

Imports the ontology module directly (not the json dumps) so verb/object
extraction here is byte-for-byte the same logic used to build/freeze it --
single source of truth, no drift between N1 and N2.

Usage (server):
    python -m src.eval.eval_naming_v2 \
        --jsonl /tmp/naming_struct_v2_probe.jsonl
"""
import argparse, json, re, statistics
from collections import Counter

from src.analysis.build_ontology import (
    extract_verbs, extract_object, norm_verb,
    STRICT_INVERSE, CONTEXTUAL_INVERSE, CONTEXTUAL_VERB_NORM, GENERIC_VERBS,
)

JSON_VERB_RE = re.compile(r'"(?:primary_verb|verb)"\s*:\s*"([^"]*)"', re.I)
JSON_OBJECT_RE = re.compile(r'"object"\s*:\s*"([^"]*)"', re.I)
JSON_SECONDARY_RE = re.compile(r'"secondary_verbs"\s*:\s*\[([^\]]*)\]', re.I)
_QUOTED = re.compile(r'"([^"]*)"')


def dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def contextual_fold(verb, obj):
    """Apply the object-conditioned verb collapse frozen in
    build_ontology.CONTEXTUAL_VERB_NORM (e.g. reinstall|sink strainer ->
    seat). NEVER applied globally -- only fires when the object matches
    exactly, and only for pairs an audit confirmed are the same action."""
    return CONTEXTUAL_VERB_NORM.get((verb, obj), verb)


def pred_fields(raw, fallback_name):
    vm = JSON_VERB_RE.search(raw)
    pred_verb = norm_verb(vm.group(1).strip().lower()) if vm else None
    om = JSON_OBJECT_RE.search(raw)
    pred_obj_text = om.group(1).strip() if om else fallback_name
    sm = JSON_SECONDARY_RE.search(raw)
    secondary = ([norm_verb(w.strip().lower()) for w in _QUOTED.findall(sm.group(1))]
                 if sm else [])
    ordered = dedup([v for v in ([pred_verb] + secondary) if v])
    pred_obj, _, _, _ = extract_object(pred_obj_text)
    return ordered, pred_obj


def gt_fields(gt_name):
    ordered = dedup(extract_verbs(gt_name))
    gt_obj, _, _, _ = extract_object(gt_name)
    return ordered, gt_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="output of eval_naming_persegment.py --structured")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.jsonl)]
    n = len(recs)

    verb_p, verb_r, verb_f1 = [], [], []
    exact_set = 0
    sec_hit = sec_total = 0
    obj_correct = obj_total = 0
    strict_inv = Counter()      # gt_verb -> # segments with a strict-inverse pred
    ctx_inv = Counter()         # (gt_verb, obj) -> # segments
    generic_pred_segs = 0
    n_with_gt_verbs = 0
    cat = Counter()             # segment-level verdict for the summary table

    for r in recs:
        raw = r["raw"]
        pred_ordered, pred_obj = pred_fields(raw, r.get("pred_name", ""))
        gt_ordered, gt_obj = gt_fields(r["gt_name"])

        obj_for_fold = gt_obj or pred_obj or ""
        gt_folded = dedup(contextual_fold(v, obj_for_fold) for v in gt_ordered)
        pred_folded = dedup(contextual_fold(v, obj_for_fold) for v in pred_ordered)
        gt_set, pred_set = set(gt_folded), set(pred_folded)

        if any(v in GENERIC_VERBS for v in pred_folded):
            generic_pred_segs += 1

        if gt_set:
            n_with_gt_verbs += 1
            tp = len(pred_set & gt_set)
            p = tp / max(len(pred_set), 1)
            rc = tp / max(len(gt_set), 1)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            verb_p.append(p); verb_r.append(rc); verb_f1.append(f1)
            if pred_set == gt_set:
                exact_set += 1

        # secondary-verb recall: denominator ONLY segments where GT has >=2
        # DISTINCT verbs after folding (primary = first in appearance order)
        if len(gt_folded) >= 2:
            secondary_gt = set(gt_folded[1:])
            sec_total += len(secondary_gt)
            sec_hit += len(secondary_gt & pred_set)

        if gt_obj:
            obj_total += 1
            if pred_obj == gt_obj:
                obj_correct += 1

        # direction error: a pred verb not covered by GT that is a known
        # inverse of some GT verb (strict = unconditional; contextual = only
        # via the object-conditioned map, i.e. only fires post-fold so it
        # reflects genuinely uncollapsed inverses like remove<->seat elsewhere)
        extra_pred = pred_set - gt_set
        hit_strict = hit_ctx = False
        for gv in gt_set:
            for pv in extra_pred:
                if pv in STRICT_INVERSE.get(gv, []):
                    strict_inv[gv] += 1
                    hit_strict = True
                elif pv in CONTEXTUAL_INVERSE.get(gv, []):
                    ctx_inv[(gv, obj_for_fold)] += 1
                    hit_ctx = True

        if not gt_set:
            cat["no_gt_verb"] += 1
        elif pred_set == gt_set:
            cat["exact"] += 1
        elif hit_strict or hit_ctx:
            cat["direction_swap"] += 1
        elif pred_set & gt_set:
            cat["partial_or_compound_omission"] += 1
        elif not pred_set:
            cat["no_pred_verb"] += 1
        else:
            cat["other_wrong"] += 1

    print(f"=== N2 naming eval (n={n}) ===")
    print(f"segments with >=1 GT verb: {n_with_gt_verbs}/{n}")
    if verb_p:
        print(f"verb-set  P={statistics.mean(verb_p):.3f}  "
              f"R={statistics.mean(verb_r):.3f}  F1={statistics.mean(verb_f1):.3f}")
        print(f"exact verb-set accuracy: {exact_set}/{n_with_gt_verbs} = "
              f"{exact_set/max(n_with_gt_verbs,1):.1%}")
    if sec_total:
        print(f"secondary-verb recall (segments with >=2 GT verbs only): "
              f"{sec_hit}/{sec_total} = {sec_hit/sec_total:.1%}")
    else:
        print("secondary-verb recall: no segments with >=2 GT verbs in this set")
    if obj_total:
        print(f"canonical-object accuracy: {obj_correct}/{obj_total} = "
              f"{obj_correct/obj_total:.1%}  (of {n} segs, {obj_total} have a "
              f"resolvable GT object)")

    print(f"\nsegment-level verdict breakdown:")
    for k, c in cat.most_common():
        print(f"  {k:28s} {c:5d}  {c/n:.1%}")

    print(f"\nstrict-inverse errors by GT verb (unconditional direction swap):")
    for v, c in strict_inv.most_common(10):
        print(f"  {v:12s} {c}")
    print(f"\ncontextual-inverse errors by (GT verb, object) "
          f"(NOT auto-collapsed -- audit before adding to CONTEXTUAL_VERB_NORM):")
    for (v, o), c in ctx_inv.most_common(10):
        print(f"  {v:12s} | {o:20s} {c}")

    print(f"\ngeneric-verb predictions (manipulate/present/display/adjust/move/"
          f"arrange -- excluded from N4 hard-negative distractors): "
          f"{generic_pred_segs}/{n} = {generic_pred_segs/n:.1%}")
    print("\nNOTE: atomicity accuracy is NOT computed here -- needs N3's "
          "hand-labeled {atomic,compound,cycle,phase} ground truth, which "
          "doesn't exist yet. Add once N3 benchmark lands.")


if __name__ == "__main__":
    main()
