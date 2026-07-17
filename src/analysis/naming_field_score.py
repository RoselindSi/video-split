"""P0 -- score the structured JSON fields directly (verb/object), not the
canonical_name text. mean_sim on canonical_name conflates several different
things (text length, shared nouns, verb correctness) into one number that we
already showed is misleading (e.g. "remove sink strainer" vs GT "Rinse and
replace sink strainer" scores 0.81 on shared nouns despite the verb being
wrong). This scores what actually matters and specifically flags the two
dominant, distinct error types:

  - direction_swap: predicted verb is the KNOWN INVERSE of the GT verb
    (remove<->replace, open<->close, ...) -- the model got the object and
    rough action family right but inverted the direction.
  - compound_omission: GT name contains multiple action words (e.g. "rinse
    and replace", "open ... and remove ...") but the single predicted verb
    only captures one of them.

Self-contained (no cross-file imports) so it runs regardless of server layout.

Usage (server):
    python -m src.analysis.naming_field_score --jsonl /tmp/naming_struct_v2_probe.jsonl
"""
import argparse, json, re, statistics

_WORD = re.compile(r"[a-zA-Z]+")
tok = lambda s: [w.lower() for w in _WORD.findall(s)]

STOP = {"the", "a", "an", "and", "or", "to", "of", "into", "onto", "on", "in",
        "with", "from", "for", "at", "by", "up", "down", "out", "off", "over",
        "then", "all", "it", "its", "this", "that", "these", "those", "each",
        "again", "first", "second", "third", "starts", "ends", "here", "step",
        "cycle", "iteration"}
ORD_RE = re.compile(r"\b\d+(st|nd|rd|th)\b|\bfirst\b|\bsecond\b|\bthird\b", re.I)

CONTROLLED_VERBS = {
    "open", "close", "remove", "insert", "replace", "unpack", "repack", "fold",
    "unfold", "coil", "uncoil", "extend", "retract", "fill", "empty", "tighten",
    "loosen", "attach", "detach", "wipe", "clean", "rinse", "scrub", "inspect",
    "rotate", "flip", "slide", "press", "pour", "adjust", "wrap", "unwrap",
    "seat", "reseat", "mount", "unmount", "unbox", "unwrap", "pick", "place",
    "put", "hold", "grab", "grasp", "retrieve",
}

INVERSE_WORD = {
    "remove": "insert", "insert": "remove", "take": "insert", "extract": "insert",
    "unpack": "repack", "unbox": "repack", "repack": "unpack", "pack": "unpack",
    "open": "close", "close": "open", "unwrap": "wrap", "wrap": "unwrap",
    "fold": "unfold", "unfold": "fold", "coil": "uncoil", "uncoil": "coil",
    "extend": "retract", "retract": "extend", "fill": "empty", "empty": "fill",
    "tighten": "loosen", "loosen": "tighten", "screw": "unscrew", "unscrew": "screw",
    "attach": "detach", "detach": "attach", "mount": "detach", "seat": "remove",
    "reseat": "remove", "pick": "put", "put": "pick", "grab": "put", "place": "remove",
}

JSON_VERB_RE = re.compile(r'"verb"\s*:\s*"([^"]*)"', re.I)
JSON_OBJECT_RE = re.compile(r'"object"\s*:\s*"([^"]*)"', re.I)


def content(name):
    return {w for w in tok(name) if w not in STOP and len(w) > 2}


def gt_verbs(gt_name):
    """All controlled verbs mentioned in the GT name, in order (for compound
    detection). Ordinal-only names (e.g. "6th iteration") yield none."""
    clean = ORD_RE.sub("", gt_name)
    return [w for w in tok(clean) if w in CONTROLLED_VERBS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="output of eval_naming_persegment.py --structured")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.jsonl)]
    n = len(recs)
    verb_correct = obj_f1s = 0
    obj_f1_list = []
    direction_swaps, compound_omissions, other_verb_errors = [], [], []

    for r in recs:
        raw = r["raw"]
        vm = JSON_VERB_RE.search(raw); om = JSON_OBJECT_RE.search(raw)
        pred_verb = vm.group(1).strip().lower() if vm else ""
        pred_obj = om.group(1).strip() if om else r["pred_name"]

        gvs = gt_verbs(r["gt_name"])
        primary_gt_verb = gvs[0] if gvs else ""

        exact = bool(primary_gt_verb) and pred_verb == primary_gt_verb
        if exact:
            verb_correct += 1

        co = content(pred_obj); cg = content(r["gt_name"]) - set(gvs)
        of = (len(co & cg) / max(len(co | cg), 1)) if (co and cg) else 0.0
        obj_f1_list.append(of)

        if primary_gt_verb and pred_verb and not exact:
            if INVERSE_WORD.get(primary_gt_verb) == pred_verb:
                direction_swaps.append((r["recording_id"], r["segment_idx"],
                                        r["gt_name"], pred_verb))
            else:
                other_verb_errors.append((r["recording_id"], r["segment_idx"],
                                          r["gt_name"], pred_verb))
        if len(gvs) >= 2 and pred_verb in gvs and pred_verb != gvs[0]:
            pass  # captured a later verb, not the primary -- still "some" match
        if len(gvs) >= 2 and pred_verb not in gvs[1:]:
            # GT has >=2 action verbs; predicted verb doesn't cover the later one(s)
            compound_omissions.append((r["recording_id"], r["segment_idx"],
                                       r["gt_name"], pred_verb, gvs))

    print(f"=== P0 field-level scoring (n={n}) ===")
    print(f"primary_verb exact-match accuracy: {verb_correct}/{n} = {verb_correct/n:.1%}")
    print(f"object F1 (JSON 'object' field vs GT content words): "
          f"{statistics.mean(obj_f1_list):.3f}")
    print(f"\ndirection_swap (pred verb == INVERSE of GT verb): "
          f"{len(direction_swaps)}/{n} = {len(direction_swaps)/n:.1%}")
    for rid, si, gt, pv in direction_swaps[:6]:
        print(f"  {rid} seg{si}: GT='{gt}' pred_verb='{pv}'")
    print(f"\ncompound_omission (GT has >=2 action verbs, pred missed the "
          f"non-primary one(s)): {len(compound_omissions)}/{n} = "
          f"{len(compound_omissions)/n:.1%}")
    for rid, si, gt, pv, gvs in compound_omissions[:6]:
        print(f"  {rid} seg{si}: GT='{gt}' (verbs={gvs}) pred_verb='{pv}'")
    print(f"\nother verb errors (neither exact nor direction-swap): "
          f"{len(other_verb_errors)}/{n} = {len(other_verb_errors)/n:.1%}")


if __name__ == "__main__":
    main()
