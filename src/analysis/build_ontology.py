"""N1 -- build the naming ontology (verbs / objects / inverse map) from GT.

Produces DRAFT ontology files under an output dir, to be human-reviewed (the
plan requires ontology stable before any big naming experiment). It:
  - extracts leading + embedded controlled verbs across all GT names, applies a
    normalization map (scrubbing->scrub, rinsing->rinse, ...) and a stoplist
    (first/second/cycle/paper/... are NOT verbs), reports the surviving verb
    frequency so a human can confirm merges vs keep-distinct;
  - extracts candidate object head-nouns (content words minus verbs/stop),
    applies an object-normalization map (power bank aliases, sink strainer
    aliases, ...);
  - emits a starter inverse-map (one-to-MANY allowed) for direction scoring.

Nothing here calls a model. Review + hand-edit the outputs before N2/N4 use them.

Usage (server):
    python -m src.analysis.build_ontology \
        --data /workspace/tr1/data_recseg/recseg_train.json \
        --data /workspace/tr1/data_recseg/recseg_val.json \
        --out_dir results/naming/ontology
"""
import argparse, json, os, re
from collections import Counter

_WORD = re.compile(r"[a-zA-Z]+")
tok = lambda s: [w.lower() for w in _WORD.findall(s)]

# not verbs even if they appear leading / look verb-ish
STOPLIST = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "pass", "cycle", "start", "starts", "end",
    "ends", "portion", "main", "full", "paper", "item", "object", "step",
    "iteration", "here", "final", "initial", "the", "a", "an", "and", "of",
    "to", "into", "onto", "on", "in", "with", "from", "for", "at", "by",
    "left", "right", "side", "front", "back", "center", "position", "positions",
    "flat", "small", "new", "second", "using", "via", "next",
}

# surface form -> canonical verb (merges inflections/synonyms that are SAME action)
VERB_NORM = {
    "scrubbing": "scrub", "rinsing": "rinse", "opening": "open", "closing": "close",
    "folding": "fold", "unfolding": "unfold", "wiping": "wipe", "cleaning": "clean",
    "retrieving": "retrieve", "taking": "retrieve", "extracting": "extract",
    "removing": "remove", "inserting": "insert", "replacing": "replace",
    "reseat": "seat", "re-seat": "seat", "reinstall": "seat", "reseating": "seat",
    "seating": "seat", "coiling": "coil", "uncoiling": "uncoil", "pressing": "press",
    "sliding": "slide", "rotating": "rotate", "flipping": "flip", "pouring": "pour",
    "adjusting": "adjust", "inspecting": "inspect", "mounting": "mount",
    "unwrapping": "unwrap", "wrapping": "wrap", "unpacking": "unpack",
    "repackaging": "repack", "repackage": "repack", "repacking": "repack",
    "unboxing": "unbox", "filling": "fill", "emptying": "empty", "rolling": "roll",
    "tucking": "tuck", "grasping": "grasp", "grabbing": "grab", "holding": "hold",
    "lifting": "lift", "placing": "place", "putting": "put", "picking": "pick",
    "extending": "extend", "retracting": "retract",
}

# the canonical verb vocabulary we keep (post-normalization). Keep pairs that
# must stay DISTINCT distinct: retrieve!=remove, clean!=rinse, seat!=replace,
# fold!=roll, open!=unbox.
CANONICAL_VERBS = {
    "open", "close", "remove", "insert", "replace", "seat", "unpack", "repack",
    "unbox", "fold", "unfold", "roll", "tuck", "coil", "uncoil", "extend",
    "retract", "fill", "empty", "tighten", "loosen", "attach", "detach", "mount",
    "wipe", "clean", "rinse", "scrub", "inspect", "rotate", "flip", "slide",
    "press", "pour", "adjust", "wrap", "unwrap", "retrieve", "extract", "pick",
    "put", "place", "hold", "grab", "grasp", "lift", "reposition", "move",
    "coil", "loop", "thread", "align",
}

# PHRASE-level object normalization (phrase -> canonical object). Longest match
# wins, so multi-word objects ("sink strainer") aren't split into single words
# ("sink","strainer"). object = the entity the action DIRECTLY acts on; tool /
# content / container are separate fields (see TOOL_WORDS / extract_object).
# Do NOT over-merge related-but-different objects (power bank != product box).
OBJECT_NORM = {
    "sink drain strainer": "sink strainer", "drain strainer": "sink strainer",
    "sink strainer": "sink strainer", "strainer": "sink strainer",
    "portable power bank": "power bank", "portable charger": "power bank",
    "power bank": "power bank", "charger": "power bank",
    "remote control": "remote control", "remote": "remote control",
    "utility knife": "utility knife", "box cutter": "utility knife",
    "tissue sheet": "tissue", "facial tissue": "tissue", "tissue paper": "tissue",
    "paper tissue": "tissue", "tissue": "tissue",
    "water bottle": "water bottle", "bottle": "water bottle",
    "product box": "product box", "charger box": "product box",
    "portable charger box": "product box", "inner box": "product box",
    "sink countertop": "countertop", "countertop": "countertop",
    "squeegee": "squeegee", "sink squeegee": "squeegee",
    "usb cable": "usb cable", "charging cable": "usb cable", "cable": "usb cable",
    "umbrella canopy": "umbrella", "folded umbrella": "umbrella", "umbrella": "umbrella",
    "mug": "mug", "cup": "mug", "bowl": "bowl", "slippers": "slippers",
    "slipper": "slippers", "phone": "phone", "smartphone": "phone",
    "tape measure": "tape measure", "telescopic rod": "telescopic rod",
    "dustpan": "dustpan", "pen": "pen", "notebook": "notebook", "sink": "sink",
    "drain": "sink drain", "sink drain": "sink drain",
}
# longest phrases first, so "sink drain strainer" matches before "strainer"
OBJECT_PHRASES = sorted(OBJECT_NORM, key=lambda p: -len(p.split()))

TOOL_WORDS = {"squeegee", "utility knife", "tweezers", "brush", "cloth", "spray bottle"}
CONTENT_WORDS = {"dirty water", "water", "debris", "dust"}
CONTAINER_WORDS = {"product box", "box", "drawer", "pouch", "packaging", "pack"}

# object-side stoplist: modifiers/positions/phases/abstract that are NOT objects
OBJECT_STOP = {
    "white", "blue", "black", "red", "gray", "grey", "green",
    "left", "right", "front", "back", "center", "side", "upper", "lower", "top",
    "section", "portion", "cycle", "pass", "phase", "start", "final", "step",
    "manipulation", "process", "interaction", "adjustment", "attempt", "part",
    "running", "folded", "wrapped", "dirty", "clean", "wet", "dry", "closed",
    "item", "object", "thing", "unit", "product", "small", "flat", "new",
    "under", "onto", "into", "position", "window", "compact", "shape", "hand",
}

# strict inverses: pure visual state reversal, object-independent -> safe to use
STRICT_INVERSE = {
    "open": ["close"], "close": ["open"],
    "fold": ["unfold"], "unfold": ["fold"],
    "coil": ["uncoil"], "uncoil": ["coil"],
    "extend": ["retract"], "retract": ["extend"],
    "fill": ["empty"], "empty": ["fill"],
    "attach": ["detach"], "detach": ["attach"],
    "wrap": ["unwrap"], "unwrap": ["wrap"],
    "tighten": ["loosen"], "loosen": ["tighten"],
}
# contextual inverses: only hold given object state/relation (installed, held...)
# -> do NOT treat as unconditional inverse; keyed by (verb, object_family)
CONTEXTUAL_INVERSE = {
    "remove": ["seat", "insert"],           # only when removed-from-installed
    "seat": ["remove"], "insert": ["remove"],
    "mount": ["remove", "detach"],
    "pick": ["put", "place"], "put": ["pick"], "lift": ["put"],
    "unbox": ["repack"], "unpack": ["repack"], "repack": ["unpack", "unbox"],
}
# procedure-level counterparts (NOT frame-level visual inverses; benchmark only)
DIRECTIONAL_PAIR = {
    "retrieve": ["return", "store"], "clean": ["dirty"],
}


def norm_verb(w):
    w = VERB_NORM.get(w, w)
    return w if w in CANONICAL_VERBS else None


def extract_verbs(name):
    return [v for v in (norm_verb(w) for w in tok(name) if w not in STOPLIST) if v]


def extract_object(name):
    """Phrase-first, longest-match. Returns (object, tool, container, unresolved).
    object = canonical entity the action directly acts on (longest OBJECT_NORM
    phrase found); tool/container pulled out separately; leftover content words
    that resolve to nothing are returned as `unresolved` for audit."""
    toks = [w for w in tok(name) if w not in STOPLIST and norm_verb(w) is None]
    text = " ".join(toks)
    obj = tool = container = None
    matched_spans = []
    for phrase in OBJECT_PHRASES:                      # longest first
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            canon = OBJECT_NORM[phrase]
            if canon in TOOL_WORDS and tool is None:
                tool = canon
            elif canon in CONTAINER_WORDS and container is None:
                container = canon
            elif obj is None:
                obj = canon
            matched_spans.append(phrase)
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    unresolved = [w for w in text.split() if w not in OBJECT_STOP and len(w) > 2]
    return obj, tool, container, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    names = []
    for p in a.data:
        for r in json.load(open(p)):
            for seg in r.get("solution", []):
                if seg and isinstance(seg[0], str):
                    names.append(seg[0].strip())

    verb_freq = Counter(); no_verb = 0
    obj_freq = Counter(); tool_freq = Counter(); container_freq = Counter()
    unresolved_freq = Counter(); no_object = 0
    for nm in names:
        vs = extract_verbs(nm)
        if vs:
            verb_freq.update(vs)
        else:
            no_verb += 1
        obj, tool, cont, unresolved = extract_object(nm)
        if obj:
            obj_freq[obj] += 1
        else:
            no_object += 1
        if tool:
            tool_freq[tool] += 1
        if cont:
            container_freq[cont] += 1
        unresolved_freq.update(unresolved)

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"canonical_verbs": sorted(CANONICAL_VERBS),
               "normalization": VERB_NORM, "stoplist": sorted(STOPLIST),
               "observed_freq": dict(verb_freq.most_common())},
              open(os.path.join(a.out_dir, "verbs.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"phrase_normalization": OBJECT_NORM, "object_stoplist": sorted(OBJECT_STOP),
               "tool_words": sorted(TOOL_WORDS), "container_words": sorted(CONTAINER_WORDS),
               "canonical_object_freq": dict(obj_freq.most_common()),
               "tool_freq": dict(tool_freq.most_common()),
               "container_freq": dict(container_freq.most_common()),
               "unresolved_audit": dict(unresolved_freq.most_common(100))},
              open(os.path.join(a.out_dir, "objects.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"strict_inverse": STRICT_INVERSE,
               "contextual_inverse": CONTEXTUAL_INVERSE,
               "directional_pair_not_strict": DIRECTIONAL_PAIR},
              open(os.path.join(a.out_dir, "inverse_map.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"names: {len(names)} | NO canonical verb: {no_verb} ({no_verb/len(names):.1%})"
          f" | NO canonical object: {no_object} ({no_object/len(names):.1%})")
    print(f"\ntop {a.top} canonical verbs (post-norm):")
    for v, c in verb_freq.most_common(a.top):
        print(f"  {c:5d}  {v}")
    print(f"\ntop canonical OBJECTS (phrase-level, longest-match):")
    for w, c in obj_freq.most_common(a.top):
        print(f"  {c:5d}  {w}")
    non_obj = sum(c for _, c in unresolved_freq.most_common(50))
    print(f"\ntop 25 UNRESOLVED tokens (audit: should be low / real missing objects):")
    for w, c in unresolved_freq.most_common(25):
        print(f"  {c:5d}  {w}")
    if tool_freq:
        print(f"\ntools: {dict(tool_freq.most_common(10))}")
    print(f"\nwrote verbs.json / objects.json / inverse_map.json -> {a.out_dir}")
    print("N1 done-criteria: top objects are phrases (no split sink/strainer), "
          "unresolved tokens mostly modifiers not real objects. REVIEW by hand "
          "before N2/N4.")


if __name__ == "__main__":
    main()
