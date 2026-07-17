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

OBJECT_NORM = {
    "portable charger": "power bank", "charger": "power bank",
    "sink drain strainer": "sink strainer", "drain strainer": "sink strainer",
    "strainer": "sink strainer", "cup": "mug", "tissue sheet": "tissue",
    "facial tissue": "tissue", "usb cable": "usb cable", "remote": "remote control",
}

# one-to-MANY inverse map for direction scoring
INVERSE_MAP = {
    "remove": ["seat", "insert", "replace"], "seat": ["remove"],
    "insert": ["remove"], "replace": ["remove"],
    "open": ["close"], "close": ["open"],
    "unpack": ["repack"], "unbox": ["repack"], "repack": ["unpack", "unbox"],
    "fill": ["empty"], "empty": ["fill"],
    "fold": ["unfold"], "unfold": ["fold"],
    "coil": ["uncoil"], "uncoil": ["coil"],
    "extend": ["retract"], "retract": ["extend"],
    "pick": ["put"], "put": ["pick"], "lift": ["put"],
    "mount": ["remove", "detach"], "attach": ["detach"], "detach": ["attach"],
    "tighten": ["loosen"], "loosen": ["tighten"],
    "wrap": ["unwrap"], "unwrap": ["wrap"],
}


def norm_verb(w):
    w = VERB_NORM.get(w, w)
    return w if w in CANONICAL_VERBS else None


def extract_verbs(name):
    return [v for v in (norm_verb(w) for w in tok(name) if w not in STOPLIST) if v]


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

    verb_freq = Counter()
    no_verb = 0
    obj_words = Counter()
    for nm in names:
        vs = extract_verbs(nm)
        if vs:
            verb_freq.update(vs)
        else:
            no_verb += 1
        vset = set(vs)
        for w in tok(nm):
            if w not in STOPLIST and w not in vset and len(w) > 2 \
                    and norm_verb(w) is None:
                obj_words[w] += 1

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"canonical_verbs": sorted(CANONICAL_VERBS),
               "normalization": VERB_NORM, "stoplist": sorted(STOPLIST),
               "observed_freq": dict(verb_freq.most_common())},
              open(os.path.join(a.out_dir, "verbs.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"normalization": OBJECT_NORM,
               "observed_head_nouns": dict(obj_words.most_common(200))},
              open(os.path.join(a.out_dir, "objects.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(INVERSE_MAP, open(os.path.join(a.out_dir, "inverse_map.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"names: {len(names)} | names with NO canonical verb: {no_verb} "
          f"({no_verb/len(names):.1%}) <- review: are these real actions the "
          f"CANONICAL_VERBS set is missing, or genuinely non-action labels?")
    print(f"\ntop {a.top} canonical verbs (post-norm):")
    for v, c in verb_freq.most_common(a.top):
        print(f"  {c:5d}  {v}")
    print(f"\ntop {a.top} candidate object head-nouns:")
    for w, c in obj_words.most_common(a.top):
        print(f"  {c:5d}  {w}")
    print(f"\nwrote verbs.json / objects.json / inverse_map.json -> {a.out_dir}")
    print("REVIEW these drafts by hand before N2/N4 consume them.")


if __name__ == "__main__":
    main()
