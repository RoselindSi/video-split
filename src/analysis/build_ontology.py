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
    "reseat": "seat", "re-seat": "seat", "reseating": "seat",
    "seating": "seat", "coiling": "coil", "uncoiling": "uncoil", "pressing": "press",
    "sliding": "slide", "rotating": "rotate", "flipping": "flip", "pouring": "pour",
    "adjusting": "adjust", "inspecting": "inspect", "mounting": "mount",
    "unwrapping": "unwrap", "wrapping": "wrap", "unpacking": "unpack",
    "repackaging": "repack", "repackage": "repack", "repacking": "repack",
    "unboxing": "unbox", "filling": "fill", "emptying": "empty", "rolling": "roll",
    "tucking": "tuck", "grasping": "grasp", "grabbing": "grab", "holding": "hold",
    "lifting": "lift", "placing": "place", "putting": "put", "picking": "pick",
    "extending": "extend", "retracting": "retract",
    "folding": "fold", "coiled": "coil", "sweeping": "sweep", "relocating": "reposition",
    "relocate": "reposition", "installing": "install", "stacking": "stack",
    "unstacking": "unstack", "unplugging": "unplug", "plugging": "plug",
    "storing": "store", "turning": "turn", "arranging": "arrange",
    "reinstalling": "reinstall", "washing": "wash", "discarding": "discard",
    "peeling": "peel", "manipulating": "manipulate", "displaying": "display",
    "stowing": "stow", "presenting": "present", "collecting": "collect",
    "returning": "return",
    "flattening": "flatten", "pressing": "press", "loosening": "loosen",
    "tightening": "tighten", "threading": "thread", "aligning": "align",
    "gathering": "gather", "spreading": "spread",
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
    "coil", "loop", "thread", "align", "flatten",
    # added from unresolved-verb audit (real actions the first pass missed)
    "sweep", "plug", "unplug", "install", "stack", "unstack", "store", "turn",
    "arrange", "wash", "peel", "manipulate", "display", "stow", "present",
    "collect", "tape", "gather", "spread", "reinstall", "discard", "return",
}

# verbs that describe an action too weakly to be a good hard-negative candidate
# (N4/N7): keep them canonical so they don't inflate NO-verb, but flag as generic
# so the benchmark builder won't offer them as distractors.
GENERIC_VERBS = {"manipulate", "present", "display", "adjust", "move", "arrange"}

# object-conditioned verb normalization: a verb collapses to another ONLY for a
# specific object family, where the audit (reinstall/replace x object dumps)
# confirmed the surface verb is really the canonical action. Applied at SCORING
# time, never globally. Everything not listed keeps its own canonical verb until
# audited. Seeded conservatively; expand only from the *_x_object audit output.
CONTEXTUAL_VERB_NORM = {
    # audit (iter4): replace/reinstall/install sink strainer all = putting the
    # strainer back into the drain = seat; remove stays (the inverse). 467+187+
    # 81 vs seat 241. Gives a clean remove<->seat direction pair on one object.
    ("reinstall", "sink strainer"): "seat",
    ("replace", "sink strainer"): "seat",
    ("install", "sink strainer"): "seat",
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
    "paper tissue": "tissue", "tissue": "tissue", "napkin": "tissue",
    "water bottle": "water bottle", "bottle": "water bottle",
    "product box": "product box", "charger box": "product box",
    "portable charger box": "product box", "inner box": "product box",
    # longer phrases FIRST so the container (head noun) wins over "portable
    # charger" -> power bank; "open the charger's PRODUCT BOX" is an action on
    # the box, the charger isn't even out yet. (audit: rec188 seg0/5, rec196
    # seg9 -- model correctly said "open box", ontology was wrongly forcing
    # GT object to power bank via "portable charger" matching first.)
    "portable charger product box": "product box",
    "portable charger packaging": "product box",
    "sink countertop": "countertop", "countertop": "countertop",
    "squeegee": "squeegee", "sink squeegee": "squeegee",
    "usb cable": "usb cable", "charging cable": "usb cable", "cable": "usb cable",
    "umbrella canopy": "umbrella", "folded umbrella": "umbrella", "umbrella": "umbrella",
    "mug": "mug", "cup": "mug", "bowl": "bowl", "slippers": "slippers",
    "slipper": "slippers", "phone": "phone", "smartphone": "phone",
    "tape measure": "tape measure", "telescopic rod": "telescopic rod",
    "dustpan": "dustpan", "pen": "pen", "notebook": "notebook", "sink": "sink",
    "drain": "sink drain", "sink drain": "sink drain",
    # added from unresolved audit (real objects the first pass missed)
    "table": "table", "wall": "wall", "screen": "screen", "cabinet": "cabinet",
    "floor": "floor", "phone case": "phone case", "case": "phone case",
    "adapter": "adapter", "blade": "blade", "power button": "button",
    "button": "button", "power bank": "power bank", "power cord": "power cord",
    "power strip": "power strip", "power adapter": "adapter",
    "tissue sheet": "tissue", "sheet": "tissue", "tissue pack": "tissue pack",
    "sink edge": "sink edge", "table edge": "table edge",
    "phone screen": "screen", "smartphone": "phone",
    # iter5 (final context audit): "lift and re-seat removable sink basin" /
    # "adjust and check sink basin fit" -- basin IS the directly-manipulated
    # object, not a modifier. "pack" alone (e.g. "tissue ... from pack") is the
    # same referent as the existing "tissue pack" phrase, just split across the
    # sentence -- map it to the same canonical so it's absorbed, not leaked.
    "sink basin": "sink basin", "basin": "sink basin", "pack": "tissue pack",
    # iter3: unambiguous phrase objects from unresolved audit (box/power/edge/
    # surface/crease/metal HELD pending *_x_context raw dumps below)
    "wall socket": "socket", "power socket": "socket", "power outlet": "socket",
    "outlet": "socket", "socket": "socket",
    "desk drawer": "drawer", "table drawer": "drawer", "drawer": "drawer",
    "desk": "desk", "counter": "countertop", "sink counter": "countertop",
    "metal tin": "tin", "tin": "tin", "tins": "tin",
    "tissue sachet": "sachet", "sachet": "sachet", "sachets": "sachet",
    "trash bin": "trash bin", "waste bin": "trash bin", "bin": "trash bin",
    "tool holder": "holder", "squeegee holder": "holder", "holder": "holder",
    # iter4 (from context dumps): power cable was being eaten by "cable"->usb
    # cable, mislabeling 325 power cables. Give it its own object; unify cord.
    "power cable": "power cable", "power cord": "power cable",
    "data cable": "usb cable",                       # same eaten-by-"cable" bug
    "storage box": "box", "box": "box", "bowls": "bowl",
    "pens": "pen",                                   # was splitting off from pen
    "faucet": "faucet", "tap": "faucet",
    "bottle cap": "cap", "cap": "cap",
}
# cleaning implements -> tool slot (not the acted-on object)
_TOOL_OBJ = {"cleaning sponge": "sponge", "sponge": "sponge",
             "cleaning cloth": "cloth", "cloth": "cloth", "rag": "cloth"}
OBJECT_NORM.update(_TOOL_OBJ)
# content = substance being moved/removed (NOT the object); keep separate
CONTENT_NORM = {"dirty water": "dirty water", "water": "water",
                "debris": "debris", "dust": "dust"}
# longest phrases first, so "sink drain strainer" matches before "strainer"
OBJECT_PHRASES = sorted(OBJECT_NORM, key=lambda p: -len(p.split()))

TOOL_WORDS = {"squeegee", "utility knife", "tweezers", "brush", "cloth",
              "spray bottle", "sponge", "rag", "holder"}
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
    "pair", "two", "one", "three", "bimanual", "both", "full", "empty",
    "portable", "disposable", "used", "each", "single", "multiple",
    # content substances (tracked separately in CONTENT_NORM, not objects)
    "water", "debris", "dust",
    # iter4: region / feature / phase / modifier words from context audit --
    # these are parts-of an object or phase titles, never standalone objects
    "edge", "crease", "surface", "sides", "opposite", "interior", "exterior",
    "half", "forth", "operation", "inspection", "batch", "metal", "face",
    "corner", "row", "column", "layer", "end", "middle",
    # iter5 (final context audit): "tea" is always a modifier of sachet/pack
    # ("tea sachet(s)", "pack tea sachets") -- content, never itself the
    # manipulated object (matches water/debris/dust). "tool" in this data is
    # always the AGENT acting on an already-resolved object ("curved wall
    # tool" -> object is "wall", already captured separately). "tabletop"
    # only ever modifies another noun ("tabletop appliance/device/items") or
    # marks location ("on right tabletop") -- never itself acted on, unlike
    # "sink basin" which IS acted on directly.
    "tea", "tool", "tabletop",
    # pure descriptors seen in the final unresolved audit, never referents
    "plastic", "cylindrical", "off", "far", "near", "neat", "handheld",
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
    Match object PHRASES on stoplist-filtered text FIRST (before removing verbs),
    so noun phrases whose head is a verb-homonym ("tape measure", "power plug")
    still match; only the LEFTOVER after phrase removal is verb-filtered for the
    unresolved audit."""
    text = " ".join(w for w in tok(name) if w not in STOPLIST)
    obj = tool = container = None
    for phrase in OBJECT_PHRASES:                      # longest first
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            canon = OBJECT_NORM[phrase]
            if canon in TOOL_WORDS and tool is None:
                tool = canon
            elif canon in CONTAINER_WORDS and container is None:
                container = canon
            elif obj is None:
                obj = canon
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    unresolved = [w for w in text.split()
                  if w not in OBJECT_STOP and len(w) > 2 and norm_verb(w) is None]
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

    # tokens the mentor flagged as ambiguous: dump raw context before deciding a
    # phrase mapping (don't guess / don't stoplist blindly).
    CONTEXT_TOKENS = {"power", "box", "edge", "surface", "crease", "metal",
                      "tape", "plug", "socket", "holder",
                      "tool", "tea", "pack", "basin", "tabletop"}
    # verbs whose object-conditioned collapse we need to see before freezing.
    PAIR_VERBS = {"replace", "reinstall", "install", "seat", "remove"}

    verb_freq = Counter(); no_verb = 0
    obj_freq = Counter(); tool_freq = Counter(); container_freq = Counter()
    unresolved_freq = Counter(); no_object = 0
    verb_obj_pairs = Counter()
    unresolved_examples = {}          # token -> [raw names]
    context_examples = {}             # token -> [raw names]
    pair_examples = {}                # verb -> Counter(object)
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
        for v in set(vs):
            verb_obj_pairs[(v, obj or "<none>")] += 1
            if v in PAIR_VERBS:
                pair_examples.setdefault(v, Counter())[obj or "<none>"] += 1
        for tkn in set(unresolved):
            ex = unresolved_examples.setdefault(tkn, [])
            if len(ex) < 10:
                ex.append(nm)
        toks = set(tok(nm))
        for tkn in CONTEXT_TOKENS & toks:
            ex = context_examples.setdefault(tkn, [])
            if len(ex) < 10:
                ex.append(nm)

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"canonical_verbs": sorted(CANONICAL_VERBS),
               "generic_verbs": sorted(GENERIC_VERBS),
               "normalization": VERB_NORM, "stoplist": sorted(STOPLIST),
               "contextual_verb_norm": {f"{v}|{o}": c
                                        for (v, o), c in CONTEXTUAL_VERB_NORM.items()},
               "observed_freq": dict(verb_freq.most_common())},
              open(os.path.join(a.out_dir, "verbs.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"phrase_normalization": OBJECT_NORM, "object_stoplist": sorted(OBJECT_STOP),
               "tool_words": sorted(TOOL_WORDS), "container_words": sorted(CONTAINER_WORDS),
               "canonical_object_freq": dict(obj_freq.most_common()),
               "tool_freq": dict(tool_freq.most_common()),
               "container_freq": dict(container_freq.most_common()),
               "unresolved_audit": dict(unresolved_freq.most_common(100)),
               "unresolved_examples": {t: unresolved_examples[t]
                                       for t, _ in unresolved_freq.most_common(40)},
               "context_examples": context_examples,
               "verb_object_pairs": {f"{v}|{o}": c
                                     for (v, o), c in verb_obj_pairs.most_common(120)},
               "pair_verb_object": {v: dict(c.most_common(20))
                                    for v, c in pair_examples.items()}},
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

    print(f"\n=== top-12 unresolved tokens + up to 5 raw examples "
          f"(decide: phrase-object / modifier / content) ===")
    for w, c in unresolved_freq.most_common(12):
        print(f"  [{c}] {w}:")
        for nm in unresolved_examples.get(w, [])[:5]:
            print(f"        {nm}")
    print(f"\n=== context dumps for ambiguous tokens (mentor-flagged) ===")
    for tkn in sorted(context_examples):
        print(f"  {tkn}:")
        for nm in context_examples[tkn][:6]:
            print(f"        {nm}")
    print(f"\n=== object-conditioned verb audit (replace/reinstall/install/seat "
          f"x object -> which collapse to seat?) ===")
    for v in ("replace", "reinstall", "install", "seat", "remove"):
        if v in pair_examples:
            top = pair_examples[v].most_common(8)
            print(f"  {v}: " + ", ".join(f"{o}={n}" for o, n in top))

    print(f"\nwrote verbs.json / objects.json / inverse_map.json -> {a.out_dir}")
    print("N1 done-criteria: top objects are phrases (no split sink/strainer), "
          "unresolved tokens mostly modifiers not real objects. REVIEW by hand "
          "before N2/N4.")


if __name__ == "__main__":
    main()
