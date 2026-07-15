"""A0.1 — Naming vocabulary / distribution analysis.

Reads the multi-seg training json (rows with `solution` = [[name, start, end], ...])
and characterizes the ground-truth naming style, so we can:
  1. define a canonical naming paradigm (verb + object) from real data,
  2. build a controlled action/object vocabulary as a reference standard,
  3. calibrate the LLM-judge + decoupled naming metric (A0.2/A0.3).

Pure stdlib (no spaCy/nltk) so it runs in any venv on the server.

Usage (server):
    python -m src.analysis.naming_vocab \
        --data /workspace/tr1/data_handtask/train_multiseg_train.json \
        --dump_names /tmp/gt_names.txt
    # add a second --data for the val split to pool both.
"""

import argparse
import json
import re
from collections import Counter

# Words that signal a lazy / non-specific name (we WANT these to be rare).
GENERIC_WORDS = {
    "object", "objects", "item", "items", "thing", "things", "stuff",
    "something", "task", "tasks", "step", "steps", "part", "parts",
    "area", "surface", "material", "materials", "element",
}

# Common English stopwords + prepositions/articles, to surface content words.
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "into", "onto", "on", "in",
    "with", "from", "for", "at", "by", "up", "down", "out", "off", "over",
    "then", "all", "it", "its", "final", "first", "second", "third", "next",
    "each", "every", "some", "this", "that", "these", "those",
}

_WORD = re.compile(r"[a-zA-Z]+")


def load_names(paths):
    names = []
    for p in paths:
        rows = json.load(open(p))
        for r in rows:
            for seg in r.get("solution", []):
                # seg = [name, start, end]
                if seg and isinstance(seg[0], str):
                    names.append(seg[0].strip())
    return names


def tokenize(s):
    return [w.lower() for w in _WORD.findall(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                    help="one or more multiseg json paths (repeat --data to pool)")
    ap.add_argument("--dump_names", default=None,
                    help="optional path to write the full sorted unique name list")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    names = load_names(args.data)
    n = len(names)
    uniq = sorted(set(names))
    print(f"=== corpus: {n} segment names, {len(uniq)} unique "
          f"({len(uniq)/max(n,1):.0%} unique) ===\n")

    # 1) length (word count) distribution
    lens = [len(tokenize(x)) for x in names]
    lens_c = Counter(lens)
    print("--- name length (words) ---")
    for k in sorted(lens_c):
        print(f"  {k:2d} words: {lens_c[k]:4d}  {'#' * (lens_c[k] * 40 // max(lens_c.values()))}")
    print(f"  mean {sum(lens)/max(n,1):.1f} words\n")

    # 2) leading verb (imperative form -> first token is usually the action)
    verbs = Counter(tokenize(x)[0] for x in names if tokenize(x))
    print(f"--- top {args.top} leading verbs (action vocabulary) ---")
    for w, c in verbs.most_common(args.top):
        print(f"  {c:4d}  {w}")
    print(f"  ({len(verbs)} distinct leading verbs)\n")

    # 3) content words (objects/modifiers), stopwords + leading verbs removed
    content = Counter()
    for x in names:
        toks = tokenize(x)
        for w in toks[1:]:                       # drop leading verb
            if w not in STOP:
                content[w] += 1
    print(f"--- top {args.top} content words (objects/modifiers) ---")
    for w, c in content.most_common(args.top):
        print(f"  {c:4d}  {w}")
    print()

    # 4) genericity: how many names lean on non-specific words
    generic_names = [x for x in names if GENERIC_WORDS & set(tokenize(x))]
    print(f"--- genericity in GT (should be LOW) ---")
    print(f"  {len(generic_names)}/{n} names contain a generic word "
          f"({len(generic_names)/max(n,1):.1%})")
    gw = Counter(w for x in generic_names for w in tokenize(x) if w in GENERIC_WORDS)
    if gw:
        print("  generic words used:", dict(gw.most_common()))
    print()

    # 5) exact-duplicate names (canonical repeats across videos -> vocabulary core)
    dup = Counter(names)
    print(f"--- top {args.top} most-repeated exact names (canonical core) ---")
    for w, c in dup.most_common(args.top):
        if c < 2:
            break
        print(f"  {c:3d}x  {w}")
    print()

    if args.dump_names:
        with open(args.dump_names, "w") as f:
            f.write("\n".join(uniq))
        print(f"wrote {len(uniq)} unique names -> {args.dump_names}")


if __name__ == "__main__":
    main()
