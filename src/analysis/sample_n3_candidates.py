"""N3 -- select a stratified ~200-300 segment candidate list for the clean
hand-labeled naming benchmark, and emit a CSV template for the human
annotator to fill in {verbs, object, state_before, state_after, atomicity}.

This script does NOT annotate anything -- it only picks a DIVERSE, BALANCED
set of segments to look at, covering:
  - the target object list (sink strainer, mug, power bank, tissue, usb cable,
    remote control, umbrella, squeegee, water bottle, slippers)
  - category tags: atomic (1 verb) / compound (>=2 verbs) / inverse-pair
    (correct verb has a known inverse also observed on this object) / cycle
    (name says "cycle"/ordinal, or an adjacent segment shares the exact same
    name) / same-object-diff-verb (this recording uses >=3 distinct verbs on
    this object) / short_duration (<=3s, a PROXY for visual ambiguity -- the
    annotator should re-tag this by eye, it's just a cheap pre-filter)

A segment can carry multiple tags; selection is quota-based per (category)
and per (object) so no single easy bucket dominates the 200-300 budget.

Usage (server):
    python -m src.analysis.sample_n3_candidates \
        --pool_data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --target_data /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/n3_candidates.csv --n_total 250
"""
import argparse, csv, json, random
from collections import defaultdict

from src.analysis.build_ontology import extract_verbs, extract_object, STRICT_INVERSE, CONTEXTUAL_INVERSE

TARGET_OBJECTS = {"sink strainer", "mug", "power bank", "tissue", "usb cable",
                  "remote control", "umbrella", "squeegee", "water bottle",
                  "slippers"}
CATEGORIES = ["atomic", "compound", "inverse_pair", "cycle",
              "same_object_diff_verb", "short_duration"]


def dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def build_pool(paths):
    pool = defaultdict(set)
    for p in paths:
        for r in json.load(open(p)):
            for seg in r.get("solution", []):
                if not (seg and isinstance(seg[0], str)):
                    continue
                verbs = dedup(extract_verbs(seg[0]))
                obj, _, _, _ = extract_object(seg[0])
                if obj:
                    pool[obj].update(verbs)
    return pool


def tag_segment(name, obj, verbs, s, e, prev_name, next_name, pool):
    tags = set()
    tags.add("compound" if len(verbs) >= 2 else "atomic")
    if "cycle" in name.lower() or name == prev_name or name == next_name:
        tags.add("cycle")
    if verbs:
        inv = set(STRICT_INVERSE.get(verbs[0], [])) | set(CONTEXTUAL_INVERSE.get(verbs[0], []))
        if inv & pool.get(obj, set()):
            tags.add("inverse_pair")
    if len(pool.get(obj, set())) >= 3:
        tags.add("same_object_diff_verb")
    if (e - s) <= 3.0:
        tags.add("short_duration")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_data", nargs="+", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = build_pool(a.pool_data)
    rng = random.Random(a.seed)

    all_items = []
    for r in json.load(open(a.target_data)):
        segs = [seg for seg in r.get("solution", []) if seg and isinstance(seg[0], str)]
        names = [seg[0] for seg in segs]
        for i, seg in enumerate(segs):
            name, s, e = seg[0], seg[1], seg[2]
            verbs = dedup(extract_verbs(name))
            obj, _, _, _ = extract_object(name)
            if obj not in TARGET_OBJECTS or not verbs:
                continue
            prev_n = names[i - 1] if i > 0 else None
            next_n = names[i + 1] if i + 1 < len(names) else None
            tags = tag_segment(name, obj, verbs, s, e, prev_n, next_n, pool)
            all_items.append({
                "recording_id": r.get("recording_id"), "video": r["video"],
                "segment_idx": i, "start": round(s, 2), "end": round(e, 2),
                "gt_name": name, "canonical_object": obj,
                "canonical_verbs_auto": ";".join(verbs), "tags": sorted(tags),
            })

    print(f"eligible segments (target object + resolvable verb): {len(all_items)}")

    # quota-based selection: fill each (category) bucket up to a per-category
    # target, capping how many items from the SAME object each bucket takes so
    # e.g. "compound" isn't 90% sink-strainer.
    per_cat_target = a.n_total // len(CATEGORIES)
    per_obj_cap_per_cat = max(2, per_cat_target // len(TARGET_OBJECTS) + 1)
    selected, selected_ids = [], set()
    for cat in CATEGORIES:
        items = [x for x in all_items if cat in x["tags"] and id(x) not in selected_ids]
        rng.shuffle(items)
        obj_cap = defaultdict(int)
        n_this_cat = 0
        for x in items:
            if n_this_cat >= per_cat_target:
                break
            if obj_cap[x["canonical_object"]] >= per_obj_cap_per_cat:
                continue
            obj_cap[x["canonical_object"]] += 1
            selected.append(x); selected_ids.add(id(x)); n_this_cat += 1
        print(f"  {cat:24s} pool={len(items):5d} selected={n_this_cat}")

    # top up to n_total if short (categories overlap heavily so quotas rarely
    # fill exactly), sampling uniformly at random from whatever's left
    if len(selected) < a.n_total:
        remaining = [x for x in all_items if id(x) not in selected_ids]
        rng.shuffle(remaining)
        for x in remaining:
            if len(selected) >= a.n_total:
                break
            selected.append(x); selected_ids.add(id(x))

    rng.shuffle(selected)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "recording_id", "segment_idx", "video", "start", "end",
                    "gt_name", "canonical_object_auto", "canonical_verbs_auto",
                    "category_tags",
                    # -- fill these in by watching the clip --
                    "verbs", "object", "state_before", "state_after", "atomicity", "notes"])
        for idx, x in enumerate(selected):
            w.writerow([idx, x["recording_id"], x["segment_idx"], x["video"],
                        x["start"], x["end"], x["gt_name"], x["canonical_object"],
                        x["canonical_verbs_auto"], ";".join(x["tags"]),
                        "", "", "", "", "", ""])

    print(f"\nwrote {len(selected)} candidates -> {a.out}")
    obj_counts = defaultdict(int)
    for x in selected:
        obj_counts[x["canonical_object"]] += 1
    print("per-object coverage:", dict(sorted(obj_counts.items(), key=lambda kv: -kv[1])))
    print("\nNOTE: 'verbs'/'object'/'canonical_verbs_auto'/'canonical_object_auto' "
          "columns are ontology-DERIVED, not human-confirmed -- the empty "
          "verbs/object/state_before/state_after/atomicity/notes columns are "
          "what the human annotator actually fills in while watching the clip "
          "(start-2s to end+2s), independently of the GT text.")


if __name__ == "__main__":
    main()
