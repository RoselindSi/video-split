"""N7 follow-up (5a) -- mine NATURALLY-OCCURRING matched pairs from GT: same
canonical object AND same primary verb, one instance atomic (that verb
alone) and one instance compound (that verb plus something else). No new
human annotation needed -- these pairs already exist in the dataset, e.g.
"rinse sink strainer" (atomic) vs "rinse and seat sink strainer" (compound).

This directly controls the confound the N7d probe found (pooled gate AUROC
was mostly explained by BETWEEN-object/verb differences, not real
within-group discrimination): by holding object+primary_verb FIXED within
each pair, any remaining score difference between the atomic and compound
member can't be explained by "this object/verb just tends to be compound" --
it has to come from something else (ideally the actual extra action, but
could still be confounded by recording/duration; report both).

Usage (server, no GPU needed):
    python -m src.eval.mine_matched_pairs \
        --data /workspace/tr1/data_recseg/recseg_train.json /workspace/tr1/data_recseg/recseg_val.json \
        --out /tmp/matched_pairs.jsonl --max_pairs_per_key 3
"""
import argparse, json, random
from collections import defaultdict

from src.analysis.build_ontology import extract_verbs, extract_object


def dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pairs_per_key", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    atomic = defaultdict(list)   # (object, verb) -> [(video, recording_id, seg_idx, s, e)]
    compound = defaultdict(list)

    for path in a.data:
        for r in json.load(open(path)):
            segs = [s for s in r.get("solution", []) if s and isinstance(s[0], str)]
            for i, (name, s, e) in enumerate(segs):
                verbs = dedup(extract_verbs(name))
                obj, _, _, _ = extract_object(name)
                if not verbs or obj is None:
                    continue
                key = (obj, verbs[0])
                item = (r["video"], r.get("recording_id"), i, s, e, name)
                (atomic if len(verbs) == 1 else compound)[key].append(item)

    shared_keys = sorted(set(atomic) & set(compound))
    print(f"(object, primary_verb) keys with BOTH atomic and compound "
          f"instances in GT: {len(shared_keys)}")

    pairs = []
    for key in shared_keys:
        a_items, c_items = list(atomic[key]), list(compound[key])
        rng.shuffle(a_items); rng.shuffle(c_items)
        n = min(len(a_items), len(c_items), a.max_pairs_per_key)
        for j in range(n):
            pairs.append({"pair_id": f"{key[0]}|{key[1]}|{j}",
                          "object": key[0], "primary_verb": key[1],
                          "atomic": {"video": a_items[j][0], "recording_id": a_items[j][1],
                                     "segment_idx": a_items[j][2], "start": a_items[j][3],
                                     "end": a_items[j][4], "gt_name": a_items[j][5]},
                          "compound": {"video": c_items[j][0], "recording_id": c_items[j][1],
                                       "segment_idx": c_items[j][2], "start": c_items[j][3],
                                       "end": c_items[j][4], "gt_name": c_items[j][5]}})

    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"wrote {len(pairs)} matched pairs (from {len(shared_keys)} keys) -> {a.out}")
    per_key = defaultdict(int)
    for p in pairs:
        per_key[(p["object"], p["primary_verb"])] += 1
    print("top 15 keys by pair count:",
          sorted(per_key.items(), key=lambda kv: -kv[1])[:15])


if __name__ == "__main__":
    main()
