"""Split train_multiseg.json into train / held-out val by whole video.

Held-out videos are NOT seen in training; used later for eval_multiseg.py.

    python -m src.data.split_multiseg --in /workspace/tr1/data_handtask/train_multiseg.json --val 14
"""

import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="/workspace/tr1/data_handtask/train_multiseg.json")
    ap.add_argument("--val", type=int, default=14, help="number of videos held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = json.load(open(args.inp))
    random.Random(args.seed).shuffle(rows)
    val = rows[: args.val]
    train = rows[args.val:]

    base = os.path.splitext(args.inp)[0]
    train_path = base + "_train.json"
    val_path = base + "_val.json"
    json.dump(train, open(train_path, "w"), ensure_ascii=False, indent=2)
    json.dump(val, open(val_path, "w"), ensure_ascii=False, indent=2)

    n_train_seg = sum(len(r["solution"]) for r in train)
    n_val_seg = sum(len(r["solution"]) for r in val)
    print(f"train: {len(train)} videos / {n_train_seg} segs -> {train_path}")
    print(f"val:   {len(val)} videos / {n_val_seg} segs -> {val_path}")


if __name__ == "__main__":
    main()
