"""One-off: dump the raw pred vs GT text for every segment where N2's
canonical-object accuracy check says "wrong", so we can eyeball whether it's
a REAL object error or an ontology-coverage gap (model used a phrasing not
yet in OBJECT_NORM, e.g. GT "sink strainer" vs model "drain filter").

Usage (server):
    python -m src.eval.dump_object_errors --jsonl /tmp/naming_struct_v2_probe.jsonl
"""
import argparse, json

from src.eval.eval_naming_v2 import pred_fields, gt_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.jsonl)]
    n_shown = 0
    for r in recs:
        pred_ordered, pred_obj = pred_fields(r["raw"], r.get("pred_name", ""))
        gt_ordered, gt_obj = gt_fields(r["gt_name"])
        if gt_obj and pred_obj != gt_obj:
            n_shown += 1
            print(f"--- {r['recording_id']} seg{r['segment_idx']} ---")
            print(f"  GT name:        {r['gt_name']}")
            print(f"  GT canon obj:   {gt_obj}")
            print(f"  pred canon obj: {pred_obj}")
            print(f"  raw model output: {r['raw'][:300]}")
            print()
    print(f"total object mismatches shown: {n_shown}")


if __name__ == "__main__":
    main()
