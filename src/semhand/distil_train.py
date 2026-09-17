"""Distil per-hand Qwen into V1's classifier slot, and test it where nothing has looked.

THE STUDENT REPLACES ONE FILE. V1's pipeline keeps its detector, its
geometric prior, OwnHold and the two-owner cap; only the classifier weights
change. So the student is trained and read exactly where `own_ctx_best.pt`
sits today.

WHAT THE SWAP EXPERIMENT DECIDED. Qwen's advantage is the view: restricted to
V1's crops it does worse than V1 (476 foreign frames against 296), while V1's
own architecture reading the whole frame with the box drops from 175 to 34.
But a whole-frame student trained on 1,013 hands memorised workstations (133
of 141 foreign errors on one unseen day). So the student reads the whole frame
AND is trained on many scenes -- which is what the teacher is for.

ARMS (V1's architecture and recipe throughout, three seeds, P averaged):
    S_wide    whole frame with the box (448x246) + 224 crop, Qwen labels
    S_v1      V1's own view + 14 geometry features, Qwen labels
              (the same control the swap experiment ran, now with 20x the data)
    S_wide_h  S_wide plus V1's 1,013 human-labelled hands
    S_wide_z  S_wide plus the human zone-labelled hands from the batches not
              held out (the annotator's own labels, fed in as the user asked)

WRITTEN BEFORE ANY STUDENT IS SCORED -- WHAT IT TAKES TO SHIP. On the held-out
zone batch, through V1's own post-processing (prior + OwnHold + cap), against
`V1 deployed` on the same detections and frames:

    (1) M2  foreign-hand frames called self        <= V1 deployed's
    (2) M1  own-hand flips per 100 pairs           <= V1 deployed's
    (3) G   own-hand frames called self            >= V1 deployed's - GUARD
    (4) at least one of M1, M2 strictly better with a 95% recording-bootstrap
        interval of the difference that excludes zero

TWO TEACHERS (added 2026-09-17, written before GPT-6 labelled the pool). The
blind audit showed Q1's remaining privacy-direction errors are mostly boxes
GPT-6 gets right (11 of 13 on the disagreement strata). Two more arms, same
view, recipe and seeds as S_wide:
    S_wide_and   train only on hands where Q1 and GPT-6 agree
    S_wide_cons  owner only when both say owner; every other hand -- a
                 disagreement or a GPT-6 "cannot tell" -- is trained as other
They are compared with S_wide on the held-out zone batches (4-6, and 7-9 once
labelled) at the post-processing the ablation chose (geom_w 0.25, no cap):
a two-teacher arm replaces S_wide only if its foreign frames called self are
fewer on 7-9 AND it still passes the four criteria against V1 deployed there.

Raw (unsmoothed) numbers are reported beside it, and fresh29 / e2e_main2 are
reported as secondary because both have been scored many times. A student that
only wins on the sets its teacher was measured on does not ship.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random

import numpy as np

from src.semhand.crossv1 import Views, fit, loader, predict, read_rows

ARMS = ("S_wide", "S_v1", "S_wide_h", "S_wide_z", "S_wide_and", "S_wide_cons")
VIEW = {"S_wide": "VQ", "S_v1": "VV", "S_wide_h": "VQ", "S_wide_z": "VQ",
        "S_wide_and": "VQ", "S_wide_cons": "VQ"}


def gpt6_labels(root):
    """-> {id: True | False | None} from gpt6_teacher's answers."""
    out = {}
    p = os.path.join(root, "gpt6", "answers.jsonl")
    for line in open(p):
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d["wearer"]
    return out


def two_teacher(q1, g6, mode):
    """Combine Q1 (1 wearer / 0 other) with GPT-6 (True / False / None)."""
    out = {}
    for k, y in q1.items():
        if k not in g6:
            continue
        g = g6[k]
        if mode == "and":
            if g is not None and int(g) == y:
                out[k] = y
        else:                                            # cons
            out[k] = 1 if (y == 1 and g is True) else 0
    return out


def qwen_labels(root, arm="Q1"):
    out = {}
    for f in glob.glob(os.path.join(root, "qwen", f"{arm}_*.jsonl")):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = int(d["p"] >= 0.5)
    return out


def zone_labels(path):
    """-> {id: y} from `selfother.labels` output (1 = the wearer's own hand)."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("label") in ("0", "1"):
            out[f"{r['rec']}|{r['frame']}|{r['tid']}"] = int(r["label"])
    return out


def with_labels(rows, labels):
    keep = []
    for r in rows:
        if r["id"] in labels:
            keep.append(dict(r, y=labels[r["id"]], tag=r["id"].split("|")[0]))
    return keep


def write_preds(root, arm, trows, p):
    """Add or replace one arm's column in <root>/student/pred.csv."""
    d = os.path.join(root, "student")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "pred.csv")
    old = {}
    if os.path.exists(path):
        old = {r["id"]: r for r in csv.DictReader(open(path))}
    cols = [c for c in (list(old[next(iter(old))].keys()) if old else ["id"]) if c != "id"]
    cols = sorted(set(cols) | {arm})
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + cols)
        for n, r in enumerate(trows):
            row = dict(old.get(r["id"], {}))
            row[arm] = f"{p[n]:.6f}"
            w.writerow([r["id"]] + [row.get(c, "") for c in cols])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distil_root", default="/workspace/distil")
    ap.add_argument("--bank_root", default="/workspace/semhand", help="V1's human rows")
    ap.add_argument("--zone_root", action="append", default=None,
                    help="roots whose zone labels join training (not the held-out batch)")
    ap.add_argument("--zone_labels", action="append", default=None)
    ap.add_argument("--test_root", action="append", default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="/workspace/distil/student")
    ap.add_argument("--predict_only", action="store_true",
                    help="load <out>/<arm>_seed*.pt and only predict on --test_root")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    if a.predict_only:
        from src.rig import own_ctx
        tests = {root: read_rows(root, "test") for root in (a.test_root or [])}
        for arm in a.arms.split(","):
            models = []
            for path in sorted(glob.glob(os.path.join(a.out, f"{arm}_seed*.pt"))):
                ck = torch.load(path, map_location=device, weights_only=False)
                m = own_ctx.build(ck["arm"]).to(device)
                m.load_state_dict(ck["state"])
                models.append(m)
            if not models:
                raise SystemExit(f"没有 {arm} 的权重")
            for root, trows in tests.items():
                write_preds(root, arm, trows, predict(models, trows, VIEW[arm], device))
                print(f"  {arm} ({len(models)} seeds) -> {root}: {len(trows)} 只手", flush=True)
        return

    teach = with_labels(read_rows(a.distil_root, "test"), qwen_labels(a.distil_root))
    print(f"teacher rows {len(teach)}（自己 {sum(r['y'] == 1 for r in teach)} / 别人 "
          f"{sum(r['y'] == 0 for r in teach)}），{len({r['tag'] for r in teach})} 段录像", flush=True)
    human = read_rows(a.bank_root, "bank")
    zone = []
    for root, lab in zip(a.zone_root or [], a.zone_labels or []):
        z = with_labels(read_rows(root, "test"), zone_labels(lab))
        print(f"zone rows {root}: {len(z)}（别人 {sum(r['y'] == 0 for r in z)}）", flush=True)
        zone += z
    tests = {root: read_rows(root, "test") for root in (a.test_root or [])}

    extra = {}
    if any(x in a.arms for x in ("S_wide_and", "S_wide_cons")):
        q1 = qwen_labels(a.distil_root)
        g6 = gpt6_labels(a.distil_root)
        base_rows = read_rows(a.distil_root, "test")
        for mode in ("and", "cons"):
            lab = two_teacher(q1, g6, mode)
            extra[f"S_wide_{mode}"] = with_labels(base_rows, lab)
            r = extra[f"S_wide_{mode}"]
            print(f"two-teacher {mode}: {len(r)} 行（自己 {sum(x['y'] == 1 for x in r)} / 别人 "
                  f"{sum(x['y'] == 0 for x in r)}）；GPT-6 覆盖 {len(g6)}", flush=True)

    report = {}
    for arm in a.arms.split(","):
        if arm in extra:
            rows = extra[arm]
        else:
            rows = teach + (human if arm == "S_wide_h" else zone if arm == "S_wide_z" else [])
        view = VIEW[arm]
        models = []
        for seed in range(a.seeds):
            m, best = fit(view, rows, seed, device, epochs=a.epochs)
            torch.save({"state": {k: v.cpu() for k, v in m.state_dict().items()},
                        "arm": "both_geom" if view == "VV" else "both", "view": view,
                        "train_rows": len(rows), "dev": best},
                       os.path.join(a.out, f"{arm}_seed{seed}.pt"))
            models.append(m)
            report.setdefault(arm, []).append(best)
            print(f"  {arm} seed {seed}: dev F1 {best['f1']:.3f} (epoch {best['epoch']}), "
                  f"{len(rows)} 行", flush=True)
        for root, trows in tests.items():
            write_preds(root, arm, trows, predict(models, trows, view, device))
            print(f"  {arm} -> {root}: {len(trows)} 只手", flush=True)
    json.dump(report, open(os.path.join(a.out, "dev.json"), "w"), indent=1, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
