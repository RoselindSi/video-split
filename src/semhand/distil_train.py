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

TWO TEACHERS (added 2026-09-17; revised the same day before any of it ran,
because no GPT-6 API access exists -- only GPT-6's finished shared labels).
GPT-6 labelled monocular frames from recordings our pool never touches, so the
agreement filter runs on GPT-6's hands: Q1 judges them (`g6_prep.py`), and the
blind audit showed Q1's remaining privacy-direction errors are mostly boxes
GPT-6 gets right. Two arms, same view, recipe and seeds as S_wide, each
trained on the pool's Q1 rows PLUS rows from GPT-6's frames:
    S_wide_g6     GPT-6's label on every sampled hand
    S_wide_g6and  only the hands where Q1 and GPT-6 agree
Monocular frames are letterboxed into the panorama-shaped input, not stretched.
A third arm, added 2026-09-17 before any GPT-6 student was scored, because the
incumbent S_wide_h carries V1's 1,013 human hands and the two arms above do not:
    S_wide_hg6and  the pool's Q1 rows + V1's 1,013 human hands + the agreeing
                   GPT-6 hands
It is read by the same rule as the other two (as amended in `g6_final.py`).
They are compared with S_wide on the held-out zone batches (4-6, and 7-9 once
labelled) at the post-processing the ablation chose (geom_w 0.25, no cap):
an arm replaces S_wide only if its foreign frames called self are fewer on
7-9 AND it still passes the four criteria against V1 deployed there.

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

from src.semhand import crossv1
from src.semhand.crossv1 import Views, fit, loader, predict, read_rows

ARMS = ("S_wide", "S_v1", "S_wide_h", "S_wide_z", "S_wide_g6", "S_wide_g6and", "S_wide_hg6and",
        "S_wide_g6_v2")
VIEW = {"S_wide": "VQ", "S_v1": "VV", "S_wide_h": "VQ", "S_wide_z": "VQ",
        "S_wide_g6": "VQ", "S_wide_g6and": "VQ", "S_wide_hg6and": "VQ", "S_wide_g6_v2": "VQ"}


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


def g6_rows(root, mode):
    """Rows from GPT-6's monocular frames (g6_prep), labelled by GPT-6 or by agreement."""
    q1 = qwen_labels(root)
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "fresh", "*", "index.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            ident = f"{r['rec']}|{r['frame']}|{r['tid']}"
            g = 1 if r["gpt6"] == "owner" else 0
            if mode == "and" and q1.get(ident) != g:
                continue
            rows.append({"id": ident, "image": r["image"], "mono": True, "y": g, "tag": r["recording"],
                         "box": [float(r[c]) for c in ("x0", "y0", "x1", "y1")],
                         "g": np.zeros(14, np.float32)})
    return rows


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
    ap.add_argument("--distil_root", action="append", default=None,
                    help="repeatable; every root's Q1-labelled rows are pooled")
    ap.add_argument("--bank_root", default="/workspace/semhand", help="V1's human rows")
    ap.add_argument("--zone_root", action="append", default=None,
                    help="roots whose zone labels join training (not the held-out batch)")
    ap.add_argument("--zone_labels", action="append", default=None)
    ap.add_argument("--test_root", action="append", default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_list", default=None,
                    help="comma list: train only these seeds (one process per seed, run in parallel)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="/workspace/distil/student")
    ap.add_argument("--g6_root", default="/workspace/g6teach")
    # THE THIRD CLASS IS A ROOT, NOT A FLAG. Its rows are y=2 and they are
    # read by the same `read_rows(root, "bank")` as everything else, so the
    # only thing that changes is the number of answers the head has. Passing
    # it without --n_out 3 is refused below rather than silently training a
    # two-way head on three-way labels, which would put every machine part
    # into whichever class the loss finds cheaper.
    ap.add_argument("--nothand_root", action="append", default=None,
                    help="a bank of y=2 rows from `semhand.neg_bank`")
    ap.add_argument("--n_out", type=int, default=2,
                    help="3 adds `not a hand` as a third answer")
    ap.add_argument("--predict_only", action="store_true",
                    help="load <out>/<arm>_seed*.pt and only predict on --test_root")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    a.distil_root = a.distil_root or ["/workspace/distil"]
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

    teach = [r for root in a.distil_root
             for r in with_labels(read_rows(root, "test"), qwen_labels(root))]
    print(f"teacher rows {len(teach)}（自己 {sum(r['y'] == 1 for r in teach)} / 别人 "
          f"{sum(r['y'] == 0 for r in teach)}），{len({r['tag'] for r in teach})} 段录像", flush=True)
    human = read_rows(a.bank_root, "bank")
    zone = []
    for root, lab in zip(a.zone_root or [], a.zone_labels or []):
        z = with_labels(read_rows(root, "test"), zone_labels(lab))
        print(f"zone rows {root}: {len(z)}（别人 {sum(r['y'] == 0 for r in z)}）", flush=True)
        zone += z
    tests = {root: read_rows(root, "test") for root in (a.test_root or [])}

    nothand = []
    for root in (a.nothand_root or []):
        rows_nh = read_rows(root, "bank")
        # The control rows the harvest wrote alongside the negatives are hands
        # whose OWNERSHIP nobody labelled. They exist to be predicted on, not
        # trained on: given a y of 1 they would teach "anything from the new
        # pass that is a hand is the wearer's", which is the shortcut this
        # pool was built to test for.
        rows_nh = [r for r in rows_nh if int(r.get("y", -1)) == 2]
        print(f"nothand rows {root}: {len(rows_nh)}", flush=True)
        nothand += rows_nh
    if nothand and a.n_out < 3:
        raise SystemExit("--nothand_root needs --n_out 3: a two-way head "
                         "trained on y=2 rows puts every machine part into "
                         "whichever class the loss finds cheaper")
    if a.n_out >= 3 and not nothand:
        raise SystemExit("--n_out 3 with no --nothand_root: the third class "
                         "would have no examples and never fire")

    extra = {}
    for arm, mode in (("S_wide_g6", "all"), ("S_wide_g6and", "and"), ("S_wide_hg6and", "and"),
                      ("S_wide_g6_v2", "all")):
        if arm in a.arms.split(","):
            g = g6_rows(a.g6_root, mode)
            extra[arm] = teach + g + (human if arm == "S_wide_hg6and" else [])
            print(f"{arm}: 蒸馏池 {len(teach)} + GPT-6 图 {len(g)} 行（GPT-6 图里 自己 "
                  f"{sum(x['y'] == 1 for x in g)} / 别人 {sum(x['y'] == 0 for x in g)}）", flush=True)

    report = {}
    for arm in a.arms.split(","):
        if arm in extra:
            rows = extra[arm]
        else:
            rows = teach + (human if arm == "S_wide_h" else zone if arm == "S_wide_z" else [])
        rows = rows + nothand
        if nothand:
            n2 = sum(1 for r in rows if int(r.get("y", -1)) == 2)
            print(f"{arm}: 训练池 {len(rows)} 行 —— 别人 "
                  f"{sum(1 for r in rows if int(r.get('y', -1)) == 0)} / 自己 "
                  f"{sum(1 for r in rows if int(r.get('y', -1)) == 1)} / 不是手 "
                  f"{n2}（{n2 / max(1, len(rows)):.1%}，现实约 7%，"
                  f"不做逆频率加权）", flush=True)
        view = VIEW[arm]
        models = []
        seeds = [int(x) for x in a.seed_list.split(",")] if a.seed_list else range(a.seeds)
        for seed in seeds:
            m, best = fit(view, rows, seed, device, epochs=a.epochs,
                          n_out=a.n_out)
            torch.save({"state": {k: v.cpu() for k, v in m.state_dict().items()},
                        "arm": "both_geom" if view == "VV" else "both",
                        "n_out": a.n_out, "view": view,
                        "train_rows": len(rows), "dev": best},
                       os.path.join(a.out, f"{arm}_seed{seed}.pt"))
            models.append(m)
            report.setdefault(arm, []).append(best)
            print(f"  {arm} seed {seed}: dev F1 {best['f1']:.3f} (epoch {best['epoch']}), "
                  f"{len(rows)} 行", flush=True)
        for root, trows in tests.items():
            if a.n_out < 3:
                write_preds(root, arm, trows,
                            predict(models, trows, view, device))
                continue
            # A DIFFERENT COLUMN AND A DIFFERENT NUMBER, both on purpose.
            #
            # The column: writing into `S_wide_g6` would overwrite the
            # incumbent's predictions with the challenger's, in the file the
            # comparison reads. The evaluation would then compare a model
            # against itself and report a tie.
            #
            # The number: `predict` returns P(class 1) out of the softmax, and
            # under three classes that is P(owner) with the rest split between
            # `other` and `not a hand`. Against a two-class P(owner) it is
            # deflated by whatever mass the third class took, so every
            # threshold moves for a reason that has nothing to do with
            # ownership. The ownership column is renormalised over the two
            # classes that ARE ownership, and the third is its own column so
            # it can be read rather than inferred from what is missing.
            pr = crossv1.predict_multi(models, trows, view, device)
            own = pr[:, 1] / np.maximum(1e-9, pr[:, 0] + pr[:, 1])
            write_preds(root, arm + "_3c", trows, own)
            write_preds(root, arm + "_nothand", trows, pr[:, 2])
            print(f"  {arm} -> {root}: {len(trows)} 只手", flush=True)
    tag = f"_{a.arms}_{a.seed_list}".replace(",", "-") if a.seed_list else ""
    json.dump(report, open(os.path.join(a.out, f"dev{tag}.json"), "w"), indent=1, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
