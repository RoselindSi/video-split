"""Score Qwen3.8 against V1's package labels, beside V1 and the trained arms.

EVERY MODEL ON THE SAME HANDS. A labelled hand counts only if every model in
the table produced a verdict for it: Qwen answered its frame with valid JSON,
V1's loader kept the row, and the stereo arms have a pair for it. Nothing is
compared across different subsets.

TRAINING PACKAGES ARE REPORTED AS WHAT THEY ARE. `otherpkg_1` and
`trainpkg_T1` are V1's training data, and V1p13/V1s13 were fitted on them
too, so their scores there are in-sample and are printed that way (V1p13 and
V1s13 are not scored there at all). `testpkg_2` is held out from every trained
model here, and is the fair column. The zone arms never saw any package.

OWN-HAND RECALL FIRST, as the priority says: how often the wearer's own hand
is recognised as the wearer's. Foreign-hand recall and precision sit beside
it, and a per-frame check asks whether the whole frame came out right.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np

TRAIN_PKGS = {"otherpkg_1", "trainpkg_T1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--recut_root", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v1p", action="append", default=[])
    ap.add_argument("--v1s", action="append", default=[])
    ap.add_argument("--zone_ckpt", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from src.rig import own_ctx
    from src.selfother.model import load_trained
    from src.selfother.train import predict
    from src.v1stereo.evaluate import load_v1, probs
    from src.v1stereo.train import StereoPairs, common_rows
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Qwen: stem -> owner? ------------------------------------------
    answers = {}
    for line in open(a.answers):
        if line.strip():
            d = json.loads(line)
            answers[d["id"]] = d
    qwen, frame_of, meta = {}, {}, collections.Counter()
    for m in a.manifest:
        for line in open(m):
            it = json.loads(line)
            ans = answers.get(it["id"])
            if ans is None:
                meta["unanswered_frames"] += 1
                continue
            meta["frames"] += 1
            if not ans["ok"]:
                meta["invalid_json_frames"] += 1
                continue
            wearer = set(ans["parsed"]["wearer"])
            if len(wearer) > 2:
                meta["frames_more_than_two_wearer"] += 1
            for b in it["boxes"]:
                if b.get("label") in ("owner", "other"):
                    qwen[b["stem"]] = b["n"] in wearer
                    frame_of[b["stem"]] = it["id"]

    # ---- V1 and the trained arms: stem -> P(owner) ---------------------
    preds = {"Qwen3.8-27B": {s: float(v) for s, v in qwen.items()}}
    labels, pkg_of = {}, {}
    for pkg in a.pkg:
        name = os.path.basename(pkg.rstrip("/"))
        rows = own_ctx.load([pkg], verbose=False)
        for r in rows:
            labels[r["stem"]] = r["y"]
            pkg_of[r["stem"]] = name
        m, _ = load_v1(a.v1, device)
        p = probs(m, own_ctx.Pairs(rows, augment=False), device)
        preds.setdefault("V1 shipped", {}).update(
            {r["stem"]: float(x) for r, x in zip(rows, p)})
        srows, _ = common_rows([pkg], a.recut_root)
        if name not in TRAIN_PKGS:
            for tag, paths, kind in (("V1p13", a.v1p, "pano"), ("V1s13", a.v1s, "stereo")):
                if not paths:
                    continue
                ps = []
                for path in paths:
                    mm, _ = load_v1(path, device)
                    ds = (own_ctx.Pairs(srows, augment=False) if kind == "pano"
                          else StereoPairs(a.recut_root, srows, augment=False))
                    ps.append(probs(mm, ds, device))
                preds.setdefault(tag, {}).update(
                    {r["stem"]: float(x) for r, x in zip(srows, np.mean(ps, 0))})
        by_arm = collections.defaultdict(list)
        for path in a.zone_ckpt:
            net, meta_ck = load_trained(path, device)
            init = "v1s" if "V1s13" in (meta_ck.get("v1_ckpt") or "") else meta_ck["init"]
            arm = f"{meta_ck['input']}/{init}"
            by_arm[arm].append(predict(net, a.recut_root, "", srows, meta_ck["input"],
                                       device, 256, 4))
        for arm, ps in by_arm.items():
            preds.setdefault(arm, {}).update(
                {r["stem"]: float(x) for r, x in zip(srows, np.mean(ps, 0))})
        print(f"  {name}: V1 行 {len(rows)}，有双目对 {len(srows)}", flush=True)

    # ---- score on the common set, per package group --------------------
    report = {"qwen_meta": dict(meta), "groups": {}}
    groups = {"训练包 (V1 训练集内)": TRAIN_PKGS,
              "testpkg_2 (留出)": {"testpkg_2"}}
    print(f"\nQwen: {dict(meta)}")
    for gname, pkgs in groups.items():
        models = [m for m in preds
                  if not (m in ("V1p13", "V1s13") and pkgs & TRAIN_PKGS)]
        stems = [s for s in labels if pkg_of[s] in pkgs
                 and all(s in preds[m] for m in models)]
        if not stems:
            continue
        y = np.array([labels[s] for s in stems])
        print(f"\n=== {gname}: {len(stems)} 只手（自己 {int((y == 1).sum())} / 别人 "
              f"{int((y == 0).sum())}），{len({frame_of.get(s) for s in stems})} 帧 ===")
        report["groups"][gname] = {}
        for mname in models:
            p = np.array([preds[mname][s] for s in stems]) >= 0.5
            own, oth = y == 1, y == 0
            own_rec = float(p[own].mean()) if own.any() else float("nan")
            oth_rec = float((~p[oth]).mean()) if oth.any() else float("nan")
            tp = int((~p & oth).sum()); fp = int((~p & own).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            f1 = 2 * prec * oth_rec / (prec + oth_rec) if prec == prec and prec + oth_rec else float("nan")
            fr = collections.defaultdict(list)
            for s, pi, yi in zip(stems, p, y):
                fr[frame_of.get(s)].append(bool(pi) == bool(yi))
            frame_ok = float(np.mean([all(v) for v in fr.values()]))
            report["groups"][gname][mname] = {"own_recall": own_rec, "other_recall": oth_rec,
                                              "other_prec": prec, "other_f1": f1,
                                              "frame_all_right": frame_ok, "n": len(stems)}
            print(f"  {mname:<14} 自己的手召回 {own_rec:6.1%}  别人的手 召回 {oth_rec:6.1%} "
                  f"精度 {prec:6.1%} F1 {f1:.3f}  整帧全对 {frame_ok:6.1%}")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
