"""The earlier table's arms, rescored on the sampled hands this experiment uses.

Runs in `/workspace/venv_rig`. Reference rows only: none of them enters a
verdict. Each is predicted exactly as `v1stereo.evaluate` predicted it (same
checkpoints, same stereo pairs or panorama crops, same seed averaging), then
restricted to the hands in `fresh/*/index.csv`, so `evaluate.py` can put them
in the same table with the same raw and held forms.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/workspace/semhand")
    ap.add_argument("--dump", default="/workspace/own_dump_fresh.csv")
    ap.add_argument("--stereo_root", default="/workspace/selfother")
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--pano_root", default="/workspace/v1stereo/pano_fresh")
    a = ap.parse_args()
    import torch
    from src.selfother.crops import read_index
    from src.selfother.model import load_trained
    from src.selfother.train import predict
    from src.v1stereo.evaluate import DetPano, g13_from_dump, load_v1, probs
    from src.v1stereo.train import StereoPairs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    want = set()
    for p in glob.glob(os.path.join(a.root, "fresh", "*", "index.csv")):
        want |= {(r["rec"], r["frame"], r["tid"])
                 for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"}
    dump = {(r["rec"], r["frame"], r["tid"]): r
            for r in csv.DictReader(open(a.dump, encoding="utf-8"))}
    pano = {}
    for p in glob.glob(os.path.join(a.pano_root, "*", "index.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["status"] == "ok":
                pano[(r["rec"], r["frame"], r["tid"])] = r
    rows = []
    for r in read_index(a.stereo_root, a.split):
        k = (r["rec"], r["frame"], r["tid"])
        if r["status"] == "ok" and k in want:
            rows.append(dict(r, g=g13_from_dump(dump[k], pano[k]["pano_w"], pano[k]["pano_h"]),
                             hand_path=pano[k]["hand_path"], ctx_path=pano[k]["ctx_path"]))
    print(f"参照臂：{len(rows)}/{len(want)} 只手", flush=True)

    cols = {}
    for name, pat, kind in (("V1p13", "/workspace/v1stereo/ckpt/V1p13_seed*.pt", "pano"),
                            ("V1s13", "/workspace/v1stereo/ckpt/V1s13_seed*.pt", "stereo")):
        ps = []
        for path in sorted(glob.glob(pat)):
            m, _ = load_v1(path, device)
            ds = (DetPano(a.pano_root, rows) if kind == "pano"
                  else StereoPairs(os.path.join(a.stereo_root, a.split), rows, augment=False))
            ps.append(probs(m, ds, device))
        cols[name] = np.mean(ps, 0)
    zone = {}
    for path in sorted(glob.glob("/workspace/selfother/ckpt/A_imagenet_seed*.pt")
                       + glob.glob("/workspace/selfother/ckpt/B_imagenet_seed*.pt")
                       + glob.glob("/workspace/v1stereo/zone_ckpt/*_seed*.pt")):
        net, meta = load_trained(path, device)
        init = "v1s" if "V1s13" in (meta.get("v1_ckpt") or "") else meta["init"]
        zone.setdefault(f"{meta['input']}/{init}", []).append(
            predict(net, a.stereo_root, a.split, rows, meta["input"], device, 256, 4))
    for name, ps in sorted(zone.items()):
        cols[name] = np.mean(ps, 0)
    os.makedirs(os.path.join(a.root, "refs"), exist_ok=True)
    with open(os.path.join(a.root, "refs", "pred.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + list(cols))
        for n, r in enumerate(rows):
            w.writerow([f"{r['rec']}|{r['frame']}|{r['tid']}"] + [f"{cols[c][n]:.6f}" for c in cols])
    print(f"-> {a.root}/refs/pred.csv  {list(cols)}")


if __name__ == "__main__":
    main()
