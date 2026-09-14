"""Train V1p13 and V1s13: V1's recipe, V1's rows, two different images.

ONE LOOP, TWO DATASETS. The optimiser, learning rate, batch, epochs,
brightness/contrast augmentation, recording split and dev other-F1 selection
are V1's (`own_ctx.run_epoch`, `own_ctx.split_by_recording`, `own_ctx.scores`
are called, not copied), and so is the network (`own_ctx.build`). The loop
itself is `own_ctx.train_arm` with one change: it takes its datasets as
arguments, because `train_arm` builds a panorama dataset internally and
forces every image square, which would squash a 128x256 pair.

THE SAME ROWS FOR BOTH. A row enters only if V1's loader kept it AND its
stereo pair was cut, so nothing the stereo arm lost is quietly kept by the
panorama arm.

EVERY SEED IS SAVED. V1 shipped seed 0 of its best arm; here all three are
kept, because stability across seeds is part of what is being asked, and seed
0 is the one used to initialise the zone arms, as V1's was.
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np

GEOM13 = ("box_cx", "box_cy", "box_w", "box_h", "dir_x", "dir_y", "exit_x",
          "exit_y", "edge_bottom", "edge_left", "edge_right", "edge_top",
          "conf")


def geom13(row14):
    """V1's 14-vector (own_ctx.GEOM order) -> the 13 without hand_span."""
    from src.rig import own_ctx
    at = {n: i for i, n in enumerate(own_ctx.GEOM)}
    return np.array([row14[at[n]] for n in GEOM13], np.float32)


class StereoPairs:
    """(hand pair, context pair, geometry, label), augmented as V1 augments."""

    def __init__(self, root, rows, augment):
        self.root, self.rows, self.augment = root, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        import torch
        from src.selfother.model import to_tensor
        cv2.setNumThreads(1)
        r = self.rows[i]
        hand = cv2.imread(os.path.join(self.root, r["b_path"]))
        ctx = cv2.imread(os.path.join(self.root, r["a_path"]))
        if hand is None or ctx is None:
            raise FileNotFoundError(r["b_path"])
        if self.augment:                       # own_ctx.Pairs, same ranges
            a = 1.0 + random.uniform(-0.30, 0.30)
            b = random.uniform(-28, 28)
            hand = np.clip(hand.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            ctx = np.clip(ctx.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return (to_tensor(hand), to_tensor(ctx),
                torch.from_numpy(r["g"]), int(r.get("y", 0)))


def common_rows(pkgs, root):
    """V1's rows that also have a stereo pair, with 13-d geometry attached."""
    from src.rig import own_ctx
    from src.v1stereo.recut import read_rows
    rows = own_ctx.load(pkgs, verbose=False)
    stereo = {}
    for pkg in pkgs:
        for r in read_rows(root, os.path.basename(pkg.rstrip("/"))):
            if r["status"] == "ok":
                stereo[r["stem"]] = r
    out = []
    for r in rows:
        s = stereo.get(r["stem"])
        if s is None:
            continue
        out.append(dict(r, g=geom13(r["g"]), a_path=s["a_path"],
                        b_path=s["b_path"]))
    return out, len(rows)


def train_variant(variant, tr, dv, root, seed, epochs=12, lr=3e-4):
    """own_ctx.train_arm, with the datasets passed in."""
    import torch
    from src.rig import own_ctx
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = own_ctx.build("both_geom", n_geom=len(GEOM13)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if variant == "pano":
        dtr = own_ctx.Pairs(tr, augment=True)
        ddv = own_ctx.Pairs(dv, augment=False)
    else:
        dtr = StereoPairs(root, tr, augment=True)
        ddv = StereoPairs(root, dv, augment=False)
    best, best_state = None, None
    for e in range(epochs):
        _, _, tl = own_ctx.run_epoch(model, dtr, device, opt)
        p, y, _ = own_ctx.run_epoch(model, ddv, device)
        s = own_ctx.scores(p, y)
        print(f"    epoch {e + 1:>2}  loss {tl:.3f}  dev other-f1 {s['f1']:.3f}  "
              f"prec {s['prec']:.3f}  rec {s['rec']:.3f}", flush=True)
        if best is None or (s["f1"] == s["f1"] and s["f1"] > best["f1"]):
            best = s
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--root", required=True, help="recut.py output root")
    ap.add_argument("--variant", choices=("pano", "stereo"), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    import torch
    from src.rig import own_ctx
    torch.set_num_threads(8)

    rows, n_v1 = common_rows(a.pkg, a.root)
    tr, dv = own_ctx.split_by_recording(rows, 0.25)
    name = {"pano": "V1p13", "stereo": "V1s13"}[a.variant]
    print(f"[{name}] V1 的行 {n_v1}，有双目对的 {len(rows)}；train {len(tr)} "
          f"({sum(1 for r in tr if r['y'] == 0)} other, "
          f"{len({r['tag'] for r in tr})} 段)  dev {len(dv)} "
          f"({sum(1 for r in dv if r['y'] == 0)} other, "
          f"{len({r['tag'] for r in dv})} 段)", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for s in range(a.seeds):
        print(f"  seed {s}", flush=True)
        model, best = train_variant(a.variant, tr, dv, a.root, s)
        path = os.path.join(a.out_dir, f"{name}_seed{s}.pt")
        torch.save({"arm": "both_geom", "n_geom": len(GEOM13),
                    "geom": GEOM13, "variant": a.variant, "seed": s,
                    "dev": best, "rows": len(rows),
                    "state": model.state_dict()}, path)
        print(f"  seed {s} dev other-f1 {best['f1']:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
