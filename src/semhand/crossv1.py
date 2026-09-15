"""V1's architecture and recipe, trained on Qwen's view instead of its own.

Runs in `/workspace/venv_rig`. The other half of the swap (`crosscheck.py`).

THREE ARMS, ONE RECIPE, THE SAME 1,013 LABELLED HANDS:
    VV     V1's view: hand 128 + 2.5x context 128 + 14 geometry features
           (`own_ctx.build("both_geom")`) -- V1 retrained, the control
    VVng   V1's view without the geometry features (`both`) -- Qwen's view
           carries no numbers, so this separates pixels from geometry
    VQ     Qwen's view: the whole frame with the green box, at 448x246, and the
           256 px crop around the box at 224 (`both`: the frame goes through
           the context trunk, the crop through the hand trunk)

Everything else is V1's: two ImageNet ResNet18 trunks, its head, AdamW 3e-4,
12 epochs, batch 32, unweighted cross-entropy, brightness/contrast jitter on
both images, `split_by_recording(0.25)` and dev `other` F1 selection, three
seeds whose P is averaged. Training rows are V1's own (`crossprep --mode bank`);
all views are cut from the same saved JPEG frames.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random

import numpy as np

ARMS = ("VV", "VVng", "VQ")
FRAME_PX = (448, 246)
CROP_PX = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def norm_rgb(arr):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1) / 255.0
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class Views:
    def __init__(self, rows, arm, augment):
        self.rows, self.arm, self.augment = rows, arm, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        import torch
        from PIL import Image
        from src.semhand.qwen import views
        cv2.setNumThreads(1)
        r = self.rows[i]
        if self.arm == "VQ":
            frame, crop = views({"image": r["image"], "box": r["box"]})
            h = np.asarray(crop.resize((CROP_PX, CROP_PX), Image.BICUBIC))
            c = np.asarray(frame.resize(FRAME_PX, Image.BICUBIC))
        else:
            h = cv2.imread(r["hand"])[:, :, ::-1]
            c = cv2.imread(r["ctx"])[:, :, ::-1]
        if self.augment:                                  # as own_ctx.Pairs
            a, b = 1.0 + random.uniform(-0.30, 0.30), random.uniform(-28, 28)
            h = np.clip(h.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            c = np.clip(c.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        g = r["g"] if self.arm == "VV" else np.zeros(14, np.float32)
        return norm_rgb(h), norm_rgb(c), torch.from_numpy(g), int(r.get("y", -1))


def loader(rows, arm, augment, shuffle, batch=32):
    import torch
    return torch.utils.data.DataLoader(Views(rows, arm, augment), batch_size=batch, shuffle=shuffle,
                                       num_workers=6, persistent_workers=False)


def read_rows(root, mode):
    idx = {}
    for p in glob.glob(os.path.join(root, "cross", f"index_{mode}_*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["status"] == "ok":
                idx[r["id"]] = r
    frames = {}
    if mode == "bank":
        for p in glob.glob(os.path.join(root, "pkg", "index_*.csv")):
            frames.update({r["stem"]: r for r in csv.DictReader(open(p, encoding="utf-8"))})
    else:
        for p in glob.glob(os.path.join(root, "fresh", "*", "index.csv")):
            frames.update({f"{r['rec']}|{r['frame']}|{r['tid']}": r
                           for r in csv.DictReader(open(p, encoding="utf-8"))})
    rows = []
    for k, r in sorted(idx.items()):
        f = frames[k]
        row = {"id": k, "hand": r["hand"], "ctx": r["ctx"], "image": f["image"],
               "g": np.asarray(json.loads(r["geom"]), np.float32),
               "box": [float(f[c]) for c in ("x0", "y0", "x1", "y1")]}
        if mode == "bank":
            row.update(y=int(r["y"]), tag=r["tag"])
        rows.append(row)
    return rows


def fit(arm, rows, seed, device, epochs=12):
    import torch
    from src.rig import own_ctx
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tr, dv = own_ctx.split_by_recording(rows, 0.25, seed)
    model = own_ctx.build("both_geom" if arm == "VV" else "both").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    best, state = None, None
    for e in range(epochs):
        model.train()
        for h, c, g, y in loader(tr, arm, True, True):
            loss = ce(model(h.to(device), c.to(device), g.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        p = predict([model], dv, arm, device)
        sc = own_ctx.scores((p >= 0.5).astype(int), np.array([r["y"] for r in dv]))
        print(f"    {arm} seed {seed} epoch {e + 1:>2}  dev F1 {sc['f1']:.3f}", flush=True)
        if best is None or (sc["f1"] == sc["f1"] and sc["f1"] > best["f1"]):
            best = dict(sc, epoch=e + 1)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def predict(models, rows, arm, device):
    import torch
    out = []
    for m in models:
        m.eval()
    with torch.no_grad():
        for h, c, g, _ in loader(rows, arm, False, False, batch=128):
            h, c, g = h.to(device), c.to(device), g.to(device)
            out.append(np.mean([torch.softmax(m(h, c, g), 1)[:, 1].cpu().numpy() for m in models], 0))
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank_root", default="/workspace/semhand")
    ap.add_argument("--test_root", action="append", default=None,
                    help="default: /workspace/semhand_e2e and /workspace/semhand")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    a.test_root = a.test_root or ["/workspace/semhand_e2e", "/workspace/semhand"]
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    bank = read_rows(a.bank_root, "bank")
    print(f"bank {len(bank)} 行（别人 {sum(r['y'] == 0 for r in bank)}）", flush=True)
    tests = {root: read_rows(root, "test") for root in a.test_root}
    preds = {root: {} for root in tests}
    dev = {}
    for arm in a.arms.split(","):
        models = []
        for seed in range(a.seeds):
            m, best = fit(arm, bank, seed, device)
            models.append(m)
            dev.setdefault(arm, []).append(best)
            print(f"  {arm} seed {seed}: dev F1 {best['f1']:.3f} (epoch {best['epoch']})", flush=True)
        for root, rows in tests.items():
            preds[root][arm] = predict(models, rows, arm, device)
            print(f"  {arm} -> {root}: {len(rows)} 只手", flush=True)
    for root, rows in tests.items():
        d = os.path.join(root, "crossv1")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "pred.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id"] + list(preds[root]))
            for n, r in enumerate(rows):
                w.writerow([r["id"]] + [f"{preds[root][arm][n]:.6f}" for arm in preds[root]])
        json.dump(dev, open(os.path.join(d, "dev.json"), "w"), indent=1, default=float)
        print(f"-> {d}/pred.csv")


if __name__ == "__main__":
    main()
