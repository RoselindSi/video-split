"""Train one arm on zone labels: input A or B, trunk from ImageNet or from V1.

LABELS FROM ZONES ONLY. The human gold never supervises a weight. It is read
in phase one to choose how long to train, on out-of-fold predictions -- the
same use V1 made of the same gold, which is what keeps the comparison even.

PHASE ONE CHOOSES THE EPOCH COUNT, OUT OF FOLD. Recordings go into five folds;
each fold's model is scored after every epoch on recordings it never trained
on, and the epoch with the best track-level balanced accuracy against the
gold wins (ties: higher foreign-hand recall, then fewer epochs). Balanced
accuracy rather than foreign recall alone, because calling every hand foreign
scores a perfect foreign recall.

PHASE TWO IS PHASE ONE, STOPPED AT THAT EPOCH. Every recording, once per
seed, and the learning-rate schedule keeps phase one's full horizon rather
than being compressed into fewer epochs -- otherwise `epoch 7 of 12` and
`epoch 7 of 7` are different learning rates and the chosen count would not
mean what it meant when it was chosen. Nothing in phase two looks at a score.

ONE CORRECTION FOR IMBALANCE, NOT TWO STACKED. Frames are drawn so every
track counts equally -- a thirty-second track would otherwise outvote a
two-second one fifteen to one -- and the loss weights the classes by TRACK
counts. Oversampling that already favours the minority class plus
inverse-frequency weights is the double count that once pushed this project's
`other` prior to eleven times its real size.

AUGMENTATION IS APPEARANCE ONLY: brightness and contrast, as V1 does. No flip
and no rotation. On a fixed head-mounted rig orientation is a real egocentric
cue, and V1 measured it as the strongest geometric one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import os
import random

import numpy as np

from src.selfother.crops import read_index
from src.selfother.model import build, sha256, to_tensor


def read_gold(paths):
    """-> {(recording, track id): "owner" | "other"}; other verdicts dropped."""
    gold = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            t = r.get("track_truth") or r.get("human_ownership")
            if t in ("owner", "other"):
                gold[(r["rec"], str(r["tid"]))] = t
    return gold


class Crops:
    """(tensor, label) for one input kind; label -1 when the row has none."""

    def __init__(self, root, split, rows, input_kind, augment):
        self.base = os.path.join(root, split)
        self.rows = rows
        self.col = "a_path" if input_kind == "A" else "b_path"
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        cv2.setNumThreads(1)      # per loader worker; the container caps tasks
        r = self.rows[i]
        path = os.path.join(self.base, r[self.col])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        if self.augment:
            a = 1.0 + random.uniform(-0.30, 0.30)
            b = random.uniform(-28, 28)
            img = np.clip(img.astype(np.float32) * a + b, 0,
                          255).astype(np.uint8)
        y = int(r["label"]) if r.get("label") in ("0", "1") else -1
        return to_tensor(img), y


def folds_by_recording(rows, k):
    """-> {recording: fold}. Whole recordings, deterministic.

    Recordings that contain a foreign hand are dealt out first, so no fold is
    left without the class its recall is measured on."""
    has_other = {}
    for r in rows:
        has_other[r["rec"]] = has_other.get(r["rec"], False) or r["label"] == "0"
    order = (sorted(x for x, h in has_other.items() if h)
             + sorted(x for x, h in has_other.items() if not h))
    return {rec: i % k for i, rec in enumerate(order)}


def predict(net, root, split, rows, input_kind, device, batch, workers):
    """-> P(self) per row, in row order."""
    import torch
    loader = torch.utils.data.DataLoader(
        Crops(root, split, rows, input_kind, augment=False),
        batch_size=batch, shuffle=False, num_workers=workers)
    was_training = net.training
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.softmax(net(x.to(device, non_blocking=True)), 1)[:, 1]
            out.append(p.float().cpu().numpy())
    net.train(was_training)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def fit(root, split, rows, input_kind, init, v1_ckpt, epochs, horizon, seed,
        device, batch, lr, workers, eval_rows=None):
    """Train for `epochs` of a `horizon`-epoch schedule. -> (net, [probs])"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = build(input_kind, init, v1_ckpt).to(device)

    key = lambda r: (r["rec"], r["tid"])
    per_track = collections.Counter(key(r) for r in rows)
    track_label = {key(r): int(r["label"]) for r in rows}
    n_cls = collections.Counter(track_label.values())
    class_w = torch.tensor(
        [len(track_label) / (2.0 * max(n_cls[c], 1)) for c in (0, 1)],
        dtype=torch.float32, device=device)
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        [1.0 / per_track[key(r)] for r in rows], num_samples=len(rows),
        replacement=True, generator=gen)
    loader = torch.utils.data.DataLoader(
        Crops(root, split, rows, input_kind, augment=True),
        batch_size=batch, sampler=sampler, num_workers=workers,
        persistent_workers=workers > 0)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(horizon * len(loader), 1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
    history = []
    net.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(y)
        if eval_rows is not None:
            history.append(predict(net, root, split, eval_rows, input_kind,
                                   device, batch, workers))
        print(f"    epoch {ep + 1}/{epochs}  loss {total / len(rows):.4f}",
              flush=True)
    return net, history


def track_scores(rows, probs, gold):
    """Track-level recalls against gold, mean P(self) >= 0.5 is self."""
    by = collections.defaultdict(list)
    for r, p in zip(rows, probs):
        by[(r["rec"], r["tid"])].append(p)
    own = [k for k in gold if k in by and gold[k] == "owner"]
    oth = [k for k in gold if k in by and gold[k] == "other"]
    own_hit = sum(1 for k in own if np.mean(by[k]) >= 0.5)
    oth_hit = sum(1 for k in oth if np.mean(by[k]) < 0.5)
    own_r = own_hit / len(own) if own else float("nan")
    oth_r = oth_hit / len(oth) if oth else float("nan")
    return {"n": len(own) + len(oth), "own_recall": own_r,
            "other_recall": oth_r, "balanced": (own_r + oth_r) / 2,
            "own": [own_hit, len(own)], "other": [oth_hit, len(oth)]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, help="crops split to train on")
    ap.add_argument("--input", choices=("A", "B"), required=True)
    ap.add_argument("--init", choices=("imagenet", "v1"), required=True)
    ap.add_argument("--v1_ckpt", default="",
                    help="required with --init v1, refused otherwise")
    ap.add_argument("--gold", action="append", required=True,
                    help="gold on the TRAINING recordings, for epoch choice")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    if a.init == "imagenet" and a.v1_ckpt:
        ap.error("--init imagenet 不能带 --v1_ckpt")
    if a.init == "v1" and not a.v1_ckpt:
        ap.error("--init v1 需要 --v1_ckpt")
    import torch
    torch.set_num_threads(8)

    rows = [r for r in read_index(a.root, a.split)
            if r["status"] == "ok" and r["label"] in ("0", "1")]
    gold = read_gold(a.gold)
    fold = folds_by_recording(rows, a.folds)
    tracks = {(r["rec"], r["tid"]): r["label"] for r in rows}
    arm = f"{a.input}_{a.init}"
    print(f"[{arm}] {len(rows)} 个裁剪 / {len(tracks)} 条轨迹 / "
          f"{len(fold)} 段录像；轨迹里自己的手 "
          f"{sum(1 for v in tracks.values() if v == '1')}，别人的手 "
          f"{sum(1 for v in tracks.values() if v == '0')}；gold 覆盖 "
          f"{sum(1 for k in gold if k in tracks)} 条", flush=True)

    common = dict(root=a.root, split=a.split, input_kind=a.input,
                  init=a.init, v1_ckpt=a.v1_ckpt or None, device=a.device,
                  batch=a.batch, lr=a.lr, workers=a.workers)

    # -- phase one: out-of-fold, seed 0, every epoch scored ---------------
    oof = [np.full(len(rows), np.nan, np.float32)
           for _ in range(a.max_epochs)]
    for k in range(a.folds):
        tr = [r for r in rows if fold[r["rec"]] != k]
        te_idx = [i for i, r in enumerate(rows) if fold[r["rec"]] == k]
        print(f"  fold {k + 1}/{a.folds}: 训练 {len(tr)}，留出 "
              f"{len(te_idx)}", flush=True)
        _, hist = fit(rows=tr, epochs=a.max_epochs, horizon=a.max_epochs,
                      seed=0, eval_rows=[rows[i] for i in te_idx], **common)
        for ep, p in enumerate(hist):
            oof[ep][te_idx] = p
    table = [track_scores(rows, oof[ep], gold) for ep in range(a.max_epochs)]
    finite = lambda v: v if v == v else -1.0          # NaN never wins
    best = max(range(a.max_epochs),
               key=lambda e: (finite(table[e]["balanced"]),
                              finite(table[e]["other_recall"]), -e))
    print(f"\n[{arm}] 留出轨迹对 gold（选 epoch 用）:")
    for e, t in enumerate(table):
        mark = "  <- 选中" if e == best else ""
        print(f"  epoch {e + 1:2d}  平衡准确 {t['balanced']:.3f}  "
              f"别人的手 {t['other'][0]}/{t['other'][1]}  "
              f"自己的手 {t['own'][0]}/{t['own'][1]}{mark}")

    # -- phase two: all recordings, the chosen epoch count, every seed ------
    os.makedirs(a.out_dir, exist_ok=True)
    v1_sha = sha256(a.v1_ckpt) if a.v1_ckpt else ""
    for s in range(a.seeds):
        print(f"  seed {s}: 全部录像训练 {best + 1} 个 epoch", flush=True)
        net, _ = fit(rows=rows, epochs=best + 1, horizon=a.max_epochs,
                     seed=s, **common)
        import torch
        path = os.path.join(a.out_dir, f"{arm}_seed{s}.pt")
        torch.save({"state": net.state_dict(), "input": a.input,
                    "init": a.init, "epochs": best + 1,
                    "horizon": a.max_epochs, "seed": s,
                    "v1_ckpt": a.v1_ckpt, "v1_sha256": v1_sha,
                    "labels": "zone", "train_split": a.split,
                    "train_recs": sorted(fold),
                    "created": datetime.datetime.now().isoformat(
                        timespec="seconds")}, path)
        print(f"    -> {path}", flush=True)
    with open(os.path.join(a.out_dir, f"{arm}_selection.json"), "w") as f:
        json.dump({"arm": arm, "chosen_epochs": best + 1, "table": table,
                   "folds": fold}, f, indent=1)


if __name__ == "__main__":
    main()
