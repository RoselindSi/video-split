"""Train "is this a hand" with the ownership student's architecture and inputs.

WHY NOT EXTEND THE OWNERSHIP MODEL. `distil_train --n_out 3` exists and was
built for exactly this, but its positive pool is 1600x900 panorama renders
while any negative harvested from a run is a 1920x1520 raw cam3 frame. A model
given those two pools learns the rendering, not the content -- that is what
invalidated the three-class attempt once already, and it would have looked
healthy on a validation set drawn the same wrong way. Here both classes come
out of `handness_views`, which renders through `student.views`, the same
function inference calls.

WHY NOT REUSE THE EXISTING HAND-NESS STUDENT. It reads two 224x224 thumbnails
and no surrounding context, and on 100 human-confirmed non-hands it puts 31
above the deployment threshold where the teacher puts 1. Adding negatives did
not move it: 895 -> 1388 gained 0.036 and 1388 -> 2373 gained 0.000. The
diagnosis was input poverty, so this takes the ownership student's inputs --
the whole frame with the box drawn, plus the crop -- which that student
distils successfully for a different question off the same pictures.

THE TEST IS NOT ACCURACY. 93.8% of scored boxes are hands, so a model that
answers "hand" unconditionally scores 93.8% while hiding nothing. What has to
be matched is the teacher's false positive rate on things that are not hands.
Reported per box-size band as well as pooled, because half the non-hands sit
below 150px and that is precisely the band where the previous student's
discrimination collapsed -- 100% precision at 246px, 5.5% at 23-28px.

EPOCH SELECTION IS BY MEDIAN, NOT BY BEST. Reading the best epoch off the
same split that chose it reported v6 at 0.899 when the median-epoch answer was
0.828, which reversed a comparison. The checkpoint saved is the median epoch
of the second half of training, and the number reported with it is that
epoch's, not the run's minimum.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import statistics

BANDS = ((0, 60), (60, 100), (100, 150), (150, 10000))


def read_index(patterns):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            root = os.path.dirname(path)
            for row in csv.DictReader(open(path, encoding="utf-8-sig")):
                stem = row["stem"]
                hand = os.path.join(root, stem + "_h.jpg")
                ctx = os.path.join(root, stem + "_c.jpg")
                if os.path.exists(hand) and os.path.exists(ctx):
                    rows.append((hand, ctx, int(row["y"]), row["rec"],
                                 max(int(row["w_px"]), int(row["h_px"]))))
    return rows


def band_of(px):
    for lo, hi in BANDS:
        if lo <= px < hi:
            return "%d-%d" % (lo, hi)
    return "?"


def report(ps, ys, pxs, thr=0.5):
    """-> 主判据 + 按框大小分带，因为非手一半在 150px 以下。"""
    neg = [(p, px) for p, y, px in zip(ps, ys, pxs) if y == 0]
    pos = [(p, px) for p, y, px in zip(ps, ys, pxs) if y == 1]
    out = {
        "nonhand_false_positive": round(
            sum(1 for p, _ in neg if p >= thr) / max(1, len(neg)), 4),
        "hand_recall": round(
            sum(1 for p, _ in pos if p >= thr) / max(1, len(pos)), 4),
        "n_neg": len(neg), "n_pos": len(pos), "bands": {},
    }
    for lo, hi in BANDS:
        key = "%d-%d" % (lo, hi)
        nb = [p for p, px in neg if lo <= px < hi]
        if nb:
            out["bands"][key] = {
                "n": len(nb),
                "fp": round(sum(1 for p in nb if p >= thr) / len(nb), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2,
                    help="图像训练卡在读盘不是算力，一次两三个就够")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from src.rig import own_ctx
    from src.semhand.crossv1 import norm_rgb

    rows = read_index(a.index)
    counts = collections.Counter(y for _h, _c, y, _r, _p in rows)
    print("共 %d 框：非手 %d / 手 %d（非手 %.1f%%）"
          % (len(rows), counts[0], counts[1],
             100 * counts[0] / max(1, len(rows))), flush=True)
    if counts[0] < 200:
        raise SystemExit("负例 %d 个，太少，不值得训" % counts[0])

    # 按录像留出，不是按框。同一条录像相邻帧的框几乎是同一张图，按框切
    # 会把同一只手分到两边，留出集就只是在量训练误差。
    recs = sorted({r for _h, _c, _y, r, _p in rows})
    random.Random(0).shuffle(recs)
    held = set(recs[:max(1, int(len(recs) * a.holdout))])
    train_rows = [r for r in rows if r[3] not in held]
    dev_rows = [r for r in rows if r[3] in held]
    print("录像 %d：训练 %d 条 / 留出 %d 条" % (len(recs), len(recs) - len(held),
                                           len(held)), flush=True)
    print("  训练 %d 框（非手 %d）/ 留出 %d 框（非手 %d）"
          % (len(train_rows), sum(1 for r in train_rows if r[2] == 0),
             len(dev_rows), sum(1 for r in dev_rows if r[2] == 0)), flush=True)
    if not dev_rows or not any(r[2] == 0 for r in dev_rows):
        raise SystemExit("留出集里没有负例，换 holdout")

    class Boxes(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            hand, ctx, y, _rec, px = self.items[i]
            h = norm_rgb(np.asarray(Image.open(hand).convert("RGB")))
            c = norm_rgb(np.asarray(Image.open(ctx).convert("RGB")))
            return h, c, y, px

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train = len(train_rows)
    n_neg = max(1, sum(1 for r in train_rows if r[2] == 0))
    # 逆频率权重，不叠富集采样。两者一起用会重复计数 —— 上一次把 other
    # 的先验推到 11 倍就是这么来的。
    weight = torch.tensor([n_train / (2.0 * n_neg),
                           n_train / (2.0 * max(1, n_train - n_neg))],
                          dtype=torch.float32, device=device)
    print("类别权重 非手 %.2f / 手 %.2f" % tuple(weight.tolist()), flush=True)

    summary = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        model = own_ctx.build("both").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss(weight=weight)
        tl = DataLoader(Boxes(train_rows), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, pin_memory=True, drop_last=True)
        dl = DataLoader(Boxes(dev_rows), batch_size=a.batch, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
        history = []
        for epoch in range(a.epochs):
            model.train()
            for h, c, y, _px in tl:
                h, c, y = h.to(device), c.to(device), y.to(device)
                g = torch.zeros(len(h), 14, device=device)
                opt.zero_grad()
                lossf(model(h, c, g), y).backward()
                opt.step()
            model.eval()
            ps, ys, pxs = [], [], []
            with torch.no_grad():
                for h, c, y, px in dl:
                    h, c = h.to(device), c.to(device)
                    g = torch.zeros(len(h), 14, device=device)
                    p = torch.softmax(model(h, c, g), 1)[:, 1]
                    ps += p.cpu().tolist()
                    ys += y.tolist()
                    pxs += px.tolist()
            line = report(ps, ys, pxs)
            line.update(seed=seed, epoch=epoch)
            print(json.dumps(line, ensure_ascii=False), flush=True)
            # 只留后半段的权重，而且必须 clone：state_dict() 给的是活张量的
            # 引用，不克隆的话每一条历史都指向最后一轮，中位选择就是摆设。
            keep = None
            if epoch >= a.epochs // 2:
                keep = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
            history.append((line, keep))

        # 后半段的中位 epoch。看「最好 epoch」是在选它的那一份数据上读分，
        # 那个口径曾把 v6 报成 0.899 而中位口径是 0.828，结论方向都反了。
        half = [h for h in history if h[1] is not None]
        fps = [h[0]["nonhand_false_positive"] for h in half]
        median = statistics.median(fps)
        pick = min(half, key=lambda h: abs(h[0]["nonhand_false_positive"] - median))
        path = os.path.join(a.out, "handness_bin_seed%d.pt" % seed)
        torch.save({"state": pick[1], "arm": "both", "n_out": 2,
                    "seed": seed, "dev": pick[0],
                    "trained_on": "student.views on raw frames",
                    "holdout_recordings": sorted(held)}, path)
        print("seed %d 中位 epoch %d：非手误判 %.4f / 手召回 %.4f -> %s"
              % (seed, pick[0]["epoch"], pick[0]["nonhand_false_positive"],
                 pick[0]["hand_recall"], path), flush=True)
        summary.append(pick[0])

    if summary:
        print("\n=== %d 个种子，中位 epoch 口径 ===" % len(summary))
        print("  非手误判  中位 %.4f  （老师 0.01，现有 v7 学生 0.31）"
              % statistics.median(s["nonhand_false_positive"] for s in summary))
        print("  手召回    中位 %.4f"
              % statistics.median(s["hand_recall"] for s in summary))
        bands = collections.defaultdict(list)
        for s in summary:
            for key, v in s["bands"].items():
                bands[key].append(v["fp"])
        for lo, hi in BANDS:
            key = "%d-%d" % (lo, hi)
            if key in bands:
                print("    %-10s 非手误判中位 %.4f"
                      % (key + "px", statistics.median(bands[key])))
        json.dump(summary, open(os.path.join(a.out, "dev_summary.json"), "w"),
                  indent=2)


if __name__ == "__main__":
    main()
