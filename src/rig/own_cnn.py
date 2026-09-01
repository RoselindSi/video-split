"""Ownership from the hand's own pixels, keeping the detector that works.

WHAT WENT WRONG BEFORE. The production pipeline is three steps -- YOLO finds
the hands, GrabCut cuts them out, a rule decides whose they are -- and only
the third step has the coordinate dependence worth removing. The dense
segmentation head replaced all three: it was asked to find hands AND classify
them from raw pixels, starting from 124 frames. It answered by painting the
whole frame owner_arm at one class weight and nothing at all at another, which
is what a model does when it has no idea where the hands are. The detector
already knows: two hands per frame, 0.83-0.87 confidence, zero dropouts.

So this replaces ONLY the third step. Same detections, same masks, same
everything -- a small network looks at the hand crop and says whose it is.

THE CROPS ALREADY EXIST. Every labelled package stores a 192px crop per hand.
That means all 1148 labels are usable immediately, including the 677 whose
source recording was never identified: a crop needs no video. The whole
`recover` dependency disappears.

WHAT THE RULE'S ROTATED VERDICT IS, WITHOUT RENDERING ANYTHING. `hands.csv`
stores `exit_y` as a fraction of frame height, and a half turn maps it to
1 - exit_y. So the rule's answer on an upside-down frame is
`(1 - exit_y) >= 0.55`, computable from the CSV. The comparison this module
prints costs no video decoding at all.

APPEARANCE MAY NOT TRANSPORT, AND THE SPLIT IS BUILT TO CATCH THAT. Within one
recording the wearer's sleeve is one colour and everyone else's is another, so
a model can score well by memorising a shirt. The split is by RECORDING for
exactly this reason; a within-recording split would report a number that
evaporates on the next workstation.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")
RULE_FRAC = 0.55
SIZE = 128


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit(
            "torch is not installed in this environment.\n"
            "  venv_rig has opencv and numpy; training needs the training "
            "venv instead.")


def load(pkgs, verbose=True):
    """-> [row] with `_crop`, `tag`, `y`, `exit_y`, `rule`."""
    rows, missing = [], []
    for p in pkgs:
        q = os.path.join(p, "hands.csv")
        if not os.path.exists(q):
            missing.append(p)
            continue
        for r in csv.DictReader(open(q, encoding="utf-8-sig")):
            if r.get("label") not in ("owner", "other"):
                continue
            m = STEM_RE.match(r["stem"])
            if not m:
                continue
            crop = os.path.join(p, "crops", r["stem"] + ".jpg")
            if not os.path.exists(crop):
                continue
            rows.append({
                "stem": r["stem"], "tag": m.group(1), "_crop": crop,
                "y": 1 if r["label"] == "owner" else 0,
                "exit_y": float(r.get("exit_y", np.nan)),
                "rule": int(r.get("rule_owner", 0)),
                "pkg": os.path.basename(p)})
    if verbose and missing:
        print(f"  !! {len(missing)} package(s) skipped: {missing}")
    return rows


def rule_verdict(rows, k):
    """The rule's owner/other call after k quarter turns. -> np.array

    Only k=0 and k=2 are defined: a quarter turn swaps the axes, so the
    quantity the rule reads -- a height -- is no longer a height at all, and
    reporting a number there would be inventing one."""
    if k % 4 == 0:
        return np.array([r["rule"] for r in rows])
    if k % 4 == 2:
        return np.array([1 if (1.0 - r["exit_y"]) >= RULE_FRAC else 0
                         for r in rows])
    return None


def split_by_tag(rows, holdout):
    tr = [r for r in rows if r["tag"] not in holdout]
    ev = [r for r in rows if r["tag"] in holdout]
    return tr, ev


def choose_holdout(rows, want_other=8, verbose=True):
    """Pick held-out recordings that contain `other`, smallest first.

    An eval split with no `other` in it cannot score the class the whole
    feature exists for -- the first segmentation run held out two recordings
    that were 100% owner and reported a dash. Smallest-first because `other`
    is concentrated in a few recordings and spending the largest of them on
    the eval would empty the training set of the class."""
    import collections
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["tag"]][r["y"]] += 1
    have = sorted(((n_oth, tag) for tag, (n_oth, _) in by.items()
                   if n_oth > 0))
    hold, got = [], 0
    for n_oth, tag in have:
        hold.append(tag)
        got += n_oth
        if got >= want_other:
            break
    if verbose:
        print(f"  holdout {hold}  ({got} other, "
              f"{sum(sum(by[t]) for t in hold)} hands)")
    return set(hold)


class Crops:
    def __init__(self, rows, size=SIZE, augment=False, seed=0, force_rot=None):
        self.rows, self.size, self.augment = rows, size, augment
        self.force_rot = force_rot
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        torch = _torch()
        r = self.rows[i]
        img = cv2.imread(r["_crop"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["_crop"])
        img = cv2.resize(img, (self.size, self.size))
        k = self.force_rot
        if k is None:
            k = int(self.rng.integers(4)) if self.augment else 0
        if k:
            img = np.rot90(img, k)
        if self.augment and self.rng.random() < 0.5:
            img = img[:, ::-1]
        x = torch.from_numpy(
            np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        x = (x - 0.45) / 0.25
        return x, torch.tensor(r["y"], dtype=torch.long)


def build_model():
    """A small convnet trained from scratch.

    Not a pretrained backbone: the container has no route to torchvision's
    weights, and a frozen RANDOM encoder is what produced the previous run's
    two degenerate answers. 1148 crops at 128px is small enough that a scratch
    network of this size is the honest choice rather than the compromise."""
    torch = _torch()
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128), block(128, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(128, 2))


def evaluate(model, ds, device, batch=64):
    torch = _torch()
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            xs, ys = zip(*[ds[j] for j in range(i, min(i + batch, len(ds)))])
            p = model(torch.stack(xs).to(device)).argmax(1).cpu().numpy()
            P.append(p)
            Y.append(np.array([int(v) for v in ys]))
    return np.concatenate(P), np.concatenate(Y)


def scores(pred, true):
    """-> dict. Accuracy alone is 97% for always-owner, so it is not enough."""
    pred, true = np.asarray(pred), np.asarray(true)
    tp = int(((pred == 0) & (true == 0)).sum())      # `other` is the positive
    fp = int(((pred == 0) & (true == 1)).sum())
    fn = int(((pred == 1) & (true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {"acc": float((pred == true).mean()),
            "other_prec": prec, "other_rec": rec,
            "other_f1": (2 * prec * rec / (prec + rec)
                         if prec and rec and np.isfinite(prec + rec)
                         else np.nan),
            "n_other": int((true == 0).sum()), "n": len(true)}


def _row(lab, s):
    def f(v):
        return f"{v:>10.3f}" if isinstance(v, float) and np.isfinite(v) \
            else f"{'-':>10}"
    return (f"    {lab:<22}{s['n']:>5}{s['n_other']:>7}"
            f"{f(s['acc'])}{f(s['other_prec'])}{f(s['other_rec'])}"
            f"{f(s['other_f1'])}")


def report(results):
    print(f"\n    {'':<22}{'n':>5}{'other':>7}{'acc':>10}"
          f"{'oth prec':>10}{'oth rec':>10}{'oth f1':>10}")
    for lab, s in results.items():
        print(_row(lab, s))


def train(rows, holdout, epochs=30, bs=32, lr=1e-3, seed=0, rotate=True,
          device=None, out=None):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr_rows, ev_rows = split_by_tag(rows, holdout)
    if not tr_rows or not ev_rows:
        raise SystemExit("train or eval split is empty")
    n_oth = sum(1 for r in tr_rows if r["y"] == 0)
    print(f"  train {len(tr_rows)} hands ({n_oth} other), "
          f"eval {len(ev_rows)} ({sum(1 for r in ev_rows if r['y']==0)} other)")
    if n_oth < 5:
        print("  !! fewer than 5 `other` in training. Any score below is "
              "about owner only.")

    tr = Crops(tr_rows, augment=True, seed=seed, force_rot=None if rotate else 0)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # `other` is 3% of the labels. Weighted by inverse frequency and no more:
    # the dense head showed what an unbounded weight does to a minority class.
    w = torch.tensor([len(tr_rows) / max(n_oth, 1) / 2.0, 1.0],
                     dtype=torch.float32, device=device).clamp(max=20.0)
    lossf = nn.CrossEntropyLoss(weight=w)
    print(f"  class weights other={w[0]:.2f} owner={w[1]:.2f}")

    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(tr))
        tot = n = 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xs, ys = zip(*[tr[int(j)] for j in idx])
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
            n += len(idx)
        if ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}  loss {tot/max(n,1):.4f}", flush=True)

    results = {}
    for k, lab in ((0, "upright"), (2, "turned 180")):
        p, y = evaluate(model, Crops(ev_rows, force_rot=k), device)
        results[f"cnn {lab}"] = scores(p, y)
        rv = rule_verdict(ev_rows, k)
        results[f"rule {lab}"] = scores(rv, np.array([r["y"] for r in ev_rows]))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "size": SIZE}, out)
        print(f"\n  wrote {out}")
    return results


def _self_test():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok += bool(cond)

    rows = [{"tag": "A_", "y": 1, "exit_y": 0.90, "rule": 1},
            {"tag": "A_", "y": 0, "exit_y": 0.20, "rule": 0},
            {"tag": "B_", "y": 1, "exit_y": 0.70, "rule": 1},
            {"tag": "C_", "y": 1, "exit_y": 0.60, "rule": 1}]
    chk("upright verdict is the stored one",
        list(rule_verdict(rows, 0)) == [1, 0, 1, 1])
    # 0.90 -> 0.10 (other), 0.20 -> 0.80 (owner), 0.70 -> 0.30, 0.60 -> 0.40
    chk("a half turn inverts every verdict",
        list(rule_verdict(rows, 2)) == [0, 1, 0, 0])
    chk("a quarter turn has no defined verdict",
        rule_verdict(rows, 1) is None)

    tr, ev = split_by_tag(rows, {"B_"})
    chk("split is by recording", len(tr) == 3 and len(ev) == 1)

    hold = choose_holdout(rows, want_other=1, verbose=False)
    chk("holdout contains a recording with `other`", hold == {"A_"})

    s = scores(np.array([1, 1, 1, 1]), np.array([1, 0, 1, 1]))
    chk("always-owner scores 0.75 accuracy and no other recall",
        abs(s["acc"] - 0.75) < 1e-9 and s["other_rec"] == 0.0)
    s2 = scores(np.array([0, 0]), np.array([0, 1]))
    chk("precision counts the wrong `other` calls",
        abs(s2["other_prec"] - 0.5) < 1e-9)
    print(f"\n  {ok}/7")
    return ok == 7


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--holdout", action="append", default=[],
                    help="tags to hold out; default picks the smallest "
                         "recordings that contain `other`")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no_rotate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    rows = load(a.pkg)
    if not rows:
        raise SystemExit("no labelled hands with crops")
    import collections
    c = collections.Counter(r["tag"] for r in rows)
    print(f"{len(rows)} hands with crops over {len(c)} recordings, "
          f"{sum(1 for r in rows if r['y']==0)} other")
    hold = set(a.holdout) if a.holdout else choose_holdout(rows)
    res = train(rows, hold, epochs=a.epochs, seed=a.seed,
                rotate=not a.no_rotate, out=a.out)
    report(res)
    print("\n  `other` is the positive class: precision is how often a hand "
          "called foreign\n  really is one, recall is how many foreign hands "
          "were caught. Accuracy is not\n  enough -- always answering "
          "`owner` scores about 0.97 here.")
    print("  The rule rows are computed from the stored exit height, not "
          "re-rendered. A\n  half turn maps exit_y to 1-exit_y, so its "
          "turned column is what the rule\n  would actually answer on a "
          "camera mounted the other way up.")


if __name__ == "__main__":
    main()
