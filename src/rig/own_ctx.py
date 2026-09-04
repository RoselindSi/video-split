"""Ownership from the hand, from what is around it, or from both.

WHY THE CROP MAY HAVE NOTHING LEFT TO GIVE. On the frozen test set the shipped
classifier recalls 0.644 of foreign hands, and the thirty-one it misses were
audited by eye: no visible property separates them from the fifty-six it
catches, and twenty of the thirty-one are confident errors at P(owner) >= 0.70.
That is the signature of a feature ceiling rather than a boundary drawn in the
wrong place. It is also unsurprising: a colleague's hand and the wearer's hand
LOOK THE SAME, because they are both hands. Whose it is is not a property of
the hand.

WHERE IT IS A PROPERTY OF. The forearm, and where the forearm goes. The
wearer's arms enter from the bottom and the near sides of a head-mounted view
and connect to nothing visible; a colleague's arm crosses the bench and
connects to a torso. This is the finding in `Is That My Hand?`, which reports
that the context branch alone disambiguates, and in `This Hand Is My Hand`,
which combines appearance with egocentric layout and temporal constraints
rather than reading the hand by itself.

THE ABLATION HAS TO SHARE ITS DATA OR IT MEASURES THE WRONG THING. Only 1014
of the 2533 labelled hands carry both a stored box and a context frame; the
rest were labelled from crops alone and their geometry columns are empty. So
every arm here trains on the SAME 1014, including the hand-only arm. Comparing
a context model on 1014 against the shipped model on 2533 would confound the
representation with the training set, and this project has already paid once
for a comparison that did that.

AND THE SELECTION SET IS NOT THE TEST SET. Leave-one-recording-out put recall
at 0.969; the frozen independent set put it at 0.644. That gap is the reason
`testpkg_2` is scored once, at the end, and every choice -- architecture,
epochs, threshold -- is made on recordings held out of training instead.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?_)f(\d+)_h\d+$")

# The context window, as a multiple of the hand box. 2.5 was chosen to reach
# roughly a forearm's length past the wrist without swallowing the whole
# bench: at this rig's framing a forearm is about one and a half hand-widths.
# It is a starting value and the point of `--ctx_scale` is that it is cheap to
# challenge.
CTX_SCALE = 2.5

HAND_PX = 128
CTX_PX = 128

# Frame-normalised and about this hand alone. `rel_size` is deliberately
# absent: it divides by another detection, so a hand's own feature changed
# when a different hand appeared, which is the defect that produced the
# flicker in the first place.
GEOM = ("box_cx", "box_cy", "box_w", "box_h", "dir_x", "dir_y",
        "exit_x", "exit_y", "edge_bottom", "edge_left", "edge_right",
        "edge_top", "hand_span", "conf")


def _f(v, d=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load(pkgs, verbose=True):
    """-> [row] carrying a crop, a context frame, a box and geometry.

    A row is kept only when all four are present. The count this drops is
    printed rather than swallowed: it is most of the corpus, and any result
    below is a result about the part that survived."""
    rows, dropped = [], 0
    for p in pkgs:
        q = os.path.join(p, "hands.csv")
        if not os.path.exists(q):
            continue
        for r in csv.DictReader(open(q, encoding="utf-8-sig")):
            if r.get("label") not in ("owner", "other"):
                continue
            m = STEM_RE.match(r.get("stem", ""))
            if not m:
                continue
            crop = os.path.join(p, "crops", r["stem"] + ".jpg")
            ctx = os.path.join(p, "context", r["stem"] + ".jpg")
            g = [_f(r.get(c)) for c in GEOM]
            if (not os.path.exists(crop) or not os.path.exists(ctx)
                    or not all(np.isfinite(g))):
                dropped += 1
                continue
            rows.append({
                "stem": r["stem"], "tag": m.group(1), "pkg": p,
                "_crop": crop, "_ctx": ctx,
                "box": (_f(r["box_cx"]), _f(r["box_cy"]),
                        _f(r["box_w"]), _f(r["box_h"])),
                "g": np.array(g, np.float32),
                "y": 1 if r["label"] == "owner" else 0})
    if verbose:
        n_o = sum(1 for r in rows if r["y"] == 0)
        print(f"  {len(rows)} hands with crop + context + geometry "
              f"({n_o} other) over {len({r['tag'] for r in rows})} "
              f"recordings; {dropped} labelled hands dropped for missing "
              f"one of the three")
    return rows


def strip_overlay(img):
    """Remove the marks the labelling tool drew on the stored frame. -> img

    THE CONTEXT ARM WOULD OTHERWISE BE READING THE GEOMETRY ARM'S FEATURES.
    `own_label._write_sample` saves the context frame with a yellow box on the
    hand and a MAGENTA LINE FROM THE WRIST TO WHERE THE FORE ARM LEAVES THE
    FRAME. That line is `dir_x`, `dir_y`, `exit_x` and `exit_y` drawn in
    paint. Leaving it in makes `context` a picture of the geometry features
    rather than a picture of the scene, so a win for the context branch could
    not be attributed to the forearm or the torso -- the very thing the
    experiment exists to test -- and `both_geom` would receive the same cue
    twice in two forms.

    The yellow box is a different case and it also goes. It marks WHICH hand
    is the subject, which is legitimate information the crop already carries
    by being centred, but it is drawn from the same box the geometry features
    are computed from, so it is the same leak in a weaker form.

    Colour-keyed and inpainted rather than re-extracted from video: the two
    marks cover under one percent of the frame, and re-decoding a thousand
    scattered frames off network storage costs more than this measurement is
    worth. What is left is an inpainted streak, not clean pixels -- so this
    reduces the leak, it does not prove it gone."""
    import cv2
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    drawn = (((b > 150) & (r > 150) & (g < 90))       # magenta line
             | ((g > 150) & (r > 150) & (b < 90)))    # yellow box
    if not drawn.any():
        return img
    m = cv2.dilate(drawn.astype(np.uint8), np.ones((3, 3), np.uint8))
    return cv2.inpaint(img, m, 4, cv2.INPAINT_TELEA)


def context_crop(img, box, scale=CTX_SCALE):
    """-> the region `scale` times the hand box, clipped to the frame."""
    import cv2
    H, W = img.shape[:2]
    cx, cy, bw, bh = box
    half = max(bw * W, bh * H) * scale / 2.0
    x0 = int(max(0, cx * W - half))
    y0 = int(max(0, cy * H - half))
    x1 = int(min(W, cx * W + half))
    y1 = int(min(H, cy * H + half))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


class Pairs:
    """(hand, context, geometry, label) for one labelled hand."""

    def __init__(self, rows, augment=False, ctx_scale=CTX_SCALE,
                 strip=True):
        self.rows = rows
        self.augment = bool(augment)
        self.ctx_scale = float(ctx_scale)
        self.strip = bool(strip)

    def __len__(self):
        return len(self.rows)

    def _prep(self, img, px):
        import cv2
        import torch
        img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img[:, :, ::-1].copy()).float().permute(2, 0, 1)
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std

    def __getitem__(self, i):
        import cv2
        import torch
        r = self.rows[i]
        hand = cv2.imread(r["_crop"])
        ctx_img = cv2.imread(r["_ctx"])
        if ctx_img is not None and self.strip:
            ctx_img = strip_overlay(ctx_img)
        ctx = (context_crop(ctx_img, r["box"], self.ctx_scale)
               if ctx_img is not None else None)
        if hand is None:
            hand = np.zeros((HAND_PX, HAND_PX, 3), np.uint8)
        if ctx is None or ctx.size == 0:
            ctx = hand
        if self.augment:
            # APPEARANCE ONLY, AND NO HALF TURN. Brightness, contrast and
            # colour are where a model can learn a shortcut to the recording
            # rather than to the hand, so those are what get perturbed.
            # Rotation is NOT: a fixed head-mounted rig makes image
            # orientation a legitimate and strong egocentric cue, measured
            # here as the strongest single geometric one, and augmenting it
            # away would destroy evidence rather than regularise the model.
            a = 1.0 + random.uniform(-0.30, 0.30)
            b = random.uniform(-28, 28)
            hand = np.clip(hand.astype(np.float32) * a + b, 0, 255).astype(
                np.uint8)
            ctx = np.clip(ctx.astype(np.float32) * a + b, 0, 255).astype(
                np.uint8)
        return (self._prep(hand, HAND_PX), self._prep(ctx, CTX_PX),
                torch.from_numpy(r["g"]), int(r["y"]))


def build(arm, n_geom=len(GEOM)):
    """-> a model for one arm of the ablation.

    Two trunks and not one shared trunk. The hand crop and a 2.5x window are
    different distributions -- one is filled by a hand, the other is mostly
    bench -- and tying the weights would ask one set of filters to serve both.
    They are small enough that two is affordable."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    def trunk():
        m = resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Identity()
        return m

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_hand = arm in ("hand", "both", "both_geom")
            self.use_ctx = arm in ("context", "both", "both_geom")
            self.use_geom = arm in ("both_geom", "geom")
            self.hand = trunk() if self.use_hand else None
            self.ctx = trunk() if self.use_ctx else None
            d = 512 * (int(self.use_hand) + int(self.use_ctx))
            if self.use_geom:
                self.gmlp = nn.Sequential(
                    nn.Linear(n_geom, 64), nn.ReLU(), nn.Linear(64, 64),
                    nn.ReLU())
                d += 64
            self.head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(d, 128), nn.ReLU(),
                nn.Linear(128, 2))

        def forward(self, h, c, g):
            parts = []
            if self.use_hand:
                parts.append(self.hand(h))
            if self.use_ctx:
                parts.append(self.ctx(c))
            if self.use_geom:
                parts.append(self.gmlp(g))
            return self.head(torch.cat(parts, 1))

    return Net()


# The width the labelling tool wrote its context frames at. Inference has the
# full panorama, which is more than twice as wide, and a 2.5x window cut from
# it and then squeezed to 128px would carry detail the training crops never
# had. Matching the training resize is not cosmetic: it is the difference
# between the distribution the weights were fitted to and a sharper one.
CONTEXT_STORE_W = 900


def load_model(path, device=None):
    """-> (model, device, arm), or (None, None, None) when absent.

    A missing checkpoint is not an error here any more than in `own_cnn`:
    the hand-only classifier is the incumbent and stays the fallback."""
    import torch
    if not path or not os.path.exists(path):
        return None, None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    arm = ck.get("arm", "both_geom")
    m = build(arm).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, device, arm


def _geom_vector(det, shape):
    """own_label's feature vector, reordered to GEOM. -> np.float32

    Reordered by NAME and not by position. `own_label.FEATURES` has seventeen
    entries in its own order and GEOM has fourteen in another; indexing one
    with the other's positions would feed `wrist_y` where `box_cx` belongs and
    the model would still run, silently, at chance."""
    from src.rig import own_label
    f = own_label.features(det, shape)
    at = {name: i for i, name in enumerate(own_label.FEATURES)}
    return np.array([float(f[at[c]]) for c in GEOM], np.float32)


def predict(model, device, rgb, dets, ctx_scale=CTX_SCALE):
    """-> [(is_owner, p_owner)] one per detection, matching own_cnn.predict.

    The context window is cut from the frame AT THE STORED WIDTH, and no
    overlay is stripped: nothing drew on this frame. `strip_overlay` exists
    for the training crops only, and calling it here would inpaint whatever
    magenta or yellow the bench actually contains."""
    import cv2
    import torch
    if not dets:
        return []
    from src.rig.own_cnn import crop_of
    H, W = rgb.shape[:2]
    small = (cv2.resize(rgb, (CONTEXT_STORE_W,
                              int(CONTEXT_STORE_W * H / W)),
                        interpolation=cv2.INTER_AREA)
             if W > CONTEXT_STORE_W else rgb)
    ds = Pairs([], ctx_scale=ctx_scale)
    hs, cs, gs, keep = [], [], [], []
    for i, d in enumerate(dets):
        hand = crop_of(rgb, d)
        if hand is None:
            continue
        x0, y0, x1, y1 = [float(v) for v in d["box"]]
        box = ((x0 + x1) / 2.0 / W, (y0 + y1) / 2.0 / H,
               (x1 - x0) / W, (y1 - y0) / H)
        ctx = context_crop(small, box, ctx_scale)
        if ctx is None or ctx.size == 0:
            ctx = hand
        hs.append(ds._prep(hand, HAND_PX))
        cs.append(ds._prep(ctx, CTX_PX))
        gs.append(torch.from_numpy(_geom_vector(d, rgb.shape)))
        keep.append(i)
    out = [(True, 1.0)] * len(dets)
    if not hs:
        return out
    with torch.no_grad():
        pr = torch.softmax(
            model(torch.stack(hs).to(device), torch.stack(cs).to(device),
                  torch.stack(gs).to(device)), 1)[:, 1]
    for i, p in zip(keep, pr.cpu().numpy().tolist()):
        out[i] = (p >= 0.5, float(p))
    return out


def scores(pred, true):
    """`other` is the positive class."""
    pred, true = np.asarray(pred), np.asarray(true)
    tp = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 0) & (true == 1)).sum())
    fn = int(((pred == 1) & (true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and prec + rec else float("nan"))
    return {"n": len(true), "other": int((true == 0).sum()),
            "acc": float((pred == true).mean()), "prec": prec, "rec": rec,
            "f1": f1}


def split_by_recording(rows, frac=0.25, seed=0):
    """-> (train, dev). Whole recordings, never split within one.

    Stratified on whether a recording contains `other` at all, so the dev
    side cannot come back with no positives and an undefined recall."""
    by = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    with_o = sorted(t for t, v in by.items() if any(x["y"] == 0 for x in v))
    without = sorted(t for t, v in by.items() if t not in set(with_o))
    rng = random.Random(seed)
    rng.shuffle(with_o)
    rng.shuffle(without)
    dev = set(with_o[:max(1, int(round(len(with_o) * frac)))]
              + without[:int(round(len(without) * frac))])
    return ([r for r in rows if r["tag"] not in dev],
            [r for r in rows if r["tag"] in dev])


def run_epoch(model, ds, device, opt=None, batch=32, weight=None):
    import torch
    train = opt is not None
    model.train(train)
    idx = list(range(len(ds)))
    if train:
        random.shuffle(idx)
    P, Y, loss_sum = [], [], 0.0
    lossf = torch.nn.CrossEntropyLoss(weight=weight)
    for i in range(0, len(idx), batch):
        chunk = idx[i:i + batch]
        h, c, g, y = zip(*[ds[j] for j in chunk])
        h = torch.stack(h).to(device)
        c = torch.stack(c).to(device)
        g = torch.stack(g).to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        with torch.set_grad_enabled(train):
            out = model(h, c, g)
            loss = lossf(out, y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_sum += float(loss) * len(chunk)
        P.append(out.argmax(1).detach().cpu().numpy())
        Y.append(y.cpu().numpy())
    return (np.concatenate(P), np.concatenate(Y),
            loss_sum / max(len(idx), 1))


def train_arm(arm, tr, dv, epochs=12, seed=0, lr=3e-4, ctx_scale=CTX_SCALE,
              strip=True, verbose=True):
    """-> (model, best dev scores). Selection is on dev `other` F1."""
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dtr = Pairs(tr, augment=True, ctx_scale=ctx_scale, strip=strip)
    ddv = Pairs(dv, augment=False, ctx_scale=ctx_scale, strip=strip)
    best, best_state = None, None
    for e in range(epochs):
        _, _, tl = run_epoch(model, dtr, device, opt)
        p, y, _ = run_epoch(model, ddv, device)
        s = scores(p, y)
        if verbose:
            print(f"    epoch {e + 1:>2}  loss {tl:.3f}  dev f1 "
                  f"{s['f1']:.3f}  prec {s['prec']:.3f}  rec {s['rec']:.3f}")
        if best is None or (s["f1"] == s["f1"] and s["f1"] > best["f1"]):
            best = s
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def evaluate(model, rows, ctx_scale=CTX_SCALE, strip=True):
    import torch
    device = next(model.parameters()).device
    p, y, _ = run_epoch(model, Pairs(rows, augment=False,
                                     ctx_scale=ctx_scale,
                                     strip=strip), device)
    return scores(p, y), p, y


def main():
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True,
                    help="training packages")
    ap.add_argument("--test", action="append", default=[],
                    help="the frozen set. Scored ONCE, after every choice is "
                         "made on dev.")
    ap.add_argument("--arm", action="append", default=[],
                    choices=("hand", "context", "both", "both_geom", "geom"),
                    help="default runs all four of the ablation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3,
                    help="one seed cannot tell an architecture from an "
                         "initialisation on a set this size")
    ap.add_argument("--dev_frac", type=float, default=0.25)
    ap.add_argument("--ctx_scale", type=float, default=CTX_SCALE)
    ap.add_argument("--keep_overlay", action="store_true",
                    help="do NOT remove the labelling tool's drawn box and "
                         "wrist-to-exit line from the context frame. Only "
                         "for measuring how much those marks were worth.")
    ap.add_argument("--out", help="save the best arm's checkpoint here")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    arms = a.arm or ["hand", "context", "both", "both_geom"]
    rows = load(a.pkg)
    if not rows:
        raise SystemExit("no labelled hands with crop + context + geometry")
    tr, dv = split_by_recording(rows, a.dev_frac)
    print(f"  train {len(tr)} ({sum(1 for r in tr if r['y'] == 0)} other, "
          f"{len({r['tag'] for r in tr})} recordings)   "
          f"dev {len(dv)} ({sum(1 for r in dv if r['y'] == 0)} other, "
          f"{len({r['tag'] for r in dv})} recordings)")
    test = load(a.test) if a.test else []

    results = {}
    for arm in arms:
        print(f"\n=== {arm} ===")
        devs, models = [], []
        for s in range(a.seeds):
            print(f"  seed {s}")
            m, best = train_arm(arm, tr, dv, epochs=a.epochs, seed=s,
                                ctx_scale=a.ctx_scale,
                                strip=not a.keep_overlay)
            devs.append(best)
            models.append(m)
        f1 = [d["f1"] for d in devs]
        print(f"  DEV other-F1 over {a.seeds} seeds: "
              + "  ".join(f"{x:.3f}" for x in f1)
              + f"   median {sorted(f1)[len(f1) // 2]:.3f}")
        results[arm] = (devs, models)

    print(f"\n  {'arm':<12} {'dev f1 median':>14} {'dev prec':>9} "
          f"{'dev rec':>8}")
    for arm in arms:
        devs, _ = results[arm]
        f1 = sorted(d["f1"] for d in devs)
        print(f"  {arm:<12} {f1[len(f1) // 2]:>14.3f} "
              f"{np.median([d['prec'] for d in devs]):>9.3f} "
              f"{np.median([d['rec'] for d in devs]):>8.3f}")
    print("\n  Read the spread across seeds before the gap between arms. On "
          "a dev set this\n  size an architecture has to beat its own "
          "initialisation before it beats another\n  architecture.")

    if test:
        best_arm = max(arms, key=lambda k: np.median(
            [d["f1"] for d in results[k][0]]))
        print(f"\n=== FROZEN TEST, {best_arm} only ===")
        print("  Scored once. Every other arm's test number is deliberately "
              "not computed:\n  four numbers on a frozen set is a selection "
              "set with extra steps.")
        for i, m in enumerate(results[best_arm][1]):
            s, _, _ = evaluate(m, test, a.ctx_scale,
                               strip=not a.keep_overlay)
            print(f"    seed {i}  n {s['n']}  other {s['other']}  "
                  f"prec {s['prec']:.3f}  rec {s['rec']:.3f}  "
                  f"f1 {s['f1']:.3f}  acc {s['acc']:.3f}")
        print("\n  The shipped hand-only classifier on this same set: "
              "prec 0.889  rec 0.644  f1 0.747.\n  It trained on 2533 hands "
              "against this experiment's 1014, so a tie here is not a tie.")
        if a.out:
            import torch
            torch.save({"arm": best_arm,
                        "state": results[best_arm][1][0].state_dict()},
                       a.out)
            print(f"  checkpoint -> {a.out}")


def _self_test():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  ok   " if c else "  FAIL ") + m)
        ok = ok and bool(c)

    img = np.zeros((400, 800, 3), np.uint8)
    img[180:220, 380:420] = 255
    c = context_crop(img, (0.5, 0.5, 0.05, 0.10), scale=2.5)
    chk(c is not None and c.shape[0] > 40 and c.shape[1] > 40,
        "the context window is larger than the hand box")
    edge = context_crop(img, (0.01, 0.01, 0.05, 0.10), scale=2.5)
    chk(edge is not None and edge.size > 0,
        "a box against the frame edge still yields a window")

    rows = [{"tag": f"r{i}", "y": 0 if i < 6 else 1} for i in range(12)]
    tr, dv = split_by_recording(rows, 0.25, seed=0)
    chk(not ({r["tag"] for r in tr} & {r["tag"] for r in dv}),
        "no recording appears on both sides of the split")
    chk(any(r["y"] == 0 for r in dv),
        "the dev side carries at least one `other`, or recall is undefined")

    drawn = np.full((60, 60, 3), 120, np.uint8)
    drawn[10:14, 5:55] = (255, 0, 255)
    drawn[40:44, 5:55] = (0, 255, 255)
    clean = strip_overlay(drawn)
    b, g, r = (clean[:, :, i].astype(int) for i in range(3))
    left = int((((b > 150) & (r > 150) & (g < 90))
                | ((g > 150) & (r > 150) & (b < 90))).sum())
    chk(left == 0,
        "the drawn wrist-to-exit line and hand box are gone from the "
        "context frame")
    chk(strip_overlay(np.full((20, 20, 3), 100, np.uint8)).shape == (20, 20, 3),
        "a frame with no marks on it is returned unchanged in shape")

    # THE ORDER OF THE GEOMETRY VECTOR IS THE FAILURE THAT WOULD NOT SHOW.
    # `own_label.FEATURES` has seventeen names in one order, GEOM has fourteen
    # in another. Index one with the other's positions and the model receives
    # `wrist_y` where it learned `box_cx`; nothing raises, nothing looks
    # wrong, and ownership is decided at chance.
    from src.rig import own_label
    kp = np.zeros((21, 2), float)
    kp[0] = (300.0, 400.0)
    kp[1:] = (300.0, 250.0)
    det = {"box": (250, 350, 350, 450), "kp": kp, "conf": 0.77,
           "edge": "bottom", "exit": np.array([300.0, 480.0])}
    shape = (480, 800, 3)
    ref = own_label.features(det, shape)
    at = {n: i for i, n in enumerate(own_label.FEATURES)}
    got = _geom_vector(det, shape)
    chk(len(got) == len(GEOM),
        "the geometry vector has one entry per GEOM name")
    chk(all(abs(float(got[i]) - float(ref[at[c]])) < 1e-6
            for i, c in enumerate(GEOM)),
        "every geometry entry is the feature of the SAME NAME, not the same "
        "position")

    chk("rel_size" not in GEOM,
        "the detection-set-dependent cue is not among the geometry features")
    return ok


if __name__ == "__main__":
    main()
