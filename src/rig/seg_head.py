"""The 3-class ownership head: background / owner arm / other arm.

    0  background and objects
    1  the wearer's own hand and arm
    2  anyone else's hand and arm

A LEARNED HEAD BECAUSE THE HAND-WRITTEN ONE WAS DISPROVED, not because it was
awkward. On this rig the bench's front edge measures 0.21-0.25 m and the
wearer's forearm 0.23-0.30 m, so depth carries no separation between them and
every rule that asked it for one took the bench as well. Depth is an INPUT
here, weighed against appearance, arm shape, scale and position by something
that can learn what depth is worth.

FROZEN ENCODER, TRAINED DECODER. A few hundred annotated frames cannot move a
backbone without memorising them, and this corpus makes memorisation
especially cheap: the wearer wears a red wristband and a navy sleeve in nearly
every frame of some recordings, so a head with enough freedom will learn
'red wristband = owner' and report an excellent score right up until the
operator changes. The encoder stays frozen and the split is by OPERATOR, not
by recording -- a held-out recording of the same person tests almost nothing.

DEPTH ENTERS AT THE DECODER, NOT THE STEM. A frozen RGB backbone has no
channel for it, and unfreezing the stem to admit two planes would undo the
paragraph above. A small parallel pyramid over (range, validity) is
concatenated at each decoder level instead, which leaves the backbone
untouched and lets the depth pathway be ablated by zeroing one input.

THE CLASS-2 SCORE IS REPORTED BY REGIME AND NEVER AGGREGATED. The blind census
found another person's arm in 46.7% of frames but reaching the workspace in
only 1.1%, and those two cases are not the same problem: a colleague across
the bench is small, high in the frame and far, while a colleague reaching in
is the same size, colour, uniform and distance as the wearer's own arm. An
aggregate class-2 IoU is dominated by the easy 45.6% and would read as success
while the case that actually corrupts the downstream signal was never tested.
`evaluate` therefore splits by the census label and refuses to print a single
number.

WHAT THIS NEEDS AND DOES NOT HAVE YET: masks. `--smoke` exists so the whole
path -- data, fusion, loss, metrics -- can be proven on synthetic frames with
a known rule before a single mask is drawn by hand.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

CLASSES = ("background", "owner_arm", "other_arm")
IGNORE = 255

# 16:9 exactly, both sides divisible by 32 so every encoder stride divides
# evenly and no decoder level is off by a pixel.
DEFAULT_SIZE = (1024, 576)

# ImageNet statistics, because the frozen encoders are ImageNet-pretrained and
# feeding them anything else silently shifts every feature.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Range is fed in metres divided by this, so the network sees roughly unit
# scale. 4 m covers the bench and the far side of the aisle; beyond it the
# 60 mm baseline is not measuring anything anyway.
RANGE_SCALE = 4.0


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit(
            "torch is not installed in this environment.\n"
            "  venv_rig has opencv and numpy for the geometry work; training "
            "needs the\n  training venv instead. Activate that one and rerun.")


# --------------------------------------------------------------------------
# data


def load_manifest(root):
    """-> [row]. Rows without a mask are dropped and counted, because a
    silently-skipped frame is how a dataset ends up smaller than believed."""
    man = os.path.join(root, "manifest.csv")
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    keep, no_mask = [], 0
    for r in rows:
        stem = f"{r['recording']}_f{int(r['frame']):06d}.png"
        r["_rgb"] = os.path.join(root, "images", stem)
        r["_range"] = os.path.join(root, "range", stem)
        r["_mask"] = os.path.join(root, "masks", stem)
        if not os.path.exists(r["_mask"]) or not os.path.exists(r["_rgb"]):
            no_mask += 1
            continue
        keep.append(r)
    return keep, no_mask


class OwnershipDataset:
    """RGB + (range, validity) + mask, at a fixed size.

    Range is resized with NEAREST. Interpolating it would invent depths at the
    edge between a measured arm and an unmeasured bench, and an invented depth
    is indistinguishable from a measured one once it reaches the model."""

    def __init__(self, rows, size=DEFAULT_SIZE, augment=False, seed=0,
                 rotate=False, force_rot=None):
        self.rows, self.size, self.augment = rows, size, augment
        self.rotate, self.force_rot = rotate, force_rot
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        torch = _torch()
        r = self.rows[i]
        W, H = self.size
        rgb = cv2.imread(r["_rgb"], cv2.IMREAD_COLOR)
        rng16 = cv2.imread(r["_range"], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(r["_mask"], cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise FileNotFoundError(r["_rgb"])
        if rng16 is None:
            rng16 = np.zeros(rgb.shape[:2], np.uint16)
        # ROTATION, AND WHY IT REVERSES THE COMMENT BELOW. The augment block
        # used to forbid a vertical flip on the grounds that an arm entering
        # from the top is the strongest cue for class 2. That cue is exactly
        # what must not be learned. It only means "someone else's arm" in the
        # wide render, whose up axis is the fan's rotation axis; in a raw
        # module view cam1 is mounted rolled and the wearer's own arm comes in
        # from the lower left. A model that leans on frame orientation has
        # reproduced the geometric rule in weights and inherited its one
        # limitation. Rotating the training frames removes the cue, so
        # ownership has to be read from appearance and context instead.
        k = self.force_rot
        if k is None:
            k = int(self.rng.integers(4)) if (self.augment and self.rotate) \
                else 0
        if k:
            rgb = np.rot90(rgb, k)
            rng16 = np.rot90(rng16, k)
            mask = np.rot90(mask, k)
        rgb = cv2.resize(np.ascontiguousarray(rgb), (W, H),
                         interpolation=cv2.INTER_AREA)
        rng16 = np.ascontiguousarray(rng16)
        mask = np.ascontiguousarray(mask)
        rng16 = cv2.resize(rng16, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.augment and self.rng.random() < 0.5:
            rgb, rng16, mask = rgb[:, ::-1], rng16[:, ::-1], mask[:, ::-1]

        x = torch.from_numpy(
            np.ascontiguousarray(rgb[:, :, ::-1]).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        x = (x - torch.tensor(MEAN)[:, None, None]) \
            / torch.tensor(STD)[:, None, None]

        valid = (rng16 > 0).astype(np.float32)
        met = np.ascontiguousarray(rng16).astype(np.float32) / 1000.0
        d = torch.from_numpy(
            np.stack([np.clip(met / RANGE_SCALE, 0, 4) * valid, valid], 0))
        y = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))
        return x, d, y, r.get("census_label", "")


def class_weights(rows, size=DEFAULT_SIZE, cap=50.0):
    """Inverse pixel frequency, capped. -> [3] float

    Uncapped inverse frequency on this corpus reaches four figures for class 2
    and turns every faint red pixel into a colleague. The cap is a bound on
    that, not a tuned number."""
    import cv2
    counts = np.zeros(len(CLASSES), np.float64)
    for r in rows:
        m = cv2.imread(r["_mask"], cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for c in range(len(CLASSES)):
            counts[c] += (m == c).sum()
    if counts.sum() == 0:
        return np.ones(len(CLASSES), np.float32)
    freq = counts / counts.sum()
    w = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), 0.0)
    w = w / w[w > 0].min()
    return np.clip(w, 0, cap).astype(np.float32)


# --------------------------------------------------------------------------
# model


def build_encoder(name="resnet18", path=None):
    """-> (module, out_channels[list], strides[list]). Frozen by the caller.

    A missing download is reported and not worked around: a randomly
    initialised 'pretrained' encoder trains to a number that means nothing,
    and it is better to fail loudly here than to explain the number later."""
    torch = _torch()
    import torch.nn as nn

    if name == "resnet18":
        import torchvision
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=w)
            pre = True
        except Exception as e:
            print(f"  !! pretrained weights unavailable ({type(e).__name__}: "
                  f"{e}); using random init.\n     Any score from this run is "
                  f"about the decoder alone, not about the task.")
            net = torchvision.models.resnet18(weights=None)
            pre = False

        class R18(nn.Module):
            def __init__(s):
                super().__init__()
                s.stem = nn.Sequential(net.conv1, net.bn1, net.relu,
                                       net.maxpool)
                s.l1, s.l2, s.l3, s.l4 = (net.layer1, net.layer2,
                                          net.layer3, net.layer4)

            def forward(s, x):
                x = s.stem(x)
                a = s.l1(x); b = s.l2(a); c = s.l3(b); d = s.l4(c)
                return [a, b, c, d]

        m = R18()
        m.pretrained = pre
        return m, [64, 128, 256, 512], [4, 8, 16, 32]

    if name == "segformer_b0":
        from transformers import SegformerModel
        src = path or "nvidia/mit-b0"
        net = SegformerModel.from_pretrained(src)

        class SF(nn.Module):
            def __init__(s):
                super().__init__()
                s.net = net

            def forward(s, x):
                return list(s.net(x, output_hidden_states=True).hidden_states)

        m = SF()
        m.pretrained = True
        return m, [32, 64, 160, 256], [4, 8, 16, 32]

    raise SystemExit(f"unknown encoder {name!r}; have resnet18, segformer_b0")


def _depth_pyramid(strides, ch):
    """A small conv tower over (range, validity), one output per encoder
    level. Separate from the backbone so the depth pathway can be ablated by
    zeroing the input rather than by editing the model."""
    torch = _torch()
    import torch.nn as nn

    def block(c_in, c_out, n_down):
        """n_down stride-2 convs, then one at stride 1."""
        L, c = [], c_in
        for _ in range(n_down):
            L += [nn.Conv2d(c, c_out, 3, stride=2, padding=1),
                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c = c_out
        L += [nn.Conv2d(c, c_out, 3, padding=1),
              nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        return nn.Sequential(*L)

    # Each block takes the previous level's output down to the next encoder
    # stride, derived from the strides rather than assumed to double: level 0
    # needs two halvings to reach stride 4 and the rest need one each, and
    # hardcoding that would break on an encoder with a different pyramid.
    mods, c_in, prev = [], 2, 1
    for s in strides:
        n_down = int(round(np.log2(s / prev)))
        assert n_down >= 1, f"stride {s} does not follow {prev}"
        mods.append(block(c_in, ch, n_down))
        c_in, prev = ch, s
    return nn.ModuleList(mods)


def build_model(encoder="resnet18", path=None, dch=32, mid=128,
                n_classes=len(CLASSES)):
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    enc, chans, strides = build_encoder(encoder, path)
    for p in enc.parameters():
        p.requires_grad_(False)

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = enc
            s.strides = strides
            s.dpyr = _depth_pyramid(strides, dch)
            s.lat = nn.ModuleList([nn.Conv2d(c + dch, mid, 1) for c in chans])
            s.smooth = nn.ModuleList([
                nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1),
                              nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
                for _ in chans])
            s.out = nn.Conv2d(mid, n_classes, 1)

        def forward(s, x, d):
            s.enc.eval()                       # frozen: keep BN statistics still
            with torch.no_grad():
                feats = s.enc(x)
            dz, cur = [], d
            for blk in s.dpyr:
                cur = blk(cur)
                dz.append(cur)
            # Top-down: start at the coarsest level and add each finer one.
            y = None
            for i in reversed(range(len(feats))):
                f = feats[i]
                di = dz[i]
                if di.shape[-2:] != f.shape[-2:]:
                    di = F.interpolate(di, f.shape[-2:], mode="nearest")
                z = s.lat[i](torch.cat([f, di], 1))
                y = z if y is None else z + F.interpolate(
                    y, z.shape[-2:], mode="bilinear", align_corners=False)
                y = s.smooth[i](y)
            return F.interpolate(s.out(y), x.shape[-2:], mode="bilinear",
                                 align_corners=False)

    m = Head()
    m.pretrained = getattr(enc, "pretrained", False)
    return m


# --------------------------------------------------------------------------
# metrics


def confusion(pred, true, n=len(CLASSES)):
    k = (true >= 0) & (true < n)
    return np.bincount(n * true[k].astype(int) + pred[k],
                       minlength=n * n).reshape(n, n)


def iou_from_confusion(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def fp_from_confusion(cm):
    """Fraction of ALL pixels wrongly predicted as each class.

    IoU goes to NaN where a class is absent from the ground truth, which is
    exactly the regime that matters most here: 52% of frames contain nobody
    else, and the failure that would corrupt the downstream signal is calling
    the wearer's own arm somebody else's. That failure has no IoU to report --
    it needs a false-positive rate, and this is defined in every regime rather
    than only where the class happens to appear."""
    tot = max(cm.sum(), 1)
    fp = cm.sum(0) - np.diag(cm)
    return (fp.astype(np.float64) / tot)


def evaluate(model, ds, device, batch=2):
    """-> {"overall": iou[3], "by_regime": {label: iou[3]}, "support": {...}}

    Split by the census label on purpose. A single class-2 number here would
    be 45.6% easy frames and 1.1% hard ones averaged into one figure that
    describes neither."""
    torch = _torch()
    model.eval()
    cm_all = np.zeros((len(CLASSES),) * 2, np.int64)
    cm_reg, sup = {}, {}
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            xs, dsz, ys, labs = zip(*[ds[j] for j in
                                      range(i, min(i + batch, len(ds)))])
            x = torch.stack(xs).to(device)
            d = torch.stack(dsz).to(device)
            p = model(x, d).argmax(1).cpu().numpy()
            y = torch.stack(ys).numpy()
            for b, lab in enumerate(labs):
                c = confusion(p[b].ravel(), y[b].ravel())
                cm_all += c
                cm_reg[lab] = cm_reg.get(
                    lab, np.zeros_like(c)) + c
                sup[lab] = sup.get(lab, 0) + 1
    return {"overall": iou_from_confusion(cm_all).tolist(),
            "overall_fp": fp_from_confusion(cm_all).tolist(),
            "by_regime": {k: iou_from_confusion(v).tolist()
                          for k, v in sorted(cm_reg.items())},
            "fp_by_regime": {k: fp_from_confusion(v).tolist()
                             for k, v in sorted(cm_reg.items())},
            "support": sup}


def _cells(v, fmt="{:>13.3f}"):
    return "".join(fmt.format(x) if np.isfinite(x) else f"{'-':>13}"
                   for x in v)


def print_eval(res):
    for title, key, okey in (("IoU", "by_regime", "overall"),
                             ("false positives, share of all pixels",
                              "fp_by_regime", "overall_fp")):
        print(f"\n    {title}")
        print(f"    {'regime':<14}{'n':>4}" +
              "".join(f"{c:>13}" for c in CLASSES))
        for k, v in res[key].items():
            print(f"    {k or '(none)':<14}{res['support'].get(k,0):>4}"
                  + _cells(v))
        print(f"    {'-'*(18+13*len(CLASSES))}")
        print(f"    {'ALL':<14}{sum(res['support'].values()):>4}"
              + _cells(res[okey]))
    print("\n  Read the 'other_near' row of the IoU table and the "
          "'owner_only' row of the\n  false-positive table. The first is the "
          "case that is hard; the second is\n  hallucinating a colleague onto "
          "a frame that has none, which is the failure\n  that would corrupt "
          "the downstream signal on 52% of the corpus. Neither is\n  visible "
          "in the ALL row.")


# --------------------------------------------------------------------------
# train


def train(train_root, eval_root, out, encoder="resnet18", path=None,
          epochs=20, bs=4, lr=3e-4, size=DEFAULT_SIZE, seed=0, device=None,
          rotate=False, eval_rot=None):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr_rows, tr_missing = load_manifest(train_root)
    ev_rows, ev_missing = load_manifest(eval_root)
    if not tr_rows:
        raise SystemExit(
            f"no annotated frames under {train_root}.\n"
            f"  {tr_missing} manifest rows have no mask in masks/. Masks are "
            f"the one thing\n  this cannot synthesise -- draw them, or run "
            f"--smoke to exercise the code.")
    print(f"  train {len(tr_rows)} frames ({tr_missing} unannotated), "
          f"eval {len(ev_rows)} ({ev_missing} unannotated)")
    if set(f"{r['recording']}/{r['frame']}" for r in tr_rows) & \
       set(f"{r['recording']}/{r['frame']}" for r in ev_rows):
        raise SystemExit("train and eval share frames. Refusing: a head "
                         "scored on what it\n  fitted reports a number about "
                         "nothing.")

    w = class_weights(tr_rows, size)
    print(f"  class weights {dict(zip(CLASSES, w.round(2).tolist()))}")
    model = build_model(encoder, path).to(device)
    if not model.pretrained:
        print("  !! encoder is NOT pretrained; scores describe the decoder.")
    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device),
                                ignore_index=IGNORE)
    tr = OwnershipDataset(tr_rows, size, augment=True, seed=seed,
                          rotate=rotate)
    ev = OwnershipDataset(ev_rows, size, augment=False)
    ev_rot = (OwnershipDataset(ev_rows, size, augment=False,
                               force_rot=eval_rot)
              if eval_rot else None)

    os.makedirs(out, exist_ok=True)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(tr))
        tot = n = 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xs, ds_, ys, _ = zip(*[tr[int(j)] for j in idx])
            x = torch.stack(xs).to(device)
            d = torch.stack(ds_).to(device)
            y = torch.stack(ys).to(device)
            opt.zero_grad()
            loss = lossf(model(x, d), y)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx); n += len(idx)
        line = {"epoch": ep, "loss": tot / max(n, 1)}
        if ev_rows and (ep % 5 == 0 or ep == epochs):
            res = evaluate(model, ev, device)
            line["eval"] = res
            print(f"  epoch {ep:3d}  loss {line['loss']:.4f}")
            print_eval(res)
        else:
            print(f"  epoch {ep:3d}  loss {line['loss']:.4f}")
        hist.append(line)
    if ev_rot is not None:
        print(f"\n  === the same frames, turned by {eval_rot*90} degrees "
              f"===")
        rres = evaluate(model, ev_rot, device)
        print_eval(rres)
        json.dump(rres, open(os.path.join(out, f"eval_rot{eval_rot}.json"),
                             "w"), indent=1)
        up = np.asarray(hist[-1].get("eval", {}).get("overall", [np.nan] * 3))
        ro = np.asarray(rres["overall"])
        print(f"\n    class            upright        rotated         drop")
        for c, u, r in zip(CLASSES, up, ro):
            print(f"    {c:<14}{u:>10.3f}{r:>15.3f}{u-r:>13.3f}")
        if eval_rot == 2:
            print("\n  The geometric rule scores ZERO here, and not "
                  "approximately: it calls a\n  hand the wearer's when the "
                  "forearm ray leaves above 0.55 of the height,\n  and a "
                  "half turn maps every exit height h to H-h. Every call "
                  "inverts. So\n  whatever the rotated column says, it is "
                  "the whole of what the segmenter\n  has that the rule does "
                  "not.")
        else:
            print("\n  The rule does not have a score here at all -- a "
                  "quarter turn makes its\n  quantity, the exit HEIGHT, "
                  "measure the horizontal axis. That is the case\n  the raw "
                  "module views actually present.")
    torch.save({"model": model.state_dict(), "encoder": encoder,
                "size": size, "classes": CLASSES}, os.path.join(out, "head.pt"))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    print(f"\n  wrote {out}/head.pt and history.json")
    return hist


# --------------------------------------------------------------------------
# smoke


def _synth(dirpath, n, size=(128, 128), seed=0, near_frac=0.5):
    """Frames with a rule the head must find: a skin blob touching the BOTTOM
    is the wearer's, one touching the TOP is somebody else's, and they are
    identical in colour. If the pipeline works, this is learnable; if the
    fusion or the loss is wired wrong, it is not."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = size
    for d in ("images", "range", "masks"):
        os.makedirs(os.path.join(dirpath, d), exist_ok=True)
    rows = []
    for i in range(n):
        rgb = np.full((H, W, 3), (60, 90, 45), np.uint8)
        mask = np.zeros((H, W), np.uint8)
        rng16 = np.zeros((H, W), np.uint16)
        # owner arm: a bar from the bottom edge upward
        x0 = int(rng.integers(10, W - 40)); w = int(rng.integers(18, 34))
        h = int(rng.integers(H // 2, H - 8))
        rgb[H - h:, x0:x0 + w] = (110, 150, 200)
        mask[H - h:, x0:x0 + w] = 1
        rng16[H - h:, x0:x0 + w] = 300
        other = i < int(n * near_frac)
        if other:
            x1 = int(rng.integers(10, W - 40)); w1 = int(rng.integers(18, 34))
            h1 = int(rng.integers(H // 3, H // 2))
            rgb[:h1, x1:x1 + w1] = (110, 150, 200)
            mask[:h1, x1:x1 + w1] = 2
            rng16[:h1, x1:x1 + w1] = 900
        rng16[mask == 0] = np.where(
            rng.random((H, W)) < 0.5, 1500, 0).astype(np.uint16)[mask == 0]
        stem = f"synth_f{i:06d}.png"
        cv2.imwrite(os.path.join(dirpath, "images", stem), rgb)
        cv2.imwrite(os.path.join(dirpath, "range", stem), rng16)
        cv2.imwrite(os.path.join(dirpath, "masks", stem), mask)
        rows.append({"recording": "synth", "frame": i, "split": "smoke",
                     "has_depth": 1,
                     "census_label": "other_near" if other else "owner_only",
                     "rgb": f"images/{stem}", "range": f"range/{stem}",
                     "note": ""})
    with open(os.path.join(dirpath, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    return rows


def smoke(tmp, encoder="resnet18", epochs=12):
    """Prove the path end to end before a mask is drawn by hand."""
    torch = _torch()
    tr_dir = os.path.join(tmp, "train"); ev_dir = os.path.join(tmp, "eval")
    _synth(tr_dir, 24, seed=1)
    _synth(ev_dir, 8, seed=2)
    # frame ids collide between the two synthetic sets by construction; make
    # the eval side distinct so the overlap guard is testing something real.
    import shutil
    rows = list(csv.DictReader(open(os.path.join(ev_dir, "manifest.csv"))))
    for r in rows:
        new = int(r["frame"]) + 1000
        for d in ("images", "range", "masks"):
            a = os.path.join(ev_dir, d, f"synth_f{int(r['frame']):06d}.png")
            b = os.path.join(ev_dir, d, f"synth_f{new:06d}.png")
            shutil.move(a, b)
        r["frame"] = new
    with open(os.path.join(ev_dir, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    print("  synthetic rule: a skin bar from the BOTTOM is the owner's, one "
          "from the TOP\n  is somebody else's, and they are the same colour. "
          "Only position separates\n  them, so this fails unless the whole "
          "path works.\n")
    hist = train(tr_dir, ev_dir, os.path.join(tmp, "run"), encoder=encoder,
                 epochs=epochs, bs=4, lr=1e-3, size=(128, 128), seed=0)
    first, last = hist[0]["loss"], hist[-1]["loss"]
    res = hist[-1].get("eval")
    ok = []

    def chk(c, m):
        ok.append(bool(c)); print(f"  {'ok ' if c else 'FAIL'} {m}")

    chk(last < first * 0.5, f"loss fell {first:.3f} -> {last:.3f}")
    chk(res is not None, "the eval ran and produced per-regime numbers")
    if res:
        near = res["by_regime"].get("other_near")
        chk(near is not None and np.isfinite(near[2]) and near[2] > 0.5,
            f"other_arm IoU on the hard regime is "
            f"{near[2]:.3f} (>0.5 means the fusion and loss are wired)")
        chk(np.isfinite(res["overall"][1]) and res["overall"][1] > 0.5,
            f"owner_arm IoU {res['overall'][1]:.3f}")
        chk(set(res["by_regime"]) >= {"other_near", "owner_only"},
            "regimes are reported separately, never as one figure")
        fp = res["fp_by_regime"].get("owner_only")
        chk(fp is not None and fp[2] < 0.02,
            f"on frames with nobody else, other_arm is hallucinated onto "
            f"{fp[2]:.1%}\n       of pixels -- the failure IoU cannot show "
            f"because the class is absent")
    print(f"\n  {sum(ok)}/{len(ok)} checks pass.")
    return all(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotate", action="store_true",
                    help="train with random 90-degree rotations, so ownership "
                         "cannot be read off frame orientation")
    ap.add_argument("--eval_rot", type=int, default=None, choices=(0, 1, 2, 3),
                    help="evaluate with every frame turned by this many "
                         "quarter turns. The geometric rule is exactly "
                         "inverted at 2; a segmenter that holds up here does "
                         "not depend on which way the render calls down.")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic end-to-end check; needs no annotations")
    ap.add_argument("--train_root")
    ap.add_argument("--eval_root")
    ap.add_argument("--out", default="/workspace/seg_runs/run1")
    ap.add_argument("--encoder", default="resnet18",
                    choices=("resnet18", "segformer_b0"))
    ap.add_argument("--encoder_path", help="local weights dir, for offline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.smoke:
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            raise SystemExit(0 if smoke(t, a.encoder) else 1)
    if not a.train_root or not a.eval_root:
        ap.error("--train_root and --eval_root are required without --smoke")
    train(a.train_root, a.eval_root, a.out, rotate=a.rotate,
          eval_rot=a.eval_rot, encoder=a.encoder,
          path=a.encoder_path, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed)


if __name__ == "__main__":
    main()
