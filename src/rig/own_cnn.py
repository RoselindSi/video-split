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
                "frame": int(m.group(2)),
                "y": 1 if r["label"] == "owner" else 0,
                # A crop labelled through the sheet may carry no geometry: the
                # sweep writes hands.csv only when it finishes, so a package
                # labelled from crops alone has rows with these columns empty.
                # `get` defaults do not fire on an empty STRING, and an empty
                # `rule` must not become 0 -- that would be recording a verdict
                # of `other` from a rule that never saw the hand, and the rule
                # is the baseline the model is being compared against.
                "exit_y": _num(r.get("exit_y"), float),
                "rule": _num(r.get("rule_owner"), int),
                "pkg": os.path.basename(p)})
    if verbose and missing:
        print(f"  !! {len(missing)} package(s) skipped: {missing}")
    return rows


def _num(v, cast):
    """-> cast(v), or None when the cell is empty or unparseable."""
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def has_geometry(r):
    """-> True if the rule can be asked about this hand at all."""
    return r.get("rule") is not None and r.get("exit_y") is not None


def rule_verdict(rows, k):
    """The rule's owner/other call after k quarter turns. -> np.array

    Only k=0 and k=2 are defined: a quarter turn swaps the axes, so the
    quantity the rule reads -- a height -- is no longer a height at all, and
    reporting a number there would be inventing one.

    Rows with no forearm exit get None rather than a guess. They are perfectly
    good training data -- a crop and a human's verdict -- but the rule was
    never shown them, and scoring it on hands it never saw would understate
    the incumbent this model has to beat."""
    if k % 4 == 0:
        return np.array([r["rule"] if has_geometry(r) else None
                         for r in rows], dtype=object)
    if k % 4 == 2:
        return np.array([(1 if (1.0 - r["exit_y"]) >= RULE_FRAC else 0)
                         if has_geometry(r) else None
                         for r in rows], dtype=object)
    return None


def split_by_tag(rows, holdout):
    tr = [r for r in rows if r["tag"] not in holdout]
    ev = [r for r in rows if r["tag"] in holdout]
    return tr, ev


def choose_holdout(rows, want_other=8, min_rec_hands=5, verbose=True):
    """Pick held-out recordings that contain `other`, smallest first.

    An eval split with no `other` in it cannot score the class the whole
    feature exists for -- the first segmentation run held out two recordings
    that were 100% owner and reported a dash. Smallest-first because `other`
    is concentrated in a few recordings and spending the largest of them on
    the eval would empty the training set of the class.

    `min_rec_hands` EXISTS BECAUSE SMALLEST-FIRST STOPPED BEING SAFE. It was
    written when four recordings held every `other` there was. With twenty it
    picks the scraps: one run held out seven recordings of which one had a
    single hand and two had two, then reported a per-recording accuracy of
    1.000 on a sample of one. A recording too small to carry a rate should not
    be spent as though it could."""
    import collections
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["tag"]][r["y"]] += 1
    have = sorted(((n_oth, tag) for tag, (n_oth, n_own) in by.items()
                   if n_oth > 0 and n_oth + n_own >= min_rec_hands))
    if not have:                      # nothing is big enough; take what exists
        have = sorted(((n_oth, tag) for tag, (n_oth, _) in by.items()
                       if n_oth > 0))
    hold, got = [], 0
    for n_oth, tag in have:
        hold.append(tag)
        got += n_oth
        if got >= want_other:
            break
    if verbose:
        n_h = sum(sum(by[t]) for t in hold)
        print(f"  holdout {hold}  ({got} other, {n_h} hands)")
        big = max(((sum(by[t]), t) for t in hold), default=(0, ""))
        if n_h and big[0] > 0.5 * n_h:
            print(f"    !! {big[1]} alone is {big[0]}/{n_h} of the held-out "
                  f"hands. Read the per-recording\n       rows below before "
                  f"the aggregate: one recording can carry the whole number.")
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


def load_model(path, device=None):
    """-> (model, device) ready for inference, or (None, None) if absent.

    A missing checkpoint is not an error. The geometric rule is the fallback,
    and it is the incumbent being replaced rather than a stub."""
    torch = _torch()
    if not path or not os.path.exists(path):
        return None, None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    m = build_model().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, device


def crop_of(rgb, det, pad=0.6):
    """The same crop the labeller saw. -> BGR image

    `pad = 0.6 * max(w, h)` is not a tunable: it is what `own_label` wrote to
    disk, so every training example has exactly this much forearm around the
    hand. Inference on a tighter or looser crop would be a different input
    distribution from the one the weights were fitted to."""
    import cv2
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = det["box"]
    p = int(max(x1 - x0, y1 - y0) * pad)
    cx0, cy0 = max(0, x0 - p), max(0, y0 - p)
    cx1, cy1 = min(W, x1 + p), min(H, y1 + p)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    return cv2.resize(rgb[cy0:cy1, cx0:cx1], (SIZE, SIZE))


def predict(model, device, rgb, dets):
    """-> [(is_owner, p_owner)] one per detection, in order."""
    import cv2
    torch = _torch()
    if not dets:
        return []
    xs, keep = [], []
    for i, d in enumerate(dets):
        c = crop_of(rgb, d)
        if c is None:
            continue
        x = torch.from_numpy(
            np.ascontiguousarray(cv2.resize(c, (SIZE, SIZE))[:, :, ::-1]
                                 ).astype(np.float32) / 255.0).permute(2, 0, 1)
        xs.append((x - 0.45) / 0.25)
        keep.append(i)
    out = [(True, 1.0)] * len(dets)
    if not xs:
        return out
    with torch.no_grad():
        pr = torch.softmax(model(torch.stack(xs).to(device)), 1)[:, 1]
    for i, p in zip(keep, pr.cpu().numpy().tolist()):
        out[i] = (p >= 0.5, float(p))
    return out


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


def check(rows, path, verbose=True):
    """Run a saved model over crops that already carry a label. -> misses

    A demo shows a decision the eye can read but not one it can check: a frame
    where nothing is blurred looks the same whether the classifier cleared it
    or missed everything in it. An empty `other` set is the failure that hides
    best, because it renders as a clean frame.

    So this reads the shipped checkpoint over the same crops the labeller saw,
    upright and unaugmented, and prints the frame numbers where a hand a
    person called foreign was called the wearer's. Those numbers index the
    video: a missing blur can be laid at the classifier's door or taken off
    it."""
    model, device = load_model(path)
    if model is None:
        raise SystemExit(f"no checkpoint at {path}")
    pred, true = evaluate(model, Crops(rows, augment=False), device)
    res = {"all": scores(pred, true)}
    by = {}
    for r, p in zip(rows, pred):
        by.setdefault(r["tag"], []).append((p, r["y"]))
    for t, v in sorted(by.items()):
        if any(y == 0 for _, y in v):
            res[t] = scores([p for p, _ in v], [y for _, y in v])
    if verbose:
        report(res)
    miss = [r for r, p in zip(rows, pred) if r["y"] == 0 and p == 1]
    if verbose:
        print(f"\n  {len(miss)} of {sum(1 for r in rows if r['y']==0)} "
              f"`other` hands were called the wearer's.")
        seen = {}
        for r in miss:
            seen.setdefault(r["tag"], []).append(r["frame"])
        for t, f in sorted(seen.items()):
            print(f"    {t:<12} frames {sorted(set(f))}")
        print("\n  A frame listed here renders with no red box and nothing "
              "blurred, which is\n  indistinguishable on screen from a frame "
              "that never had a colleague in it.\n  Only recordings that "
              "contain `other` are broken out: the rest have no\n  positive "
              "to recall and their accuracy is the always-owner number.")
    return miss


def train(rows, holdout, epochs=30, bs=32, lr=1e-3, seed=0, rotate=True,
          device=None, out=None, class_weight="auto"):
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
    # INVERSE FREQUENCY AND ENRICHED SAMPLING DO THE SAME JOB, AND DOING BOTH
    # COUNTS IT TWICE. The weight was written when `other` was 3% of the
    # labels and the sample was whatever the sweep found. Labelling is now
    # ordered by the model's own p(other), which lifted the training set to
    # around 13% `other` against a deployment rate near 3%. Weighting the
    # already-enriched set by its own inverse frequency pushed the effective
    # prior to roughly 36% -- eleven times reality -- and a model that expects
    # a third of hands to be foreign calls a third of them foreign. That is
    # the shape of the first result off this data: recall 0.625, precision
    # 0.250. `auto` keeps the old behaviour so earlier runs stay reproducible;
    # `none` lets the enrichment be the only balancing, which is what an
    # actively sampled set wants.
    if class_weight in (None, "none"):
        w = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    elif class_weight == "auto":
        w = torch.tensor([len(tr_rows) / max(n_oth, 1) / 2.0, 1.0],
                         dtype=torch.float32, device=device).clamp(max=20.0)
    else:
        w = torch.tensor([float(class_weight), 1.0],
                         dtype=torch.float32, device=device)
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
        # Scored only where the rule has something to read. The `n` column
        # therefore differs between the cnn and rule rows, which is the honest
        # presentation: they were not asked the same number of questions.
        rv = rule_verdict(ev_rows, k)
        keep = [i for i, r in enumerate(ev_rows) if has_geometry(r)]
        if keep:
            results[f"rule {lab}"] = scores(
                np.array([int(rv[i]) for i in keep]),
                np.array([ev_rows[i]["y"] for i in keep]))

    # PER RECORDING, AND THE REASON IS A COMPETING EXPLANATION. Two of the
    # held-out recordings are 100% `other`, so a model that only recognises
    # "this does not look like anything I trained on" would score perfectly on
    # them without ever deciding ownership. The recording that contains BOTH
    # classes is the one where novelty cannot help, and it is the only row
    # that distinguishes the two accounts.
    tags = sorted({r["tag"] for r in ev_rows})
    p0, _ = evaluate(model, Crops(ev_rows, force_rot=0), device)
    per = {}
    for t in tags:
        idx = [i for i, r in enumerate(ev_rows) if r["tag"] == t]
        yy = np.array([ev_rows[i]["y"] for i in idx])
        mixed = "" if len(set(yy.tolist())) > 1 else "   (single class)"
        per[f"  {t} upright{mixed}"] = scores(p0[idx], yy)
    results.update(per)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "size": SIZE}, out)
        print(f"\n  wrote {out}")
    return results


def _self_test():
    ok = n = 0

    def chk(name, cond):
        nonlocal ok, n
        n += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok += bool(cond)

    rows = [{"tag": "A_", "y": 1, "exit_y": 0.90, "rule": 1},
            {"tag": "A_", "y": 0, "exit_y": 0.20, "rule": 0},
            {"tag": "B_", "y": 1, "exit_y": 0.70, "rule": 1},
            {"tag": "C_", "y": 1, "exit_y": 0.60, "rule": 1}]
    chk("upright verdict is the stored one",
        list(rule_verdict(rows, 0)) == [1, 0, 1, 1])
    # A crop labelled from the sheet alone has no forearm exit. It is training
    # data all the same, but the rule must not be credited or blamed for it.
    blind = rows + [{"tag": "D_", "y": 0, "exit_y": None, "rule": None}]
    chk("a hand with no geometry gets no verdict rather than a default",
        list(rule_verdict(blind, 0))[-1] is None
        and list(rule_verdict(blind, 2))[-1] is None)
    chk("empty csv cells parse to None, not to a crash or a zero",
        _num("", int) is None and _num(None, float) is None
        and _num("0", int) == 0)
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
    print(f"\n  {ok}/{n}")
    return ok == n



def leave_one_out(rows, epochs=30, seed=0, min_other=5, verbose=True):
    """Hold out each recording in turn. -> [(tag, prec, rec, n, n_other)]

    WHY A DISTRIBUTION AND NOT A NUMBER. Every cross-recording figure this
    project has quoted came from one split, and the splits disagree violently:
    holding out six recordings gave f1 0.519, holding out three gave 0.568,
    and inside those the per-recording recall ran from 0.158 to 1.000. A
    single split of a 43-recording corpus estimates the mean of that spread
    from three or six draws, which is why the number moved every time the
    split did. Held out one at a time, every recording contributes, and the
    spread itself becomes the result.

    THE SPREAD IS THE FINDING, NOT NOISE AROUND ONE. A model that works on two
    thirds of workstations and fails on the rest is a different object from one
    that is mediocre everywhere, and they have the same mean. Only the first
    can be shipped behind a check that detects the bad third.

    Recordings with fewer than `min_other` foreign hands are skipped rather
    than scored: a recall computed on two positives is a coin toss reported to
    three decimals."""
    import collections
    import time
    by = collections.Counter(r["tag"] for r in rows if r["y"] == 0)
    tags = sorted(t for t, n in by.items() if n >= min_other)
    if verbose:
        print(f"  {len(tags)} recordings carry at least {min_other} `other`; "
              f"training {len(tags)} models")
    out, t0 = [], time.time()
    for i, t in enumerate(tags, 1):
        res = train(rows, {t}, epochs=epochs, seed=seed, verbose=False)
        sc = res.get("cnn upright") or {}
        out.append((t, sc.get("other_prec", float("nan")),
                    sc.get("other_rec", float("nan")),
                    sc.get("n", 0), sc.get("n_other", 0)))
        if verbose:
            el = time.time() - t0
            print(f"    [{i}/{len(tags)}] {t:<16} "
                  f"prec {out[-1][1]:.3f}  rec {out[-1][2]:.3f}  "
                  f"({out[-1][4]} other)   "
                  f"{el:.0f}s, {el / i * (len(tags) - i):.0f}s left",
                  flush=True)
    return out


def report_loo(res, verbose=True):
    """Print the distribution, not the mean alone. -> dict"""
    rec = np.array([r[2] for r in res], float)
    prec = np.array([r[1] for r in res], float)
    ok = np.isfinite(rec)
    nan = float("nan")
    q = {"n_recordings": int(ok.sum()),
         "rec_median": float(np.median(rec[ok])) if ok.any() else nan,
         "rec_q1": float(np.percentile(rec[ok], 25)) if ok.any() else nan,
         "rec_q3": float(np.percentile(rec[ok], 75)) if ok.any() else nan,
         "rec_min": float(rec[ok].min()) if ok.any() else nan,
         "rec_max": float(rec[ok].max()) if ok.any() else nan,
         "prec_median": float(np.nanmedian(prec)) if len(prec) else nan,
         "frac_below_half": (float((rec[ok] < 0.5).mean())
                             if ok.any() else nan)}
    if verbose:
        head = f"{'recording':<18}{'other':>7}{'prec':>9}{'rec':>9}"
        print()
        print("    " + head)
        for t, p, r, n, no in sorted(
                res, key=lambda x: (x[2] if np.isfinite(x[2]) else -1)):
            ps = f"{p:>9.3f}" if np.isfinite(p) else f"{'-':>9}"
            rs = f"{r:>9.3f}" if np.isfinite(r) else f"{'-':>9}"
            print(f"    {t:<18}{no:>7}{ps}{rs}")
        print()
        print(f"  recall over {q['n_recordings']} held-out recordings: "
              f"median {q['rec_median']:.3f}  "
              f"IQR [{q['rec_q1']:.3f}, {q['rec_q3']:.3f}]  "
              f"range [{q['rec_min']:.3f}, {q['rec_max']:.3f}]")
        print(f"  precision median {q['prec_median']:.3f}")
        print(f"  {q['frac_below_half']:.0%} of recordings fall below "
              f"0.5 recall")
        print()
        print("  Read the IQR, not the median. One split of this corpus "
              "returns a number")
        print("  anywhere inside that range depending on which recordings it "
              "happened to hold")
        print("  out, which is why the earlier figures moved every time the "
              "split did.")
    return q


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
    ap.add_argument("--check", help="score a saved checkpoint on these "
                                    "packages instead of training")
    ap.add_argument("--class_weight", default="auto",
                    help="`auto` weights `other` by inverse frequency, which "
                         "double-counts an actively sampled set; `none` lets "
                         "the enrichment do the balancing; a number sets it "
                         "directly.")
    ap.add_argument("--loo", action="store_true",
                    help="hold out each recording in turn and report the "
                         "DISTRIBUTION of per-recording precision and recall. "
                         "One split cannot estimate this: the spread between "
                         "recordings is wider than any difference between "
                         "models measured so far.")
    ap.add_argument("--min_rec_hands", type=int, default=5,
                    help="recordings smaller than this are not spent on the "
                         "holdout")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    rows = load(a.pkg)
    if not rows:
        raise SystemExit("no labelled hands with crops")
    import collections
    c = collections.Counter(r["tag"] for r in rows)
    print(f"{len(rows)} hands with crops over {len(c)} recordings, "
          f"{sum(1 for r in rows if r['y']==0)} other")
    if a.loo:
        report_loo(leave_one_out(rows, epochs=a.epochs, seed=a.seed))
        raise SystemExit(0)
    if a.check:
        check(rows, a.check)
        raise SystemExit(0)
    hold = set(a.holdout) if a.holdout else choose_holdout(
        rows, min_rec_hands=a.min_rec_hands)
    res = train(rows, hold, epochs=a.epochs, seed=a.seed,
                rotate=not a.no_rotate, out=a.out,
                class_weight=a.class_weight)
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
