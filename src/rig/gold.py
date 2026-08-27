"""Hand-corrected masks, and the three-way comparison they exist to enable.

Every number the head has produced so far is an agreement score with a
pseudo-label. `owner_arm IoU 0.876` means it reproduces GrabCut plus a
bottom-fraction rule, and says nothing about whether either of them finds the
wearer's arm. One small set of hand-drawn masks converts that into a real
measurement, and the measurement that matters is not the head's score alone
but the TRIANGLE:

        gold (a person)
         /          \\
    pseudo          head

    head closer to gold than pseudo is
        the head is denoising its labels; more pseudo-labels will help
    head no closer than pseudo
        the head is faithfully reproducing the labeller's mistakes, and the
        thing to fix is the labeller

Those two lead to opposite next steps, which is why the gold is worth the
hours it costs and why guessing is not.

HALF THE FRAMES ARE ANNOTATED BLIND AND HALF ARE SEEDED, ON PURPOSE. Handing
someone the pseudo-mask to correct is several times faster than drawing from
scratch, and it also anchors them: a plausible-looking wrong mask gets
accepted, the gold quietly inherits the pseudo-label's errors, and the
triangle above collapses into a tautology. So half the package ships with the
pseudo-mask as a starting point and half ships with an empty one, assigned by
a seed the annotator does not control. If the seeded half agrees with the
pseudo-labels markedly more than the blind half does, the anchoring is
MEASURED rather than assumed, and the blind half is the one to trust.

SELECTION USES NO MODEL SCORE. The census label and the recording decide, and
nothing else. A gold set chosen by where a model was uncertain measures the
model's own idea of difficulty; a blind, regime-stratified one measures the
task. That distinction has already cost this project one wasted eval split.
"""
from __future__ import annotations

import csv
import os
import shutil

import numpy as np

CLASSES = ("background", "owner_arm")

README = """\
GOLD MASKS -- the wearer's own hand and arm, nothing else

  0  everything else: bench, parts, tools, machines, OTHER PEOPLE
  1  the WEARER's own hand and forearm

Edit <stem>_start.png and save it in place. Values must be exactly 0 and 1.
_image.png is the frame. _overlay.jpg shows the current automatic guess for
reference only -- it is often wrong, and on the frames marked `blind` in
manifest.csv there is deliberately no starting mask.

WHERE THE ARM ENDS. Include the hand, the wrist and the bare forearm. Stop at
the sleeve cuff -- the sleeve is clothing and the automatic labeller cannot
see it either, so including it here would score the head against something it
was never given.

WHAT IS NOT CLASS 1. A colleague's hand, however close. Skin-coloured objects:
the wooden turntable, the beige machine strap, tan gloves lying on the bench.
Those are class 0. They are also the automatic labeller's most common mistake,
so they are the most valuable thing to get right here.

`regime` in manifest.csv may be blank for frames that were not in the original
census. Fill it in: owner_only / other_far / other_near / none / transit.
"""


def allocate_gold(rows, n_total, min_per_regime=3, seed=0):
    """Frames to hand-label, stratified by census regime. -> [row]

    Every regime present gets at least `min_per_regime` frames if it has them,
    and the remainder is shared in proportion. Proportional alone would give
    this corpus 71% other_far and leave the case that matters -- a frame with
    nobody else in it -- with two examples."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r.get("census_label", ""), []).append(r)
    for k in by:
        by[k] = [by[k][i] for i in rng.permutation(len(by[k]))]
    take = {k: min(min_per_regime, len(v)) for k, v in by.items()}
    left = n_total - sum(take.values())
    while left > 0:
        room = {k: len(v) - take[k] for k, v in by.items() if len(v) > take[k]}
        if not room:
            break
        tot = sum(len(by[k]) for k in room)
        for k in sorted(room, key=lambda k: -len(by[k])):
            add = min(room[k], max(1, int(round(left * len(by[k]) / tot))))
            take[k] += add
            left -= add
            if left <= 0:
                break
    out = []
    for k, v in by.items():
        out += v[:take[k]]
    return out


def assign_blind(rows, seed=0):
    """Half blind, half seeded, deterministically and not by the annotator."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(rows))
    blind = set(int(i) for i in idx[:len(rows) // 2])
    for j, r in enumerate(rows):
        r["seed_mode"] = "blind" if j in blind else "seeded"
    return rows


def build_package(src_root, rows, out_dir):
    """Copy images, starting masks and overlays into a package to hand over."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "README.txt"), "w").write(README)
    man = []
    for r in rows:
        stem = f"{r['recording']}_f{int(r['frame']):06d}"
        img = os.path.join(src_root, "images", stem + ".png")
        msk = os.path.join(src_root, "masks", stem + ".png")
        ovl = os.path.join(src_root, "overlays", stem + ".jpg")
        if not os.path.exists(img):
            continue
        shutil.copyfile(img, os.path.join(out_dir, stem + "_image.png"))
        m = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # Only class 1 survives into the gold task. Class 2 was measured to be
        # furniture on this corpus, and shipping it as a starting point would
        # ask the annotator to correct a category that should not exist yet.
        start = (m == 1).astype(np.uint8)
        if r["seed_mode"] == "blind":
            start = np.zeros_like(start)
        cv2.imwrite(os.path.join(out_dir, stem + "_start.png"), start)
        if os.path.exists(ovl):
            shutil.copyfile(ovl, os.path.join(out_dir, stem + "_overlay.jpg"))
        man.append({"recording": r["recording"], "frame": int(r["frame"]),
                    "regime": r.get("census_label", ""),
                    "seed_mode": r["seed_mode"], "stem": stem, "done": ""})
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(man)
    return man


def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def score(gold_dir, src_root, head_ckpt=None, device=None):
    """-> rows of (regime, seed_mode, pseudo_vs_gold, head_vs_gold, ...)."""
    import cv2
    rows = list(csv.DictReader(open(os.path.join(gold_dir, "manifest.csv"),
                                    encoding="utf-8-sig")))
    model = None
    if head_ckpt:
        from src.rig.seg_head import build_model, MEAN, STD, RANGE_SCALE
        import torch
        ck = torch.load(head_ckpt, map_location="cpu", weights_only=False)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(ck.get("encoder", "resnet18")).to(device).eval()
        model.load_state_dict(ck["model"])
        size = tuple(ck.get("size", (1024, 576)))

    out = []
    for r in rows:
        stem = r["stem"]
        g = cv2.imread(os.path.join(gold_dir, stem + "_start.png"),
                       cv2.IMREAD_GRAYSCALE)
        p = cv2.imread(os.path.join(src_root, "masks", stem + ".png"),
                       cv2.IMREAD_GRAYSCALE)
        if g is None or p is None:
            continue
        if not str(r.get("done", "")).strip():
            continue
        gold = g == 1
        pseudo = p == 1
        row = {"regime": r.get("regime", ""), "seed_mode": r["seed_mode"],
               "stem": stem, "gold_px": int(gold.sum()),
               "pseudo_vs_gold": iou_binary(pseudo, gold)}
        if model is not None:
            import torch
            img = cv2.imread(os.path.join(src_root, "images", stem + ".png"))
            rng16 = cv2.imread(os.path.join(src_root, "range", stem + ".png"),
                               cv2.IMREAD_UNCHANGED)
            W, H = size
            rs = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            if rng16 is None:
                rng16 = np.zeros(img.shape[:2], np.uint16)
            rr = cv2.resize(rng16, (W, H), interpolation=cv2.INTER_NEAREST)
            x = torch.from_numpy(
                np.ascontiguousarray(rs[:, :, ::-1]).astype(np.float32) / 255.
            ).permute(2, 0, 1)
            x = (x - torch.tensor(MEAN)[:, None, None]) \
                / torch.tensor(STD)[:, None, None]
            valid = (rr > 0).astype(np.float32)
            met = rr.astype(np.float32) / 1000.0
            d = torch.from_numpy(np.stack(
                [np.clip(met / RANGE_SCALE, 0, 4) * valid, valid], 0))
            with torch.no_grad():
                pr = model(x[None].to(device), d[None].to(device))
                pr = pr.argmax(1)[0].cpu().numpy().astype(np.uint8)
            pr = cv2.resize(pr, (gold.shape[1], gold.shape[0]),
                            interpolation=cv2.INTER_NEAREST) == 1
            row["head_vs_gold"] = iou_binary(pr, gold)
            row["head_vs_pseudo"] = iou_binary(pr, pseudo)
        out.append(row)
    return out


def report(rows):
    if not rows:
        print("  no completed gold frames yet (manifest 'done' column empty)")
        return
    def m(sel, k):
        v = [r[k] for r in sel if np.isfinite(r.get(k, np.nan))]
        return (np.mean(v), len(v)) if v else (float("nan"), 0)

    has_head = any("head_vs_gold" in r for r in rows)
    print(f"\n  {'group':<16}{'n':>4}{'pseudo/gold':>14}"
          + (f"{'head/gold':>12}{'head/pseudo':>14}" if has_head else ""))
    groups = [("ALL", rows)]
    for k in sorted({r["regime"] for r in rows}):
        groups.append((k or "(blank)", [r for r in rows if r["regime"] == k]))
    for k in ("blind", "seeded"):
        groups.append((k, [r for r in rows if r["seed_mode"] == k]))
    for name, sel in groups:
        pg, n = m(sel, "pseudo_vs_gold")
        line = f"  {name:<16}{n:>4}{pg:>14.3f}"
        if has_head:
            hg, _ = m(sel, "head_vs_gold")
            hp, _ = m(sel, "head_vs_pseudo")
            line += f"{hg:>12.3f}{hp:>14.3f}"
        print(line)

    b, _ = m([r for r in rows if r["seed_mode"] == "blind"], "pseudo_vs_gold")
    s, _ = m([r for r in rows if r["seed_mode"] == "seeded"], "pseudo_vs_gold")
    if np.isfinite(b) and np.isfinite(s):
        print(f"\n  ANCHORING  seeded {s:.3f} vs blind {b:.3f} "
              f"(gap {s-b:+.3f})")
        print("     A large positive gap means the starting mask pulled the "
              "annotator toward\n     it, and the BLIND rows are the ones to "
              "quote.")
    if has_head:
        hg, _ = m(rows, "head_vs_gold")
        pg, _ = m(rows, "pseudo_vs_gold")
        print(f"\n  THE TRIANGLE  head/gold {hg:.3f}  vs  pseudo/gold "
              f"{pg:.3f}")
        if np.isfinite(hg) and np.isfinite(pg):
            print("     " + ("the head is CLOSER to truth than its own "
                             "labels -- it is denoising,\n     and more "
                             "pseudo-labels are worth generating."
                             if hg > pg + 0.02 else
                             "the head is NOT closer to truth than its "
                             "labels -- it reproduces the\n     labeller, "
                             "and the labeller is what to fix."))


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("select", "score"))
    ap.add_argument("--src", required=True,
                    help="a seg_auto split directory (images/ masks/ "
                         "overlays/ manifest.csv) -- use the EVAL split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--min_per_regime", type=int, default=3)
    ap.add_argument("--head")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.mode == "select":
        rows = list(csv.DictReader(
            open(os.path.join(a.src, "manifest.csv"), encoding="utf-8-sig")))
        if not rows:
            raise SystemExit(f"{a.src}/manifest.csv is empty")
        from collections import Counter
        print(f"{len(rows)} candidate frames, regimes "
              f"{dict(Counter(r.get('census_label','') for r in rows))}")
        if a.n > len(rows):
            print(f"  !! asked for {a.n} but only {len(rows)} exist in this "
                  f"split.\n     Taking all of them. To go bigger the "
                  f"held-out recordings need more\n     frames rendered, and "
                  f"they must still be chosen without a model score.")
        sel = allocate_gold(rows, min(a.n, len(rows)), a.min_per_regime, a.seed)
        sel = assign_blind(sel, a.seed)
        man = build_package(a.src, sel, a.out)
        print(f"\n  {len(man)} frames -> {a.out}")
        print(f"    regimes {dict(Counter(r['regime'] for r in man))}")
        print(f"    {sum(1 for r in man if r['seed_mode']=='blind')} blind, "
              f"{sum(1 for r in man if r['seed_mode']=='seeded')} seeded")
        print("\n  Edit <stem>_start.png in place, values 0 and 1, then put "
              "any mark in the\n  'done' column of manifest.csv. Half have an "
              "empty starting mask on purpose:\n  a correction task anchors, "
              "and this measures how much rather than hoping.")
    else:
        rows = score(a.out, a.src, a.head)
        report(rows)


if __name__ == "__main__":
    main()
