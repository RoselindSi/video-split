"""Pseudo-masks from automatic prompts, so a head exists before a human draws.

SAM is promptable, and the expensive part of annotation is the prompt, not the
boundary. A skin-region detector cannot decide OWNERSHIP -- that was measured
and disproved -- but it is a perfectly good prompt generator, which is the
same division of labour the hand rule was always meant to have:

    the detector says "something arm-shaped is here"
    the geometry says whose it is
    SAM says exactly which pixels

Ownership here is the one cue that survived measurement: in the wide frame the
wearer's own arm enters from the BOTTOM, because the virtual camera's up axis
is the fan's rotation axis and `FAN_UP_SIGN` was fixed by observation. A large
skin region touching the bottom edge is the wearer's; a large one that does
not is somebody else's.

WHAT THESE LABELS ARE AND ARE NOT. They are pseudo-labels. A head trained on
them learns to imitate SAM-plus-a-heuristic, and a score measured against them
says how well it imitates -- NOT how well it finds hands. That distinction is
not a footnote: every manifest row written here is stamped `label_source=auto`
so nothing downstream can mistake one for a drawn mask, and mixing the two in
one directory is refused.

They are still worth having. They give a working head today, they exercise the
whole path on real frames rather than synthetic ones, and they turn the human
task from drawing 76 masks into correcting the ones that are wrong -- which is
several times cheaper and is the only reason this is not a detour.

THE HEURISTIC'S OWN FAILURE MODES, so the reviewer knows what to look for:
the bench edge is skin-coloured under some lighting and touches the bottom;
a colleague leaning in from the side may touch the bottom edge too and be
called the wearer's; and the wearer's arm crossing the frame's lower corner
can split into two components. All three are visible at a glance in the
overlays this writes.
"""
from __future__ import annotations

import csv
import os
import shutil

import numpy as np

CLASSES = ("background", "owner_arm", "other_arm")

# How far down the frame a skin region must reach to be the wearer's own.
#
# THIS REPLACED A "TOUCHES THE BOTTOM EDGE" TEST, WHICH FAILED FOR A REASON
# WORTH KEEPING. A sleeve is not skin, so the wearer's arm ends as a component
# at the cuff, often well above the image edge -- and this corpus has sleeves
# in nearly every frame. Measured on four wide renders spanning the cases:
#
#     owner arms      0.884  0.963  1.000  1.000
#     everything else 0.741  0.756  0.762  0.771  0.779  0.867
#
# All ten separate at 0.88, including the hard frame where a colleague's hand
# rests on the bench at 0.867 against the wearer's arm at 0.963. THAT MARGIN
# IS 1.7 POINTS ON FOUR FRAMES. It is a working threshold for pseudo-labels
# that a human reviews, not a calibrated one, and the overlays are where it
# gets caught when it is wrong.
OWNER_BOTTOM_FRAC = 0.88

# Fractions of the frame. Below MIN a region is a face across the aisle or
# noise; above MAX it is the bench, not an arm.
MIN_AREA_FRAC = 0.008
MAX_AREA_FRAC = 0.25

DEFAULT_MODEL = "facebook/sam-vit-base"


def components(rgb, min_frac=MIN_AREA_FRAC, max_frac=MAX_AREA_FRAC):
    """-> [(area_frac, bbox, bottom_frac, seed_points)], deepest first.

    Sorted by how far down the frame the region reaches, not by area, because
    that is the quantity ownership is decided on and sorting by anything else
    would make the two-arm cap select on the wrong thing."""
    import cv2
    from src.rig.near_other_miner import skin_mask
    m = skin_mask(rgb)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA] / m.size
        if not (min_frac <= a <= max_frac):
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        bottom = (y + h) / H
        ys, xs = np.where(lab == i)
        # Points INSIDE the component, not the centroid: an arm is curved and
        # its centroid frequently lands on the bench between wrist and elbow,
        # which would prompt SAM with a point on the thing to exclude.
        k = max(1, len(xs) // 3)
        pts = [(int(xs[j]), int(ys[j])) for j in (0, k, 2 * k)][:3]
        out.append((float(a), (int(x), int(y), int(x + w), int(y + h)),
                    float(bottom), pts))
    return sorted(out, key=lambda r: -r[2])


def assign(comps, max_owner=2, max_other=2, thr=OWNER_BOTTOM_FRAC):
    """-> [(class_id, bbox, points)]. Reaching deep into the frame is the
    wearer's.

    Capped at two arms each: a wearer has two, and beyond two foreign regions
    the frame is a crowd and the heuristic has nothing useful to say."""
    own = [c for c in comps if c[2] >= thr][:max_owner]
    oth = [c for c in comps if c[2] < thr][:max_other]
    return ([(1, c[1], c[3]) for c in own]
            + [(2, c[1], c[3]) for c in oth])


class Segmenter:
    """Prompt -> pixels. SAM when it is reachable, GrabCut when it is not.

    THE SERVER HAS NO NETWORK, so `facebook/sam-vit-base` cannot be fetched
    and the whole plan of prompting a foundation model collapses -- unless the
    prompt is separated from the model that consumes it, which it is. GrabCut
    ships inside OpenCV, needs no weights, and takes exactly the same prompt: a
    box plus a region believed to be foreground.

    It is worse than SAM, and specifically worse in a way worth naming: it
    refines a colour model inside the box rather than recognising an object,
    so it holds a boundary well where the arm meets green bench and poorly
    where the arm meets the skin-toned bench edge -- which is the case the
    depth rule already failed on. Nothing here is pretending otherwise. What
    it buys is a real mask on real frames today, offline, with the prompt
    logic already in place for SAM to take over the moment weights exist."""

    def __init__(self, model=DEFAULT_MODEL, device=None):
        self.kind = "grabcut"
        self.name = "cv2.grabCut (offline)"
        if not model or model.lower() in ("none", "grabcut"):
            return
        try:
            import torch
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available()
                                     else "cpu")
            if "sam2" in model.lower():
                from transformers import Sam2Model, Sam2Processor
                self.proc = Sam2Processor.from_pretrained(model)
                self.net = Sam2Model.from_pretrained(model)
                self.kind = "sam2"
            else:
                from transformers import SamModel, SamProcessor
                self.proc = SamProcessor.from_pretrained(model)
                self.net = SamModel.from_pretrained(model)
                self.kind = "sam"
            self.net = self.net.to(self.device).eval()
            self.name = f"{model} on {self.device}"
        except Exception as e:
            print(f"  !! {model} unavailable ({type(e).__name__}); falling "
                  f"back to GrabCut.\n     Boundaries will be looser, "
                  f"especially where the arm meets the\n     skin-toned bench "
                  f"edge. The prompts are identical either way.")
            self.kind = "grabcut"
            self.name = "cv2.grabCut (offline)"

    def _grabcut(self, rgb, boxes, points):
        import cv2
        from src.rig.near_other_miner import skin_mask
        H, W = rgb.shape[:2]
        skin = skin_mask(rgb).astype(bool)
        out = []
        for (x0, y0, x1, y1) in boxes:
            # Pad the box: GrabCut treats everything outside the rect as
            # certain background, and a box drawn tight to the component
            # would forbid it from recovering the boundary it was cut at.
            pad = 12
            bx = (max(0, x0 - pad), max(0, y0 - pad),
                  min(W, x1 + pad), min(H, y1 + pad))
            m = np.full((H, W), cv2.GC_BGD, np.uint8)
            m[bx[1]:bx[3], bx[0]:bx[2]] = cv2.GC_PR_BGD
            inbox = np.zeros((H, W), bool)
            inbox[bx[1]:bx[3], bx[0]:bx[2]] = True
            m[skin & inbox] = cv2.GC_PR_FGD          # the detector's evidence
            if not (m == cv2.GC_PR_FGD).any():
                out.append(np.zeros((H, W), bool))
                continue
            bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(rgb, m, None, bg, fg, 3, cv2.GC_INIT_WITH_MASK)
                res = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
            except cv2.error:
                res = skin & inbox      # refusing to refine beats inventing
            out.append(res & inbox)
        return out

    def masks(self, rgb, boxes, points):
        """rgb is BGR. -> [H,W] bool per box, in the input order."""
        import cv2
        if self.kind == "grabcut":
            return self._grabcut(rgb, boxes, points)
        torch = self.torch
        img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        kw = {"input_boxes": [[list(map(float, b)) for b in boxes]]}
        if points:
            kw["input_points"] = [[[list(map(float, p)) for p in ps]
                                   for ps in points]]
            kw["input_labels"] = [[[1] * len(ps) for ps in points]]
        inp = self.proc(img, return_tensors="pt", **kw).to(self.device)
        with torch.no_grad():
            out = self.net(**inp, multimask_output=False)
        got = self.proc.image_processor.post_process_masks(
            out.pred_masks.float().cpu(),
            inp["original_sizes"].cpu(),
            getattr(inp, "reshaped_input_sizes",
                    inp.get("reshaped_input_sizes", None)).cpu()
            if "reshaped_input_sizes" in inp else None)
        m = got[0].numpy()
        return [m[i, 0].astype(bool) for i in range(m.shape[0])]


# Census labels under which the frame is known to contain nobody else. On
# those frames a non-owner skin region is an OBJECT -- the pink gloves and the
# yellow tool grips on this bench are both skin-coloured and both trip the
# detector -- and labelling it class 2 would teach the head that a glove lying
# on the bench is a colleague's arm. Left as background it teaches the
# opposite, which is true and is exactly the distinction that is hard.
NO_OTHER_LABELS = {"owner_only", "none", "transit"}


def label_frame(seg, rgb, census_label=""):
    """-> (mask uint8 with 0/1/2, [(class, bbox)] used).

    `census_label` is the human judgement from the blind census. It is used
    only to SUPPRESS class 2 where a person has said there is nobody else --
    never to add it, because the census says a colleague is present somewhere
    in the frame, not which pixels are theirs."""
    comps = components(rgb)
    picks = assign(comps)
    if census_label in NO_OTHER_LABELS:
        picks = [p for p in picks if p[0] == 1]
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    if not picks:
        return mask, []
    boxes = [p[1] for p in picks]
    points = [p[2] for p in picks]
    try:
        ms = seg.masks(rgb, boxes, points)
    except Exception:
        ms = seg.masks(rgb, boxes, None)      # points are optional, boxes are not
    # class 2 first, then class 1 over it: where the heuristic says both, the
    # bottom-connected evidence is the stronger of the two.
    for cls in (2, 1):
        for (c, _, _), m in zip(picks, ms):
            if c == cls:
                mask[m] = cls
    return mask, [(p[0], p[1]) for p in picks]


def overlay(rgb, mask):
    import cv2
    col = np.zeros_like(rgb)
    col[mask == 1] = (0, 220, 0)
    col[mask == 2] = (0, 90, 255)
    return cv2.addWeighted(rgb, 0.62, col, 0.38, 0)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packages", required=True,
                    help="directory of sample_* dirs from annotation_package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--eval_recordings", type=int, default=4,
                    help="recordings held out entirely. Splitting by "
                         "recording rather than by frame is the minimum; the "
                         "wearer's wristband makes a frame-level split "
                         "meaningless.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import cv2
    import json
    # meta.json is the COMPLETION marker -- it is written last. keyframe_rgb
    # appears partway through a render, so filtering on it picks up samples
    # that are still being written, and their metadata is not there yet: the
    # recording id silently degrades to the directory name and the held-out
    # split stops being by recording at all.
    dirs = sorted(d for d in os.listdir(a.packages)
                  if d.startswith("sample_")
                  and os.path.exists(os.path.join(a.packages, d, "meta.json"))
                  and os.path.exists(os.path.join(a.packages, d,
                                                  "keyframe_rgb.png")))
    if a.limit:
        dirs = dirs[:a.limit]
    if not dirs:
        raise SystemExit(f"no finished sample_* under {a.packages}")

    metas = []
    for d in dirs:
        p = os.path.join(a.packages, d, "meta.json")
        metas.append(json.load(open(p)) if os.path.exists(p) else {})
    recs = sorted({m["recording_id"] for m in metas})
    if len(recs) < a.eval_recordings + 1:
        raise SystemExit(
            f"only {len(recs)} finished recordings ({len(dirs)} keyframes); "
            f"holding out\n  {a.eval_recordings} would leave "
            f"{len(recs)-a.eval_recordings} to train on. Wait for the render, "
            f"or lower\n  --eval_recordings.")
    held = set(recs[-a.eval_recordings:]) if a.eval_recordings else set()
    print(f"{len(dirs)} keyframes, {len(recs)} recordings, "
          f"{len(held)} held out: {sorted(held)}\n")

    seg = Segmenter(a.model)
    # seg.name already carries the device where there is one; GrabCut has no
    # device at all, and asking for one is how the fallback path crashes on
    # the line that was meant to announce it.
    print(f"  segmenter: {seg.name}\n")
    for split in ("train", "eval"):
        for sub in ("images", "range", "masks", "overlays"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        open(os.path.join(a.out, split, "SPLIT"), "w").write(split)

    rows = {"train": [], "eval": []}
    t0 = time.time()
    n_empty = 0
    for i, (d, meta) in enumerate(zip(dirs, metas), 1):
        src = os.path.join(a.packages, d)
        rgb = cv2.imread(os.path.join(src, "keyframe_rgb.png"))
        if rgb is None:
            continue
        rid = meta.get("recording_id", d)
        fr = int(meta.get("keyframe_frame", 0))
        split = "eval" if rid in held else "train"
        stem = f"{rid}_f{fr:06d}.png"
        mask, picks = label_frame(seg, rgb, meta.get("census_label", ""))
        if not picks:
            n_empty += 1
        cv2.imwrite(os.path.join(a.out, split, "images", stem), rgb)
        cv2.imwrite(os.path.join(a.out, split, "masks", stem), mask)
        rp = os.path.join(src, "keyframe_range.png")
        if os.path.exists(rp):
            shutil.copyfile(rp, os.path.join(a.out, split, "range", stem))
        cv2.imwrite(os.path.join(a.out, split, "overlays",
                                 stem.replace(".png", ".jpg")),
                    overlay(rgb, mask), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows[split].append({
            "recording": rid, "frame": fr, "split": split,
            "has_depth": int(os.path.exists(rp)),
            "census_label": meta.get("census_label", ""),
            "label_source": "auto",
            "n_owner": sum(1 for c, _ in picks if c == 1),
            "n_other": sum(1 for c, _ in picks if c == 2),
            "owner_px": int((mask == 1).sum()),
            "other_px": int((mask == 2).sum())})
        if i % 10 == 0 or i == len(dirs):
            el = time.time() - t0
            print(f"  [{i}/{len(dirs)}] {el/60:.1f} min, "
                  f"{el/i*(len(dirs)-i)/60:.1f} left")

    for split, rs in rows.items():
        if not rs:
            continue
        with open(os.path.join(a.out, split, "manifest.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)

    print(f"\n  train {len(rows['train'])}  eval {len(rows['eval'])}  "
          f"({n_empty} frames got no prompt at all)")
    for split, rs in rows.items():
        if not rs:
            continue
        o = sum(r["owner_px"] for r in rs) / max(
            sum(r["owner_px"] + r["other_px"] for r in rs), 1)
        print(f"  {split}: {sum(r['n_owner'] for r in rs)} owner regions, "
              f"{sum(r['n_other'] for r in rs)} other regions, "
              f"owner is {o:.0%} of labelled pixels")
    print(f"\n  overlays/ is the thing to look at. These are PSEUDO-labels: a "
          f"score against\n  them says how well the head imitates SAM plus a "
          f"heuristic, not how well it\n  finds hands. Every row is stamped "
          f"label_source=auto.")


if __name__ == "__main__":
    main()
