"""One verdict per hand, or one verdict per frame: which is right more often.

OWNERSHIP IS NOT A PROPERTY OF A FRAME. A hand belongs to whoever it belongs
to for as long as it is in shot, and the pipeline currently re-decides that
thirty times a second, then spends a hysteresis, a two-hand cap and a coasting
bridge trying to undo the resulting flicker. Pooling the scores of a track and
answering once should be strictly better -- but nothing in this corpus could
test it, because every labelled package was sampled at a stride of thirteen or
seventeen frames and no two labelled hands were the same hand.

`track_audit` fixed that: thirty-four tracks over four recordings, each judged
once by a person, each carrying every frame it was seen in. Zero were judged
`mixed`, so no track pools two different hands, and zero were judged `not a
hand`, so admission at 0.60 has essentially perfect precision and it is recall
that is broken. That makes these tracks clean ground truth for the only
question here: given the same per-frame scores, does answering once per track
beat answering once per frame?

WHAT POOLING CANNOT DO. It cannot rescue a hand the model is wrong about on
every frame -- if the median frame says `owner`, so does the median. Its whole
value is on tracks where the frames disagree, so the number to read is not the
headline accuracy but how many tracks were internally inconsistent at all.
That is the ceiling on what any temporal aggregation can win here.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def load_pkg(pkg):
    """-> (tracks, rows). Labels live per track; crops live per frame."""
    tracks = {}
    for r in csv.DictReader(open(os.path.join(pkg, "tracks.csv"),
                                 encoding="utf-8-sig")):
        if r.get("label"):
            tracks[int(r["tid"])] = r["label"]
    rows = []
    for r in csv.DictReader(open(os.path.join(pkg, "hands.csv"),
                                 encoding="utf-8-sig")):
        tid = int(r["tid"])
        if tid not in tracks:
            continue
        crop = os.path.join(pkg, "crops", r["stem"] + ".jpg")
        ctx = os.path.join(pkg, "context", r["stem"] + ".jpg")
        if not os.path.exists(crop):
            continue
        rows.append({"tid": tid, "stem": r["stem"], "_crop": crop,
                     "_ctx": ctx, "raw": r, "pkg": pkg,
                     "tag": os.path.basename(pkg).replace("trackpkg_", ""),
                     "y": 1 if tracks[tid] == "owner" else 0})
    return tracks, rows


def merge_labels(pkg, csv_path):
    """Write a downloaded sheet's track labels back into tracks.csv. -> n"""
    got = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        got[int(r["tid"])] = r["label"]
    q = os.path.join(pkg, "tracks.csv")
    rows = list(csv.DictReader(open(q, encoding="utf-8-sig")))
    n = 0
    for r in rows:
        lab = got.get(int(r["tid"]))
        if lab:
            r["label"] = lab
            n += 1
    with open(q, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {n} track labels -> {q}")
    return n


def scores(pred, true):
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


def frame_scores(rows, clf=None, clf_ctx=None):
    """-> np.array of P(owner) per row, from whichever model is given."""
    import cv2
    if clf_ctx:
        from src.rig import own_ctx
        model, device, _ = own_ctx.load_model(clf_ctx)
        if model is None:
            raise SystemExit(f"no checkpoint at {clf_ctx}")
        import torch
        ds = own_ctx.Pairs([], strip=False)
        out = []
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            hs, cs, gs = [], [], []
            for r in chunk:
                hand = cv2.imread(r["_crop"])
                ctximg = cv2.imread(r["_ctx"])
                box = tuple(float(r["raw"][c]) for c in
                            ("box_cx", "box_cy", "box_w", "box_h"))
                c = (own_ctx.context_crop(ctximg, box, own_ctx.CTX_SCALE)
                     if ctximg is not None else None)
                if c is None or c.size == 0:
                    c = hand
                hs.append(ds._prep(hand, own_ctx.HAND_PX))
                cs.append(ds._prep(c, own_ctx.CTX_PX))
                gs.append(torch.tensor(
                    [float(r["raw"][k]) for k in own_ctx.GEOM],
                    dtype=torch.float32))
            with torch.no_grad():
                p = torch.softmax(
                    model(torch.stack(hs).to(device),
                          torch.stack(cs).to(device),
                          torch.stack(gs).to(device)), 1)[:, 1]
            out.append(p.cpu().numpy())
        return np.concatenate(out)

    from src.rig import own_cnn
    model, device = own_cnn.load_model(clf)
    if model is None:
        raise SystemExit(f"no checkpoint at {clf}")
    import torch
    out = []
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        xs = []
        for r in chunk:
            im = cv2.imread(r["_crop"])
            x = torch.from_numpy(np.ascontiguousarray(
                cv2.resize(im, (own_cnn.SIZE, own_cnn.SIZE))[:, :, ::-1]
            ).astype(np.float32) / 255.0).permute(2, 0, 1)
            xs.append((x - 0.45) / 0.25)
        with torch.no_grad():
            p = torch.softmax(model(torch.stack(xs).to(device)), 1)[:, 1]
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def report(rows, p, name):
    """Frame-level against track-pooled, on the same scores. -> None"""
    by = {}
    for r, x in zip(rows, p):
        by.setdefault((r["tag"], r["tid"]), {"y": r["y"], "p": []})["p"].append(
            float(x))

    fr_pred = [1 if x >= 0.5 else 0 for x in p]
    fr_true = [r["y"] for r in rows]
    fr = scores(fr_pred, fr_true)

    tids = sorted(by)
    tt = [by[k]["y"] for k in tids]
    med = [1 if float(np.median(by[k]["p"])) >= 0.5 else 0 for k in tids]
    mean = [1 if float(np.mean(by[k]["p"])) >= 0.5 else 0 for k in tids]
    vote = [1 if np.mean([x >= 0.5 for x in by[k]["p"]]) >= 0.5 else 0
            for k in tids]
    # What the pipeline effectively delivers today: a per-frame verdict, so a
    # track is right only on the frames it is right on.
    per_track_frac = [np.mean([(x >= 0.5) == by[k]["y"]
                               for x in by[k]["p"]]) for k in tids]
    split = [k for k in tids
             if 0 < np.mean([x >= 0.5 for x in by[k]["p"]]) < 1]

    print(f"\n=== {name} ===")
    print(f"  {'unit':<22} {'n':>4} {'other':>6} {'acc':>7} {'prec':>7} "
          f"{'rec':>7} {'f1':>7}")
    print(f"  {'frame-level (今天)':<22} {fr['n']:>4} {fr['other']:>6} "
          f"{fr['acc']:>7.3f} {fr['prec']:>7.3f} {fr['rec']:>7.3f} "
          f"{fr['f1']:>7.3f}")
    for lab, pred in (("track median", med), ("track mean", mean),
                      ("track majority", vote)):
        s = scores(pred, tt)
        print(f"  {lab:<22} {s['n']:>4} {s['other']:>6} {s['acc']:>7.3f} "
              f"{s['prec']:>7.3f} {s['rec']:>7.3f} {s['f1']:>7.3f}")
    print(f"\n  {len(split)} of {len(tids)} tracks answered BOTH ways across "
          f"their own frames.")
    print(f"    Pooling can only act on those; on the rest it repeats the "
          f"frame verdict.\n    Per-track fraction of frames correct: median "
          f"{np.median(per_track_frac):.3f}, "
          f"worst {min(per_track_frac):.3f}")
    if split:
        worst = sorted(split, key=lambda k: np.mean(
            [(x >= 0.5) == by[k]["y"] for x in by[k]["p"]]))[:4]
        print("    most inconsistent: " + ", ".join(
            f"{t}#{i} {np.mean([(x >= 0.5) == by[(t, i)]['y'] for x in by[(t, i)]['p']]):.2f}"
            for t, i in worst))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", action="append", required=True)
    ap.add_argument("--merge", action="append", default=[],
                    help="pkg=csv, to write a downloaded sheet back first")
    ap.add_argument("--clf")
    ap.add_argument("--clf_ctx")
    a = ap.parse_args()

    for spec in a.merge:
        pkg, path = spec.split("=", 1)
        merge_labels(pkg, path)

    rows, tracks = [], 0
    for p in a.pkg:
        t, r = load_pkg(p)
        tracks += len(t)
        rows += r
    if not rows:
        raise SystemExit("no labelled tracks with crops")
    print(f"  {tracks} labelled tracks, {len(rows)} crops, "
          f"{sum(1 for r in rows if r['y'] == 0)} crops on foreign hands")

    if a.clf:
        report(rows, frame_scores(rows, clf=a.clf), "hand-only CNN")
    if a.clf_ctx:
        report(rows, frame_scores(rows, clf_ctx=a.clf_ctx),
               "hand+context+geometry")
    print("\n  Track rows are the unit a person judged, so the track lines "
          "are measured\n  against ground truth directly. The frame line is "
          "the SAME truth spread over\n  that track's frames -- it is what "
          "the pipeline delivers today, not a second\n  labelling.")


if __name__ == "__main__":
    main()
