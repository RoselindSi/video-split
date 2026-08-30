"""Whose hand is it -- labelled by a person, then learned, per detection.

The rule this replaces asked where the forearm's ray leaves the frame and
called a low exit the wearer's. It works most of the time and it has been
patched once already: the wearer's right hand, far over, leaves through the
lower RIGHT edge rather than the bottom, and 4 of 48 real detections were
blurred because of it. Of 117 detections on one recording, 21 did not exit
through the bottom at all. Every version of that rule is a threshold through
two distributions that overlap whenever the wearer leans, raises a hand, or
the camera tilts.

SO THE OWNERSHIP DECISION IS LEARNED AND THE REST IS NOT. The detector already
finds hands with no dropouts at 0.83-0.87, and GrabCut already turns a box
into a boundary. Neither needs replacing. A full segmentation model would have
to relearn "where is the hand", which is the part that works, and would need
masks instead of clicks -- an order of magnitude more annotation for the same
decision.

    detector    where the hands are        works, unchanged
    GrabCut     which pixels               works, unchanged
    THIS        whose hand it is           the only part that is wrong

THE LABEL IS ONE KEYPRESS PER HAND. Drawing a mask took ten minutes a frame;
saying "mine" or "not mine" about a hand the detector has already boxed takes
a second. Several hundred labels is an hour, and several hundred is what a
fifteen-feature classifier needs.

AND THE RULE'S VERDICT IS RECORDED ALONGSIDE THE HUMAN'S. Until now the
evidence that the rule is wrong has been indirect -- a patch it needed, a
count of unusual exits. Labelling produces the direct number: how often a
person disagrees with it, and on which kinds of hand.
"""
from __future__ import annotations

import csv
import os

import numpy as np

# Sampled blind, stratified over time. Not where the rule is uncertain: a
# training set drawn from a model's own doubts teaches it its own blind spots,
# and a test set drawn that way measures the model's idea of difficulty rather
# than the task's.
FRAMES_PER_RECORDING = 60

FEATURES = ("wrist_x", "wrist_y", "box_cx", "box_cy", "box_w", "box_h",
            "dir_x", "dir_y", "exit_x", "exit_y",
            "edge_bottom", "edge_left", "edge_right", "edge_top",
            "is_left_hand", "conf", "hand_span")


def features(det, shape):
    """One detection -> a fixed-length vector. Everything normalised by the
    frame, so a hand near the lens and one across the bench are comparable."""
    H, W = shape[:2]
    kp = np.asarray(det["kp"], float)
    x0, y0, x1, y1 = [float(v) for v in det["box"]]
    wrist = kp[0]
    d = wrist - kp[1:21].mean(0)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.zeros(2)
    ex = det.get("exit")
    ex = (np.array([np.nan, np.nan]) if ex is None else np.asarray(ex, float))
    e = det.get("edge")
    span = float(np.linalg.norm(kp.max(0) - kp.min(0))) / max(W, 1)
    return np.array([
        wrist[0] / W, wrist[1] / H,
        (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H,
        d[0], d[1],
        (ex[0] / W if np.isfinite(ex[0]) else 0.0),
        (ex[1] / H if np.isfinite(ex[1]) else 0.0),
        float(e == "bottom"), float(e == "left"),
        float(e == "right"), float(e == "top"),
        float(det.get("side") == "left"), float(det["conf"]), span,
    ], dtype=np.float32)


def extract(rig, videos, out_dir, start, n_frames, model, stride=15,
            crop_px=192, tag="", verbose=True):
    """Render frames, detect, and write one crop plus one row per hand.

    Prints as it goes. The first version printed only on completion, and
    since it decodes stride*n frames from three videos before it is done --
    2700 for the defaults -- silence for ten minutes is indistinguishable
    from a hang."""
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.hand_detect import detect

    vcam = VirtualWideCamera.from_rig(rig)
    rd = ClipReader(rig, videos, start)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "context"), exist_ok=True)
    mc, rows = {}, []
    t0 = time.time()
    for k in range(n_frames):
        rgb = None
        src = rd.next(skip=(stride - 1) if k else 0)
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        dets = detect(model, rgb)
        H, W = rgb.shape[:2]
        for j, d in enumerate(dets):
            stem = f"{tag}f{start + k * stride:06d}_h{j}"
            x0, y0, x1, y1 = d["box"]
            pad = int(max(x1 - x0, y1 - y0) * 0.6)
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            cv2.imwrite(os.path.join(out_dir, "crops", stem + ".jpg"),
                        cv2.resize(rgb[cy0:cy1, cx0:cx1], (crop_px, crop_px)),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            # The whole frame with this hand marked. Ownership is a question
            # about the arm, and the arm mostly lives outside the box.
            ctx = rgb.copy()
            cv2.rectangle(ctx, (x0, y0), (x1, y1), (0, 255, 255), 4)
            if d.get("exit") is not None and np.isfinite(d["exit"]).all():
                cv2.line(ctx, tuple(np.asarray(d["kp"][0], int)),
                         tuple(np.asarray(d["exit"], int)), (255, 0, 255), 3)
            cv2.imwrite(os.path.join(out_dir, "context", stem + ".jpg"),
                        cv2.resize(ctx, (900, int(900 * H / W))),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            row = {"stem": stem, "frame": start + k * stride, "hand": j,
                   "side": d.get("side", ""), "conf": round(d["conf"], 3),
                   "edge": d.get("edge") or "", "rule_owner": int(
                       bool(d.get("owner"))), "label": ""}
            for name, v in zip(FEATURES, features(d, rgb.shape)):
                row[name] = float(v)
            rows.append(row)
        if verbose and ((k + 1) % 10 == 0 or k + 1 == n_frames):
            el = time.time() - t0
            print(f"    [{k+1}/{n_frames}] {len(rows)} hands, {el:.0f}s, "
                  f"{el/(k+1)*(n_frames-k-1):.0f}s left", flush=True)
    rd.close()
    if not rows:
        raise SystemExit(f"no hands found in {n_frames} frames from {start}")
    with open(os.path.join(out_dir, "hands.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def mine_disagreements(rig, videos, out_dir, start, n_frames, model, clf,
                       stride=15, crop_px=192, tag="", verbose=True):
    """Extract only the hands where the classifier and the rule disagree.

    MINED, SO TRAINING ONLY. Selecting frames by where a model is unsure
    teaches it its own blind spots and measures its own idea of difficulty; a
    score computed on mined frames is about the miner. The blind packages stay
    the evaluation set. This is the same split the census line already had to
    learn once, and it cost an eval split to learn it.

    Why it is worth mining at all: on 148 blind labels the two agree
    everywhere, so blind sampling would need thousands of hands to turn up the
    cases that separate them. The first one it did turn up -- a colleague's
    hand entering from the upper left, which the classifier called the
    wearer's at p=0.996 -- is exactly the combination the training set has
    none of: `edge=left` appears 12 times there and is `owner` every time."""
    import time
    import cv2
    from src.rig.geometry import VirtualWideCamera
    from src.rig.render_wide import render
    from src.rig.seam_fix import ClipReader
    from src.rig.hand_detect import detect

    vcam = VirtualWideCamera.from_rig(rig)
    rd = ClipReader(rig, videos, start)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "context"), exist_ok=True)
    mc, rows, t0, seen = {}, [], time.time(), 0
    for k in range(n_frames):
        src = rd.next(skip=(stride - 1) if k else 0)
        if not src:
            break
        try:
            rgb, _, _, _ = render(rig, vcam, src, 0.6, map_cache=mc)
        except TypeError:
            rgb, _, _, _ = render(rig, vcam, src, 0.6)
        dets = detect(model, rgb, clf=clf)
        seen += len(dets)
        H, W = rgb.shape[:2]
        for j, d in enumerate(dets):
            if bool(d.get("owner")) == bool(d.get("rule_owner")):
                continue
            stem = f"{tag}f{start + k * stride:06d}_h{j}"
            _write_sample(out_dir, stem, rgb, d, crop_px)
            row = {"stem": stem, "frame": start + k * stride, "hand": j,
                   "side": d.get("side", ""), "conf": round(d["conf"], 3),
                   "edge": d.get("edge") or "",
                   "rule_owner": int(bool(d.get("rule_owner"))),
                   "clf_owner": int(bool(d.get("owner"))),
                   "clf_p": round(float(d.get("owner_p", -1)), 4),
                   "label": ""}
            for name, v in zip(FEATURES, features(d, rgb.shape)):
                row[name] = float(v)
            rows.append(row)
        if verbose and ((k + 1) % 20 == 0 or k + 1 == n_frames):
            el = time.time() - t0
            print(f"    [{k+1}/{n_frames}] {seen} hands, {len(rows)} "
                  f"disagreements, {el:.0f}s", flush=True)
    rd.close()
    if not rows:
        print(f"  no disagreements in {seen} hands. The two decide alike "
              f"here; there is\n  nothing to mine and nothing to learn from "
              f"this segment.")
        return []
    with open(os.path.join(out_dir, "hands.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _write_sample(out_dir, stem, rgb, d, crop_px):
    """The crop and the whole frame with this hand marked."""
    import cv2
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = d["box"]
    pad = int(max(x1 - x0, y1 - y0) * 0.6)
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    cv2.imwrite(os.path.join(out_dir, "crops", stem + ".jpg"),
                cv2.resize(rgb[cy0:cy1, cx0:cx1], (crop_px, crop_px)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    ctx = rgb.copy()
    cv2.rectangle(ctx, (x0, y0), (x1, y1), (0, 255, 255), 4)
    if d.get("exit") is not None and np.isfinite(d["exit"]).all():
        cv2.line(ctx, tuple(np.asarray(d["kp"][0], int)),
                 tuple(np.asarray(d["exit"], int)), (255, 0, 255), 3)
    cv2.imwrite(os.path.join(out_dir, "context", stem + ".jpg"),
                cv2.resize(ctx, (900, int(900 * H / W))),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


def label_ui(pkg):
    """Keyboard labelling. o = mine, x = not mine, s = skip, u = undo, q."""
    import cv2
    path = os.path.join(pkg, "hands.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    todo = [i for i, r in enumerate(rows) if not str(r["label"]).strip()]
    if not todo:
        print("  everything is already labelled")
        return
    print(f"  {len(todo)} unlabelled of {len(rows)}\n"
          "  o = the WEARER's   x = SOMEBODY ELSE's   s = skip   "
          "u = undo   q = quit")
    print("  The rule's own verdict is deliberately NOT shown -- seeing it "
          "would make\n  the disagreement rate a measure of how persuasive it "
          "looks.\n")

    def save():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    pos, done = 0, 0
    while pos < len(todo):
        i = todo[pos]
        r = rows[i]
        crop = cv2.imread(os.path.join(pkg, "crops", r["stem"] + ".jpg"))
        ctx = cv2.imread(os.path.join(pkg, "context", r["stem"] + ".jpg"))
        if ctx is None:
            pos += 1
            continue
        h = ctx.shape[0]
        c = cv2.resize(crop, (h, h)) if crop is not None else np.zeros(
            (h, h, 3), np.uint8)
        vis = np.hstack([ctx, c])
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(vis, f"[{done+1}/{len(todo)}] {r['stem']}  "
                    f"{r['side']} {r['conf']}   o=mine  x=other  s=skip  u=undo",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("whose hand", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("u") and pos > 0:
            pos -= 1
            rows[todo[pos]]["label"] = ""
            done = max(0, done - 1)
            continue
        if k == ord("o"):
            r["label"] = "owner"
        elif k == ord("x"):
            r["label"] = "other"
        elif k == ord("s"):
            r["label"] = "skip"
        else:
            continue
        pos += 1
        done += 1
        if done % 10 == 0:
            save()
    save()
    cv2.destroyAllWindows()
    n = sum(1 for r in rows if r["label"] in ("owner", "other"))
    print(f"\n  {n} labelled, saved to {path}")


def label_grid(pkg, cols=3, rows_n=2, cell=460):
    """Six at a time: everything is the wearer's unless you say otherwise.

    79% of hands are the wearer's, so a keypress each spends four fifths of
    the effort confirming the obvious. Here a page defaults to `owner` and the
    digits mark the exceptions -- the annotator still LOOKS at every hand,
    which is what keeps the labels unbiased; only the typing is reduced."""
    import cv2
    path = os.path.join(pkg, "hands.csv")
    allrows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    todo = [i for i, r in enumerate(allrows) if not str(r["label"]).strip()]
    if not todo:
        print("  everything is already labelled")
        return
    per = cols * rows_n
    print(f"  {len(todo)} unlabelled, {per} per page\n"
          f"  1-{per} toggle a hand to OTHER   space = commit page   "
          f"b = back   q = quit\n"
          f"  Untouched hands are recorded as the WEARER'S. Look at every "
          f"one anyway --\n  the default is there to save typing, not "
          f"looking.\n")

    def save():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)

    page, done = 0, 0
    while page * per < len(todo):
        idx = todo[page * per:(page + 1) * per]
        marked = set()
        while True:
            tiles = []
            for n, i in enumerate(idx):
                r = allrows[i]
                im = cv2.imread(os.path.join(pkg, "context",
                                             r["stem"] + ".jpg"))
                if im is None:
                    im = np.zeros((cell, cell, 3), np.uint8)
                im = cv2.resize(im, (cell, int(cell * im.shape[0] /
                                               im.shape[1])))
                col = (0, 90, 255) if n in marked else (0, 200, 0)
                cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1),
                              col, 6)
                cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(im, f"{n+1}  {'OTHER' if n in marked else 'mine'}"
                            f"  {r['side']}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                tiles.append(im)
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT) for t in tiles]
            while len(tiles) < per:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[r_ * cols:(r_ + 1) * cols])
                              for r_ in range(rows_n)])
            bar = np.zeros((34, grid.shape[1], 3), np.uint8)
            cv2.putText(bar, f"page {page+1}/{-(-len(todo)//per)}   "
                        f"{done} labelled   space=commit  b=back  q=quit",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("whose hands", np.vstack([bar, grid]))
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                save()
                cv2.destroyAllWindows()
                print(f"\n  {done} labelled, saved")
                return
            if k == ord("b"):
                page = max(0, page - 1)
                break
            if k == ord(" "):
                for n, i in enumerate(idx):
                    allrows[i]["label"] = "other" if n in marked else "owner"
                done += len(idx)
                page += 1
                save()
                break
            if ord("1") <= k <= ord("9"):
                n = k - ord("1")
                if n < len(idx):
                    marked.symmetric_difference_update({n})
    save()
    cv2.destroyAllWindows()
    print(f"\n  {done} labelled, saved to {path}")


def report_rule(rows):
    """How often the human and the geometric rule disagree, and where."""
    lab = [r for r in rows if r.get("label") in ("owner", "other")]
    if not lab:
        print("  nothing labelled yet")
        return
    y = np.array([r["label"] == "owner" for r in lab])
    p = np.array([str(r["rule_owner"]) == "1" for r in lab])
    print(f"\n  {len(lab)} labelled   human says owner {y.mean():.1%}")
    print(f"  rule agrees with human on {(y == p).mean():.1%}")
    print(f"    rule says other, human says owner : {int((y & ~p).sum())}")
    print(f"    rule says owner, human says other : {int((~y & p).sum())}")
    by = {}
    for r, yy, pp in zip(lab, y, p):
        e = r.get("edge") or "?"
        by.setdefault(e, [0, 0])
        by[e][0] += 1
        by[e][1] += int(yy != pp)
    print(f"\n  {'forearm exit':<14}{'n':>5}{'rule wrong':>12}")
    for e, (n, w) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"  {e:<14}{n:>5}{w:>8} ({w/max(n,1):.0%})")
    print("\n  This is the number the rule never had. Every previous argument "
          "against it\n  was indirect: a patch it needed, a count of unusual "
          "exits.")


def train(rows, seed=0):
    """A small classifier on the geometric features. -> (model, report)

    FIFTEEN FEATURES AND NOT THE RAW KEYPOINTS. Forty more numbers from 21
    joint positions would let a few hundred labels be memorised; what
    ownership depends on is where the hand is and which way the arm runs, and
    those are already in here. The raw keypoints are kept in the package for
    when there are thousands of labels rather than hundreds."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    lab = [r for r in rows if r.get("label") in ("owner", "other")]
    if len(lab) < 40:
        raise SystemExit(f"only {len(lab)} labels; this needs a few hundred")
    n_oth = sum(1 for r in lab if r["label"] == "other")
    if n_oth == 0 or n_oth == len(lab):
        raise SystemExit(
            f"every label is the same class ({len(lab)} of them). A "
            f"classifier fitted to\n  one class predicts that class and "
            f"scores 100%, which is a fact about the\n  labels and not about "
            f"the model. Label a package that contains both --\n  the rule's "
            f"`rule_owner` column shows which ones might.")
    X = np.array([[float(r[f]) for f in FEATURES] for r in lab], np.float32)
    y = np.array([r["label"] == "owner" for r in lab], int)
    # Grouped by FRAME: two hands in one frame are not independent, and a
    # random split would put one in train and the other in test and report a
    # score that no new frame will reproduce.
    g = np.array([int(r["frame"]) for r in lab])
    rule = np.array([str(r["rule_owner"]) == "1" for r in lab], int)

    n_split = min(5, len(np.unique(g)))
    accs, rule_accs = [], []
    for tr, te in GroupKFold(n_splits=n_split).split(X, y, g):
        m = GradientBoostingClassifier(random_state=seed, max_depth=2,
                                       n_estimators=120)
        m.fit(X[tr], y[tr])
        accs.append(float((m.predict(X[te]) == y[te]).mean()))
        rule_accs.append(float((rule[te] == y[te]).mean()))
    # Does it differ from the rule on held-out data, or has it just learned
    # the rule? With 100% agreement on the labels the two are indistinguishable
    # by accuracy alone, and the honest way to say so is to count where their
    # PREDICTIONS differ rather than to report two identical scores.
    diff = 0
    for tr, te in GroupKFold(n_splits=n_split).split(X, y, g):
        m = GradientBoostingClassifier(random_state=seed, max_depth=2,
                                       n_estimators=120).fit(X[tr], y[tr])
        diff += int((m.predict(X[te]) != rule[te]).sum())
    final = GradientBoostingClassifier(random_state=seed, max_depth=2,
                                       n_estimators=120).fit(X, y)
    imp = sorted(zip(FEATURES, final.feature_importances_),
                 key=lambda kv: -kv[1])[:6]
    return final, {"n": len(lab), "n_other": int(n_oth),
                   "differs_from_rule": diff,
                   "owner_frac": float(y.mean()),
                   "cv_acc": float(np.mean(accs)),
                   "cv_acc_std": float(np.std(accs)),
                   "rule_acc": float(np.mean(rule_accs)),
                   "folds": n_split, "top_features": imp}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("extract", "mine", "label", "report",
                                     "train"))
    ap.add_argument("--clf", help="mine only: the classifier to disagree with")
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=FRAMES_PER_RECORDING)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--tag", default="")
    ap.add_argument("--also", action="append", default=[],
                    help="additional package directories to pool. Ownership "
                         "varies by workstation, so one recording's labels "
                         "describe one workstation.")
    ap.add_argument("--out", help="where to write own_clf.pkl (default: "
                                  "inside --pkg)")
    ap.add_argument("--grid", action="store_true",
                    help="label six at a time; a page defaults to the "
                         "wearer's and the digits mark the exceptions. 79%% "
                         "of hands are the wearer's, so this is most of the "
                         "typing removed and none of the looking.")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    a = ap.parse_args()

    if a.mode == "mine":
        if not a.calibration or not a.video or not a.clf:
            ap.error("mine needs --calibration, --video and --clf")
        from ultralytics import YOLO
        from src.rig.calibration import RigCalibration
        from src.rig.hand_detect import load_owner_clf
        rig = RigCalibration(a.calibration)
        rows = mine_disagreements(
            rig, dict(s.split("=", 1) for s in a.video), a.pkg, a.start, a.n,
            YOLO(a.weights), load_owner_clf(a.clf), a.stride, tag=a.tag)
        if rows:
            import collections
            c = collections.Counter(r["edge"] for r in rows)
            print(f"\n  {len(rows)} disagreements -> {a.pkg}   exits {dict(c)}")
            print("  MINED: these are training data only. A score computed on "
                  "frames chosen by\n  where a model was unsure is a score "
                  "about the model's own doubts.")
        return

    if a.mode == "extract":
        if not a.calibration or not a.video:
            ap.error("extract needs --calibration and --video")
        from ultralytics import YOLO
        from src.rig.calibration import RigCalibration
        rig = RigCalibration(a.calibration)
        rows = extract(rig, dict(s.split("=", 1) for s in a.video), a.pkg,
                       a.start, a.n, YOLO(a.weights), a.stride, tag=a.tag)
        n_rule = sum(r["rule_owner"] for r in rows)
        print(f"  {len(rows)} hands over {a.n} frames -> {a.pkg}")
        print(f"  the rule calls {n_rule} of them the wearer's "
              f"({n_rule/max(len(rows),1):.0%})")
        print("\n  Label them with `label`. The rule's verdict is stored but "
              "NOT shown while\n  labelling -- seeing it would turn the "
              "disagreement rate into a measure of\n  how convincing the rule "
              "looks.")
        return

    rows = list(csv.DictReader(open(os.path.join(a.pkg, "hands.csv"),
                                    encoding="utf-8-sig")))
    missing = []
    for extra in a.also:
        q = os.path.join(extra, "hands.csv")
        if not os.path.exists(q):
            # A package that failed to extract must not stop a training run
            # over the dozen that succeeded -- but it is named, because a
            # silently smaller dataset is how a score drifts without anyone
            # noticing which recordings are in it.
            missing.append(extra)
            continue
        rows += list(csv.DictReader(open(q, encoding="utf-8-sig")))
    if missing:
        print(f"  !! {len(missing)} package(s) have no hands.csv and were "
              f"skipped:")
        for m in missing:
            print(f"     {m}")
    if a.mode == "label":
        (label_grid(a.pkg) if a.grid else label_ui(a.pkg))
    elif a.mode == "report":
        report_rule(rows)
    else:
        import json
        import pickle
        m, rep = train(rows)
        report_rule(rows)
        print(f"\n  {rep['n']} labels ({rep['n_other']} other), "
              f"{rep['owner_frac']:.0%} owner, "
              f"{rep['folds']}-fold grouped BY FRAME")
        print(f"  learned  {rep['cv_acc']:.3f} +/- {rep['cv_acc_std']:.3f}")
        print(f"  rule     {rep['rule_acc']:.3f}")
        print(f"  the two make DIFFERENT predictions on "
              f"{rep['differs_from_rule']} of {rep['n']} held-out hands")
        if rep["differs_from_rule"] == 0:
            print("    -- so the classifier has reproduced the rule exactly. "
                  "It will behave\n       identically until it is given "
                  "labels the rule gets wrong.")
        print(f"  top features: " + ", ".join(f"{k} {v:.2f}"
                                              for k, v in rep["top_features"]))
        outp = a.out or os.path.join(a.pkg, "own_clf.pkl")
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "wb") as f:
            pickle.dump({"model": m, "features": FEATURES}, f)
        json.dump(rep, open(outp.replace(".pkl", ".json"), "w"),
                  indent=1, default=float)
        print(f"\n  wrote {outp}")
        print("  The comparison that matters is learned vs rule on the SAME "
              "held-out frames.\n  If they tie, the rule was fine and the "
              "labels bought a measurement rather\n  than a model -- which is "
              "still worth having, and cost an hour.")


if __name__ == "__main__":
    main()
