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
            crop_px=192, tag=""):
    """Render frames, detect, and write one crop plus one row per hand."""
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
        rd.close() if False else None
    rd.close()
    with open(os.path.join(out_dir, "hands.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


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
    final = GradientBoostingClassifier(random_state=seed, max_depth=2,
                                       n_estimators=120).fit(X, y)
    imp = sorted(zip(FEATURES, final.feature_importances_),
                 key=lambda kv: -kv[1])[:6]
    return final, {"n": len(lab), "owner_frac": float(y.mean()),
                   "cv_acc": float(np.mean(accs)),
                   "cv_acc_std": float(np.std(accs)),
                   "rule_acc": float(np.mean(rule_accs)),
                   "folds": n_split, "top_features": imp}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("extract", "label", "report", "train"))
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--calibration")
    ap.add_argument("--video", action="append", default=[],
                    metavar="FILEKEY=PATH")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--n", type=int, default=FRAMES_PER_RECORDING)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--tag", default="")
    ap.add_argument("--weights",
                    default="/shared/models/HaWoR/weights/external/detector.pt")
    a = ap.parse_args()

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
    if a.mode == "label":
        label_ui(a.pkg)
    elif a.mode == "report":
        report_rule(rows)
    else:
        import json
        import pickle
        m, rep = train(rows)
        report_rule(rows)
        print(f"\n  {rep['n']} labels, {rep['owner_frac']:.0%} owner, "
              f"{rep['folds']}-fold grouped BY FRAME")
        print(f"  learned  {rep['cv_acc']:.3f} +/- {rep['cv_acc_std']:.3f}")
        print(f"  rule     {rep['rule_acc']:.3f}")
        print(f"  top features: " + ", ".join(f"{k} {v:.2f}"
                                              for k, v in rep["top_features"]))
        with open(os.path.join(a.pkg, "own_clf.pkl"), "wb") as f:
            pickle.dump({"model": m, "features": FEATURES}, f)
        json.dump(rep, open(os.path.join(a.pkg, "own_clf.json"), "w"),
                  indent=1, default=float)
        print(f"\n  wrote {a.pkg}/own_clf.pkl")
        print("  The comparison that matters is learned vs rule on the SAME "
              "held-out frames.\n  If they tie, the rule was fine and the "
              "labels bought a measurement rather\n  than a model -- which is "
              "still worth having, and cost an hour.")


if __name__ == "__main__":
    main()
